"""Compose generated waveforms into a 200 MHz / 245.76 MSps SigMF recording.

Reads a YAML config describing frequency blocks (60 MHz pitch, 50 MHz usable,
5 MHz guards) and the waveforms in each. For every block it lays out, in time:
    [ ZC ] -> [ OFDM metadata burst ] -> [ data: stacked waveforms ]
frequency-shifts the block to its center, and sums all blocks. Writes:
    <name>.sigmf-data / <name>.sigmf-meta            -> TX truth (full annotations)
    <name>.rx.sigmf-data / <name>.rx.sigmf-meta      -> "received" (no annotations)

The metadata burst encodes each waveform's class, variation, position (relative
to the ZC), bandwidth, and power so the receiver can rebuild the annotations.
"""

from __future__ import annotations
import os
import json
from functools import lru_cache
import numpy as np
import yaml
from scipy.io import loadmat
from sigmf import SigMFFile

import geometry as geo
import protocol as proto
from zc_sync import load_zc, freq_shift

FS = geo.FS


# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def load_waveform(mat_path):
    """Return (iq complex64, info dict) for a library waveform .mat + .json.
    Cached: a waveform reused across durations/groups is read from disk once.
    Callers must treat the returned arrays as read-only (the composer only ever
    copies via tile/cut/peak-norm/freq-shift)."""
    iq = np.asarray(loadmat(mat_path)["f_sig"]).ravel().astype(np.complex64)
    json_path = os.path.splitext(mat_path)[0] + ".json"
    with open(json_path) as f:
        md = json.load(f)
    info = {
        "class": str(md.get("class", "Unknown")),
        "name": str(md.get("waveformName", os.path.basename(mat_path))),
        "variation": str(md.get("variationsExplored", md.get("modulationSetting", ""))),
        "occ_bw_hz": float(md.get("designedOccupiedBandwidthHz", 0.0)),
        "modulation": str(md.get("modulation", "")),
    }
    return iq, info


def _rms(x):
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def _scale_rms(x, target):
    r = _rms(x)
    return x * (target / r) if r > 0 else x


def _peak_norm(x):
    p = float(np.max(np.abs(x)))
    return x / p if p > 0 else x


def active_extent(iq, thresh_db=-60.0):
    """First sample offset and count of the active (above -thresh_db) region."""
    a = np.abs(iq)
    pk = float(a.max())
    if pk <= 0:
        return 0, int(iq.size)
    idx = np.where(a > pk * 10 ** (thresh_db / 20.0))[0]
    if idx.size == 0:
        return 0, int(iq.size)
    return int(idx[0]), int(idx[-1] - idx[0] + 1)


def _target_samples(w):
    """Target sample count for a waveform from duration_ms/duration_samples, or None."""
    if w.get("duration_ms") is not None:
        return int(round(float(w["duration_ms"]) * 1e-3 * FS))
    if w.get("duration_samples") is not None:
        return int(w["duration_samples"])
    return None


def _placed_len(w):
    t = _target_samples(w)
    return t if t is not None else int(w["iq"].size)


def _tile_or_cut(iq, target):
    """Scale a waveform to `target` samples: cut if longer, tile (repeat) if shorter."""
    if iq.size >= target:
        return iq[:target].copy()
    reps = int(np.ceil(target / iq.size))
    return np.tile(iq, reps)[:target]


def parse_groups(cfg):
    """Return a list of time groups; each is a list of frequency blocks.

    Supports `time_groups: [{blocks: [...]}, ...]` and the single-group shorthand
    `blocks: [...]` (one time group)."""
    if cfg.get("time_groups"):
        return [g["blocks"] for g in cfg["time_groups"]]
    return [cfg["blocks"]]


# --------------------------------------------------------------------------- #
def plan_group(blocks, lib):
    """Parse + validate the frequency blocks of one time group into entries."""
    entries = []
    used_slots = {}
    for bi, block in enumerate(blocks):
        slots = sorted(block["slots"])
        for s in slots:
            if s < 0 or s > 2:
                raise ValueError(f"block {bi}: slot {s} out of range 0..2")
            if s in used_slots:
                raise ValueError(
                    f"block {bi}: slot {s} already used by block {used_slots[s]}; "
                    f"each frequency slot may hold only one block")
            used_slots[s] = bi
        if len(slots) > 1 and not geo.contiguous(slots):
            raise ValueError(f"block {bi}: multi-slot blocks must be contiguous, got {slots}")

        wfs = []
        for w in block["waveforms"]:
            iq, info = load_waveform(os.path.join(lib, w["mat"]))
            wfs.append({"iq": iq, "info": info,
                        "power_db": float(w.get("power_db", 0.0)),
                        "duration_ms": w.get("duration_ms"),
                        "duration_samples": w.get("duration_samples"),
                        "mat": w["mat"]})

        fc = geo.merged_center(slots)
        if len(slots) > 1:
            if len(wfs) != 1:
                raise ValueError(
                    f"block {bi}: a multi-slot (wide) block must contain exactly one "
                    f"waveform, got {len(wfs)}; split them into separate blocks")
            occ = wfs[0]["info"]["occ_bw_hz"]
            need = geo.required_blocks(occ)
            if need != len(slots):
                raise ValueError(
                    f"block {bi}: waveform '{wfs[0]['info']['name']}' occupies "
                    f"{occ/1e6:.1f} MHz which needs {need} slot(s), but {len(slots)} "
                    f"were assigned; use {need} contiguous slots")
            centers = [fc]
        else:
            occ = max(w["info"]["occ_bw_hz"] for w in wfs)
            if occ > geo.usable_bw(1) + 1.0:
                raise ValueError(
                    f"block {bi}: waveform occupies {occ/1e6:.1f} MHz > 50 MHz; assign "
                    f"it a multi-slot block (set 'slots' to {geo.required_blocks(occ)} "
                    f"contiguous indices)")
            centers = geo.stack_layout([w["info"]["occ_bw_hz"] for w in wfs],
                                       fc, geo.usable_bw(1))
        entries.append({"slots": slots, "fc": fc, "wfs": wfs, "centers": centers})
    return entries


# --------------------------------------------------------------------------- #
def compose(config_path):
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name", "composite")
    out_dir = os.path.join(cfg_dir, cfg.get("output_dir", "composites"))
    os.makedirs(out_dir, exist_ok=True)
    ref_rms = float(cfg.get("ref_rms", 0.15))
    peak_norm = float(cfg.get("peak_normalize", 0.95))

    lib = os.path.join(cfg_dir, cfg.get("library_root", "."))
    # guard1: metadata -> waveform; guard2: waveform -> next group's ZC
    guard1 = int(round(float(cfg.get("guard1_ms", 2.5)) * 1e-3 * FS))
    guard2 = int(round(float(cfg.get("guard2_ms", cfg.get("guard_time_ms", 5.0))) * 1e-3 * FS))
    power_mode = str(cfg.get("power_mode", "peak_db"))   # "peak_db" or "original"
    profile = str(cfg.get("metadata_profile", proto.DEFAULT_PROFILE))  # "fast" or "robust"
    # Global TX-waveform power offset (dB), applied to every data waveform's
    # amplitude *only* -- the ZC and metadata bursts (at ref_rms) are untouched.
    # The final peak-normalize scales the whole composite uniformly, so this
    # exactly sets how far the waveforms sit below the ZC/metadata reference.
    wf_gain_db = float(cfg.get("waveform_gain_db", 0.0))
    wf_gain = 10.0 ** (wf_gain_db / 20.0)

    zc = load_zc(os.path.join(cfg_dir, cfg["zc_file"]))
    zc_len = zc.size
    meta_start = proto.meta_start(zc_len)
    data_start = proto.data_start(zc_len, guard1, profile)
    zc_scaled = _scale_rms(zc, ref_rms)

    groups = parse_groups(cfg)
    planned = [plan_group(blocks, lib) for blocks in groups]

    # group length = data_start + longest (tile/cut-scaled) waveform; groups are
    # concatenated in time separated by guard2 samples of silence.
    group_lens = []
    for entries in planned:
        maxlen = max((_placed_len(w) for e in entries for w in e["wfs"]), default=0)
        group_lens.append(data_start + maxlen)
    starts, cur = [], 0
    for i, gl in enumerate(group_lens):
        starts.append(cur)
        cur += gl + (guard2 if i < len(group_lens) - 1 else 0)
    total = cur
    comp = np.zeros(total, dtype=np.complex64)

    annotations = []
    nwf = 0
    for gi, (entries, gstart) in enumerate(zip(planned, starts)):
        for e in entries:
            fc = e["fc"]
            # ---- ZC ----
            comp[gstart:gstart + zc_len] += freq_shift(zc_scaled, fc, FS)
            annotations.append(dict(start=gstart, length=zc_len, label="ZC",
                                    flo=fc - 25e6, fhi=fc + 25e6,
                                    extra={"wfgt:kind": "zadoff_chu",
                                           "wfgt:block_center_hz": fc,
                                           "wfgt:time_group": gi}))
            # ---- data waveforms (tiled/cut to target length if requested) ----
            descs = []
            for w, cwf in zip(e["wfs"], e["centers"]):
                orig_len = int(w["iq"].size)
                tgt = _target_samples(w)
                if tgt is not None:
                    iqp = _tile_or_cut(w["iq"], tgt)
                    rs, n = data_start, iqp.size            # box spans the new length
                else:
                    iqp = w["iq"]
                    a0, an = active_extent(iqp)
                    rs, n = data_start + a0, an             # box spans the active extent
                if power_mode == "original":
                    x = iqp.astype(np.complex64)            # as-stored amplitude
                else:
                    x = (_peak_norm(iqp) * (10 ** (float(w.get("power_db", 0.0)) / 20.0))).astype(np.complex64)
                if wf_gain != 1.0:
                    x = (x * np.complex64(wf_gain)).astype(np.complex64)
                comp[gstart + data_start:gstart + data_start + x.size] += freq_shift(x, cwf, FS)
                occ = w["info"]["occ_bw_hz"]
                nwf += 1
                eff_pdb = float(w.get("power_db", 0.0)) + wf_gain_db   # placed power label
                descs.append({"cls": w["info"]["class"], "var": w["info"]["name"],
                              "rs": int(rs), "n": int(n), "ol": orig_len,
                              "fc": float(cwf), "bw": float(occ),
                              "pdb": eff_pdb})
                annotations.append(dict(
                    start=gstart + rs, length=int(n),
                    label=w["info"]["class"], flo=cwf - occ / 2, fhi=cwf + occ / 2,
                    extra={"wfgt:kind": "waveform",
                           "wfgt:class": w["info"]["class"],
                           "wfgt:variation": w["info"]["name"],
                           "wfgt:modulation": w["info"]["modulation"],
                           "wfgt:occupied_bw_hz": occ,
                           "wfgt:power_db": eff_pdb,
                           "wfgt:length_samples": int(n),
                           "wfgt:original_length_samples": orig_len,
                           "wfgt:block_center_hz": fc,
                           "wfgt:time_group": gi,
                           "wfgt:source_mat": w["mat"]}))
            # ---- metadata burst ----
            descriptor = {"v": 1, "fc": float(fc), "slots": e["slots"], "g": gi, "wfs": descs}
            mb = _scale_rms(proto.encode_metadata(descriptor, profile), ref_rms)
            comp[gstart + meta_start:gstart + meta_start + mb.size] += freq_shift(mb, fc, FS)
            annotations.append(dict(start=gstart + meta_start, length=int(mb.size), label="METADATA",
                                    flo=fc - 13e6, fhi=fc + 13e6,
                                    extra={"wfgt:kind": "metadata",
                                           "wfgt:block_center_hz": fc,
                                           "wfgt:time_group": gi,
                                           "wfgt:num_waveforms": len(descs)}))

    # ---- peak-normalize the whole composite (preserves relative powers) ----
    pk = float(np.max(np.abs(comp)))
    if peak_norm > 0 and pk > 0:
        comp *= (peak_norm / pk)
    comp = comp.astype(np.complex64)

    base = os.path.join(out_dir, name)
    rx_meta = base + ".rx.sigmf-meta"
    _write_sigmf(base, comp, FS, annotations, cfg)        # TX truth pair
    _write_bare_rx_meta(rx_meta, base + ".sigmf-data", FS) # bare RX meta (uses TX data for clean Rx)

    summary = {"name": name, "samples": total, "duration_s": total / FS,
               "peak": pk, "n_groups": len(planned), "n_waveforms": nwf,
               "n_annotations": len(annotations),
               "tx_meta": base + ".sigmf-meta", "rx_meta": rx_meta,
               "data": base + ".sigmf-data"}
    print(f"composed '{name}': {total} samples ({total/FS*1e3:.2f} ms), "
          f"{len(planned)} time groups, {nwf} waveforms, {len(annotations)} annotations, peak {pk:.3f}")
    print(f"  TX truth : {base}.sigmf-meta")
    print(f"  RX (bare): {rx_meta}")
    return summary


def _write_sigmf(base, iq, fs, annotations, cfg):
    data_path = base + ".sigmf-data"
    iq.astype("<c8").tofile(data_path)
    meta = SigMFFile(data_file=data_path, global_info={
        "core:datatype": "cf32_le", "core:sample_rate": fs, "core:version": "1.0.0",
        "core:description": "200 MHz composite of generated waveforms (TX ground truth)",
        "core:author": "compose.py"})
    meta.add_capture(0, metadata={"core:frequency": 0.0})
    for a in annotations:
        md = {"core:freq_lower_edge": a["flo"], "core:freq_upper_edge": a["fhi"],
              "core:label": a["label"]}
        md.update(a.get("extra", {}))
        meta.add_annotation(a["start"], a["length"], metadata=md)
    meta.tofile(base + ".sigmf-meta", overwrite=True)


def _write_bare_rx_meta(meta_path, data_path, fs):
    """Write a receiver-side .sigmf-meta with only receiver params, no annotations.
    Pairs with `data_path` (the clean TX data, or a channel-applied capture)."""
    n = os.path.getsize(data_path) // 8  # complex64 = 8 bytes/sample
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": fs,
                   "core:version": "1.0.0", "core:num_channels": 1,
                   "core:description": "Received 200 MHz composite (annotations to be recovered)"},
        "captures": [{"core:sample_start": 0, "core:frequency": 0.0,
                      "rx:hardware": "simulated USRP front-end", "rx:rf_center_freq_hz": 0.0}],
        "annotations": [],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    import sys
    compose(sys.argv[1])
