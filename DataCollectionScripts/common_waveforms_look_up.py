#!/usr/bin/env python3
"""Coarse estimated labels for detector regions, looked up in the common-waveforms manifest.

The Holohub signal detector (Holohub-Signal-Detection, applications/usrp_wideband_signal_detection) marks
every emission as "signal". This tool measures each detected region (absolute centre frequency and
bandwidth, duty / persistence across frames, burst length, and - when the raw IQ is at hand - the in-band
spectral shape) and compares it with every entry of common_ota_waveforms_manifest_usa.yaml. When exactly
one entry matches closely, the region gets an *estimated* label prefixed with "~" (e.g. "~FM_broadcast");
everything unusual, off its normal band or ambiguous stays "signal".

Inputs (pick one)
  <dir>_preprocessed            output of preprocess_dataset.py (snippets/*.sigmf-meta + manifest.csv)
  <capture>.sigmf-meta          any SigMF recording whose annotations carry core:freq_lower_edge/upper_edge
  <capture folder> --masks M    masks/<M>.npz written by dino_fine_tuning/src/run_ota_detections.py

Outputs
  lookup_regions.csv / lookup_summary.json next to the input (or --report-dir), one row per region with the
  measured features, the best entry, its confidence and the evidence. Unless --no-write, the estimated label
  is also written into the SigMF annotations: core:label = "~<label>" (the detector's own label is kept in
  argus:detector_label, together with argus:estimate_confidence / argus:estimate_entry / argus:estimate_evidence),
  and the label column of preprocess_dataset.py's manifest.csv files is updated.

Examples
  python3 common_waveforms_look_up.py ~/captures/site_a_preprocessed
  python3 common_waveforms_look_up.py ~/captures/site_a/n320_fm_*/n320_fm_*.sigmf-meta --no-write
  python3 common_waveforms_look_up.py ~/Datasets/ARGUS/OOD_captures/OTA/n320_hdtv_uhf_565-585MHz_c575M_r20M_g55_ch1 --masks M12S
  python3 common_waveforms_look_up.py --check-manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest_io as mio  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "common_ota_waveforms_manifest_usa.yaml"
VERSION = "common_waveforms_look_up.py 1.0 (ARGUS-RFAI-Dataset)"
ESTIMATE_PREFIX = "~"


# =========================================================================================== regions
@dataclass
class Region:
    capture: str
    f_lo: float                 # absolute Hz
    f_hi: float
    t0: float                   # s from the start of the capture
    dur: float                  # s
    frame: int                  # frame number in the capture (512-row frames)
    rate: float                 # capture sample rate
    fc_cap: float               # capture centre frequency
    bin_hz: float               # detector frequency resolution
    row_s: float                # detector time resolution
    frame_s: float
    meta_path: Path | None = None
    ann_index: int | None = None
    det_label: str = ""
    track: int = -1

    @property
    def fc(self) -> float:
        return 0.5 * (self.f_lo + self.f_hi)

    @property
    def bw(self) -> float:
        return self.f_hi - self.f_lo


@dataclass
class Track:
    tid: int
    regions: list[Region]                 # every member region (incl. absorbed edge-flicker fragments)
    core: list[Region] = field(default_factory=list)   # the regions that define the band (before absorption)
    fc: float = 0.0
    bw: float = 0.0
    f_lo: float = 0.0
    f_hi: float = 0.0
    duty: float = 0.0            # median per-frame fraction of the frame rows covered
    persistence: float = 0.0     # fraction of analysed frames in which the track appears
    burst_s: float = 0.0         # median single-region duration
    bursts_per_frame: float = 0.0
    n_frames_total: int = 1
    temporal: str = "unknown"    # continuous | tdd | intermittent | bursty
    cut: str = ""                # "lo" / "hi" when the box touches the capture edge (bandwidth is a lower bound)
    forced_label: str = ""       # set by cross-track passes (spur comb)
    forced_evidence: str = ""
    iq: dict = field(default_factory=dict)

    @property
    def rate(self):
        return self.regions[0].rate

    @property
    def bin_hz(self):
        return self.regions[0].bin_hz

    @property
    def row_s(self):
        return self.regions[0].row_s

    @property
    def fc_cap(self):
        return self.regions[0].fc_cap


def _track_stats(t: Track, n_frames_total: int):
    # band edges: medians for a few regions; for many (a carrier whose box flickers at the edges, or a host that
    # absorbed its fragments) the 15th / 85th percentiles of the member edges, i.e. close to the union
    core = t.core or t.regions
    los = np.array([x.f_lo for x in core])
    his = np.array([x.f_hi for x in core])
    if len(los) >= 5:
        t.f_lo, t.f_hi = float(np.percentile(los, 15)), float(np.percentile(his, 85))
    else:
        t.f_lo, t.f_hi = float(np.median(los)), float(np.median(his))
    t.fc = 0.5 * (t.f_lo + t.f_hi)
    t.bw = t.f_hi - t.f_lo
    frames = defaultdict(float)
    counts = Counter()
    for r in t.regions:
        frames[r.frame] += r.dur
        counts[r.frame] += 1
    frame_s = t.regions[0].frame_s
    duties = [min(1.0, d / frame_s) for d in frames.values()]
    t.duty = float(np.median(duties))
    t.n_frames_total = max(n_frames_total, len(frames))
    t.persistence = len(frames) / t.n_frames_total
    t.burst_s = float(np.median([r.dur for r in t.regions]))
    t.bursts_per_frame = float(np.mean(list(counts.values())))
    if t.duty >= 0.85 and t.persistence >= 0.75:
        t.temporal = "continuous"
    elif t.duty >= 0.55 and t.persistence >= 0.75:
        t.temporal = "tdd"
    elif t.duty >= 0.9 and len(frames) >= 2:
        t.temporal = "sporadic_full"      # fills every frame it is seen in, but a weak carrier the detector drops in other frames
    elif t.duty < 0.55 or t.persistence < 0.5:
        t.temporal = "bursty"
    else:
        t.temporal = "intermittent"
    r0 = t.regions[0]
    cap_lo, cap_hi = r0.fc_cap - r0.rate / 2, r0.fc_cap + r0.rate / 2
    t.cut = "lo" if t.f_lo <= cap_lo + 1.01 * r0.bin_hz else ("hi" if t.f_hi >= cap_hi - 1.01 * r0.bin_hz else "")


def build_tracks(regions: list[Region], n_frames_total: int, iou_min: float = 0.5) -> list[Track]:
    """Group regions across frames by frequency overlap (IoU of the band edges), then absorb edge-flicker
    fragments (short-lived slivers touching a continuous carrier) into their host track."""
    tracks: list[Track] = []
    for r in sorted(regions, key=lambda x: (x.fc, x.frame)):
        best, best_iou = None, 0.0
        for t in tracks:
            inter = min(t.f_hi, r.f_hi) - max(t.f_lo, r.f_lo)
            if inter <= 0:
                continue
            union = max(t.f_hi, r.f_hi) - min(t.f_lo, r.f_lo)
            iou = inter / union
            if iou > best_iou:
                best, best_iou = t, iou
        if best is not None and best_iou >= iou_min:
            best.regions.append(r)
            best.f_lo = float(np.median([x.f_lo for x in best.regions]))
            best.f_hi = float(np.median([x.f_hi for x in best.regions]))
        else:
            tracks.append(Track(len(tracks), [r], f_lo=r.f_lo, f_hi=r.f_hi))
    for t in tracks:
        t.core = list(t.regions)
        _track_stats(t, n_frames_total)
    # second pass: fragments -> continuous hosts (they join the host's regions for duty / persistence but
    # not its core, so the host's band stays that of the carrier, not of the slivers)
    hosts = [t for t in tracks if t.temporal == "continuous"]
    if hosts:
        keep = []
        for t in tracks:
            if t.temporal == "continuous":
                keep.append(t)
                continue
            bin_hz = t.bin_hz
            host = None
            for h in hosts:
                overlap = min(h.f_hi, t.f_hi) - max(h.f_lo, t.f_lo)
                gap = -overlap
                if overlap >= 0.5 * min(h.bw, t.bw) or (gap <= 1.0 * bin_hz and t.bw <= 4 * bin_hz):
                    if t.persistence < 0.75 or t.bw <= 4 * bin_hz:
                        host = h
                        break
            if host is not None:
                host.regions.extend(t.regions)
            else:
                keep.append(t)
        tracks = keep
        for t in hosts:
            _track_stats(t, n_frames_total)
    for i, t in enumerate(tracks):
        t.tid = i
        for r in t.regions:
            r.track = i
    return tracks


def comb_pass(tracks: list[Track], manifest: dict, explained: set | None = None):
    """Cross-track: a regular comb of >= min_lines narrow persistent lines on one spacing -> spur_comb.

    `explained` holds the ids of tracks a manifest entry already explains with high confidence (e.g. FM
    stations, which sit on a 200 kHz raster and would otherwise be mistaken for comb teeth); they are never
    comb candidates. A candidate line must be narrow in the mask (<= max_width_bins) and, when the IQ says
    the emission is strong and wide, it is excluded too. The grid tolerance is at most 5 % of the spacing and
    the members must occupy at least half of the grid points they span."""
    entries = [e for e in manifest["entries"] if e.get("kind") == "cross_track" and e.get("comb")]
    if not entries or not tracks:
        return
    e = entries[0]
    c = e["comb"]
    bin_hz = tracks[0].bin_hz
    explained = explained or set()

    def is_line(t: Track) -> bool:
        if t.tid in explained or t.bw > c["max_width_bins"] * bin_hz + 1e-6:
            return False
        if t.iq.get("snr_db", 0.0) >= 6.0 and t.iq.get("obw99_hz") and not t.iq.get("line") and t.iq["obw99_hz"] > 3 * bin_hz:
            return False          # a strong, genuinely wide emission (a weak station still passes: its width is unknown)
        return True

    narrow = [t for t in tracks if is_line(t)]
    lines = [t for t in narrow if t.temporal in ("continuous", "tdd")]
    if len(lines) < c["min_lines"]:
        return
    fcs = np.array([t.fc for t in lines])
    best = None
    for s0 in sorted(c["spacings_hz"], reverse=True):     # coarsest grid first: a 0.5 MHz grid trivially holds a 1 MHz comb
        if s0 < 3 * bin_hz:
            continue
        tol = min(c["tolerance_bins"] * bin_hz, 0.05 * s0)
        for i in range(len(fcs)):
            s, f0 = float(s0), float(fcs[i])
            members = np.array([], dtype=int)
            # the real spacing is rarely the nominal value (a 1.003 MHz comb drifts off a 1.000 MHz grid after
            # a dozen lines): refine spacing and origin by least squares on the members, twice
            for _ in range(3):
                k = np.round((fcs - f0) / s)
                resid = np.abs(fcs - f0 - k * s)
                members = np.where(resid <= tol)[0]
                if len(members) < 3:
                    break
                km = k[members]
                if np.ptp(km) == 0:
                    break
                A = np.vstack([np.ones(len(km)), km]).T
                f0, s = np.linalg.lstsq(A, fcs[members], rcond=None)[0]
                if abs(s - s0) > 0.02 * s0:          # refuse to wander to a different grid
                    s, f0 = float(s0), float(fcs[i])
                    break
            if len(members) >= c["min_lines"] and (best is None or len(members) > len(best[1])):
                k = np.round((fcs - f0) / s)
                resid = np.abs(fcs - f0 - k * s)
                span_steps = int(np.ptp(k[members])) + 1
                if len(members) / span_steps < c.get("min_fill", 0.5):
                    continue          # a few lines far apart on a flexible grid are not a comb
                best = (s, members, float(np.median(resid[members])), f0, s0, tol)
    if best is None:
        return
    s, members, res, f0, s0, tol = best
    n = len(members)
    tag = f"{s / 1e6:.4g}MHz"
    for j in members:
        t = lines[j]
        t.forced_label = f"{e['label']}_{tag}"
        t.forced_evidence = f"{n} persistent lines <= {c['max_width_bins']} bins wide on a {s / 1e6:.4f} MHz grid (nominal {s0 / 1e6:g} MHz; median residual {res / 1e3:.1f} kHz)"
    # the same lines flicker in and out of the mask in other frames: narrow tracks on the grid that recur in
    # at least half the frames are the comb too (a hopper revisits one channel far less often)
    for t in narrow:
        if t.forced_label or t.persistence < 0.5 or t.bw > 4 * bin_hz:
            continue
        k = round((t.fc - f0) / s)
        if abs(t.fc - f0 - k * s) <= tol:
            t.forced_label = f"{e['label']}_{tag}"
            t.forced_evidence = f"intermittent narrow line on the {s / 1e6:.4f} MHz comb grid ({n} persistent lines), persistence {t.persistence:.2f}"


# =========================================================================================== IQ features
_DTYPES = {"ci16_le": ("<i2", 2, 4), "ci16": ("<i2", 2, 4), "cf32_le": ("<c8", 1, 8), "cf32": ("<c8", 1, 8),
           "cf64_le": ("<c16", 1, 16), "ci8": ("i1", 2, 2), "ci8_le": ("i1", 2, 2)}


def read_iq(path: Path, dtype: str, start: int, count: int) -> np.ndarray:
    np_t, per, bps = _DTYPES.get(dtype, _DTYPES["cf32_le"])
    n_total = path.stat().st_size // bps
    start = max(0, min(start, n_total - 1))
    count = max(0, min(count, n_total - start))
    raw = np.fromfile(path, dtype=np_t, count=count * per, offset=start * bps)
    if per == 2:
        return raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    return raw.astype(np.complex64)


def psd_db(x: np.ndarray, nfft: int) -> np.ndarray:
    """Welch PSD (Hann, 50 % overlap), fftshifted, dB."""
    n = (len(x) - nfft) // (nfft // 2) + 1
    if n < 1:
        nfft = max(64, 1 << int(math.floor(math.log2(max(len(x), 64)))))
        n = 1
    win = np.hanning(nfft).astype(np.float32)
    acc = np.zeros(nfft, dtype=np.float64)
    for k in range(n):
        seg = x[k * (nfft // 2): k * (nfft // 2) + nfft]
        if len(seg) < nfft:
            break
        X = np.fft.fft(seg * win)
        acc += np.abs(X) ** 2
    acc /= max(n, 1)
    return 10 * np.log10(np.fft.fftshift(acc) + 1e-20)


def pilot_grids_from_manifest(manifest: dict) -> list[dict]:
    out = []
    for e in manifest["entries"]:
        for g in e.get("pilot_grids", []) or []:
            out.append({"tag": e["id"], "grid_offset": float(g["grid_offset_hz"]), "spacing": float(g["spacing_hz"]),
                        "pilot_offset": float(g["pilot_offset_hz"]), "band": (float(g["band_hz"][0]), float(g["band_hz"][1]))})
    return out


def iq_features(track: Track, data_path: Path, dtype: str, pilot_grids: list[dict] | None = None) -> dict:
    """In-band spectral shape of a track from the capture IQ: flatness, crest, pilot spike, 99 % OBW, SNR,
    and pilot carriers found at the positions a channel grid prescribes (ATSC)."""
    r0 = min(track.regions, key=lambda r: r.frame)
    rate = track.rate
    span = max(min(r0.dur, 0.02), 64 / rate)
    start = int(r0.t0 * rate)
    count = int(span * rate)
    if count < 256:
        # tiny burst: widen to its 512-row frame neighbourhood around the burst
        start = max(0, start - 256)
        count = max(count + 512, 2048)
    x = read_iq(data_path, dtype, start, count)
    if len(x) < 256:
        return {}
    want_bin = max(track.bw / 48.0, rate / 2 ** 20)
    nfft = int(2 ** math.ceil(math.log2(rate / want_bin)))
    nfft = int(min(max(nfft, 256), 2 ** 20, 1 << int(math.floor(math.log2(len(x))))))
    p = psd_db(x, nfft)
    bin_hz = rate / nfft
    f = track.fc_cap - rate / 2 + (np.arange(nfft) + 0.5) * bin_hz
    inb = (f >= track.f_lo) & (f <= track.f_hi)
    if inb.sum() < 8:
        return {}
    floor = float(np.percentile(p, 20))
    pin = p[inb]
    n_in = len(pin)
    central = pin[int(0.1 * n_in): max(int(0.9 * n_in), int(0.1 * n_in) + 4)]
    med = float(np.median(pin))
    feats = {"psd_nfft": nfft, "psd_bin_hz": bin_hz, "snr_db": round(med - floor, 1),
             "flatness_db": round(float(np.std(central)), 2), "crest_db": round(float(np.max(pin) - med), 1)}
    # ATSC 8-VSB pilot: spike near the lower channel edge. The detector box of a channel cut by the capture
    # edge may start below the channel, so search the lower half of such a box (else the lowest 15 %).
    frac = 0.5 if track.cut == "hi" else 0.15
    low = pin[: max(3, int(frac * n_in))]
    k = int(np.argmax(low))
    neigh = np.concatenate([low[max(0, k - 6): max(0, k - 2)], low[k + 3: k + 7]])
    pilot_rel = float(low[k] - (np.median(neigh) if len(neigh) else med))
    feats["pilot_db"] = round(pilot_rel, 1)
    feats["pilot_offset_hz"] = round(float(f[inb][k] - track.f_lo), 0)
    above = pin[k + 4:]
    flat_above = float(np.std(above[int(0.1 * len(above)): max(int(0.9 * len(above)), int(0.1 * len(above)) + 4)])) if len(above) >= 12 else feats["flatness_db"]
    feats["flatness_above_pilot_db"] = round(flat_above, 2)
    # 99 % occupied bandwidth and power centroid of the emission that contains the box: look at the band
    # plus a margin, keep the contiguous run of bins above the floor that overlaps the box, then integrate.
    # The box edges from a mask are quantised and flicker; the PSD gives the centre to a few kHz.
    margin = max(0.25 * track.bw, 2 * track.bin_hz)
    sel = np.where((f >= track.f_lo - margin) & (f <= track.f_hi + margin))[0]
    if len(sel) >= 8:
        ps = p[sel]
        thr = floor + max(3.0, 0.3 * max(feats["snr_db"], 0.0))
        above = ps > thr
        box = (f[sel] >= track.f_lo) & (f[sel] <= track.f_hi)
        if above.any() and box.any():
            # start from the strongest bin inside the box and grow while above threshold
            k0 = int(np.argmax(np.where(box, ps, -np.inf)))
            if above[k0]:
                # grow from the peak while above threshold, but stop at a valley: once the PSD has dropped
                # >= 6 dB from its running minimum and rises again, the next bin belongs to a neighbour
                # (adjacent FM stations 200 kHz apart are contiguous above the floor but have a valley between)
                a, vmin = k0, ps[k0]
                while a > 0 and above[a - 1]:
                    if ps[a - 1] > vmin + 6.0:
                        break
                    a -= 1
                    vmin = min(vmin, ps[a])
                b, vmin = k0, ps[k0]
                while b < len(ps) - 1 and above[b + 1]:
                    if ps[b + 1] > vmin + 6.0:
                        break
                    b += 1
                    vmin = min(vmin, ps[b])
                run = slice(a, b + 1)
                lin = np.clip(10 ** (ps[run] / 10) - 10 ** (floor / 10), 0, None)
                if lin.sum() > 0:
                    fs = f[sel][run]
                    c = np.cumsum(lin) / lin.sum()
                    lo = fs[int(np.searchsorted(c, 0.005))]
                    hi = fs[min(int(np.searchsorted(c, 0.995)), len(fs) - 1)]
                    feats["obw99_hz"] = round(float(hi - lo), 0)
                    feats["fc_iq_hz"] = round(float(np.sum(fs * lin) / lin.sum()), 1)
                    feats["line"] = bool(hi - lo <= 4 * bin_hz)     # a carrier / spur: a few PSD bins wide
                    feats["obw_reliable"] = bool(a > 0 and b < len(ps) - 1)   # the run ended inside the window
    # pilot carriers on a known channel grid (ATSC 8-VSB: lower channel edge + 309.441 kHz), resolved at <= 6 kHz
    hits = []
    for g in pilot_grids or []:
        if not (g["band"][0] <= track.fc <= g["band"][1]) and not (g["band"][0] <= track.f_lo <= g["band"][1] or g["band"][0] <= track.f_hi <= g["band"][1]):
            continue
        nfft_p = int(min(2 ** 20, 2 ** math.ceil(math.log2(rate / 6.0e3))))
        if nfft_p > len(x):
            nfft_p = 1 << int(math.floor(math.log2(len(x))))
        pp = psd_db(x, nfft_p)
        bp = rate / nfft_p
        fp = track.fc_cap - rate / 2 + (np.arange(nfft_p) + 0.5) * bp
        k0 = math.ceil((track.f_lo - g["grid_offset"] - g["pilot_offset"]) / g["spacing"])
        k1 = math.floor((track.f_hi - g["grid_offset"] - g["pilot_offset"]) / g["spacing"])
        for k in range(k0, k1 + 1):
            f_p = g["grid_offset"] + k * g["spacing"] + g["pilot_offset"]
            win = (fp >= f_p - 15e3) & (fp <= f_p + 15e3)
            ref = (fp >= f_p + 80e3) & (fp <= f_p + 500e3)           # flat in-band data just above the pilot
            if win.any() and ref.sum() >= 4 and pp[win].max() - np.median(pp[ref]) >= 8.0:
                hits.append((g["tag"], round(f_p / 1e6, 4)))
    if pilot_grids:
        feats["pilots"] = hits
    # spectral shape verdict
    if feats["snr_db"] < 3:
        shape = "weak"
    elif feats["pilot_db"] >= 8 and min(feats["flatness_db"], feats["flatness_above_pilot_db"]) <= 4.5:
        shape = "flat_top_pilot"
    elif feats["flatness_db"] <= 3.0:
        shape = "flat_top"
    elif feats["crest_db"] >= 6 or feats["flatness_db"] >= 5:
        shape = "peaked"
    else:
        shape = "flat_top"
    feats["shape"] = shape
    return feats


# =========================================================================================== scoring
@dataclass
class Match:
    entry: dict
    conf: float
    verified: set
    evidence: list
    requires_ok: bool

    @property
    def label(self) -> str:
        return self.entry["label"]


def _gauss(d: float, s: float, floor: float = 0.0) -> float:
    return max(floor, math.exp(-(d / max(s, 1e-9)) ** 2))


def score_entry(t: Track, e: dict) -> Match | None:
    ev: list[str] = []
    verified: set = set()
    conf = 1.0
    merged_n = 0
    pilots = [f for tag, f in t.iq.get("pilots", []) if tag == e["id"]] if e.get("pilot_grids") else []
    # ---- 1. band gate -----------------------------------------------------------------------
    if e.get("relative_to_capture_center"):
        off = t.fc - t.fc_cap
        if abs(off) > 1.5 * t.bin_hz:
            return None
        ev.append(f"offset from capture centre {off / 1e3:+.1f} kHz")
        bins = t.bw / t.bin_hz
        bb = e.get("bandwidth_bins", {"min": 1, "max": 4})
        if bins > bb["max"] + 0.5:
            return None
        verified.add("bandwidth")
        ev.append(f"width {bins:.1f} bins")
    else:
        if e.get("kind") == "cross_track":
            return None
        bw = e["bandwidth_hz"]
        # a box cut by the capture edge: its width is a lower bound and its centre is unknown; estimate
        # the centre from the visible edge and the entry's nominal width
        fc_eval = t.fc
        bw_eval = t.bw
        bw_from_iq = False
        if t.cut == "hi":
            fc_eval = t.f_lo + 0.5 * bw["nominal"]
        elif t.cut == "lo":
            fc_eval = t.f_hi - 0.5 * bw["nominal"]
        elif t.iq.get("fc_iq_hz") and t.iq.get("snr_db", 0.0) >= 6.0 and t.iq.get("obw_reliable"):
            # the PSD centroid / 99 % width of the emission are far more precise than quantised mask edges;
            # an emission much narrower than its box (a spur line inside a 3-bin box) is judged by its true width.
            # (not used when the PSD run reached the edge of the analysis window: it then includes a neighbour)
            if t.iq.get("obw99_hz", t.bw) <= 2.5 * t.bw:
                bw_eval = float(t.iq["obw99_hz"])
                fc_eval = float(t.iq["fc_iq_hz"])
                bw_from_iq = True
                ev.append(f"IQ centroid {fc_eval / 1e6:.4f} MHz, OBW99 {bw_eval / 1e3:.0f} kHz" + (" (line, far narrower than its box)" if bw_eval < 0.25 * t.bw else ""))
        if bw["nominal"] < t.bin_hz and not t.cut and not e.get("band_exclusive"):
            # a sub-bin box cannot be told from other narrow emitters by its width; in an exclusive band the
            # allocation itself identifies the service, elsewhere the estimate is weaker
            conf *= 0.85
            ev.append("below the frequency resolution of this capture")
        tol = max(t.bin_hz, 0.02 * t.bw)
        if t.cut:
            inside = any(lo - tol <= fc_eval <= hi + tol for lo, hi in e["freq_ranges_hz"])
            partial = inside
        else:
            inside = any(lo - tol <= t.f_lo and t.f_hi <= hi + tol for lo, hi in e["freq_ranges_hz"])
            partial = any(lo <= t.fc <= hi for lo, hi in e["freq_ranges_hz"])
        if not inside and not partial:
            return None
        if not inside:
            conf *= 0.8
            ev.append("band edge crosses the allocation")
        # ---- 2. bandwidth ---------------------------------------------------------------------
        slack = max(2.0 * t.bin_hz, 0.10 * bw["nominal"])
        ras = e.get("raster")
        merged_n = 0
        if ras and e.get("raster_multiples") and not t.cut and t.bw > bw["max"]:
            # adjacent channels merged into one box (e.g. three 6 MHz TV channels): accept n x spacing when the
            # lower edge sits on a channel edge of the grid (box width, not the PSD width, which may run on
            # into contiguous neighbours)
            sp = ras["spacing_hz"]
            n_ch = int(round(t.bw / sp))
            edge0 = ras["offset_hz"] - 0.5 * sp
            d_edge = abs((t.f_lo - edge0) - round((t.f_lo - edge0) / sp) * sp)
            if 2 <= n_ch <= int(e["raster_multiples"]) and abs(t.bw - n_ch * sp) <= max(2.5 * t.bin_hz, 0.08 * sp) and d_edge <= max(1.5 * t.bin_hz, 0.03 * sp):
                merged_n = n_ch
                bw_eval, bw_from_iq = t.bw, False
        if pilots:
            # pilot carriers at grid positions identify the waveform regardless of how the box came out
            # (merged channels, overshooting edges, cut by the capture edge); bandwidth is verified in step 6
            ev.append(f"box {t.bw / 1e6:.2f} MHz spans {max(1, int(round(t.bw / (e['pilot_grids'][0]['spacing_hz']))))} channel(s)")
        elif t.cut:
            cap_lo, cap_hi = t.fc_cap - t.rate / 2, t.fc_cap + t.rate / 2
            whole_capture_inside = any(lo - tol <= cap_lo and cap_hi <= hi + tol for lo, hi in e["freq_ranges_hz"])
            if t.bw > bw["max"] + slack and not (merged_n or whole_capture_inside):
                return None
            conf *= 0.9
            ev.append(f"cut at the capture {t.cut} edge: visible {t.bw / 1e3:.0f} kHz, centre assumed {fc_eval / 1e6:.4g} MHz")
            if whole_capture_inside and e.get("band_exclusive"):
                verified.add("bandwidth")   # everything the receiver sees lies inside this exclusive allocation
                ev.append("the whole capture span lies inside the allocation")
        elif merged_n:
            verified.add("bandwidth")
            fc_eval = t.f_lo + 0.5 * merged_n * ras["spacing_hz"]   # not a channel centre; skip the raster test below
            ev.append(f"{merged_n} adjacent {ras['spacing_hz'] / 1e6:g} MHz channels merged into one box ({bw_eval / 1e6:.2f} MHz), lower edge on the channel grid")
        else:
            # the TRUE width is an interval: a mask box is quantised to bins and each edge may be one bin
            # generous (a 12 kHz NBFM channel at 24 kHz bins shows as a 1-3 bin box); a PSD width is +-10 %
            if bw_from_iq and t.iq.get("line"):
                w_lo, w_hi = 0.9 * bw_eval, 1.1 * bw_eval                       # a spectral line: its width is known
            elif bw_from_iq:
                # the 99 % power width depends on modulation depth (an FM station can measure 80 kHz in a quiet
                # passage): the true width lies between the OBW and the detector box
                w_lo, w_hi = 0.9 * bw_eval, max(1.1 * bw_eval, t.bw + 0.5 * t.bin_hz)
            elif bw_eval <= 3.0 * t.bin_hz:
                w_lo, w_hi = 0.0, bw_eval + 0.5 * t.bin_hz          # a 1-3 bin box: anything up to its width
            else:
                w_lo, w_hi = bw_eval - 2.5 * t.bin_hz, bw_eval + 0.5 * t.bin_hz
            if w_lo <= bw["max"] and w_hi >= bw["min"]:
                verified.add("bandwidth")
            else:
                d = (bw["min"] - w_hi) if w_hi < bw["min"] else (w_lo - bw["max"])
                f = _gauss(d, slack)
                if f < 0.25:
                    return None
                conf *= f
        if not t.cut:
            ev.append(f"bw {bw_eval / 1e3:.0f} kHz vs {bw['min'] / 1e3:.0f}-{bw['max'] / 1e3:.0f}")
        sets = e.get("bandwidth_sets_hz")
        if sets and not t.cut:
            ok = any(lo <= w_hi + slack and hi >= w_lo - slack for lo, hi in sets)
            if ok:
                ev.append("bw in an allowed channel-width set")
            else:
                conf *= 0.35
                verified.discard("bandwidth")
                ev.append("bw outside every channel-width set")
    # ---- 3. temporal ----------------------------------------------------------------------------
    cls = t.temporal
    want = e["temporal"]
    table = {
        "continuous": {"continuous": 1.0, "sporadic_full": 0.8, "tdd": 0.7, "intermittent": 0.45, "bursty": 0.15},
        "bursty": {"bursty": 1.0, "intermittent": 0.85, "sporadic_full": 0.5, "tdd": 0.4, "continuous": 0.15},
        "tdd": {"tdd": 1.0, "continuous": 0.85, "sporadic_full": 0.8, "intermittent": 0.7, "bursty": 0.3},
        "either": {"bursty": 1.0, "intermittent": 1.0, "sporadic_full": 1.0, "tdd": 1.0, "continuous": 1.0},
    }[want]
    f = table.get(cls, 0.5)
    conf *= f
    if (want == "continuous" and cls in ("continuous", "sporadic_full")) or (want == "bursty" and cls in ("bursty", "intermittent")) \
            or (want == "tdd" and cls in ("tdd", "continuous", "sporadic_full")) or want == "either":
        verified.add("temporal")
    ev.append(f"{cls} (duty {t.duty:.2f}, persistence {t.persistence:.2f})")
    # ---- 4. burst duration ------------------------------------------------------------------------
    b = e.get("burst_s")
    frame_s = t.regions[0].frame_s
    if b and cls != "continuous" and want != "tdd":
        if t.row_s > 2 * b["max"]:
            ev.append("burst length below the time resolution")
        elif t.burst_s >= 0.9 * frame_s:
            # a transmission longer than the 21 ms frame shows as whole-frame regions: its true length is unknown
            if b["max"] >= 0.9 * frame_s:
                verified.add("burst")
                ev.append("bursts span whole frames (length >= frame)")
            else:
                conf *= _gauss(frame_s - b["max"], 2 * max(2.0 * t.row_s, 0.25 * b["max"]), 0.3)
                ev.append(f"bursts span whole frames but the entry's bursts are <= {b['max'] * 1e3:.2f} ms")
        else:
            tol_lo = max(2.0 * t.row_s, 0.25 * b["min"])
            tol_hi = max(2.0 * t.row_s, 0.25 * b["max"])
            if b["min"] - tol_lo <= t.burst_s <= b["max"] + tol_hi:
                verified.add("burst")
                ev.append(f"burst {t.burst_s * 1e3:.2f} ms ok")
            else:
                d, tol = ((b["min"] - t.burst_s), tol_lo) if t.burst_s < b["min"] else ((t.burst_s - b["max"]), tol_hi)
                conf *= _gauss(d, 2 * tol, 0.3)
                ev.append(f"burst {t.burst_s * 1e3:.2f} ms vs {b['min'] * 1e3:.2f}-{b['max'] * 1e3:.2f} ms")
    # ---- 5. channel raster / channel plan -----------------------------------------------------------
    ras = e.get("raster")
    chans = e.get("channel_centers_hz")
    fc_grid = fc_eval if not e.get("relative_to_capture_center") else t.fc
    grid_checked = False
    if merged_n or pilots:
        verified.add("raster")          # merged box: lower edge already on the grid; pilots: grid positions verified
        ras = None
        chans = None if pilots else chans
    if ras:
        spacing, off = ras["spacing_hz"], ras["offset_hz"]
        if t.bin_hz <= spacing / 3.0:
            grid_checked = True
            k = round((fc_grid - off) / spacing)
            d = abs(fc_grid - (off + k * spacing))
            tol = max(1.5 * t.bin_hz, spacing / 20.0)
            if d <= tol:
                verified.add("raster")
                ev.append(f"on the {spacing / 1e3:g} kHz raster ({d / 1e3:.1f} kHz off)")
            else:
                conf *= _gauss(d - tol, 2 * tol, 0.15)
                ev.append(f"off raster by {d / 1e3:.1f} kHz")
    if chans:
        chans_arr = np.asarray(chans, dtype=float)
        sep = float(np.min(np.diff(np.sort(chans_arr)))) if len(chans_arr) > 1 else 10 * t.bin_hz
        if t.bin_hz <= sep / 3.0 or len(chans_arr) == 1:
            j = int(np.argmin(np.abs(chans_arr - fc_grid)))
            d = abs(fc_grid - chans_arr[j])
            tol = e.get("channel_tolerance_hz") or max(1.5 * t.bin_hz, 0.03 * max(bw_eval, e["bandwidth_hz"]["nominal"]))
            if d <= tol:
                verified.add("channels")
                if not grid_checked:
                    grid_checked = True
                ev.append(f"channel {chans_arr[j] / 1e6:.4g} MHz ({d / 1e3:.1f} kHz off)")
            elif not ("raster" in verified):
                conf *= _gauss(d - tol, 2 * tol, 0.15)
                ev.append(f"nearest channel {chans_arr[j] / 1e6:.4g} MHz is {d / 1e3:.0f} kHz away")
    if "raster" in verified and "channels" not in verified and chans:
        # LoRa-style: raster OR channel list satisfies the grid requirement
        verified.add("channels")
    if "channels" in verified and ras and "raster" not in verified:
        verified.add("raster")
    # ---- 6. spectral shape (IQ) -----------------------------------------------------------------------
    sig = e.get("iq_signature", "any")
    if pilots:
        # pilot carriers exactly where the channel plan puts them: decisive for the waveform AND the grid,
        # whatever the box looks like (merged channels, box cut by the capture edge, overshooting edges)
        verified.update({"iq", "bandwidth", "raster", "channels"})
        ev.append(f"{len(pilots)} pilot carrier(s) on the channel grid at {', '.join(f'{p:.3f}' for p in pilots[:6])} MHz")
        sig = "any"
    elif e.get("pilot_grids") and t.iq.get("snr_db", 0.0) >= 8.0 and "pilots" in t.iq and not t.cut:
        conf *= 0.9                      # strong, resolved, yet no pilot: ATSC 3.0 (no pilot) or something else
        ev.append("no pilot carrier found on the channel grid")
    if t.iq and t.iq.get("shape") not in (None, "weak") and sig not in ("any", "pulsed", "chirp"):
        shape = t.iq["shape"]
        # OFDM / VSB carriers really are flat-topped, so "peaked" contradicts them; analog shapes vary
        # with modulation depth (a 20 ms WFM average can look flat), so a flat reading only mildly
        # contradicts a "peaked" expectation
        if sig == "flat_top":
            f = {"flat_top": 1.0, "flat_top_pilot": 0.9, "peaked": 0.6}.get(shape, 0.8)
        elif sig == "flat_top_pilot":
            f = {"flat_top_pilot": 1.0, "flat_top": 0.9, "peaked": 0.6}.get(shape, 0.8)
        else:  # peaked
            f = {"peaked": 1.0, "flat_top": 0.8, "flat_top_pilot": 0.7}.get(shape, 0.8)
        conf *= f
        if f >= 0.9:
            verified.add("iq")
            if t.cut and (sig == "flat_top_pilot" and shape == "flat_top_pilot" or e.get("band_exclusive")):
                verified.add("bandwidth")   # the pilot (or the exclusive band + matching shape) identifies the waveform although the box is cut
        ev.append(f"PSD {shape} (flatness {t.iq['flatness_db']} dB, crest {t.iq['crest_db']} dB"
                  + (f", pilot {t.iq['pilot_db']} dB at +{t.iq['pilot_offset_hz'] / 1e3:.0f} kHz" if shape == "flat_top_pilot" else "") + ")")
    # ---- 7. priors ------------------------------------------------------------------------------------
    if not e.get("band_exclusive"):
        conf *= 0.92
    if e.get("tier") == "B":
        conf *= 0.97
    req_ok = all(r in verified for r in e.get("requires", []))
    return Match(e, conf, verified, ev, req_ok)


THRESH = {"A": 0.75, "B": 0.85}


def decide(t: Track, manifest: dict, margin: float = 0.10) -> tuple[Match | None, list[Match], str]:
    if t.forced_label:
        e = next(x for x in manifest["entries"] if x.get("kind") == "cross_track")
        fake = dict(e)
        fake["label"] = t.forced_label
        return Match(fake, 0.95, {"comb"}, [t.forced_evidence], True), [], "ok"
    matches = [m for m in (score_entry(t, e) for e in manifest["entries"]) if m is not None]
    matches.sort(key=lambda m: -m.conf)
    eligible = [m for m in matches if m.requires_ok and m.entry["tier"] in THRESH and m.conf >= THRESH[m.entry["tier"]]]
    if not eligible:
        why = "no entry passes" if not matches else f"best {matches[0].label} {matches[0].conf:.2f}" + ("" if matches[0].requires_ok else " (required features unverified)")
        return None, matches, why
    # specific entries beat catch-alls
    eligible.sort(key=lambda m: (m.entry.get("catch_all", False), -m.conf))
    best = eligible[0]
    for m in matches:
        if m.label == best.label or m is best:
            continue
        if best.entry.get("catch_all") or not m.entry.get("catch_all"):
            if m.conf >= best.conf - margin and m.entry["tier"] in THRESH:
                return None, matches, f"ambiguous: {best.label} {best.conf:.2f} vs {m.label} {m.conf:.2f}"
    return best, matches, "ok"


# =========================================================================================== inputs
def _detector_cols(rate: float, detector: str | None, override: int | None) -> int:
    if override:
        return override
    if detector and "dino" in detector:
        return mio.NFFT_MODEL
    if detector and "coherent" in detector:
        return mio.app_fft_size(rate)
    return mio.NFFT_MODEL


def regions_from_preprocessed(root: Path, cols_override: int | None) -> dict[str, dict]:
    """<root>/<capture>/snippets/*.sigmf-meta -> {capture: {regions, n_frames, data_path, dtype, rate, fc}}"""
    summary = {}
    sp = root / "preprocess_summary.json"
    detector = None
    if sp.exists():
        try:
            summary = json.loads(sp.read_text())
            detector = summary.get("detector")
        except json.JSONDecodeError:
            pass
    out: dict[str, dict] = {}
    # one folder per capture, directly under root or (preprocess_dataset.py --recursive) mirrored deeper
    cap_dirs = sorted({p.parent for p in root.rglob("*.source.sigmf-meta") if (p.parent / "snippets").is_dir()})
    for cap_dir in cap_dirs:
        src = sorted(cap_dir.glob("*.source.sigmf-meta"))
        if not src:
            continue
        smeta = json.loads(src[0].read_text())
        rate = float(smeta["global"]["core:sample_rate"])
        fc_cap = float((smeta.get("captures") or [{}])[0].get("core:frequency", 0.0))
        dtype = smeta["global"].get("core:datatype", "cf32_le")
        cols = _detector_cols(rate, detector, cols_override)
        geo = mio.frame_geometry(rate, cols)
        stem = src[0].name[: -len(".source.sigmf-meta")]
        # the original recording, if it is still where the summary says
        data_path = None
        for cand in (summary.get("captures", {}) or {}).get(stem, {}).get("input", None), summary.get("dataset_dir"):
            pass
        ds = summary.get("dataset_dir")
        if ds:
            for c in Path(ds).expanduser().rglob(f"{stem}.sigmf-data"):
                data_path = c
                break
        regions: list[Region] = []
        frames = set()
        for mp in sorted((cap_dir / "snippets").glob("*.sigmf-meta")):
            meta = json.loads(mp.read_text())
            g = meta.get("global", {})
            for i, a in enumerate(meta.get("annotations", [])):
                lo, hi = a.get("core:freq_lower_edge"), a.get("core:freq_upper_edge")
                if lo is None or hi is None:
                    continue
                o_start = a.get("wfgt:orig_sample_start", g.get("wfgt:orig_sample_start", 0))
                o_count = a.get("wfgt:orig_sample_count", 0)
                if not o_count:  # fall back to the snippet's own length at its own rate
                    sr = float(a.get("wfgt:snippet_sample_rate", g.get("core:sample_rate", rate)) or rate)
                    o_count = int(a.get("core:sample_count", 0)) * rate / sr
                fr = a.get("wfgt:frame_number")
                t0 = float(o_start) / rate
                if fr is None:
                    fr = int(t0 // geo["frame_s"])
                frames.add(int(fr))
                regions.append(Region(stem, float(lo), float(hi), t0, float(o_count) / rate, int(fr), rate, fc_cap,
                                      geo["bin_hz"], geo["row_s"], geo["frame_s"], mp, i, a.get("core:label", "")))
        n_frames = 0
        fm = cap_dir / "frame_manifest.csv"
        if fm.exists():
            with open(fm, newline="") as f:
                n_frames = sum(1 for _ in csv.DictReader(f))
        if not n_frames:
            oes = cap_dir / "offline_eval_summary.json"
            if oes.exists():
                try:
                    j = json.loads(oes.read_text())
                    n_frames = int(j.get("frames_processed") or j.get("frames") or 0)
                except (json.JSONDecodeError, ValueError):
                    pass
        if not n_frames and frames:
            n_frames = max(frames) - min(frames) + 1
        out[stem] = {"regions": regions, "n_frames": max(n_frames, 1), "data_path": data_path, "dtype": dtype,
                     "rate": rate, "fc": fc_cap, "dir": cap_dir, "geometry": geo}
    return out


def regions_from_sigmf(meta_path: Path, cols_override: int | None) -> dict[str, dict]:
    meta = json.loads(meta_path.read_text())
    g = meta["global"]
    rate = float(g["core:sample_rate"])
    fc_cap = float((meta.get("captures") or [{}])[0].get("core:frequency", 0.0))
    dtype = g.get("core:datatype", "cf32_le")
    geo = mio.frame_geometry(rate, cols_override)
    stem = meta_path.name[: -len(".sigmf-meta")]
    data_path = meta_path.with_name(stem + ".sigmf-data")
    regions = []
    frames = set()
    for i, a in enumerate(meta.get("annotations", [])):
        lo, hi = a.get("core:freq_lower_edge"), a.get("core:freq_upper_edge")
        if lo is None or hi is None:
            continue
        s0 = int(a.get("core:sample_start", 0))
        cnt = int(a.get("core:sample_count", 0))
        fr = a.get("wfgt:frame_number", s0 // geo["frame_samples"])
        frames.add(int(fr))
        regions.append(Region(stem, float(lo), float(hi), s0 / rate, cnt / rate, int(fr), rate, fc_cap, geo["bin_hz"], geo["row_s"], geo["frame_s"],
                              meta_path, i, a.get("core:label", "")))
    n_total = data_path.stat().st_size // _DTYPES.get(dtype, _DTYPES["cf32_le"])[2] if data_path.exists() else 0
    n_frames = max(1, n_total // geo["frame_samples"]) if n_total else (max(frames) - min(frames) + 1 if frames else 1)
    return {stem: {"regions": regions, "n_frames": n_frames, "data_path": data_path if data_path.exists() else None, "dtype": dtype,
                   "rate": rate, "fc": fc_cap, "dir": meta_path.parent, "geometry": geo}}


def regions_from_masks(cap_dir: Path, model: str, min_pixels: int = 6) -> dict[str, dict]:
    from scipy import ndimage
    metas = sorted(p for p in cap_dir.glob("*.sigmf-meta") if "synth" not in p.name)
    if not metas:
        raise FileNotFoundError(f"no *.sigmf-meta in {cap_dir}")
    meta = json.loads(metas[0].read_text())
    g = meta["global"]
    rate = float(g["core:sample_rate"])
    fc_cap = float((meta.get("captures") or [{}])[0].get("core:frequency", 0.0))
    dtype = g.get("core:datatype", "cf32_le")
    z = np.load(cap_dir / "masks" / f"{model}.npz")
    cols = int(z["cols"])
    rows = int(z["rows"])
    masks = np.unpackbits(z["packed"], axis=-1)[:, :, :cols]
    frame_index = z["frame_index"]
    FS = int(z["frame_samples"])
    geo = {"rate_hz": rate, "cols": cols, "rows": rows, "frame_samples": FS, "frame_s": FS / rate, "row_s": FS / rows / rate, "bin_hz": rate / cols}
    stem = metas[0].name[: -len(".sigmf-meta")]
    regions = []
    for k, fidx in enumerate(frame_index):
        lab, n = ndimage.label(masks[k], structure=np.ones((3, 3), dtype=int))
        if n == 0:
            continue
        sizes = ndimage.sum(masks[k], lab, index=np.arange(1, n + 1))
        for j, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sizes[j - 1] < min_pixels or sl is None:
                continue
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            f_lo = fc_cap - rate / 2 + c0 * geo["bin_hz"]
            f_hi = fc_cap - rate / 2 + c1 * geo["bin_hz"]
            t0 = int(fidx) * geo["frame_s"] + r0 * geo["row_s"]
            regions.append(Region(stem, f_lo, f_hi, t0, (r1 - r0) * geo["row_s"], int(fidx), rate, fc_cap, geo["bin_hz"], geo["row_s"], geo["frame_s"],
                                  None, None, "mask"))
    data_path = metas[0].with_name(stem + ".sigmf-data")
    return {stem: {"regions": regions, "n_frames": len(frame_index), "data_path": data_path if data_path.exists() else None, "dtype": dtype,
                   "rate": rate, "fc": fc_cap, "dir": cap_dir, "geometry": geo}}


# =========================================================================================== run
def run_capture(stem: str, cap: dict, manifest: dict, use_iq: bool, margin: float, max_iq_tracks: int = 400) -> tuple[list[Track], list[dict]]:
    tracks = build_tracks(cap["regions"], cap["n_frames"])
    if use_iq and cap.get("data_path") and len(tracks) <= max_iq_tracks:
        grids = pilot_grids_from_manifest(manifest)
        for t in tracks:
            try:
                t.iq = iq_features(t, cap["data_path"], cap["dtype"], grids)
            except Exception as ex:  # noqa: BLE001
                t.iq = {"error": f"{type(ex).__name__}: {str(ex)[:80]}"}
    # tracks that a manifest entry already explains with high confidence are not comb candidates
    explained = set()
    for t in tracks:
        best, _, _ = decide(t, manifest, margin)
        if best and best.conf >= 0.9:
            explained.add(t.tid)
    comb_pass(tracks, manifest, explained)
    rows = []
    for t in tracks:
        best, matches, why = decide(t, manifest, margin)
        est = (ESTIMATE_PREFIX + best.label) if best else "signal"
        top = matches[0] if matches else None
        row = {"capture": stem, "track": t.tid, "n_regions": len(t.regions), "frames_present": len({r.frame for r in t.regions}),
               "frames_total": t.n_frames_total, "fc_hz": round(t.fc, 1), "bw_hz": round(t.bw, 1), "f_lo_hz": round(t.f_lo, 1), "f_hi_hz": round(t.f_hi, 1),
               "offset_from_center_hz": round(t.fc - t.fc_cap, 1), "duty": round(t.duty, 3), "persistence": round(t.persistence, 3),
               "burst_s": round(t.burst_s, 6), "temporal": t.temporal, "cut": t.cut, "bin_hz": round(t.bin_hz, 1), "row_s": round(t.row_s, 7),
               "iq_shape": t.iq.get("shape", ""), "iq_error": t.iq.get("error", ""), "iq_snr_db": t.iq.get("snr_db", ""), "iq_flatness_db": t.iq.get("flatness_db", ""),
               "iq_crest_db": t.iq.get("crest_db", ""), "iq_pilot_db": t.iq.get("pilot_db", ""), "iq_obw99_hz": t.iq.get("obw99_hz", ""),
               "estimate": est, "entry": best.entry["id"] if best else "", "confidence": round(best.conf, 3) if best else "",
               "evidence": "; ".join(best.evidence) if best else why,
               "runner_up": (f"{top.label} {top.conf:.2f}" if (top and (not best or top.label != best.label)) else
                             (f"{matches[1].label} {matches[1].conf:.2f}" if best and len(matches) > 1 else "")),
               "_best": best, "_track": t}
        rows.append(row)
    return tracks, rows


def write_labels(rows: list[dict], root_for_manifest: Path | None):
    """Write ~labels into the SigMF annotations referenced by the regions and refresh manifest.csv label columns."""
    by_meta: dict[Path, list[tuple[int, dict]]] = defaultdict(list)
    for row in rows:
        best = row["_best"]
        for r in row["_track"].regions:
            if r.meta_path is None or r.ann_index is None:
                continue
            by_meta[r.meta_path].append((r.ann_index, {"best": best, "row": row}))
    label_by_key: dict[tuple, str] = {}
    n_written = 0
    for mp, items in by_meta.items():
        meta = json.loads(mp.read_text())
        anns = meta.get("annotations", [])
        for idx, info in items:
            if idx >= len(anns):
                continue
            a = anns[idx]
            best = info["best"]
            orig = a.get("argus:detector_label", a.get("core:label", ""))
            a["argus:detector_label"] = orig
            if best:
                a["core:label"] = ESTIMATE_PREFIX + best.label
                a["argus:estimate_entry"] = best.entry["id"]
                a["argus:estimate_confidence"] = round(best.conf, 3)
                a["argus:estimate_evidence"] = "; ".join(best.evidence)[:400]
            else:
                a["core:label"] = "signal"
                for k in ("argus:estimate_entry", "argus:estimate_confidence", "argus:estimate_evidence"):
                    a.pop(k, None)
            a["argus:estimate_source"] = VERSION
            label_by_key[(mp.name[: -len(".sigmf-meta")], str(a.get("core:sample_start", 0)), str(a.get("core:freq_lower_edge")))] = a["core:label"]
            n_written += 1
        mp.write_text(json.dumps(meta, indent=2) + "\n")
    # manifest.csv (root + per capture) label columns
    if root_for_manifest:
        for mcsv in [root_for_manifest / "manifest.csv"] + sorted(root_for_manifest.glob("*/manifest.csv")):
            if not mcsv.exists():
                continue
            with open(mcsv, newline="") as f:
                rd = csv.DictReader(f)
                fields = rd.fieldnames or []
                rws = list(rd)
            if "label" not in fields:
                continue
            ch = 0
            for rw in rws:
                stem = Path(rw.get("file", "")).name
                stem = stem[: -len(".sigmf-data")] if stem.endswith(".sigmf-data") else stem
                key = (stem, str(rw.get("sample_start", "0")), str(_fmt_num(rw.get("freq_lower_edge_hz"))))
                lab = label_by_key.get(key)
                if lab is None:
                    # tolerate float formatting differences
                    for (s2, ss, fl), v in label_by_key.items():
                        if s2 == stem and ss == str(rw.get("sample_start", "0")) and _close(fl, rw.get("freq_lower_edge_hz")):
                            lab = v
                            break
                if lab is not None and rw["label"] != lab:
                    rw["label"] = lab
                    ch += 1
            if ch:
                with open(mcsv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rws)
    return n_written


def _fmt_num(x):
    try:
        v = float(x)
        return int(v) if v == int(v) else v
    except (TypeError, ValueError):
        return x


def _close(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1.0
    except (TypeError, ValueError):
        return False


def summarize(rows: list[dict]) -> dict:
    per_cap: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        per_cap[r["capture"]][r["estimate"]] += 1
    tot = Counter(r["estimate"] for r in rows)
    return {"tracks": len(rows), "labelled": sum(v for k, v in tot.items() if k != "signal"), "labels": dict(tot.most_common()),
            "per_capture": {c: dict(v.most_common()) for c, v in per_cap.items()}}


def check_manifest(m: dict) -> int:
    bad = 0
    for e in m["entries"]:
        if e["tier"] not in ("A", "B", "C"):
            print(f"{e['id']}: tier {e['tier']}")
            bad += 1
        if e["temporal"] not in ("continuous", "bursty", "either", "tdd"):
            print(f"{e['id']}: temporal {e['temporal']}")
            bad += 1
        if not e.get("relative_to_capture_center") and e.get("kind") != "cross_track" and "bandwidth_hz" not in e:
            print(f"{e['id']}: missing bandwidth_hz")
            bad += 1
        for lo, hi in e["freq_ranges_hz"]:
            if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo < hi):
                print(f"{e['id']}: bad range {lo}-{hi}")
                bad += 1
        for r in e.get("requires", []):
            if r not in ("raster", "channels", "bandwidth", "temporal", "burst", "iq"):
                print(f"{e['id']}: unknown requires item {r}")
                bad += 1
            if r == "raster" and not e.get("raster") and not e.get("channel_centers_hz"):
                print(f"{e['id']}: requires raster but has none")
                bad += 1
            if r == "channels" and not e.get("channel_centers_hz") and not e.get("raster"):
                print(f"{e['id']}: requires channels but has none")
                bad += 1
    print(f"{len(m['entries'])} entries, {bad} problems")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="<dir>_preprocessed | <capture>.sigmf-meta | <capture folder> (with --masks)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--masks", default=None, help="model name: read <input>/masks/<model>.npz (run_ota_detections.py layout)")
    ap.add_argument("--cols", type=int, default=None, help="detector mask columns (default: 1024 for the fine-tuned DINO, app FFT size for coherent_power)")
    ap.add_argument("--no-iq", action="store_true", help="do not read the raw IQ for spectral-shape features")
    ap.add_argument("--no-write", action="store_true", help="report only; leave the SigMF metadata untouched")
    ap.add_argument("--margin", type=float, default=0.10, help="ambiguity margin between the two best entries")
    ap.add_argument("--report-dir", default=None, help="where to write lookup_regions.csv / lookup_summary.json (default: next to the input)")
    ap.add_argument("--check-manifest", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    m = mio.load_manifest(a.manifest)
    if a.check_manifest:
        return check_manifest(m)
    if not a.input:
        ap.error("input is required")
    inp = Path(a.input).expanduser()
    if a.masks:
        caps = regions_from_masks(inp, a.masks)
        report_dir = Path(a.report_dir) if a.report_dir else inp
        can_write = False
    elif inp.is_file() and inp.name.endswith(".sigmf-meta"):
        caps = regions_from_sigmf(inp, a.cols)
        report_dir = Path(a.report_dir) if a.report_dir else inp.parent
        can_write = True
    elif inp.is_dir():
        caps = regions_from_preprocessed(inp, a.cols)
        report_dir = Path(a.report_dir) if a.report_dir else inp
        can_write = True
    else:
        ap.error(f"cannot read {inp}")
    report_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for stem, cap in caps.items():
        tracks, rows = run_capture(stem, cap, m, not a.no_iq, a.margin)
        all_rows.extend(rows)
        if not a.quiet:
            c = Counter(r["estimate"] for r in rows)
            print(f"{stem}: {len(cap['regions'])} regions -> {len(tracks)} tracks; " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    n_written = 0
    if can_write and not a.no_write:
        n_written = write_labels(all_rows, inp if inp.is_dir() else None)
    fields = [k for k in all_rows[0].keys() if not k.startswith("_")] if all_rows else ["capture"]
    with open(report_dir / "lookup_regions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    summ = summarize(all_rows)
    summ.update({"tool": VERSION, "manifest": Path(m["_path"]).name, "input": str(inp), "annotations_written": n_written,
                 "iq_features": not a.no_iq})
    (report_dir / "lookup_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    if not a.quiet:
        print(f"tracks {summ['tracks']}, labelled {summ['labelled']}: {summ['labels']}  -> {report_dir / 'lookup_regions.csv'}"
              + (f"; {n_written} annotations updated" if n_written else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
