"""Build the ORDERED comprehensive composition config.

Every library waveform & variation, at its original (unscaled) power. Ordered:
sectioned by class (fixed class order), and within each class sorted by
bandwidth (largest first). Classes never share time (each time group is
single-class). Within a class, waveforms are frequency-packed (3 x 50 MHz slots,
stacking) emitted largest-bandwidth-first. The whole ordered set is then repeated
for each duration in DURATIONS_MS (every waveform tiled/cut to that length).
"""

from __future__ import annotations
import csv
import os
from collections import deque
import yaml

import geometry as geo
import protocol as proto

FS = geo.FS
CLASS_ORDER = ["BPSK", "QPSK", "16QAM", "Narrowband_FM", "Broadband_FM",
               "Bluetooth", "OFDM", "802_11ax", "5G_Downlink"]
DURATIONS_MS = [20.0, 10.0, 5.0, 1.0, 0.2, 0.04]
STACK = 3
GUARD1_MS, GUARD2_MS = 2.5, 5.0


def _read_manifest(path):
    by_class = {c: [] for c in CLASS_ORDER}
    with open(path) as f:
        for r in csv.DictReader(f):
            c = r["class"]
            if c in by_class:
                by_class[c].append({"mat": r["matFile"],
                                     "occ": float(r["occupiedBandwidthHz"]),
                                     "name": r["waveformName"]})
    return by_class


def _pack_class(wfs):
    """Frequency-pack one class (largest BW first) into time groups."""
    wfs = sorted(wfs, key=lambda w: -w["occ"])
    for w in wfs:
        w["nslots"] = geo.required_blocks(w["occ"])
    threes = [w for w in wfs if w["nslots"] == 3]
    twos = [w for w in wfs if w["nslots"] == 2]
    onesq = deque([w for w in wfs if w["nslots"] == 1])  # BW-desc

    def take_stack():
        out, bsum = [], 0.0
        while onesq and len(out) < STACK:
            w = onesq[0]
            add = w["occ"] + (geo.GUARD if out else 0.0)
            if bsum + add <= geo.usable_bw(1) + 1.0:
                out.append(w); bsum += add; onesq.popleft()
            else:
                break
        return out

    groups = []
    for w in threes:
        groups.append([{"slots": [0, 1, 2], "wfs": [w]}])
    for w in twos:
        blocks = [{"slots": [0, 1], "wfs": [w]}]
        free = take_stack()
        if free:
            blocks.append({"slots": [2], "wfs": free})
        groups.append(blocks)
    while onesq:
        blocks = []
        for slot in (0, 1, 2):
            st = take_stack()
            if st:
                blocks.append({"slots": [slot], "wfs": st})
        if blocks:
            groups.append(blocks)
    return groups


def build(manifest_csv, out_yaml, lib_rel, zc_rel,
          name="comprehensive_ordered", waveform_gain_db=0.0):
    by_class = _read_manifest(manifest_csv)
    # pack each class once (same packing reused for every duration)
    packed = {c: _pack_class(by_class[c]) for c in CLASS_ORDER}

    time_groups = []
    for dur in DURATIONS_MS:
        for c in CLASS_ORDER:
            for grp in packed[c]:
                blocks = []
                for blk in grp:
                    blocks.append({"slots": blk["slots"],
                                   "waveforms": [{"mat": w["mat"], "duration_ms": dur}
                                                 for w in blk["wfs"]]})
                time_groups.append({"blocks": blocks})

    cfg = {"name": name, "output_dir": "composites",
           "zc_file": zc_rel, "library_root": lib_rel,
           "guard1_ms": GUARD1_MS, "guard2_ms": GUARD2_MS,
           "power_mode": "original", "ref_rms": 0.95, "peak_normalize": 0.95,
           "waveform_gain_db": float(waveform_gain_db),   # waveforms only; ZC/meta untouched
           "time_groups": time_groups}
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)

    # size estimate
    groups_per_pass = sum(len(packed[c]) for c in CLASS_ORDER)
    zc_len = 2955
    data_start = proto.data_start(zc_len, int(round(GUARD1_MS * 1e-3 * FS)))
    g2 = int(round(GUARD2_MS * 1e-3 * FS))
    total = 0
    for dur in DURATIONS_MS:
        dn = int(round(dur * 1e-3 * FS))
        total += groups_per_pass * (data_start + dn + g2)
    nwf = sum(len(by_class[c]) for c in CLASS_ORDER)
    print(f"classes: " + ", ".join(f"{c}={len(by_class[c])}" for c in CLASS_ORDER))
    print(f"{nwf} waveforms x {len(DURATIONS_MS)} durations = {nwf*len(DURATIONS_MS)} placements")
    print(f"{groups_per_pass} time groups/pass x {len(DURATIONS_MS)} = {groups_per_pass*len(DURATIONS_MS)} groups")
    print(f"estimated: {total} samples, {total/FS:.2f} s, {total*8/1e9:.2f} GB")
    return out_yaml


if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    # optional 1st arg = waveform power offset (dB); names the config accordingly
    gain = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    name = "comprehensive_ordered" if gain == 0.0 else f"comprehensive_ordered_{gain:g}dB"
    build(os.path.join(here, "..", "generated_waveforms_24576", "waveform_manifest.csv"),
          os.path.join(here, name + ".yaml"),
          "../generated_waveforms_24576",
          "../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat",
          name=name, waveform_gain_db=gain)
