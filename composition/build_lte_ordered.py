"""Build the ordered LTE-only composition configs.

Companion to build_comprehensive_ordered.py, but the library is *only* the LTE
variations (generate_lte_24576.m). Same ordered recipe: waveforms sorted by
bandwidth (largest first) and frequency-packed into time groups (3 x 50 MHz
slots, up to 3 stacked per block), then the whole set repeated for each duration
in DURATIONS_MS. Emits two configs mirroring comprehensive_ordered:
  lte_ordered.yaml         -- original waveform power
  lte_ordered_-30dB.yaml   -- waveforms 30 dB below the ZC/metadata reference

Usage:
  python build_lte_ordered.py          # both configs (0 dB and -30 dB)
"""

from __future__ import annotations
import csv
import os
from collections import deque
import yaml

import geometry as geo
import protocol as proto

FS = geo.FS
DURATIONS_MS = [20.0, 10.0, 5.0, 1.0, 0.2, 0.04]
STACK = 3
GUARD1_MS, GUARD2_MS = 2.5, 5.0

# matFile paths in lte_manifest.csv are relative to this root.
LIB_REL = "../generated_waveforms_24576"


def _read_lte(manifest_csv):
    wfs = []
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            if r["class"] == "LTE":
                wfs.append({"mat": r["matFile"],
                            "occ": float(r["occupiedBandwidthHz"]),
                            "name": r["waveformName"]})
    return wfs


def _pack(wfs):
    """Frequency-pack the LTE class (largest BW first) into time groups."""
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


def build(manifest_csv, out_yaml, lib_rel, zc_rel, name, waveform_gain_db=0.0):
    wfs = _read_lte(manifest_csv)
    if not wfs:
        raise SystemExit(f"No LTE waveforms found in {manifest_csv} (run generate_lte_24576.m first)")
    packed = _pack(wfs)

    time_groups = []
    for dur in DURATIONS_MS:
        for grp in packed:
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
           "waveform_gain_db": float(waveform_gain_db),
           "time_groups": time_groups}
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)

    # size estimate
    zc_len = 2955
    data_start = proto.data_start(zc_len, int(round(GUARD1_MS * 1e-3 * FS)))
    g2 = int(round(GUARD2_MS * 1e-3 * FS))
    total = 0
    for dur in DURATIONS_MS:
        dn = int(round(dur * 1e-3 * FS))
        total += len(packed) * (data_start + dn + g2)
    print(f"{len(wfs)} LTE waveforms -> {len(packed)} groups/pass x {len(DURATIONS_MS)} "
          f"= {len(packed)*len(DURATIONS_MS)} time groups")
    print(f"  {name}: est {total} samples, {total/FS:.2f} s, {total*8/1e9:.2f} GB  -> {out_yaml}")
    return out_yaml


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = os.path.join(here, LIB_REL, "lte_manifest.csv")
    zc_rel = "../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat"
    build(manifest, os.path.join(here, "lte_ordered.yaml"), LIB_REL, zc_rel,
          "lte_ordered", waveform_gain_db=0.0)
    build(manifest, os.path.join(here, "lte_ordered_-30dB.yaml"), LIB_REL, zc_rel,
          "lte_ordered_-30dB", waveform_gain_db=-30.0)
