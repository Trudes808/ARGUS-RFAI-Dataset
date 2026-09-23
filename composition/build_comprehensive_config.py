"""Build a comprehensive composition config packing the ENTIRE waveform library.

Every library waveform is placed exactly once, with a varied power level (no
power-level duplication). Waveforms are bin-packed into time groups of three
50 MHz frequency slots: sub-50 MHz waveforms are stacked in frequency within a
slot, 2-slot (<=110 MHz) and 3-slot (<=170 MHz) waveforms span merged slots, and
the free slot of a 2-slot group is filled with stacked narrow waveforms.
"""

from __future__ import annotations
import csv
import os
import sys
from collections import deque
import yaml

import geometry as geo

STACK = 3                       # max waveforms stacked per block (metadata budget)
POWERS = [0.0, -3.0, -6.0, -9.0, -12.0]


def build(manifest_csv, out_yaml, lib_rel, zc_rel):
    rows = []
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            rows.append({"mat": r["matFile"], "occ": float(r["occupiedBandwidthHz"]),
                         "cls": r["class"], "name": r["waveformName"]})
    for r in rows:
        r["nslots"] = geo.required_blocks(r["occ"])

    threes = [r for r in rows if r["nslots"] == 3]
    twos = [r for r in rows if r["nslots"] == 2]
    ones = sorted([r for r in rows if r["nslots"] == 1], key=lambda r: -r["occ"])
    onesq = deque(ones)

    pcount = [0]
    def nextp():
        p = POWERS[pcount[0] % len(POWERS)]; pcount[0] += 1
        return p

    def take_stack():
        wfs, bsum = [], 0.0
        while onesq and len(wfs) < STACK:
            w = onesq[0]
            add = w["occ"] + (geo.GUARD if wfs else 0.0)
            if bsum + add <= geo.usable_bw(1) + 1.0:
                wfs.append({"mat": w["mat"], "power_db": nextp()}); bsum += add
                onesq.popleft()
            else:
                break
        return wfs

    groups = []
    for w in threes:
        groups.append({"blocks": [{"slots": [0, 1, 2],
                                   "waveforms": [{"mat": w["mat"], "power_db": nextp()}]}]})
    for w in twos:
        blocks = [{"slots": [0, 1], "waveforms": [{"mat": w["mat"], "power_db": nextp()}]}]
        free = take_stack()
        if free:
            blocks.append({"slots": [2], "waveforms": free})
        groups.append({"blocks": blocks})
    while onesq:
        blocks = []
        for slot in (0, 1, 2):
            wfs = take_stack()
            if wfs:
                blocks.append({"slots": [slot], "waveforms": wfs})
        if blocks:
            groups.append({"blocks": blocks})

    cfg = {"name": "comprehensive", "output_dir": "composites",
           "zc_file": zc_rel, "library_root": lib_rel, "guard_time_ms": 25,
           "ref_rms": 0.15, "peak_normalize": 0.95, "time_groups": groups}
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)

    nwf = len(rows)
    print(f"{nwf} waveforms ({len(threes)} 3-slot, {len(twos)} 2-slot, {len(ones)} 1-slot) "
          f"-> {len(groups)} time groups -> {out_yaml}")
    return out_yaml


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    build(os.path.join(here, "..", "generated_waveforms_24576", "waveform_manifest.csv"),
          os.path.join(here, "comprehensive.yaml"),
          "../generated_waveforms_24576",
          "../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat")
