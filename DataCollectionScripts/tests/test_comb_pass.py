#!/usr/bin/env python3
"""Unit test of the cross-track spur-comb pass (`common_waveforms_look_up.comb_pass`) on synthetic track sets.

Cases: the N320 bench comb (1.000 MHz teeth at x.27 MHz, painted 6-10 bins wide, flickering in the mask) inside
the 850 MHz uplink band next to real uplink blobs; a 5 MHz paging capture with five teeth next to two strong real
pagers; a dense uplink band of a hundred narrow bursty tracks (no comb may form by chance); a crowded FM band on
its 200 kHz raster (no comb). The classifier's `explained` set is evidence-based and tested separately; here it is
the set of tracks with strong evidence (the pagers).

    python3 tests/test_comb_pass.py [--seeds 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common_waveforms_look_up as lk  # noqa: E402
import manifest_io as mio  # noqa: E402


def mk(tid: int, fc: float, bw: float, pers: float, rate: float, fc_cap: float, snr: float = 5.0, obw: float | None = None,
       temporal: str = "bursty") -> lk.Track:
    bin_hz = rate / 1024
    r = lk.Region("t", fc - bw / 2, fc + bw / 2, 0.0, 1e-3, 0, rate, fc_cap, bin_hz, 2e-5, 1e-2)
    t = lk.Track(tid, [r], [r], fc, bw, fc - bw / 2, fc + bw / 2, 0.5, pers, 1e-3, 1.0, 47, temporal)
    t.iq = {"snr_db": snr, "obw99_hz": obw if obw is not None else 2.5 * bw, "line": False}
    return t


def run(tracks: list[lk.Track], manifest: dict, explained: set | None = None) -> dict[int, str]:
    for t in tracks:
        t.forced_label = ""
    lk.comb_pass(tracks, manifest, explained or set())
    return {t.tid: t.forced_label for t in tracks if t.forced_label}


def case_uplink_850(manifest: dict, rng: np.random.Generator) -> tuple[bool, str]:
    rate, fc_cap, b = 25e6, 836.5e6, 25e6 / 1024
    tracks, teeth = [], set()
    for k in range(10):
        fc = 839.27e6 + k * 1.0e6 + rng.uniform(-5e3, 5e3)
        tracks.append(mk(len(tracks), fc, rng.integers(6, 11) * b, rng.uniform(0.5, 0.98), rate, fc_cap, snr=rng.uniform(4.5, 6.5)))
        teeth.add(tracks[-1].tid)
    for _ in range(15):                                   # real uplink blobs: wide, short-lived
        tracks.append(mk(len(tracks), rng.uniform(824.5e6, 838.5e6), rng.uniform(0.5e6, 3e6), rng.uniform(0.05, 0.3), rate, fc_cap, snr=rng.uniform(-5, 12)))
    tracks.append(mk(len(tracks), 843.77e6, 7 * b, 0.8, rate, fc_cap))      # one stray persistent line at a half step: not a 0.5 MHz comb
    stray = tracks[-1].tid
    n = 0
    while n < 6:                                          # narrow bursty tracks off the grid
        fc = rng.uniform(824.5e6, 848.5e6)
        if abs(((fc - 839.27e6) / 1e6 + 0.5) % 1.0 - 0.5) * 1e6 > 80e3:
            tracks.append(mk(len(tracks), fc, rng.integers(6, 11) * b, rng.uniform(0.1, 0.45), rate, fc_cap))
            n += 1
    forced = run(tracks, manifest)
    ok = set(forced) == teeth and set(forced.values()) == {"spur_comb_1MHz"} and stray not in forced
    return ok, f"850 uplink: {len(forced)} forced of {len(teeth)} teeth ({sorted(set(forced.values()))}), stray half-step line {'clean' if stray not in forced else 'TAKEN'}"


def case_paging(manifest: dict) -> tuple[bool, str]:
    rate, fc_cap = 5e6, 930.3e6
    tracks = []
    for fc in (928.2731e6, 929.2697e6, 930.2712e6, 931.2717e6, 932.2727e6):
        tracks.append(mk(len(tracks), fc, 29e3, 1.0, rate, fc_cap, snr=7.0, temporal="continuous"))
    pagers = []
    for fc, snr in ((929.6164e6, 25.0), (931.062e6, 32.0)):
        tracks.append(mk(len(tracks), fc, 29e3, 1.0, rate, fc_cap, snr=snr, obw=16e3, temporal="continuous"))
        pagers.append(tracks[-1].tid)
    forced = run(tracks, manifest, explained=set(pagers))      # the pagers decode: strong evidence
    ok = set(forced) == {0, 1, 2, 3, 4}
    return ok, f"paging 929: {len(forced)} forced ({sorted(forced)}), pagers {pagers} {'clean' if not set(pagers) & set(forced) else 'MISLABELLED'}"


def case_dense_uplink(manifest: dict, rng: np.random.Generator) -> tuple[bool, str]:
    rate, fc_cap = 20e6, 707e6
    tracks = [mk(i, rng.uniform(697.2e6, 716.8e6), rng.uniform(150e3, 320e3), rng.uniform(0.5, 0.9), rate, fc_cap, snr=rng.uniform(-6, 10))
              for i in range(140)]
    forced = run(tracks, manifest)
    return not forced, f"dense uplink: {len(forced)} forced"


def case_fm_band(manifest: dict, rng: np.random.Generator) -> tuple[bool, str]:
    rate, fc_cap = 24.576e6, 98e6
    slots = rng.choice(100, size=30, replace=False)
    tracks = []
    for i, sl in enumerate(sorted(slots)):
        snr = rng.uniform(3, 15)
        tracks.append(mk(i, 88.1e6 + 0.2e6 * sl + rng.uniform(-6e3, 6e3), rng.uniform(180e3, 250e3), 1.0, rate, fc_cap, snr=snr,
                         obw=180e3 if snr >= 10 else None, temporal="continuous"))
    strong = {t.tid for t in tracks if t.iq["snr_db"] >= 8}       # pilot measured -> explained by the classifier
    forced = run(tracks, manifest, explained=strong)
    return not forced, f"FM band: {len(forced)} forced of {len(tracks) - len(strong)} weak stations"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    manifest = mio.load_library()
    results = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        results.append(("uplink_850", seed) + case_uplink_850(manifest, rng))
        results.append(("dense_uplink", seed) + case_dense_uplink(manifest, rng))
        results.append(("fm_band", seed) + case_fm_band(manifest, rng))
    results.append(("paging", 0) + case_paging(manifest))
    n_ok = sum(1 for r in results if r[2])
    for name, seed, ok, msg in results:
        if not ok or seed == 0:
            print(f"{'ok  ' if ok else 'FAIL'} {name} seed {seed}: {msg}")
    print(f"{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
