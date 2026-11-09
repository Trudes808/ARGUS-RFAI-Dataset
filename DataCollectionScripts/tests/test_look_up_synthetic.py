#!/usr/bin/env python3
"""Synthetic accuracy test of the common-waveform look-up (matcher logic only, no detector, no IQ).

For every labelable manifest entry (tier A/B) this draws tracks that obey the entry's physics (band,
channel raster, bandwidth set, duty / persistence, burst length), quantised to a random capture geometry
(sample rate -> bin width and row time, +-1 bin edge noise), and checks that the look-up returns the entry's
label (recall). It then draws negatives from the same entry with ONE property violated (off band, off
grid, 2.5x bandwidth as when two carriers merge, wrong temporal class) plus "unusual" emitters (random
off-grid OFDM bursts, hoppers in quiet bands) and checks that NO label (or at least not that label) is
produced (false-label rate). Results: tests/results/synthetic_lookup_eval.md

  python3 tests/test_look_up_synthetic.py [--n 150] [--seed 1]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import manifest_io as mio  # noqa: E402
import common_waveforms_look_up as lk  # noqa: E402

RATES = [5.0e6, 10.0e6, 12.5e6, 20.0e6, 24.576e6, 25.0e6, 31.25e6, 50.0e6, 61.44e6, 100.0e6, 200.0e6, 245.76e6]


def pick_fc(rng, e, on_grid=True):
    ranges = e["freq_ranges_hz"]
    lo, hi = ranges[rng.integers(len(ranges))]
    if on_grid and e.get("channel_centers_hz") and (not e.get("raster") or rng.random() < 0.5):
        ch = [c for c in e["channel_centers_hz"] if lo <= c <= hi] or e["channel_centers_hz"]
        return float(rng.choice(ch))
    if on_grid and e.get("raster"):
        s, off = e["raster"]["spacing_hz"], e["raster"]["offset_hz"]
        k_lo, k_hi = int(np.ceil((lo - off) / s)), int(np.floor((hi - off) / s))
        if k_hi >= k_lo:
            return float(off + rng.integers(k_lo, k_hi + 1) * s)
    return float(rng.uniform(lo, hi))


def pick_bw(rng, e):
    sets = e.get("bandwidth_sets_hz")
    if sets:
        a, b = sets[rng.integers(len(sets))]
        return float(rng.uniform(a, b))
    bw = e["bandwidth_hz"]
    return float(rng.uniform(bw["min"], bw["max"]))


def make_track(rng, fc, bw, temporal, burst_range, rate=None):
    """Quantise to a capture geometry and build a Track with one representative region per frame."""
    rate = rate or float(rng.choice([r for r in RATES if r >= 1.3 * bw] or [RATES[-1]]))
    geo = mio.frame_geometry(rate)
    bin_hz, row_s, frame_s = geo["bin_hz"], geo["row_s"], geo["frame_s"]
    fc_cap = fc + rng.uniform(-0.38, 0.38) * rate
    # quantised edges with +-1 bin noise
    f_lo = fc_cap - rate / 2 + (np.floor((fc - bw / 2 - fc_cap + rate / 2) / bin_hz) + rng.integers(-1, 2)) * bin_hz
    f_hi = fc_cap - rate / 2 + (np.ceil((fc + bw / 2 - fc_cap + rate / 2) / bin_hz) + rng.integers(-1, 2)) * bin_hz
    if f_hi - f_lo < bin_hz:
        f_hi = f_lo + bin_hz
    n_frames = 12
    regions = []
    if temporal == "continuous":
        pres = rng.uniform(0.85, 1.0)
        for k in range(n_frames):
            if rng.random() < pres or k == 0:
                regions.append(lk.Region("syn", f_lo, f_hi, k * frame_s, frame_s, k, rate, fc_cap, bin_hz, row_s, frame_s))
    elif temporal == "tdd":
        # 5 ms TDD frames: ~3.5 ms downlink slots separated by ~1.5 ms uplink gaps, every frame
        for k in range(n_frames):
            t0 = 0.0
            while t0 < frame_s - 1e-6:
                dur = min(rng.uniform(3.0e-3, 4.0e-3), frame_s - t0)
                regions.append(lk.Region("syn", f_lo, f_hi, k * frame_s + t0, dur, k, rate, fc_cap, bin_hz, row_s, frame_s))
                t0 += 5.0e-3
    else:  # bursty
        pres = rng.uniform(0.1, 0.9)
        bmin, bmax = burst_range
        for k in range(n_frames):
            if rng.random() < pres or k == 0:
                dur = float(np.exp(rng.uniform(np.log(bmin), np.log(bmax))))
                dur = max(row_s, np.ceil(dur / row_s) * row_s)
                if dur >= frame_s:  # a transmission longer than the frame fills whole frames
                    regions.append(lk.Region("syn", f_lo, f_hi, k * frame_s, frame_s, k, rate, fc_cap, bin_hz, row_s, frame_s))
                    continue
                nb = int(rng.integers(1, 4))
                for j in range(nb):
                    if k * frame_s + j * dur * 2 + dur <= (k + 1) * frame_s:
                        regions.append(lk.Region("syn", f_lo, f_hi, k * frame_s + j * dur * 2, dur, k, rate, fc_cap, bin_hz, row_s, frame_s))
    tracks = lk.build_tracks(regions, n_frames)
    return tracks[0]


def label_of(track, manifest):
    best, _, why = lk.decide(track, manifest)
    return (best.label if best else None), why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--manifest", default=str(HERE.parent / "common_ota_waveforms_manifest_usa.yaml"))
    ap.add_argument("--out", default=str(HERE / "results" / "synthetic_lookup_eval.md"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    m = mio.load_manifest(a.manifest)
    entries = [e for e in m["entries"] if e["tier"] in ("A", "B") and not e.get("relative_to_capture_center") and e.get("kind") != "cross_track"]
    by_id = mio.entry_by_id(m)
    parent_label = {e["label"]: by_id[e["parent"]]["label"] for e in entries if e.get("parent")}
    rows = []
    tot = Counter()
    conf_matrix = Counter()
    for e in entries:
        temporal = e["temporal"]
        burst = (e.get("burst_s") or {"min": 1e-4, "max": 1e-2})
        burst_range = (burst["min"], burst["max"])
        # ---------------- positives
        hit = other = none = coarse = 0
        for _ in range(a.n):
            tmode = temporal if temporal != "either" else rng.choice(["continuous", "bursty"])
            t = make_track(rng, pick_fc(rng, e), pick_bw(rng, e), tmode, burst_range)
            lab, _ = label_of(t, m)
            if lab == e["label"]:
                hit += 1
            elif lab is None:
                none += 1
            elif lab == parent_label.get(e["label"]):
                coarse += 1          # the generic parent (e.g. ISM_900MHz_device for LoRa) is still correct
            else:
                other += 1
                conf_matrix[(e["label"], lab)] += 1
        # ---------------- negatives (one violated property each)
        neg = defaultdict(lambda: [0, 0, 0])  # kind -> [n, labelled_as_entry, labelled_other]
        for _ in range(a.n):
            kind = rng.choice(["off_band", "off_grid", "merged_bw", "wrong_temporal"])
            fc, bw = pick_fc(rng, e), pick_bw(rng, e)
            tmode = temporal if temporal != "either" else rng.choice(["continuous", "bursty"])
            if kind == "off_band":
                span = max(hi for _, hi in e["freq_ranges_hz"]) - min(lo for lo, _ in e["freq_ranges_hz"])
                fc = fc + rng.choice([-1, 1]) * (span + rng.uniform(0.02, 0.3) * fc)
                fc = float(np.clip(fc, 2e6, 7.1e9))
            elif kind == "off_grid":
                if e.get("raster"):
                    fc += e["raster"]["spacing_hz"] * rng.uniform(0.3, 0.7)
                elif e.get("channel_centers_hz"):
                    ch = np.sort(np.asarray(e["channel_centers_hz"], dtype=float))
                    sep = float(np.min(np.diff(ch))) if len(ch) > 1 else 2 * bw
                    fc += sep * rng.uniform(0.3, 0.7)
                else:
                    continue  # no grid to violate
            elif kind == "merged_bw":
                bw *= 2.5
            else:
                tmode = {"continuous": "bursty", "bursty": "continuous", "tdd": "bursty"}.get(tmode, "bursty")
                if temporal == "either":
                    continue
            t = make_track(rng, fc, bw, tmode, burst_range)
            lab, _ = label_of(t, m)
            neg[kind][0] += 1
            if lab == e["label"]:
                neg[kind][1] += 1
            elif lab is not None and lab != parent_label.get(e["label"]):
                neg[kind][2] += 1
        rows.append({"id": e["id"], "label": e["label"], "tier": e["tier"], "temporal": temporal, "requires": ",".join(e.get("requires", [])),
                     "hit": hit, "none": none, "other": other, "coarse": coarse, "neg": dict(neg)})
        tot["pos"] += a.n
        tot["hit"] += hit
        tot["coarse"] += coarse
        tot["other"] += other
        for k, (n_, same, oth) in neg.items():
            tot["neg"] += n_
            tot["neg_same"] += same
            tot["neg_other"] += oth
        print(f"{e['id']:<24} recall {hit / a.n:5.2f}  coarse {coarse / a.n:5.2f}  mislabel {other / a.n:5.2f}  negatives: " +
              " ".join(f"{k}:{same}/{n_}({oth})" for k, (n_, same, oth) in neg.items()), flush=True)
    # ---------------- unusual emitters (should stay unlabelled)
    unusual = {"drone OFDM 10/20 MHz off-grid at 2.4 GHz": 0, "narrow hopper in the 1.3 GHz quiet band": 0, "40 MHz OFDM at 5.9 GHz": 0,
               "continuous 2 MHz carrier at 1.5 GHz": 0}
    u_lab = Counter()
    for _ in range(a.n):
        for name in list(unusual):
            if name.startswith("drone"):
                bw = float(rng.choice([10e6, 20e6]))
                fc = rng.uniform(2405e6, 2478e6)
                while min(abs(fc - c) for c in range(2412_000_000, 2463_000_000, 5_000_000)) < 1.5e6:
                    fc = rng.uniform(2405e6, 2478e6)
                t = make_track(rng, fc, bw, "bursty", (0.5e-3, 5e-3))
            elif name.startswith("narrow hopper"):
                t = make_track(rng, rng.uniform(1340e6, 1360e6), rng.uniform(50e3, 300e3), "bursty", (0.2e-3, 5e-3))
            elif name.startswith("40 MHz"):
                t = make_track(rng, rng.uniform(5880e6, 5920e6), 40e6, "bursty", (0.5e-3, 5e-3))
            else:
                t = make_track(rng, rng.uniform(1450e6, 1520e6), 2e6, "continuous", (1, 1))
            lab, _ = label_of(t, m)
            if lab is not None:
                unusual[name] += 1
                u_lab[(name, lab)] += 1
    # ---------------- report
    L = ["# Synthetic look-up test", "",
         f"{len(entries)} labelable entries, {a.n} positives and {a.n} single-violation negatives each (seed {a.seed}); tracks quantised to a random capture rate "
         f"from {', '.join(f'{r / 1e6:g}' for r in RATES)} MSps with +-1 bin edge noise; no IQ features. A negative keeps every property of the entry except one: "
         "`off_band` (centre moved outside the allocation), `off_grid` (centre between channels), `merged_bw` (2.5x the bandwidth, as when adjacent carriers merge into one box), "
         "`wrong_temporal` (continuous <-> bursty).", "",
         f"**Positives:** exact label {tot['hit'] / tot['pos']:.3f} ({tot['hit']}/{tot['pos']}), generic parent label {tot['coarse'] / tot['pos']:.3f}, "
         f"mislabelled as another entry {tot['other'] / tot['pos']:.3f} ({tot['other']}); the rest stay \"signal\".  ",
         f"**Negatives:** labelled as the violated entry {tot['neg_same'] / max(tot['neg'], 1):.3f} ({tot['neg_same']}/{tot['neg']}), labelled as some other entry {tot['neg_other'] / max(tot['neg'], 1):.3f}.", "",
         "Low recall for narrow channels (NBFM voice, pagers, AIS) is expected at coarse capture rates: a 12 kHz channel at 200 MSps is a one-bin box and its raster cannot be checked, "
         "so the look-up keeps \"signal\" unless the band alone identifies the service.", "",
         "| entry | tier | temporal | requires | recall | parent label | mislabelled | off_band | off_grid | merged_bw | wrong_temporal |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def cell(k):
            v = r["neg"].get(k)
            return "-" if not v else f"{v[1]}/{v[0]}" + (f" (+{v[2]} other)" if v[2] else "")
        L.append(f"| {r['id']} | {r['tier']} | {r['temporal']} | {r['requires']} | {r['hit'] / a.n:.2f} | {r['coarse'] / a.n:.2f} | {r['other'] / a.n:.2f} | {cell('off_band')} | {cell('off_grid')} | {cell('merged_bw')} | {cell('wrong_temporal')} |")
    L += ["", "Negative cells read `labelled-as-this-entry / drawn (+labelled as another entry)`. `merged_bw` is only a true violation for entries whose bandwidth range is narrower than 2.5x; "
          "for cellular blocks a 2.5x-wide box is still a legal (multi-carrier) downlink. `off_grid` at coarse rates is unresolvable and is accepted for band-exclusive entries.", "",
          "## Unusual emitters (must stay \"signal\")", "", "| emitter | labelled | as |", "|---|---|---|"]
    for name, n_ in unusual.items():
        labs = ", ".join(f"{lab} {v}" for (nm, lab), v in u_lab.items() if nm == name) or "-"
        L.append(f"| {name} | {n_}/{a.n} | {labs} |")
    if conf_matrix:
        L += ["", "## Positives mislabelled as another entry", "", "| true | predicted | n |", "|---|---|---|"]
        for (t_, p_), n_ in conf_matrix.most_common():
            L.append(f"| {t_} | {p_} | {n_} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"\nrecall {tot['hit'] / tot['pos']:.3f}, mislabel {tot['other'] / tot['pos']:.3f}, negatives labelled-as-entry {tot['neg_same'] / max(tot['neg'], 1):.3f}, "
          f"unusual labelled: {unusual}\nwrote {a.out}")


if __name__ == "__main__":
    main()
