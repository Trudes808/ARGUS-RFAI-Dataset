#!/usr/bin/env python3
"""Synthetic end-to-end test of the LUT classification: generate captures for every library entry that has a
recipe, classify them from their ground-truth boxes, and score labels and descriptor verification per entry.

    python3 tests/test_lut_synthetic.py --out /tmp/lut_synth --draws 4 --snr 10 30
    python3 tests/test_lut_synthetic.py --out /tmp/lut_synth --reuse        # score an existing generation

Writes tests/results/lut_synthetic_eval.md and lut_synthetic_rows.csv (one row per synthetic capture with the
measured SDW summary next to the truth, for the synthetic-vs-real comparison).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import manifest_io as mio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--dur", type=float, default=1.0)
    ap.add_argument("--snr", nargs=2, type=float, default=[10.0, 30.0])
    ap.add_argument("--entries", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reuse", action="store_true", help="do not regenerate; classify / score what is in --out")
    ap.add_argument("--no-classify", action="store_true", help="score the existing report only")
    ap.add_argument("--results", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    res_dir = Path(a.results)
    res_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    if not a.reuse:
        cmd = [py, str(HERE.parent / "synth_lut_scenes.py"), "--out", str(out), "--draws", str(a.draws), "--dur", str(a.dur), "--snr", str(a.snr[0]), str(a.snr[1]), "--seed", str(a.seed)]
        if a.entries:
            cmd += ["--entries", *a.entries]
        subprocess.run(cmd, check=True)
    if not a.no_classify:
        subprocess.run([py, str(HERE.parent / "lut_classify.py"), str(out), "--no-write", "--report-dir", str(out / "report"), "--quiet"], check=True)
    lib = mio.load_library()
    by_id = mio.entry_by_id(lib)
    truths = {}
    for tp in out.glob("*.truth.json"):
        t = json.loads(tp.read_text())
        truths[tp.name[: -len(".truth.json")]] = t
    rows = list(csv.DictReader(open(out / "report" / "lut_regions.csv")))
    # the track that overlaps the emission most (synthetic captures hold one emission; the DC spur may add a track)
    per_cap: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_cap[r["capture"]].append(r)
    results = []
    for stem, t in sorted(truths.items()):
        cands = per_cap.get(stem) or per_cap.get(stem.rsplit(".", 1)[0], [])     # older generations truncated the stem at its last dot
        best = None
        for r in cands:
            lo, hi = float(r["f_lo_hz"]), float(r["f_hi_hz"])
            ov = min(hi, t["f_emission_hz"] + t["bw_hz"] / 2) - max(lo, t["f_emission_hz"] - t["bw_hz"] / 2)
            if ov > 0 and (best is None or ov > best[0]):
                best = (ov, r)
        r = best[1] if best else None
        e = by_id[t["entry"]]
        want = "~" + e["label"]
        got = r["estimate"] if r else "no track"
        base = got.split("/")[0]
        status = "exact" if base == want else ("parent" if (r and r.get("entry") and (by_id.get(r["entry"], {}).get("id") == e.get("parent") or e["id"] == by_id.get(r["entry"], {}).get("parent"))) else ("signal" if got == "signal" else ("none" if not r else "wrong")))
        results.append({"capture": stem, "entry": t["entry"], "label": want, "snr_db": round(t["snr_db"], 1), "rate_hz": t["rate_hz"], "bw_hz": t["bw_hz"], "n_bursts": t["n_bursts"],
                        "estimate": got, "status": status, "confidence": r["confidence"] if r else "", "v1_confidence": r["v1_confidence"] if r else "",
                        "descriptor_factor": r["descriptor_factor"] if r else "", "verified": r["verified"] if r else "", "contradictions": r["contradictions"] if r else "",
                        "sdw_class": r["sdw_class"] if r else "", "sdw_snr_db": r["sdw_snr_db"] if r else "", "sdw_finst_line": r["sdw_finst_line"] if r else "", "sdw_abs2_line": r["sdw_abs2_line"] if r else "",
                        "sdw_sq_spacing": r["sdw_sq_spacing"] if r else "", "sdw_ofdm_scs_hz": r["sdw_ofdm_scs_hz"] if r else "", "sdw_ofdm_cp_us": r["sdw_ofdm_cp_us"] if r else "",
                        "sdw_burst_ms": r["sdw_burst_ms"] if r else "", "sdw_pri_ms": r["sdw_pri_ms"] if r else "", "sdw_gap_us": r["sdw_gap_us"] if r else "", "sdw_period_ms": r["sdw_period_ms"] if r else "",
                        "sdw_chirp_hz_s": r["sdw_chirp_hz_s"] if r else "", "sdw_dev_hz": r["sdw_dev_hz"] if r else "", "sdw_pulse_us": r["sdw_pulse_us"] if r else "",
                        "evidence": (r["evidence"] if r else "")[:500]})
    with open(res_dir / "lut_synthetic_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    # per-entry table
    per = defaultdict(Counter)
    ver = defaultdict(Counter)
    n_ver = Counter()
    for r in results:
        per[r["entry"]][r["status"]] += 1
        if r["verified"]:
            for v in r["verified"].split(";"):
                ver[r["entry"]][v] += 1
        n_ver[r["entry"]] += 1
    tot = Counter(r["status"] for r in results)
    n = len(results)
    L = ["# LUT classification on synthetic captures", "",
         f"{n} synthetic captures ({a.draws} draws per entry, {a.dur} s, SNR {a.snr[0]:g}-{a.snr[1]:g} dB, seed {a.seed}), classified from their ground-truth boxes "
         f"(detector bypassed) by `lut_classify.py` with `{Path(lib['_library_path']).name}`.", "",
         "| outcome | captures | share |", "|---|---|---|"]
    for k in ("exact", "parent", "signal", "wrong", "none"):
        L.append(f"| {k} | {tot.get(k, 0)} | {100 * tot.get(k, 0) / max(n, 1):.1f} % |")
    L += ["", "exact = the entry's own label (variant suffixes allowed); parent = the parent / child entry's label; signal = left unlabelled; wrong = another label.", "",
          "| entry | draws | exact | parent | signal | wrong | verified descriptor fields (count over draws) |", "|---|---|---|---|---|---|---|"]
    for eid in sorted(per):
        c = per[eid]
        vs = ", ".join(f"{k} {v}" for k, v in ver[eid].most_common())
        L.append(f"| {eid} | {n_ver[eid]} | {c.get('exact', 0)} | {c.get('parent', 0)} | {c.get('signal', 0)} | {c.get('wrong', 0) + c.get('none', 0)} | {vs} |")
    wrong = [r for r in results if r["status"] in ("wrong",)]
    if wrong:
        L += ["", "Wrong labels:", ""]
        for r in wrong:
            L.append(f"- {r['capture']}: {r['estimate']} ({r['confidence']}) for {r['label']}: {r['evidence'][:200]}")
    (res_dir / "lut_synthetic_eval.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"wrote {res_dir / 'lut_synthetic_eval.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
