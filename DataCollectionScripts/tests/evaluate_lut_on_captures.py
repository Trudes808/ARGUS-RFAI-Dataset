#!/usr/bin/env python3
"""Score a lut_classify.py run on real captures (a sweep or the stress set) and compare its measured parameters
with the synthetic run.

    python3 tests/evaluate_lut_on_captures.py --report <dir with lut_regions.csv> [--synthetic tests/results/lut_synthetic_rows.csv]
            [--out tests/results/lut_real_eval.md]

Real captures have no ground truth, so every label is checked for plausibility against the capture span (the
service must be allowed inside the span) and the share of labels that carry at least one verified descriptor
field is reported. For each label the distributions of the measured parameters (symbol-rate line, OFDM subcarrier
spacing, burst length, interval, gap) are tabulated next to the same label's synthetic distribution.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import manifest_io as mio  # noqa: E402

ARTEFACTS = re.compile(r"^(DC_spur_LO_leakage|spur_comb_.*)$")
STEM_RE = re.compile(r"_c(?P<fc>[0-9p.]+)M_r(?P<rate>[0-9p.]+)M")


def span_of(stem: str) -> tuple[float, float] | None:
    m = STEM_RE.search(stem)
    if not m:
        return None
    fc = float(m.group("fc").replace("p", ".")) * 1e6
    rate = float(m.group("rate").replace("p", ".")) * 1e6
    return fc - rate / 2, fc + rate / 2


def plausible(lib: dict, f_lo: float, f_hi: float) -> set[str]:
    out = set()
    for e in lib["entries"]:
        if e.get("relative_to_capture_center") or e.get("kind") == "cross_track":
            continue
        if any(lo < f_hi and hi > f_lo for lo, hi in e["freq_ranges_hz"]):
            out.add(e["label"])
    return out


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def dist(vals: list[float]) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    v = np.array(v)
    if len(v) == 1:
        return f"{v[0]:.4g}"
    return f"{np.median(v):.4g} [{np.percentile(v, 10):.4g}, {np.percentile(v, 90):.4g}] n={len(v)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, nargs="+", help="directories holding lut_regions.csv from lut_classify.py")
    ap.add_argument("--synthetic", default=str(HERE / "results" / "lut_synthetic_rows.csv"))
    ap.add_argument("--out", default=str(HERE / "results" / "lut_real_eval.md"))
    ap.add_argument("--title", default="LUT classification on real captures")
    a = ap.parse_args()
    lib = mio.load_library()
    rows = []
    for rd in a.report:
        rows += list(csv.DictReader(open(Path(rd) / "lut_regions.csv")))
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r["capture"]].append(r)
    L = [f"# {a.title}", "", f"Reports: {', '.join(f'`{r}`' for r in a.report)} ({len(rows)} tracks in {len(by_cap)} captures).", "",
         "| capture | tracks | measured | labelled | with verified SDW | inconsistent | labels |", "|---|---|---|---|---|---|---|"]
    tot = Counter()
    bad = []
    label_rows = defaultdict(list)
    for cap, rs in sorted(by_cap.items()):
        span = span_of(cap)
        allowed = plausible(lib, *span) if span else None
        lab = [r for r in rs if r["estimate"] != "signal"]
        ver = [r for r in lab if r["verified"]]
        inc = []
        for r in lab:
            base = r["estimate"][1:].split("/")[0]
            if allowed is not None and not ARTEFACTS.match(base) and base not in allowed:
                inc.append(r)
        c = Counter(r["estimate"] for r in lab)
        meas = sum(1 for r in rs if r.get("sdw_measured") == "True")
        L.append(f"| `{cap[:60]}` | {len(rs)} | {meas} | {len(lab)} | {len(ver)} | {len(inc)} | {', '.join(f'{k} {v}' for k, v in c.most_common(6))} |")
        tot.update({"tracks": len(rs), "measured": meas, "labelled": len(lab), "verified": len(ver), "inconsistent": len(inc)})
        bad += [(cap, r) for r in inc]
        for r in lab:
            label_rows[r["estimate"].split("/")[0]].append(r)
    L += ["", f"Totals: {tot['tracks']} tracks, {tot['measured']} measured, {tot['labelled']} labelled, {tot['verified']} with at least one verified descriptor field, "
          f"{tot['inconsistent']} inconsistent with the capture span (precision by plausibility {1 - tot['inconsistent'] / max(tot['labelled'], 1):.3f})."]
    if bad:
        L += ["", "Inconsistent labels:", ""]
        for cap, r in bad:
            L.append(f"- `{cap}`: {r['estimate']} ({r['confidence']}) at {float(r['fc_hz']) / 1e6:.4f} MHz: {r['evidence'][:200]}")
    # verified-field tallies per label
    L += ["", "## Verified descriptor fields per label", "", "| label | tracks | fields verified (count) | contradictions (count) |", "|---|---|---|---|"]
    for lab in sorted(label_rows):
        vc, cc = Counter(), Counter()
        for r in label_rows[lab]:
            vc.update(v for v in r["verified"].split(";") if v)
            cc.update(v.split("(")[0] for v in r["contradictions"].split(";") if v)
        L.append(f"| {lab} | {len(label_rows[lab])} | {', '.join(f'{k} {v}' for k, v in vc.most_common())} | {', '.join(f'{k} {v}' for k, v in cc.most_common())} |")
    # parameter distributions: real vs synthetic
    syn = defaultdict(list)
    sp = Path(a.synthetic)
    if sp.exists():
        for r in csv.DictReader(open(sp)):
            if r["status"] in ("exact", "parent"):
                syn[r["estimate"].split("/")[0]].append(r)
    fields = [("sdw_snr_db", "SNR dB"), ("sdw_finst_line", "f_inst line Hz"), ("sdw_abs2_line", "|y|^2 line Hz"), ("sdw_sq_spacing", "y^2 spacing Hz"),
              ("sdw_ofdm_scs_hz", "OFDM SCS Hz"), ("sdw_ofdm_cp_us", "CP us"), ("sdw_burst_ms", "burst ms"), ("sdw_pri_ms", "interval ms"), ("sdw_gap_us", "gap us"),
              ("sdw_period_ms", "period ms"), ("sdw_dev_hz", "deviation Hz"), ("sdw_chirp_hz_s", "chirp Hz/s"), ("sdw_pulse_us", "pulse us")]
    L += ["", "## Measured parameters: real vs synthetic (median [p10, p90])", ""]
    for lab in sorted(label_rows):
        real = label_rows[lab]
        s = syn.get(lab, [])
        L += [f"### {lab} (real {len(real)}, synthetic {len(s)})", "", "| parameter | real | synthetic |", "|---|---|---|"]
        for key, name in fields:
            rv = [_num(r.get(key)) for r in real]
            sv = [_num(r.get(key)) for r in s]
            if all(v is None for v in rv) and all(v is None for v in sv):
                continue
            L.append(f"| {name} | {dist(rv)} | {dist(sv)} |")
        L.append("")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L[: len(by_cap) + 8]))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
