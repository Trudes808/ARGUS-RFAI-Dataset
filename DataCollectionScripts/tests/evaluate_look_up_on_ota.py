#!/usr/bin/env python3
"""Score common_waveforms_look_up.py on the N320 OTA stress captures.

Every capture folder (`n320_<band>_...`) holds the raw IQ plus detector masks written by
dino_fine_tuning/src/run_ota_detections.py (masks/<MODEL>.npz). The band of each capture is known, so
every estimated label can be checked for *plausibility*: a label is consistent when the service it names
can legally be on the air inside the capture span (e.g. "~FM_broadcast" inside 88-108 MHz), inconsistent
otherwise (e.g. "~FM_broadcast" in a 2.4 GHz capture). Receiver artefacts (DC spur, spur comb) are
allowed everywhere. For bands that only one service may use (FM, TV, cellular downlinks) the fraction
of continuous, correctly-sized tracks that received the label is reported as a recall proxy.

  python3 tests/evaluate_look_up_on_ota.py --root ~/Datasets/ARGUS/OOD_captures/OTA --models M12S M13S M7S \
        --out tests/results/ota_lookup_eval.md
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import manifest_io as mio  # noqa: E402
import common_waveforms_look_up as lk  # noqa: E402

ARTEFACTS = re.compile(r"^(DC_spur_LO_leakage|spur_comb_.*)$")

# label -> list of absolute RF ranges where that label is plausible (derived from the manifest at run time)
def plausible_labels(manifest: dict, f_lo: float, f_hi: float) -> set[str]:
    out = set()
    for e in manifest["entries"]:
        if e.get("relative_to_capture_center") or e.get("kind") == "cross_track":
            continue
        if any(lo < f_hi and hi > f_lo for lo, hi in e["freq_ranges_hz"]):
            out.add(e["label"])
    return out


# exclusive-band captures: (regex on the capture name, manifest entry id whose label every continuous
# in-range track should get)
RECALL_TARGETS = [
    (r"fm_broadcast", "fm_broadcast"),
    (r"hdtv_uhf", "atsc_tv_uhf"),
    (r"cell700_downlink", "cell_700_dl"),
    (r"cell850_downlink", "cell_850_dl"),
    (r"cell1900_pcs_downlink", "cell_1900_dl"),
    (r"cell2100_aws_downlink", "cell_aws_dl"),
]


def evaluate(root: Path, model: str, manifest: dict, use_iq: bool) -> dict:
    by_id = mio.entry_by_id(manifest)
    res = {"model": model, "captures": [], "labels": Counter(), "inconsistent": [], "n_tracks": 0, "n_labelled": 0, "n_consistent": 0}
    for d in sorted(p for p in root.glob("n320_*") if p.is_dir() and (p / "masks" / f"{model}.npz").exists()):
        caps = lk.regions_from_masks(d, model)
        stem, cap = next(iter(caps.items()))
        f_lo, f_hi = cap["fc"] - cap["rate"] / 2, cap["fc"] + cap["rate"] / 2
        allowed = plausible_labels(manifest, f_lo, f_hi)
        tracks, rows = lk.run_capture(stem, cap, manifest, use_iq, 0.10)
        c = Counter(r["estimate"] for r in rows)
        n_lab = sum(v for k, v in c.items() if k != "signal")
        bad = []
        for r in rows:
            if r["estimate"] == "signal":
                continue
            lab = r["estimate"][1:]
            if ARTEFACTS.match(lab) or lab in allowed:
                continue
            bad.append(r)
        # recall proxy on exclusive bands
        recall = ""
        for pat, eid in RECALL_TARGETS:
            if re.search(pat, d.name):
                e = by_id[eid]
                bw = e["bandwidth_hz"]
                cand = [r for r in rows if r["temporal"] == "continuous" and bw["min"] <= float(r["bw_hz"]) <= bw["max"]
                        and any(lo <= float(r["fc_hz"]) <= hi for lo, hi in e["freq_ranges_hz"])]
                hit = [r for r in cand if r["estimate"] == "~" + e["label"]]
                recall = f"{len(hit)}/{len(cand)}"
                res.setdefault("recall", Counter())
                res["recall"][e["label"] + "_hit"] += len(hit)
                res["recall"][e["label"] + "_cand"] += len(cand)
                break
        res["captures"].append({"capture": d.name, "rate": cap["rate"], "tracks": len(rows), "labelled": n_lab, "labels": dict(c.most_common()),
                                "inconsistent": len(bad), "recall": recall})
        res["labels"].update({k: v for k, v in c.items() if k != "signal"})
        res["n_tracks"] += len(rows)
        res["n_labelled"] += n_lab
        res["n_consistent"] += n_lab - len(bad)
        for r in bad:
            res["inconsistent"].append({"capture": d.name, **{k: r[k] for k in ("fc_hz", "bw_hz", "temporal", "estimate", "confidence", "evidence")}})
        print(f"  {model} {d.name}: {len(rows)} tracks, {n_lab} labelled, {len(bad)} inconsistent, recall {recall}  {dict(c.most_common(4))}", flush=True)
    return res


def write_md(results: list[dict], out: Path, root: Path, use_iq: bool):
    L = ["# Common-waveform look-up on the N320 OTA stress captures", "",
         f"Input: `{root}`; detector masks `masks/<model>.npz` (threshold 0.6, 12 frames per capture) converted to regions with 8-connectivity, "
         f"tracked across frames, then labelled by `common_waveforms_look_up.py` ({'with' if use_iq else 'without'} IQ spectral features). "
         "A label is *consistent* when the service can legally be on the air inside the capture span; receiver artefacts (DC spur, spur comb) are always consistent. "
         "*Recall* counts continuous tracks of the right width inside an exclusive band (FM, TV, cellular downlink) that received the band's label.", ""]
    L += ["| model | tracks | labelled | consistent | inconsistent | precision (consistent / labelled) | exclusive-band recall |", "|---|---|---|---|---|---|---|"]
    for r in results:
        prec = r["n_consistent"] / max(r["n_labelled"], 1)
        rc = r.get("recall", Counter())
        hit = sum(v for k, v in rc.items() if k.endswith("_hit"))
        cand = sum(v for k, v in rc.items() if k.endswith("_cand"))
        L.append(f"| {r['model']} | {r['n_tracks']} | {r['n_labelled']} | {r['n_consistent']} | {len(r['inconsistent'])} | {prec:.3f} | {hit}/{cand} |")
    for r in results:
        L += ["", f"## {r['model']}", "", "| capture | MSps | tracks | labelled | inconsistent | recall | labels |", "|---|---|---|---|---|---|---|"]
        for c in r["captures"]:
            labs = ", ".join(f"{k} {v}" for k, v in c["labels"].items() if k != "signal") or "-"
            L.append(f"| `{c['capture']}` | {c['rate'] / 1e6:.2f} | {c['tracks']} | {c['labelled']} | {c['inconsistent']} | {c['recall'] or '-'} | {labs} |")
        if r["inconsistent"]:
            L += ["", f"Inconsistent labels ({r['model']}):", ""]
            for b in r["inconsistent"]:
                L.append(f"- `{b['capture']}`: {b['estimate']} ({b['confidence']}) at {float(b['fc_hz']) / 1e6:.4f} MHz, bw {float(b['bw_hz']) / 1e3:.0f} kHz, {b['temporal']}: {b['evidence'][:160]}")
        L += ["", "Label totals: " + ", ".join(f"{k} {v}" for k, v in r["labels"].most_common())]
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "Datasets/ARGUS/OOD_captures/OTA"))
    ap.add_argument("--models", nargs="+", default=["M12S"])
    ap.add_argument("--manifest", default=str(HERE.parent / "common_ota_waveforms_manifest_usa.yaml"))
    ap.add_argument("--no-iq", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results" / "ota_lookup_eval.md"))
    a = ap.parse_args()
    m = mio.load_manifest(a.manifest)
    results = [evaluate(Path(a.root).expanduser(), mdl, m, not a.no_iq) for mdl in a.models]
    write_md(results, Path(a.out), Path(a.root).expanduser(), not a.no_iq)


if __name__ == "__main__":
    main()
