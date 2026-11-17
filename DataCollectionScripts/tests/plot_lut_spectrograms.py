#!/usr/bin/env python3
"""Spectrograms of the test captures with the detector's boxes and the LUT labels (and decodes) drawn on top.

    python3 tests/plot_lut_spectrograms.py --report <dir with lut_regions.csv> <capture source> [...] [--select ...]
            [--out tests/results/spectrograms] [--montage synth.jpg --ncols 3]

The capture sources are what lut_classify.py read (a <dir>_preprocessed folder, a folder of SigMF captures, or a
capture folder with --masks M12S); the report is the lut_regions.csv that run produced. Tracks are rebuilt exactly as
the classifier built them and joined to the report rows by track id, so every panel shows the detector's regions
(thin boxes), the classifier's label on each labelled track, and "[decoded]" on tracks whose protocol decoder
passed. The rendering (one-hue waterfall, categorical boxes, per-panel legend) is plot_look_up_overlays.py's.
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
import common_waveforms_look_up as lk  # noqa: E402
import plot_look_up_overlays as plo  # noqa: E402


def load_rows(report_dirs: list[str]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for rd in report_dirs:
        with open(Path(rd).expanduser() / "lut_regions.csv", newline="") as f:
            for r in csv.DictReader(f):
                out[r["capture"]][int(r["track"])] = r
    return out


def labelled_tracks(cap: dict, rows: dict[int, dict]) -> list[tuple[lk.Track, str, float]]:
    tracks = lk.build_tracks(cap["regions"], cap["n_frames"])
    out = []
    for t in tracks:
        r = rows.get(t.tid)
        if r is None:
            out.append((t, "signal", float("nan")))
            continue
        lab = r["estimate"]
        if r.get("decoded"):
            lab = f"{lab} [decoded {r['decoded']}]"
        conf = float(r["confidence"]) if r.get("confidence") not in (None, "") else float("nan")
        out.append((t, lab, conf))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="capture sources (see lut_classify.py)")
    ap.add_argument("--report", nargs="+", required=True, help="report dir(s) holding lut_regions.csv")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--select", nargs="*", default=None, help="capture name patterns (regex); panel order follows the patterns")
    ap.add_argument("--out", default=str(HERE / "results" / "spectrograms"))
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--render-cols", type=int, default=1024)
    ap.add_argument("--texts-per-label", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--no-single", action="store_true")
    ap.add_argument("--montage", default=None, help="tile the selected captures into this file (.jpg recommended)")
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--montage-dpi", type=int, default=110)
    ap.add_argument("--title", default=None)
    ap.add_argument("--only-labelled", action="store_true", help="skip captures with no labelled track")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"], "savefig.facecolor": plo.SURFACE,
                         "figure.facecolor": plo.SURFACE})
    rows = load_rows(a.report)
    caps = plo.collect_inputs(a.inputs, a.masks, a.cols)
    best = {}
    for stem, cap in caps:
        if stem not in best or len(cap["regions"]) > len(best[stem]["regions"]):
            best[stem] = cap
    caps = [(s, c) for s, c in caps if best[s] is c and s in rows and c.get("data_path")]
    if a.select:
        pats = [re.compile(s) for s in a.select]
        ranked = [(next((i for i, p in enumerate(pats) if p.search(s)), None), s, c) for s, c in caps]
        caps = [(s, c) for i, s, c in sorted((r for r in ranked if r[0] is not None), key=lambda r: r[0])]
    if not caps:
        print("nothing to plot (no capture of the inputs appears in the report)", file=sys.stderr)
        return 1
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    labelled = {}
    for stem, cap in caps:
        labelled[stem] = labelled_tracks(cap, rows[stem])
        c = Counter(lab for _, lab, _ in labelled[stem] if lab != "signal")
        print(f"{stem}: {len(cap['regions'])} regions -> {len(labelled[stem])} tracks; " + (", ".join(f"{k} {v}" for k, v in c.most_common()) or "nothing labelled"), flush=True)
    if a.only_labelled:
        caps = [(s, c) for s, c in caps if any(lab != "signal" for _, lab, _ in labelled[s])]
    if not a.no_single:
        for stem, cap in caps:
            fig = plt.figure()
            (ax, lax, cax), = plo._layout(fig, 1, 1, 7.0, 4.4, colorbar=True)
            plo.draw_panel(ax, lax, stem, cap, labelled[stem], a.max_frames, a.render_cols, small=False, cax=cax, texts_per_label=a.texts_per_label)
            path = out / f"{stem}.png"
            fig.savefig(path, dpi=a.dpi)
            plt.close(fig)
            print(f"  wrote {path}")
    if a.montage:
        n = len(caps)
        ncols = max(1, min(a.ncols, n))
        nrows = int(np.ceil(n / ncols))
        fig = plt.figure()
        cells = plo._layout(fig, ncols, nrows, 5.6, 3.9, colorbar=False, top=0.7)
        for (stem, cap), (ax, lax, _) in zip(caps, cells):
            plo.draw_panel(ax, lax, stem, cap, labelled[stem], a.max_frames, min(a.render_cols, 768), small=True, texts_per_label=a.texts_per_label)
        for ax, lax, _ in cells[n:]:
            ax.axis("off")
            lax.axis("off")
        title = a.title or f"Detector boxes, LUT labels and decodes ({caps[0][1]['source']})"
        fig.text(0.5, 1 - 0.22 / fig.get_size_inches()[1], title, ha="center", va="top", fontsize=10, color=plo.INK, fontweight=600)
        mpath = Path(a.montage).expanduser()
        if not mpath.is_absolute() and mpath.parent == Path("."):
            mpath = out / mpath
        mpath.parent.mkdir(parents=True, exist_ok=True)
        if mpath.suffix.lower() in (".jpg", ".jpeg"):
            fig.savefig(mpath, dpi=a.montage_dpi, pil_kwargs={"quality": 88, "optimize": True})
        else:
            fig.savefig(mpath, dpi=a.montage_dpi)
        plt.close(fig)
        print(f"  wrote {mpath} ({mpath.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
