#!/usr/bin/env python3
"""Draw each capture as a waterfall with the detector's boxes and the look-up's "~labels" on top.

    python3 plot_look_up_overlays.py <dir>_preprocessed                       # preprocess_dataset.py output
    python3 plot_look_up_overlays.py <capture>.sigmf-meta [...]               # any annotated SigMF (soft_label.py output)
    python3 plot_look_up_overlays.py <capture folder | folder of captures> --masks M12S
    python3 plot_look_up_overlays.py <root> --masks M12S --select fm_broadcast_88 hdtv_uhf_499 ism2g4_2387 \
            --montage bands.png --ncols 3

One PNG per capture (`<out>/<capture>__<source>.png`) and, with --montage, one figure that tiles the selected
captures so the bands can be compared side by side. In every panel frequency runs across the page and the
detector's frames are stacked down it (only the frames the detector ran: the mask arrays of
run_ota_detections.py sample 12 frames of a capture, preprocess_dataset.py covers them all). The waterfall is a
single-hue ramp (darker = stronger), every detected region is a thin box, and the estimated label is written
once per tracked emission. Unlabelled detections keep a gray box and no text: they are "signal".

The labels come from common_waveforms_look_up.py, run here on the same regions (so the figure always reflects
the current manifest), or with --written-labels from the `core:label` already stored in the annotations.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_waveforms_look_up as lk  # noqa: E402
import manifest_io as mio  # noqa: E402

# ------------------------------------------------------------------------------------------------ palette
# Light chart surface and ink; the waterfall is the one-hue blue ramp (near-surface -> dark = weak -> strong), and
# the boxes of labelled emissions take the remaining categorical slots in fixed order (blue is reserved for the
# waterfall), assigned per panel by label frequency: every panel carries its own legend. Palette and ordering follow
# the dataviz reference instance; identity is also carried by the text label, never by colour alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
SIGNAL_BOX = "#898781"                   # unlabelled detections ("signal")
OTHER_BOX = "#0b0b0b"                    # labelled emissions beyond the seven coloured slots
LABEL_SLOTS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RAMP = ["#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]

_SUFFIX = re.compile(r"_c\d+p?\d*M_r\d+p?\d*M_g\d+_ch\d+$")


# ------------------------------------------------------------------------------------------------ inputs
def _meta_for(stem: str, cap: dict) -> dict:
    d = cap.get("dir")
    if not d:
        return {}
    for cand in (Path(d) / f"{stem}.sigmf-meta", Path(d) / f"{stem}.source.sigmf-meta"):
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except json.JSONDecodeError:
                return {}
    return {}


def _frames_all(cap: dict) -> list[int]:
    FS = cap["geometry"]["frame_samples"]
    n = cap["n_frames"]
    if cap.get("data_path"):
        bps = lk._DTYPES.get(cap["dtype"], lk._DTYPES["cf32_le"])[2]
        n_data = int(np.ceil(cap["data_path"].stat().st_size / bps / FS))
        n = min(max(n, 1), n_data) if n_data else n
    return list(range(n))


def collect_inputs(paths: list[str], masks: str | None, cols: int | None) -> list[tuple[str, dict]]:
    """-> [(stem, cap)] with cap["frames"] (frames the detector ran) and cap["source"] (text for the panel)."""
    out: list[tuple[str, dict]] = []

    def add_masks(d: Path):
        caps = lk.regions_from_masks(d, masks)
        z = np.load(d / "masks" / f"{masks}.npz")
        for stem, cap in caps.items():
            cap["frames"] = [int(i) for i in z["frame_index"]]
            cap["source"] = f"masks {masks}"
            out.append((stem, cap))

    for p in paths:
        p = Path(p).expanduser()
        if masks:
            if (p / "masks" / f"{masks}.npz").exists():
                add_masks(p)
            elif p.is_dir():
                subs = sorted(q for q in p.iterdir() if q.is_dir() and (q / "masks" / f"{masks}.npz").exists())
                if not subs:
                    print(f"   skip {p}: no masks/{masks}.npz here or in a sub-folder", file=sys.stderr)
                for q in subs:
                    add_masks(q)
            else:
                print(f"   skip {p}: not a folder", file=sys.stderr)
        elif p.is_file() and p.name.endswith(".sigmf-meta"):
            for stem, cap in lk.regions_from_sigmf(p, cols).items():
                cap["frames"] = _frames_all(cap)
                cap["source"] = "annotations"
                out.append((stem, cap))
        elif p.is_dir():
            if (p / "preprocess_summary.json").exists() or any((q / "snippets").is_dir() for q in p.iterdir() if q.is_dir()):
                det = ""
                try:
                    det = json.loads((p / "preprocess_summary.json").read_text()).get("detector", "")
                except (OSError, json.JSONDecodeError):
                    pass
                for stem, cap in lk.regions_from_preprocessed(p, cols).items():
                    cap["frames"] = _frames_all(cap)
                    cap["source"] = f"preprocess {det}".strip()
                    out.append((stem, cap))
            else:
                metas = sorted(q for q in p.glob("*.sigmf-meta") if not q.name.endswith(".source.sigmf-meta"))
                if not metas:
                    print(f"   skip {p}: no *.sigmf-meta", file=sys.stderr)
                for mp in metas:
                    for stem, cap in lk.regions_from_sigmf(mp, cols).items():
                        cap["frames"] = _frames_all(cap)
                        cap["source"] = "annotations"
                        out.append((stem, cap))
        else:
            print(f"   skip {p}: not found", file=sys.stderr)
    return out


# ------------------------------------------------------------------------------------------------ labels
def label_tracks(stem: str, cap: dict, manifest: dict, use_iq: bool, margin: float, written: bool) -> list[tuple[lk.Track, str, float]]:
    """-> [(track, label or 'signal', confidence)]"""
    if written:
        tracks = lk.build_tracks(cap["regions"], cap["n_frames"])
        res = []
        for t in tracks:
            c = Counter(r.det_label for r in t.regions if r.det_label.startswith(lk.ESTIMATE_PREFIX))
            res.append((t, c.most_common(1)[0][0] if c else "signal", float("nan")))
        return res
    _, rows = lk.run_capture(stem, cap, manifest, use_iq, margin)
    return [(r["_track"], r["estimate"], float(r["confidence"]) if r["confidence"] != "" else float("nan")) for r in rows]


# ------------------------------------------------------------------------------------------------ waterfall
def waterfall_strips(cap: dict, frames: list[int], strip_rows: int, cols_out: int) -> tuple[list[np.ndarray], int]:
    """One (strip_rows, cols) dB array per frame: Hann FFT of every detector row, rows averaged, bins max-pooled."""
    geo = cap["geometry"]
    FS = int(geo["frame_samples"])
    rows = int(geo.get("rows", 512))
    nfft = FS // rows
    q = max(1, nfft // cols_out)
    cols = nfft // q
    p = max(1, rows // strip_rows)
    win = np.hanning(nfft).astype(np.float32)
    strips = []
    for f in frames:
        x = lk.read_iq(cap["data_path"], cap["dtype"], f * FS, FS)
        n = len(x) // nfft
        P = np.full((rows, nfft), np.nan, dtype=np.float32)
        if n:
            X = np.fft.fft(x[: n * nfft].reshape(n, nfft) * win, axis=1)
            P[:n] = np.fft.fftshift((X.real ** 2 + X.imag ** 2).astype(np.float32), axes=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            P = np.nanmean(P[: (rows // p) * p].reshape(rows // p, p, nfft), axis=1)          # time: mean
            if q > 1:
                P = np.nanmax(P[:, : cols * q].reshape(P.shape[0], cols, q), axis=2)          # frequency: max
        strips.append(10 * np.log10(P + 1e-12))
    return strips, cols


def _pick_frames(frames: list[int], max_frames: int) -> list[int]:
    if len(frames) <= max_frames:
        return list(frames)
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, max_frames)).astype(int))
    return [frames[i] for i in idx]


def _strip_rows_for(n_strips: int, target_rows: int = 768) -> int:
    for sr in (512, 256, 128, 64, 32, 16):
        if sr * n_strips <= target_rows:
            return sr
    return 16


# ------------------------------------------------------------------------------------------------ drawing
def band_title(stem: str, cap: dict) -> str:
    g = _meta_for(stem, cap).get("global", {})
    return str(g.get("argus:band_label") or _SUFFIX.sub("", stem))


def colours_for(labelled: list[tuple[lk.Track, str, float]]) -> dict:
    """Label -> box colour for one panel: the seven slots in fixed order, most frequent label first; the rest black."""
    counts = Counter(lab for _, lab, _ in labelled if lab != "signal")
    order = sorted(counts, key=lambda k: (-counts[k], k))
    return {lab: LABEL_SLOTS[i] for i, lab in enumerate(order[: len(LABEL_SLOTS)])}


def draw_panel(ax, lax, stem: str, cap: dict, labelled: list[tuple[lk.Track, str, float]], max_frames: int,
               cols_out: int, small: bool, cax=None, texts_per_label: int = 6):
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    fs_tick, fs_lab, fs_txt, fs_leg = (6, 6.5, 5.5, 6) if small else (7, 7.5, 6.5, 7)
    frames = _pick_frames(cap["frames"], max_frames)
    geo = cap["geometry"]
    frame_s = geo["frame_s"]
    strip_rows = _strip_rows_for(len(frames))
    strips, cols = waterfall_strips(cap, frames, strip_rows, cols_out)
    colour_of = colours_for(labelled)
    # stack with a two-row surface gap wherever consecutive strips are not consecutive frames
    blocks, y_top, y = [], {}, 0
    for k, (f, s) in enumerate(zip(frames, strips)):
        if k and f != frames[k - 1] + 1:
            blocks.append(np.full((2, cols), np.nan, dtype=np.float32))
            y += 2
        y_top[f] = y
        blocks.append(s)
        y += strip_rows
    img = np.vstack(blocks)
    finite = img[np.isfinite(img)]
    vmin, vmax = (np.percentile(finite, 2.0), np.percentile(finite, 99.8)) if finite.size else (0, 1)
    cmap = LinearSegmentedColormap.from_list("argus_blue", RAMP)
    cmap.set_bad(SURFACE)
    rate, fc = cap["rate"], cap["fc"]
    f_lo_m, f_hi_m = (fc - rate / 2) / 1e6, (fc + rate / 2) / 1e6
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="antialiased",
                   extent=[f_lo_m, f_hi_m, img.shape[0], 0], origin="upper")
    ax.set_facecolor(SURFACE)
    # boxes
    counts: Counter = Counter()
    anchors = []
    for t, lab, conf in labelled:
        colour = SIGNAL_BOX if lab == "signal" else colour_of.get(lab, OTHER_BOX)
        lw = 0.6 if lab == "signal" else 1.1
        counts[lab] += 1
        ys = []
        for r in t.regions:
            f = int(np.floor(r.t0 / frame_s + 1e-9))
            if f not in y_top:
                continue
            y0 = y_top[f] + (r.t0 - f * frame_s) / frame_s * strip_rows
            h = max(r.dur / frame_s * strip_rows, 0.6)
            ax.add_patch(mpatches.Rectangle((r.f_lo / 1e6, y0), r.bw / 1e6, h, fill=False, edgecolor=colour, linewidth=lw,
                                            alpha=0.95 if lab != "signal" else 0.8, zorder=3))
            ys.append((y0, y0 + h))
        if lab != "signal" and ys:
            anchors.append((t.fc / 1e6, min(a for a, _ in ys), max(b for _, b in ys), lab, len(ys)))
    # text on at most `texts_per_label` tracks of each label (the most visible ones); the colour and the legend
    # identify the rest. Neighbours alternate between the bottom and the top of the track.
    keep = []
    for lab in {a[3] for a in anchors}:
        keep += sorted((a for a in anchors if a[3] == lab), key=lambda a: -a[4])[: max(0, texts_per_label)]
    for i, (x, ya, yb, lab, _) in enumerate(sorted(keep)):
        if i % 2 == 0:
            ax.text(x, yb, lab, rotation=90, ha="center", va="bottom", fontsize=fs_txt, color=INK, clip_on=True, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
        else:
            ax.text(x, ya, lab, rotation=90, ha="center", va="top", fontsize=fs_txt, color=INK, clip_on=True, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
    # axes chrome
    ax.set_xlim(f_lo_m, f_hi_m)
    ax.set_ylim(img.shape[0], 0)
    ticks = [y_top[f] + strip_rows / 2 for f in frames]
    step = max(1, int(np.ceil(len(frames) / (6 if small else 12))))
    ax.set_yticks(ticks[::step])
    ax.set_yticklabels([f"{f * frame_s * 1e3:.0f}" for f in frames[::step]])
    ax.set_ylabel("frame start (ms)", fontsize=fs_lab, color=INK2)
    ax.set_xlabel("frequency (MHz)", fontsize=fs_lab, color=INK2)
    ax.tick_params(labelsize=fs_tick, colors=MUTED, length=2.5, width=0.6)
    for sp in ax.spines.values():
        sp.set_color(AXIS)
        sp.set_linewidth(0.6)
    n_lab = sum(v for k, v in counts.items() if k != "signal")
    ax.set_title(band_title(stem, cap), fontsize=fs_lab + 1.5, color=INK, loc="left", pad=14 if small else 16, fontweight=600)
    ax.text(0, 1.012, f"{fc / 1e6:.3f} MHz centre · {rate / 1e6:g} MSps · {cap['source']} · {len(frames)} of {len(cap['frames'])} frames shown · "
            f"{len(labelled)} tracks, {n_lab} labelled", transform=ax.transAxes, fontsize=fs_tick, color=INK2, va="bottom", ha="left")
    # legend (in its own axes below the plot)
    lax.axis("off")
    handles, texts = [], []
    for lab in sorted((k for k in counts if k != "signal"), key=lambda k: (-counts[k], k)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor=colour_of.get(lab, OTHER_BOX), linewidth=1.4))
        texts.append(f"{lab} ({counts[lab]})")
    if counts.get("signal"):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor=SIGNAL_BOX, linewidth=1.0))
        texts.append(f"signal, unlabelled ({counts['signal']})")
    if not handles:
        texts = ["no detections"]
        handles = [mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="none")]
    lax.legend(handles, texts, loc="upper left", ncol=3 if small else 4, fontsize=fs_leg, frameon=False, handlelength=1.3,
               handleheight=0.9, columnspacing=1.2, borderaxespad=0, labelcolor=INK2)
    if cax is not None:
        cb = ax.figure.colorbar(im, cax=cax)
        cb.set_label("dB (relative)", fontsize=fs_lab, color=INK2)
        cb.ax.tick_params(labelsize=fs_tick, colors=MUTED, length=2.5, width=0.6)
        cb.outline.set_edgecolor(AXIS)
        cb.outline.set_linewidth(0.6)
    return counts


def _layout(fig, ncols: int, nrows: int, panel_w: float, panel_h: float, colorbar: bool, top: float = 0.3):
    """Manual grid: per cell a plot axes, a legend axes below it, optionally a colour-bar axes to the right."""
    left, right, bottom = 0.75, 0.25 + (0.55 if colorbar else 0.0), 0.2
    gap_x, gap_y, legend_h, title_h = 0.75, 0.45, 0.62, 0.45
    cell_w = panel_w + gap_x
    cell_h = panel_h + legend_h + title_h + gap_y
    W = left + ncols * cell_w - gap_x + right
    H = top + nrows * cell_h - gap_y + bottom
    fig.set_size_inches(W, H)
    cells = []
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        x0 = left + c * cell_w
        y_plot_bottom = H - top - r * cell_h - title_h - panel_h
        ax = fig.add_axes([x0 / W, y_plot_bottom / H, panel_w / W, panel_h / H])
        lax = fig.add_axes([x0 / W, (y_plot_bottom - 0.38 - legend_h) / H, panel_w / W, legend_h / H])
        cax = fig.add_axes([(x0 + panel_w + 0.08) / W, y_plot_bottom / H, 0.12 / W, panel_h / H]) if colorbar else None
        cells.append((ax, lax, cax))
    return cells


def _quantize(path: Path, colours: int = 256):
    """Palette PNG (no dither): a fraction of the size of the full-colour file, no visible change on these figures."""
    try:
        from PIL import Image
    except ImportError:
        return
    im = Image.open(path).convert("RGB").quantize(colors=max(8, min(colours, 256)), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(path, optimize=True)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="<dir>_preprocessed | <capture>.sigmf-meta | folder(s) of SigMF | capture folder(s) with --masks")
    ap.add_argument("--manifest", default=str(lk.DEFAULT_MANIFEST))
    ap.add_argument("--masks", default=None, help="model name: read masks/<model>.npz (run_ota_detections.py layout)")
    ap.add_argument("--cols", type=int, default=None, help="detector mask columns for annotation inputs (see common_waveforms_look_up.py)")
    ap.add_argument("--out", default=None, help="output folder (default: lookup_overlays/ next to the first input)")
    ap.add_argument("--select", nargs="*", default=None, help="only captures whose name matches one of these patterns (regex); panels follow this order")
    ap.add_argument("--written-labels", action="store_true", help="use the core:label already written by the look-up instead of re-running it")
    ap.add_argument("--no-iq", action="store_true", help="look-up without IQ spectral features")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--max-frames", type=int, default=16, help="frames per panel (evenly subsampled when the detector ran more)")
    ap.add_argument("--texts-per-label", type=int, default=6, help="write the label text on at most this many tracks per label; the rest are identified by colour and legend")
    ap.add_argument("--render-cols", type=int, default=1024, help="waterfall columns")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-single", action="store_true", help="skip the per-capture figures (montage only)")
    ap.add_argument("--montage", default=None, help="write one tiled figure of all selected captures to this file (.png, or .jpg for a much smaller file)")
    ap.add_argument("--ncols", type=int, default=3, help="montage columns")
    ap.add_argument("--montage-dpi", type=int, default=110)
    ap.add_argument("--no-quantize", action="store_true", help="keep the montage as a full-colour PNG (larger file)")
    ap.add_argument("--quantize-colours", type=int, default=256, help="palette size of the quantised PNG montage")
    ap.add_argument("--jpeg-quality", type=int, default=88, help="quality when the montage path ends in .jpg")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"], "savefig.facecolor": SURFACE,
                         "figure.facecolor": SURFACE})

    manifest = mio.load_manifest(a.manifest)
    caps = collect_inputs(a.inputs, a.masks, a.cols)
    # the same capture may appear in several inputs (a re-run of the detector): keep the one with the most regions
    best = {}
    for stem, cap in caps:
        if stem not in best or len(cap["regions"]) > len(best[stem]["regions"]):
            best[stem] = cap
    caps = [(stem, cap) for stem, cap in caps if best[stem] is cap]
    if a.select:                      # keep only matching captures, in the order of the patterns
        pats = [re.compile(s) for s in a.select]
        ranked = [(next((i for i, p in enumerate(pats) if p.search(s)), None), s, c) for s, c in caps]
        caps = [(s, c) for i, s, c in sorted((r for r in ranked if r[0] is not None), key=lambda r: r[0])]
    caps = [(s, c) for s, c in caps if c.get("data_path")] or caps
    missing = [s for s, c in caps if not c.get("data_path")]
    for s in missing:
        print(f"   skip {s}: the recording (.sigmf-data) was not found, nothing to draw", file=sys.stderr)
    caps = [(s, c) for s, c in caps if c.get("data_path")]
    if not caps:
        print("nothing to plot", file=sys.stderr)
        return 1
    first = Path(a.inputs[0]).expanduser()
    out = Path(a.out).expanduser() if a.out else (first.parent if first.is_file() else first) / "lookup_overlays"
    out.mkdir(parents=True, exist_ok=True)

    labelled: dict[str, list] = {}
    for stem, cap in caps:
        labelled[stem] = label_tracks(stem, cap, manifest, not a.no_iq, a.margin, a.written_labels)
        c = Counter(lab for _, lab, _ in labelled[stem] if lab != "signal")
        print(f"{stem}: {len(cap['regions'])} regions -> {len(labelled[stem])} tracks; " + (", ".join(f"{k} {v}" for k, v in c.most_common()) or "nothing labelled"),
              flush=True)
        if len(c) > len(LABEL_SLOTS):
            print(f"   {len(c) - len(LABEL_SLOTS)} labels beyond the seven coloured slots are drawn in black: {sorted(c, key=lambda k: -c[k])[len(LABEL_SLOTS):]}")

    src_tag = re.sub(r"[^A-Za-z0-9]+", "_", caps[0][1]["source"]).strip("_")
    if not a.no_single:
        for stem, cap in caps:
            fig = plt.figure()
            (ax, lax, cax), = _layout(fig, 1, 1, 7.0, 4.4, colorbar=True)
            draw_panel(ax, lax, stem, cap, labelled[stem], a.max_frames, a.render_cols, small=False, cax=cax, texts_per_label=a.texts_per_label)
            path = out / f"{stem}__{src_tag}.png"
            fig.savefig(path, dpi=a.dpi)
            plt.close(fig)
            print(f"  wrote {path}")
    if a.montage:
        n = len(caps)
        ncols = max(1, min(a.ncols, n))
        nrows = int(np.ceil(n / ncols))
        fig = plt.figure()
        cells = _layout(fig, ncols, nrows, 5.6, 3.9, colorbar=False, top=0.7)
        for (stem, cap), (ax, lax, _) in zip(caps, cells):
            draw_panel(ax, lax, stem, cap, labelled[stem], a.max_frames, min(a.render_cols, 768), small=True, texts_per_label=a.texts_per_label)
        for ax, lax, _ in cells[n:]:
            ax.axis("off")
            lax.axis("off")
        fig.text(0.5, 1 - 0.22 / fig.get_size_inches()[1], f"Detector boxes and look-up labels across bands ({caps[0][1]['source']})",
                 ha="center", va="top", fontsize=10, color=INK, fontweight=600)
        mpath = Path(a.montage).expanduser()
        if not mpath.is_absolute() and mpath.parent == Path("."):
            mpath = out / mpath
        mpath.parent.mkdir(parents=True, exist_ok=True)
        if mpath.suffix.lower() in (".jpg", ".jpeg"):      # the waterfalls are noise-like: JPEG is several times smaller than PNG
            fig.savefig(mpath, dpi=a.montage_dpi, pil_kwargs={"quality": a.jpeg_quality, "optimize": True})
            plt.close(fig)
        else:
            fig.savefig(mpath, dpi=a.montage_dpi)
            plt.close(fig)
            if not a.no_quantize:
                _quantize(mpath, a.quantize_colours)
        print(f"  wrote {mpath} ({mpath.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
