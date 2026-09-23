# SigMF → Spectrogram Detection Boxes

Two command-line tools that turn SigMF signal annotations into object-detection
training data on an STFT spectrogram, and let you visualize / score the result:

| Script | Role |
| --- | --- |
| `sigmf_annotations_to_detection_boxes.py` | **Generator.** Reads a `.sigmf-meta` (+ optional `.sigmf-data`), maps each annotation's time/frequency span onto a spectrogram image grid, and writes YOLO labels, a COCO/DINO JSON, CSV manifests, and (optionally) the matching spectrogram PNG tiles. |
| `visualize_detection_boxes.py` | **Inspector / scorer.** Draws those boxes over the spectrogram tiles, compares them against model predictions (COCO or YOLO), and reports IoU / precision / recall. Can also synthesize jittered "predictions" for a smoke test. |

Both tools have been validated end-to-end against the SigMF pairs in
`holoscan_waveform_generation/composition/composites/` — see
[Validation results](#validation-results).

---

## Contents

- [Requirements](#requirements)
- [Input data: SigMF pairs and the composites naming convention](#input-data-sigmf-pairs-and-the-composites-naming-convention)
- [The spectrogram coordinate system](#the-spectrogram-coordinate-system)
- [Tool 1 — `sigmf_annotations_to_detection_boxes.py`](#tool-1--sigmf_annotations_to_detection_boxespy)
  - [Quick start](#quick-start)
  - [Outputs](#outputs)
  - [Option reference](#option-reference)
  - [Class modes (important)](#class-modes-important)
  - [Time-box modes](#time-box-modes)
  - [Tiling](#tiling)
- [Tool 2 — `visualize_detection_boxes.py`](#tool-2--visualize_detection_boxespy)
  - [Quick start](#quick-start-1)
  - [Prediction inputs and matching](#prediction-inputs-and-matching)
  - [Overlay legend](#overlay-legend)
  - [Outputs](#outputs-1)
- [End-to-end worked example](#end-to-end-worked-example)
- [Validation results](#validation-results)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- Python 3.10+ with **NumPy** and **Pillow (PIL)**.
- The project conda env `holoscan_compose` already has both
  (NumPy ≥ 2.x, Pillow ≥ 12.x). Run everything through it:

```bash
conda run -n holoscan_compose python -B sigmf_annotations_to_detection_boxes.py --help
conda run -n holoscan_compose python -B visualize_detection_boxes.py --help
```

NumPy is required by both scripts. Pillow is only needed when you render
spectrogram PNGs (`--write-spectrograms`) or draw overlays.

Throughout this guide, set a shell variable for the composites folder so the
commands are copy-pasteable from either repo:

```bash
COMPOSITES=/Users/bqn82/Documents/holoscan/holoscan_waveform_generation/composition/composites
```

> Run the commands from the directory that contains the two scripts
> (`holoscan_tx_dev/`, or `holoscan_waveform_generation/composition/` for the
> copies). Otherwise prefix the script names with their full path.

---

## Input data: SigMF pairs and the composites naming convention

A SigMF recording is a **pair**: a `.sigmf-meta` JSON sidecar and a `.sigmf-data`
binary IQ file with the same stem. The generator needs the **meta** for
annotations; it uses the **data** only to (a) measure the exact recording length
and (b) render spectrogram PNGs.

The `composites/` folder contains several stems, each in a few variants. **Only
the base variant carries annotations** — this matters:

| Stem variant | Has annotations? | Has IQ data? | Use as detector input? |
| --- | --- | --- | --- |
| `<name>.sigmf-meta` / `.sigmf-data` | ✅ yes | ✅ yes | **Yes — this is the one to use.** Full pipeline incl. spectrogram PNGs. |
| `<name>.rxch.sigmf-meta` / `.sigmf-data` | ❌ no (header only) | ✅ yes | No — produces 0 boxes (it is a stripped receive-channel capture). |
| `<name>.rxch.decoded.sigmf-meta` | ✅ yes | ❌ no | Labels only (no PNGs). Duration inferred from annotations. |
| `<name>.rx.sigmf-meta` | ❌ no | ❌ no | No. |
| `<name>.rx.decoded.sigmf-meta` | ✅ yes | ❌ no | Labels only (no PNGs). |

So the canonical inputs are the **base** stems. All are `cf32_le` (complex
float32) at a 245.76 MHz sample rate:

| Base stem | Data | Samples | Annotations | `core:label` classes present |
| --- | ---: | ---: | ---: | --- |
| `example_full160` | 0.62 MB | 0.078 M | 3 | ZC, METADATA, 802_11ax |
| `example_wide80` | 1.54 MB | 0.192 M | 7 | ZC, METADATA, BPSK, Bluetooth, 802_11ax |
| `example_5g100` | 25.06 MB | 3.13 M | 6 | ZC, METADATA, 5G_Downlink, OFDM |
| `example_mixed` | 39.81 MB | 4.98 M | 11 | ZC, METADATA, Broadband_FM, Narrowband_FM, 5G_Downlink, QPSK, OFDM |
| `example_timegroups` | 76.67 MB | 9.58 M | 16 | ZC, METADATA, 5G_Downlink, BPSK, Bluetooth, QPSK, OFDM |
| `comprehensive` | 4.08 GB | 510.5 M | 597 | ZC, METADATA, BPSK, QPSK, 16QAM, 5G_Downlink, OFDM, 802_11ax, Bluetooth, Broadband_FM, Narrowband_FM |

Each annotation provides `core:sample_start`, `core:sample_count`,
`core:freq_lower_edge`, `core:freq_upper_edge`, and a human label in
`core:label` (plus `wfgt:*` extensions). `ZC` = Zadoff–Chu sync; `METADATA` =
the in-band metadata-modem burst.

---

## The spectrogram coordinate system

Boxes are placed on an STFT image with **top-left origin** (the convention YOLO
and COCO expect):

- **X = time**, increasing to the **right**. One column per STFT frame.
  Number of frames `≈ 1 + (total_samples - fft_size) / hop_size`.
- **Y = frequency**, with **high baseband frequency at the top** and **low at
  the bottom**. One row per FFT bin kept inside the displayed band.
- `df = sample_rate / fft_size` is the frequency resolution (e.g. 245.76 MHz /
  4096 ≈ 60 kHz per bin). Time resolution is `hop_size / sample_rate` per column.

The full image is `freq_bins` tall × `total_frames` wide, then optionally cut
into tiles. These captures are **very wide** (time ≫ frequency), so tiling along
time is usually necessary — see [Tiling](#tiling).

---

## Tool 1 — `sigmf_annotations_to_detection_boxes.py`

### Quick start

Full pipeline (labels **and** spectrogram PNGs) on a small composite, classed by
the clean `core:label` field, tiled to 1024-bin × 512-frame images:

```bash
COMPOSITES=/Users/bqn82/Documents/holoscan/holoscan_waveform_generation/composition/composites

conda run -n holoscan_compose python -B sigmf_annotations_to_detection_boxes.py \
  "$COMPOSITES/example_mixed.sigmf-meta" \
  --outdir ./out/mixed \
  --fft-size 2048 --hop-size 512 \
  --tile-freq-bins 1024 --tile-frames 512 \
  --class-mode field --class-field core:label \
  --write-spectrograms
```

The `.sigmf-data` is auto-discovered next to the meta. You may also pass it
explicitly as the second positional argument. Omit `--write-spectrograms` (and
you can omit the data file entirely) when you only need labels.

### Outputs

Everything lands under `--outdir`:

| Path | What |
| --- | --- |
| `dino_coco.json` | COCO/DINO ground truth. `images`, `annotations` (`bbox = [x, y, w, h]` in pixels), `categories`, and an `info.spectrogram` block recording fft/hop/sample-rate/freq-extent so the geometry is reproducible. Each annotation also carries `sigmf:*` provenance (label, sample span, Hz edges) and `spectrogram:visible_fraction`. |
| `yolo_labels/*.txt` | One file per tile image. Lines are `class_id x_center y_center width height`, all normalized 0–1. Empty tiles get no file unless `--include-empty-tiles`. |
| `classes.txt` | Class names, one per line; line number = `class_id`. |
| `dataset.yaml` | Ultralytics-style dataset stub (`names:` map). |
| `boxes.csv` | Flat table of every tile-clipped box: pixel coords, YOLO coords, source SigMF label, sample span, Hz edges, visible fraction. |
| `spectrogram_tiles.csv` | Tile manifest: `image_id`, `file_name`, `width`, `height`, and the tile's `(x_frame_start, y_bin_start_top)` offset in the full image. |
| `summary.json` | Run summary: layout, tile count, box count, category count. |
| `images/*.png` | Spectrogram tiles (only with `--write-spectrograms`). File names encode position: `<stem>_x<frame>_y<bin>_w<W>_h<H>.png`. |

The COCO image records, YOLO label files, and PNG file names all key off the
same tile names, so the three representations stay aligned.

### Option reference

**Spectrogram layout**

| Option | Default | Meaning |
| --- | --- | --- |
| `--fft-size`, `--nfft` | 4096 | FFT size = number of frequency bins (full band). |
| `--hop-size`, `--hop` | `fft_size / 4` | STFT hop in samples. Smaller = more time columns. |
| `--freq-min-hz`, `--freq-max-hz` | full Nyquist | Crop the displayed band. Accepts `25e6`, `25MHz`, `-40MHz`, etc. |
| `--time-box-mode` | `stft-overlap` | How a sample span maps to columns — see below. |

**Tiling**

| Option | Default | Meaning |
| --- | --- | --- |
| `--tile-frames` | full width | Tile width in STFT frames. Omit for a single full-width image. |
| `--tile-freq-bins` | full height | Tile height in frequency bins. Omit for full height. |
| `--tile-overlap-frames` | 0 | Horizontal overlap between tiles. |
| `--tile-overlap-freq-bins` | 0 | Vertical overlap between tiles. |
| `--include-empty-tiles` | off | Also emit tiles (and empty label files) that contain no boxes. |
| `--min-visible-fraction` | 0.0 | Drop a tile-clipped box if less than this fraction of its full-image area survives the clip. |

**Boxes / classes**

| Option | Default | Meaning |
| --- | --- | --- |
| `--class-mode` | `single` | `single` \| `field` \| `waveform-class` — see below. |
| `--class-name` | `signal` | Class for `single` mode, and the fallback for the others. |
| `--class-field` | `compose:waveform_name` | Annotation key read in `field` mode. **For these composites use `core:label`.** |
| `--exclude-pn` | off | Skip PN sync / preamble annotations (matched by `sync_pn`, `pn9_bpsk_sync`, `pn_preamble`). Does **not** remove `ZC`/`METADATA`. |
| `--min-box-width-px` | 1.0 | Expand boxes narrower than this (in frames) so sub-pixel signals stay visible. |
| `--min-box-height-px` | 1.0 | Expand boxes shorter than this (in bins). |

**Spectrogram rendering** (only used with `--write-spectrograms`)

| Option | Default | Meaning |
| --- | --- | --- |
| `--write-spectrograms` | off | Render PNG tiles matching the COCO image records (requires the `.sigmf-data`). |
| `--image-dirname` | `images` | Subdirectory for the PNGs. |
| `--spectrogram-colormap` | `viridis` | `viridis` or `gray`. |
| `--spectrogram-dynamic-range-db` | 80 | Displayed dynamic range below the per-tile max. |
| `--spectrogram-percentile` | 99.5 | Percentile used as the bright end (vmax) per tile. |
| `--spectrogram-vmin-db` / `--spectrogram-vmax-db` | auto | Fixed dB scaling instead of per-tile auto. |
| `--max-spectrogram-tiles` | 0 (all) | Render only the first N tiles — handy for a quick preview of a huge capture. |
| `--verbose` | off | Print rendering progress. |

### Class modes (important)

- **`single`** — every box gets one class (`--class-name`, default `signal`).
  Use for a pure energy/"is-there-a-signal" detector.
- **`field`** — use a SigMF annotation field verbatim as the class. The default
  field is `compose:waveform_name`, **which does not exist in these composites**;
  annotations here label the class in `core:label`. So pass
  `--class-mode field --class-field core:label` to get clean per-waveform
  classes (`5G_Downlink`, `QPSK`, `802_11ax`, `ZC`, `METADATA`, …). This is the
  recommended mode for the composites.
- **`waveform-class`** — infer a normalized family from several fields (handles
  the long generated labels from the PN-sync survey, e.g.
  `payload_evt00001_slot00001_5G_Downlink_snr_30dB...` → `5G_Downlink`).
  Families it knows: `5G_Downlink/Uplink`, `LTE_Downlink/Uplink`, `802_11ax/ac/n`,
  `Bluetooth`, `Broadband_FM`, `Broadcast_FM`, `Narrowband_FM`, `16/64/256QAM`,
  `QPSK`, `BPSK`, `OFDM`. Labels it does not recognize (e.g. `ZC`, `METADATA`)
  fall back to `--class-name` (`signal`). Best for the PN-sync survey captures;
  for the composites here `field`+`core:label` is cleaner.

### Time-box modes

`--time-box-mode` controls how an annotation's `[sample_start, sample_stop)`
range becomes a span of STFT columns:

- **`stft-overlap`** (default) — cover every column whose FFT window touches the
  signal. Slightly wider boxes; matches what you actually *see* in the
  spectrogram, so it is the safest for training.
- **`sample`** — nominal sample-accurate mapping (`start/hop … stop/hop`), no
  window-edge expansion. Tighter boxes.
- **`frame-center`** — columns whose window *center* falls inside the span.

### Tiling

These recordings are extremely wide in time (e.g. `comprehensive` is ~498k STFT
frames at fft=4096/hop=1024 but only 4096 bins tall). A single image would be
unusable, so tile along time with `--tile-frames`. By default, tiles with no
boxes are dropped — only tiles containing at least one annotation are written
(258 of ~487 possible columns for `comprehensive`). Add `--include-empty-tiles`
if your training pipeline needs negative tiles too. Use `--tile-overlap-frames`
to keep signals that straddle a tile boundary visible in both tiles, and
`--min-visible-fraction` to discard slivers left by the clip.

---

## Tool 2 — `visualize_detection_boxes.py`

### Quick start

**Ground-truth overlays** on the rendered spectrogram tiles (no predictions):

```bash
conda run -n holoscan_compose python -B visualize_detection_boxes.py \
  ./out/mixed/dino_coco.json \
  --image-root ./out/mixed/images \
  --outdir ./out/mixed_viz \
  --max-images 0          # 0 = render every image
```

**Compare against model predictions** (COCO file with `score` per box):

```bash
conda run -n holoscan_compose python -B visualize_detection_boxes.py \
  ./out/mixed/dino_coco.json \
  --pred-coco /path/to/model_predictions.json \
  --image-root ./out/mixed/images \
  --outdir ./out/mixed_eval \
  --iou-threshold 0.5 --select worst --max-images 32
```

**Smoke test with no model** — synthesize jittered predictions from the GT:

```bash
conda run -n holoscan_compose python -B visualize_detection_boxes.py \
  ./out/mixed/dino_coco.json \
  --image-root ./out/mixed/images \
  --outdir ./out/mixed_random \
  --generate-random-predictions \
  --max-images 12 --select random
```

If you omit `--image-root`, boxes are drawn on a plain dark canvas — useful for
checking pure geometry without rendering PNGs first.

### Prediction inputs and matching

Provide predictions one of three (mutually exclusive) ways:

- `--pred-coco FILE.json` — COCO predictions; image IDs must be a subset of the
  ground-truth image IDs. Optional `score` per annotation drives match priority.
- `--pred-yolo-label-dir DIR` — a directory of YOLO `.txt` files named to match
  the tile PNGs. Class index maps through the GT category order; an optional 6th
  column is the confidence (else `--default-yolo-score`, default 1.0).
- `--generate-random-predictions` — jitter the GT into fake predictions
  (`--random-drop-probability`, `--random-jitter-fraction`,
  `--random-false-positive-rate`, `--random-seed`, …). Writes them to
  `--random-output` or `random_predictions.json`.

Matching is **greedy per image, per category**, highest-score-first, accepting a
pair when `IoU ≥ --iou-threshold` (default 0.5). Each match becomes a true
positive (TP), false positive (FP, unmatched prediction), or false negative
(FN, unmatched ground truth). Per-class and overall precision / recall / mean
IoU are written to `comparison_summary.json`, and every pairing to `matches.csv`.

`--select` chooses which images get overlay PNGs when `--max-images` limits the
count: `first`, `random`, or `worst` (most FP+FN first — best for error review).

### Overlay legend

| Drawing | Meaning |
| --- | --- |
| Solid **green** box | Ground truth, matched (TP) |
| Solid **orange** box | Ground truth, missed (FN) |
| Dashed **blue** box | Prediction, matched (TP) — annotated with score and IoU |
| Dashed **red** box | Prediction, false positive (FP) — annotated with score |

When there are no predictions, all ground truth is drawn solid green.

### Outputs

| Path | What |
| --- | --- |
| `overlays/*_boxes.png` | One annotated image per selected tile. |
| `comparison_summary.json` | Box counts + overall and per-category TP/FP/FN, precision, recall, mean IoU. |
| `matches.csv` | Every GT↔pred pairing with IoU and result (`tp`/`fp`/`fn`). |
| `random_predictions.json` | Only with `--generate-random-predictions`. |

---

## End-to-end worked example

From the directory containing the scripts:

```bash
COMPOSITES=/Users/bqn82/Documents/holoscan/holoscan_waveform_generation/composition/composites

# 1) Annotations -> boxes + spectrogram tiles
conda run -n holoscan_compose python -B sigmf_annotations_to_detection_boxes.py \
  "$COMPOSITES/example_mixed.sigmf-meta" \
  --outdir ./out/mixed \
  --fft-size 2048 --hop-size 512 \
  --tile-freq-bins 1024 --tile-frames 512 \
  --class-mode field --class-field core:label \
  --write-spectrograms

# 2) Sanity-check the geometry by overlaying GT on the tiles
conda run -n holoscan_compose python -B visualize_detection_boxes.py \
  ./out/mixed/dino_coco.json \
  --image-root ./out/mixed/images \
  --outdir ./out/mixed_viz --max-images 0

# 3) Dry-run a scoring report with synthetic predictions
conda run -n holoscan_compose python -B visualize_detection_boxes.py \
  ./out/mixed/dino_coco.json \
  --image-root ./out/mixed/images \
  --outdir ./out/mixed_eval \
  --generate-random-predictions --select worst --max-images 6
```

Open `./out/mixed_viz/overlays/*.png` to confirm the green boxes sit on the
signal energy, then point step 2/3's `--pred-coco` at your real detector output.

---

## Validation results

Both tools were exercised against the composites with the `holoscan_compose`
env. All checks passed.

**Unit tests** (`tests/test_sigmf_detection_boxes.py`,
`tests/test_visualize_detection_boxes.py`):

```bash
conda run -n holoscan_compose python -m unittest \
  tests.test_sigmf_detection_boxes tests.test_visualize_detection_boxes -v
# Ran 6 tests ... OK
```

**End-to-end on real composites:**

| Check | Input | Result |
| --- | --- | --- |
| Full pipeline + tiled PNGs | `example_full160` (fft1024/hop256) | 3 source boxes → 10 tile boxes, 4 tiles, 4 PNGs (256×512 RGB) |
| Full pipeline, 7 classes | `example_mixed` (fft2048/hop512) | 11 → 63 boxes, 21 tiles, 21 PNGs |
| Labels only, large capture | `comprehensive` 4.08 GB, 597 anns (`waveform-class`) | 597 → 980 boxes, 258 tiles, 10 classes, **0.5 s** |
| GT overlays on real tiles | `example_mixed` | 21 overlays; boxes land on the signal energy |
| Scoring vs synthetic preds | `example_mixed`, IoU 0.3 | TP=50 FP=6 FN=13, precision 0.89 / recall 0.79 / mean IoU 0.82 |
| YOLO round-trip (labels fed back as preds) | `example_mixed` | TP=63 FP=0 FN=0, precision/recall/IoU = **1.000** (label↔COCO consistency) |
| Meta-only mode (no `.sigmf-data`) | `example_timegroups.rx.decoded` | Length inferred from annotations; boxes written, no PNGs |
| Annotation-less input | `example_mixed.rxch` | 0 boxes / 0 tiles (expected — `.rxch` has no annotations) |
| COCO well-formedness | `example_mixed` | No dangling image/category refs; full `info.spectrogram` block present |

---

## Troubleshooting

- **0 boxes / 0 tiles.** The meta has no annotations. The `.rxch` and `.rx`
  metas are header-only; use the **base** stem (or a `.decoded` sidecar).
- **`--write-spectrograms` errors with "requires an existing .sigmf-data file".**
  PNG rendering needs the IQ data. Use a base stem that has its `.sigmf-data`, or
  drop `--write-spectrograms` for labels-only output.
- **All boxes collapse to one class `signal` in `field` mode.** The default
  `--class-field compose:waveform_name` is absent here. Use `--class-field
  core:label`.
- **`waveform-class` lumps `ZC`/`METADATA` into `signal`.** Expected — they are
  not waveform families. Use `field`+`core:label` if you want them as distinct
  classes.
- **Images are absurdly wide / few tiles.** You did not tile along time. Add
  `--tile-frames` (e.g. 512–1024). Remember tiles without boxes are dropped
  unless `--include-empty-tiles`.
- **Boxes look slightly larger than the signal.** That is `stft-overlap` mode
  including window-edge columns. Switch to `--time-box-mode sample` for tighter
  boxes.
- **Spectrogram looks washed out / too dark.** Tune
  `--spectrogram-percentile` and `--spectrogram-dynamic-range-db`, or pin
  `--spectrogram-vmin-db`/`--spectrogram-vmax-db`.
- **Huge captures are slow to render.** Use `--max-spectrogram-tiles N` for a
  preview, or generate labels first (no `--write-spectrograms`) and render only
  the tiles you care about later.
- **`ModuleNotFoundError: numpy` / `PIL`.** You are not in the `holoscan_compose`
  env. Prefix commands with `conda run -n holoscan_compose`.
