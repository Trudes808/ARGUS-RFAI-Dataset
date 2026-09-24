# ARGUS-RFAI-dataset

Code that generates the transmit signals and ground-truth labels for the **USRP X410
Wideband Signal Detection** datasets (Texas Data Repository). The data itself (waveform
library, composites, captures) is published in TDR; this repo holds only the methods to
regenerate it. Detection methods that use the data (e.g. Holohub-Signal-Detection) live in
their own repos.

The pipeline has three stages:

1. **Generate** a library of standards / modulation / FM waveforms at **245.76 MSps**
   (MATLAB), each built at its native rate and rationally resampled.
2. **Compose** them into a **200 MHz baseband** SigMF recording, frequency- and
   time-multiplexed, with each block tagged by a Zadoff-Chu sync plus a decodable OFDM
   metadata burst (Python).
3. **Decode**: on reception (no annotations), sync the ZCs, decode the metadata, and write
   per-waveform ground-truth annotations back into the SigMF meta. This is how every
   published label was made; see **[`composition/DECODE.md`](composition/DECODE.md)**.

```
 MATLAB                                   Python (conda: holoscan_compose)
 ────────────────────────────────        ──────────────────────────────────────
 generate_waveforms_24576.m   ─┐          build_*_ordered.py ── dataset configs (.yaml)
 generate_lte_24576.m          ├─> generated_waveforms_24576/          │
 generate_zadoff_chu_50mhz.m  ─┴─> generated_sync_sequences/           ▼
                                          compose.py  ── 200 MHz .sigmf-data/-meta
                                                 │
                                                 ▼  transmit OTA, capture at the RX
                                          decode.py   ── recovers annotations into
                                                         the received .sigmf-meta
```

## Which files produced the published datasets

| Dataset | Produced by |
|---|---|
| Tx Waveform Library (287 waveforms) | `generate_waveforms_24576.m` (default `Mode`) → `generated_waveforms_24576/` |
| LTE waveforms (18) | `generate_lte_24576.m` → `generated_waveforms_24576/` (`lte_manifest.csv`) |
| Transmitted composites | `composition/comprehensive_ordered{,_-30dB,_-60dB}.yaml` and `lte_ordered{,_-30dB}.yaml` → `compose.py`. Which config was transmitted for which capture: [`composition/README.md`](composition/README.md#dataset-configs) |
| Attenuation-sweep + LTE-holdout labels | `composition/decode.py`, default settings ([`DECODE.md`](composition/DECODE.md)) |
| ZC sync sequence | `generate_zadoff_chu_50mhz.m` → `generated_sync_sequences/` (committed; needed to decode) |

## 1. Prerequisites

### MATLAB (waveform generation)
- **MATLAB R2025b** (or compatible) with toolboxes: **Communications, DSP System,
  Signal Processing, 5G, WLAN, Bluetooth, LTE**.
- The **"OFDM Transmitter and Receiver" example helpers** (`helperOFDMTx`,
  `helperOFDMRx`, …). Install once with:
  ```matlab
  openExample('comm/OFDMEndToEndExample')
  ```
  The scripts auto-locate the helpers under `~/Documents/MATLAB/Examples/*/comm/
  OFDMEndToEndExample` and `matlabroot/examples/...`. If they live elsewhere, set:
  ```bash
  export OFDM_HELPER_DIR=/path/to/OFDMEndToEndExample   # before launching MATLAB
  ```
- `Disco_Snail_easter_egg.mp3` (audio source for the FM waveforms) in the repo root.

### Python (composition + decoding)
```bash
conda env create -f composition/environment.yml
conda activate holoscan_compose
```
(numpy, scipy, **sigmf** ≥1.2, pyyaml, matplotlib. Pip equivalent:
`pip install "numpy>=1.26" "scipy>=1.11" "sigmf>=1.2" pyyaml matplotlib`.)

### Paths and portability

No machine-specific paths are hardcoded. Everything resolves relative to the script or
config file, so the only requirement is to **keep the repo layout intact**:

```
ARGUS-RFAI-Dataset/
├── generate_*.m, Disco_Snail_easter_egg.mp3
├── generated_sync_sequences/        ZC .mat (committed)
├── generated_waveforms_24576/       waveform library (generated or downloaded from TDR; gitignored)
└── composition/                     *.py, *.yaml, decode/export .m
    └── composites/                  compose.py output (gitignored)
```

- **Waveform library**: must live at `generated_waveforms_24576/` in the repo root. The
  generators write there by default, the builders read `../generated_waveforms_24576/`
  relative to `composition/`, and every YAML has `library_root: ../generated_waveforms_24576`.
  If you download the TDR *Tx Waveform Library*, unpack it there. To use another location,
  pass `"OutputRoot"` to the generators and edit `library_root` in the YAMLs.
- **ZC sync file**: YAMLs use `zc_file: ../generated_sync_sequences/...`; `decode.py` takes
  it as an argument.
- **YAML paths** (`library_root`, `zc_file`, `output_dir`) are relative to the YAML's own
  directory, so `compose.py` can be run from any working directory.
- **Python**: run scripts in `composition/` directly (`python composition/compose.py
  composition/comprehensive_ordered.yaml` works from the repo root); they import their
  sibling modules. Paths you pass on the command line (captures, metas) are relative to your
  working directory.
- **MATLAB**: the generators write relative to their own location regardless of the
  current folder; call them from the repo root (or `addpath` it). The decode/export scripts
  live in `composition/`: `cd composition` or `addpath composition` first.
- **FM audio source**: `Disco_Snail_easter_egg.mp3` in the repo root (generators and
  `export_fm_audio` default to it; override with `"AudioFile"`). Each FM waveform's metadata
  stores the absolute audio path from the machine that generated it. If that path doesn't
  exist (e.g. the TDR library, generated on macOS), `decode_waveforms_24576` falls back to the
  mp3 of the same name in the repo root.
- **OFDM example helpers**: found via, in order, the MATLAB path, `OFDM_HELPER_DIR`,
  `$HOME/Documents/MATLAB/Examples/*/comm/OFDMEndToEndExample` (`%USERPROFILE%` on Windows),
  then `matlabroot/examples`. No Java needed, so headless `matlab -batch` works.

### Cross-platform reproducibility

Verified 2026-09-24 by regenerating everything with MATLAB R2026b on Linux and comparing to
the published library (generated on macOS):
- Identical: the ZC, every waveform's payload bits, and the IQ of 272 / 287 waveforms (to
  float precision), plus all five YAMLs from the builders.
- **FM (9 waveforms)** differs: MP3 decoders differ by platform (Linux returns the audio
  530 samples / 12 ms later than macOS, with slightly different PCM), so the FM IQ is not
  bit-exact. For the same reason, audio-QA of the *TDR* FM waveforms on Linux reports a low
  score (~6 dB SNR, corr ~0.85) because the reference is misaligned; the audio itself is
  intact. Regenerated-and-decoded on the same machine, FM scores 44–99 dB.
- **Bluetooth EDR2M / EDR3M / LE500K (6 waveforms)** differ with the same config and bits
  (Bluetooth Toolbox version).
- For bit-exact inputs, use the TDR library. Labels do not depend on waveform IQ (only on
  lengths and bandwidths), so regenerated composites decode to annotations identical to the
  published ones.

If you slice signals out of a composite or capture using the annotations, filter each slice
to its annotated band (± up to half the 5 MHz guard) before demodulating; stacked neighbours
sit 5 MHz away and otherwise leak into receivers without a matched filter (e.g. the
raised-cosine single-carrier waveforms).

## 2. Workflow

### Step 1: MATLAB, generate the library + sync sequence
```matlab
generate_waveforms_24576     % 287-waveform library -> generated_waveforms_24576/
generate_lte_24576           % 18 LTE waveforms     -> generated_waveforms_24576/
generate_zadoff_chu_50mhz    % 50 MHz ZC            -> generated_sync_sequences/
```
Instead of regenerating, you can download the library from the TDR *Tx Waveform Library*
dataset into `generated_waveforms_24576/`.

Optional per-waveform QA (from `composition/`):
```matlab
cd composition
decode_waveforms_24576("../generated_waveforms_24576")   % BER (digital, expect 0) + FM audio SNR
decode_lte_24576("../generated_waveforms_24576")         % LTE BER, expect 0
export_fm_audio                                          % listenable FM WAVs -> ../recovered_fm_audio/
```

### Step 2: Python, build the configs and compose
```bash
cd composition
python build_comprehensive_ordered.py        # comprehensive_ordered.yaml
python build_comprehensive_ordered.py -30    # comprehensive_ordered_-30dB.yaml
python build_comprehensive_ordered.py -60    # comprehensive_ordered_-60dB.yaml
python build_lte_ordered.py                  # lte_ordered.yaml + lte_ordered_-30dB.yaml
python compose.py comprehensive_ordered.yaml # -> composites/<name>.sigmf-{data,meta} + <name>.rx.sigmf-meta
```
The committed YAMLs are the configs that were transmitted; the builders reproduce them.
Recipe, config schema and time layout: [`composition/README.md`](composition/README.md).

### Step 3: Python, decode a received capture
```bash
python decode.py <capture>.sigmf-data <capture>.sigmf-meta \
                 ../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat
```
Overwrites the meta's annotations in place. Full options, the annotation schema, and how to
check a re-decode against the published labels: [`composition/DECODE.md`](composition/DECODE.md).

## 3. Repository layout

| Path | Purpose |
|------|---------|
| `generate_waveforms_24576.m` | Generate all 9 classes @245.76 MSps (PN9 / MP3 sources), `.mat` + `.json` + manifest. |
| `generate_lte_24576.m` | LTE downlink RMC waveforms (6 bandwidths × 3 modulations) @245.76 MSps + `lte_manifest.csv`. |
| `generate_zadoff_chu_50mhz.m` | 50 MHz Zadoff-Chu CAZAC sync sequence. |
| `Disco_Snail_easter_egg.mp3` | FM audio source. |
| `generated_sync_sequences/` | ZC sync `.mat`/`.json` (committed). |
| `composition/build_comprehensive_ordered.py`, `build_lte_ordered.py` | Write the dataset configs. |
| `composition/*.yaml` | The transmitted dataset configs. |
| `composition/compose.py` | Build a 200 MHz SigMF composite from a config. |
| `composition/decode.py` | Recover annotations from a capture. See `DECODE.md`. |
| `composition/metadata_modem.py`, `zc_sync.py`, `geometry.py`, `protocol.py` | OFDM metadata modem, ZC sync, frequency plan, shared TX/RX layout. |
| `composition/decode_waveforms_24576.m`, `decode_lte_24576.m` | Per-waveform receive chains → BER (digital, LTE) or FM audio quality. |
| `composition/export_fm_audio.m` | Listenable multi-second FM WAVs through the same FM chain. |

Not committed (regenerable or published in TDR; see `.gitignore`):
`generated_waveforms_24576/`, `composition/composites/`, `recovered_fm_audio/`.

## 4. Notes
- IQ is `complex64` (`cf32_le`) at 245.76 MSps throughout.
- Frequency plan: three 50 MHz blocks on a 60 MHz pitch (centers −60/0/+60 MHz, 5 MHz
  guards); >50 MHz waveforms merge slots (≤110/170 MHz for 2/3 slots). See
  `composition/README.md`.
- Composing a comprehensive config writes ~12 GB.
