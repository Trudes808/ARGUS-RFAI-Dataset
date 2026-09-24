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

No source paths are hardcoded except the auto-located MATLAB OFDM helpers. Python paths are
relative to the config/script location, so keep the repo layout intact.

## 2. Workflow

### Step 1: MATLAB, generate the library + sync sequence
```matlab
generate_waveforms_24576     % 287-waveform library -> generated_waveforms_24576/
generate_lte_24576           % 18 LTE waveforms     -> generated_waveforms_24576/
generate_zadoff_chu_50mhz    % 50 MHz ZC            -> generated_sync_sequences/
```
Instead of regenerating, you can download the library from the TDR *Tx Waveform Library*
dataset into `generated_waveforms_24576/`.

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

Not committed (regenerable or published in TDR; see `.gitignore`):
`generated_waveforms_24576/`, `composition/composites/`.

## 4. Notes
- IQ is `complex64` (`cf32_le`) at 245.76 MSps throughout.
- Frequency plan: three 50 MHz blocks on a 60 MHz pitch (centers −60/0/+60 MHz, 5 MHz
  guards); >50 MHz waveforms merge slots (≤110/170 MHz for 2/3 slots). See
  `composition/README.md`.
- Composing a comprehensive config writes ~12 GB.
