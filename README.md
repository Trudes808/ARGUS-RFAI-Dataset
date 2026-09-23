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
 generate_waveforms_24576.m   ─┐          compose.py  ── 200 MHz .sigmf-data/-meta
   -> generated_waveforms_24576/          (config-driven)        (TX ground truth)
 generate_zadoff_chu_50mhz.m  ─┘                 │
   -> generated_sync_sequences/                  ▼  transmit OTA, capture at the RX
 decode_waveforms_24576.m  (BER/FM QA)     decode.py  ── recovers annotations into
 export_fm_audio.m         (listen)                      the received .sigmf-meta
```

## Which files produced the published datasets

| Dataset | Produced by |
|---|---|
| Tx Waveform Library (287 waveforms) | `generate_waveforms_24576.m` (default `Mode`) → `generated_waveforms_24576/` |
| Transmitted composite for the attenuation sweep | `composition/build_comprehensive_ordered.py` → `comprehensive_ordered.yaml` (1,722 waveforms, 444 time groups) → `compose.py` (driver: `run_ordered.py`) |
| Attenuation-sweep + LTE-holdout labels | `composition/decode.py`, default settings ([`DECODE.md`](composition/DECODE.md)) |
| ZC sync sequence | `generate_zadoff_chu_50mhz.m` → `generated_sync_sequences/` (committed; needed to decode) |

## 1. Prerequisites

### MATLAB (waveform generation + per-waveform BER/FM validation)
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
- `Disco_Snail_easter_egg.mp3` (audio source for the FM waveforms) in the repo root. Any
  mono/stereo audio works; pass another with `"AudioFile"`, but the published FM waveforms
  used this file.

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
generate_waveforms_24576                 % full library -> generated_waveforms_24576/
generate_zadoff_chu_50mhz                % 50 MHz ZC -> generated_sync_sequences/
% optional quality checks:
decode_waveforms_24576('generated_waveforms_24576')                      % BER, expect 0
decode_waveforms_24576('generated_waveforms_24576','Channel','wireless') % BER through a channel
export_fm_audio                          % listenable recovered-FM WAVs -> recovered_fm_audio/
```
`generate_waveforms_24576('Mode','smoke')` makes a small, fast subset for a quick check.
Instead of regenerating, you can download the library from the TDR *Tx Waveform Library*
dataset into `generated_waveforms_24576/`.

### Step 2: Python, compose a 200 MHz recording
```bash
cd composition
python compose.py example_timegroups.yaml        # -> composites/<name>.sigmf-{data,meta} + <name>.rx.sigmf-meta
```
The dataset composite (ordered comprehensive set: original powers, sectioned by class then
largest→smallest bandwidth, classes time-separated, repeated at 20/10/5/1/0.2/0.04 ms):
```bash
python build_comprehensive_ordered.py            # writes comprehensive_ordered.yaml
python run_ordered.py                            # compose + clean decode check
```
Config schema and time layout: [`composition/README.md`](composition/README.md).

### Step 3: Python, decode a received capture
```bash
python decode.py <capture>.sigmf-data <capture>.sigmf-meta \
                 ../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat
```
Overwrites the meta's annotations in place. Full options, the annotation schema, and how to
check a re-decode against the published labels: [`composition/DECODE.md`](composition/DECODE.md).

### One-shot validation (compose + clean + channel + report)
```bash
python run_example.py        # 5 example configs (incl. time groups), clean + channel
python run_comprehensive.py  # entire library, clean + channel (multi-GB)
```

### OTA power sweeps
The composite is generated **once**. To sweep power:
- **Whole-signal power** (vary SDR output gain or an attenuator, as in the attenuation
  sweep): transmit the one composite at each level, capture each, and **decode each
  capture**. Decoding is power-invariant (the ZC/metadata are constant-power and the
  detector metric is normalized).
- **Per-waveform power**: set `power_db` per waveform in the config and recompose.

## 3. Repository layout

| Path | Purpose |
|------|---------|
| `generate_waveforms_24576.m` | Generate all 9 classes @245.76 MSps (PN9 / MP3 sources), `.mat` + `.json` + manifest. |
| `decode_waveforms_24576.m` | Per-waveform receive chain → BER (digital) or FM audio QA; clean & channel. |
| `generate_zadoff_chu_50mhz.m` | 50 MHz Zadoff-Chu CAZAC sync sequence. |
| `export_fm_audio.m` | Render listenable recovered-FM WAVs. |
| `Disco_Snail_easter_egg.mp3` | FM audio source. |
| `generated_sync_sequences/` | ZC sync `.mat`/`.json` (committed). |
| `composition/compose.py` | Build a 200 MHz SigMF composite from a YAML config. |
| `composition/decode.py`, `fine_comb_decode.py` | Recover annotations from a capture (fast / low-SNR). See `DECODE.md`. |
| `composition/metadata_modem.py`, `zc_sync.py`, `geometry.py`, `protocol.py`, `channel.py` | OFDM metadata modem, ZC sync, frequency plan, shared TX/RX layout, channel model. |
| `composition/build_comprehensive_config.py`, `build_comprehensive_ordered.py` | Pack the whole library into a config (varied power / ordered). |
| `composition/run_example.py`, `run_comprehensive.py`, `run_ordered.py` | End-to-end validation drivers. |
| `composition/*.yaml` | Composition configs (`comprehensive_ordered.yaml` = the dataset composite). |
| `composition/sigmf_annotations_to_detection_boxes.py`, `visualize_detection_boxes.py` | Convert annotations to tiled detection-box labels + preview them (`DETECTION_BOXES.md`). |

Not committed (regenerable or published in TDR; see `.gitignore`):
`generated_waveforms_24576/`, `composition/composites/`, `recovered_fm_audio/`.

## 4. Expected results

- Library: 287 waveforms; `decode_waveforms_24576` → **BER 0** on a clean resample and
  through the mild wireless channel; FM audio SNR ~38–92 dB.
- Composition decode (clean & wireless channel): **every waveform's class, variation, time
  position, frequency edges, and length recovered** (validated 287/287 on the full library,
  ≤1 sample timing error through the channel). Decoded `.sigmf-meta` passes `sigmf`
  validation.

## 5. Notes
- IQ is `complex64` (`cf32_le`) at 245.76 MSps throughout.
- Frequency plan: three 50 MHz blocks on a 60 MHz pitch (centers −60/0/+60 MHz, 5 MHz
  guards); >50 MHz waveforms merge slots (≤110/170 MHz for 2/3 slots). See
  `composition/README.md`.
- Composing the full library takes a few minutes and ~6 GB RAM.
