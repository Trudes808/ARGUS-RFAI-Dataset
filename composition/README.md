# 200 MHz Composition Pipeline (SigMF)

Composes the generated 245.76 MSps waveforms into a 200 MHz baseband SigMF
recording, then recovers ground-truth annotations at the receiver by syncing the
Zadoff-Chu sequences and decoding an embedded OFDM metadata burst (see
[`DECODE.md`](DECODE.md)).

Run from this directory in the `holoscan_compose` conda env
(`conda env create -f environment.yml`).

## Frequency plan
Three 50 MHz blocks on a **60 MHz pitch** (5 MHz guard each side), centered at
**−60 / 0 / +60 MHz**. A waveform wider than 50 MHz merges contiguous block
slots; merged usable bandwidth = `60·N − 10` MHz → **50 / 110 / 170 MHz** for
N = 1 / 2 / 3 slots (fits 80 and 160 MHz waveforms). Within one block, up to three
sub-50 MHz waveforms are stacked in frequency with 5 MHz guards.

## Time layout
The composition is a sequence of **time groups**, each a full frequency-multiplexed
set of blocks. Within a group, every block uses the same time window:
`ZC · metadata · [guard1] · waveform(s) · [guard2]`
- **ZC**: the 50 MHz Zadoff-Chu sequence at the block center, the timing reference.
- **Metadata**: OFDM burst (QPSK, reference-symbol channel estimate + pilots,
  repetition FEC + CRC-32) encoding each waveform's class, variation, position
  **relative to the ZC**, bandwidth, and power.
- **Data**: the waveforms.
- `guard1_ms` (2.5 ms) is the metadata→waveform guard; `guard2_ms` (5 ms) is the
  guard before the next group's ZC.

ZC and metadata are held at a fixed RMS (`ref_rms`) for reliable detection.

## Files
- `build_comprehensive_ordered.py`, `build_lte_ordered.py` — write the dataset configs
  (see below).
- `compose.py` — read a config → build the composite → write SigMF (TX truth + bare RX meta).
- `decode.py` — sync ZCs, decode metadata, rebuild annotations into a received meta.
- `geometry.py` — frequency plan, slot/merge math, frequency stacking + fit checks.
- `protocol.py` — shared TX/RX layout constants + metadata (de)serialization.
- `metadata_modem.py` — the OFDM metadata modem (encode/decode bytes).
- `zc_sync.py` — ZC load + frequency-shifted matched-filter synchronization.

## Dataset configs
Both builders use the same **ordered** recipe: `power_mode: original` (as-stored
amplitudes), waveforms sorted by class then largest→smallest bandwidth and
frequency-packed into time groups, with the whole set repeated at 20 / 10 / 5 / 1 /
0.2 / 0.04 ms (tiling short waveforms / cutting long ones). The dB variants are
identical to the 0 dB config except `waveform_gain_db`, which scales the waveforms only
(ZC and metadata stay at `ref_rms`), so the sync still decodes at high attenuation.

| Config | Built by | Waveforms | Time groups | Transmitted for |
|---|---|---|---|---|
| `comprehensive_ordered.yaml` | `python build_comprehensive_ordered.py` | 1,722 (287 × 6 durations) | 444 | `attenuation_dB_{0..30}` |
| `comprehensive_ordered_-30dB.yaml` | `python build_comprehensive_ordered.py -30` | 1,722 | 444 | `attenuation_dB_30_v2`, `_{35..60}` |
| `comprehensive_ordered_-60dB.yaml` | `python build_comprehensive_ordered.py -60` | 1,722 | 444 | `attenuation_dB_60v2`, `_{65..85}` |
| `lte_ordered.yaml` | `python build_lte_ordered.py` (writes both LTE configs) | 108 (18 × 6 durations) | 18 | `lte_attenuation_dB_{0..25}` |
| `lte_ordered_-30dB.yaml` | `python build_lte_ordered.py` | 108 | 18 | `lte_attenuation_dB_30_v2`, `_{35..55}`, `_60v3` |

The capture → config mapping is visible in the captures themselves as the ZC-to-waveform
power ratio (≈0 dB for 0 dB configs, ≈+30 dB for −30 dB, falling at high attenuation as
noise dominates the waveform measurement).

The builders read the library manifests in `../generated_waveforms_24576/`
(`waveform_manifest.csv` from `generate_waveforms_24576.m`, `lte_manifest.csv` from
`generate_lte_24576.m`).

## Usage
```bash
python compose.py comprehensive_ordered.yaml   # -> composites/comprehensive_ordered.{sigmf-data,sigmf-meta} + .rx.sigmf-meta
python decode.py <capture>.sigmf-data <capture>.sigmf-meta \
                 ../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat
```
Composing the comprehensive config writes ~12 GB (6.1 s at 245.76 MSps).

## Config schema
```yaml
name: comprehensive_ordered
output_dir: composites
zc_file: ../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat
library_root: ../generated_waveforms_24576
guard1_ms: 2.5           # guard: metadata -> waveform
guard2_ms: 5.0           # guard: waveform -> next group's ZC
power_mode: original     # as-stored amplitudes ("peak_db" = peak-normalize, then apply power_db)
ref_rms: 0.95            # fixed ZC/metadata RMS
peak_normalize: 0.95     # final composite peak (preserves relative powers)
waveform_gain_db: 0.0    # applied to waveforms only; -30.0 / -60.0 for the dB variants
time_groups:
  - blocks:
      - slots: [0]       # frequency slots this block occupies (0,1,2)
        waveforms:       # multiple -> stacked in frequency with 5 MHz guards
          - {mat: "BPSK/....mat", duration_ms: 20.0}   # tiled/cut to 20 ms
      - slots: [1, 2]    # contiguous slots -> one wide waveform (e.g. 80 MHz)
        waveforms:
          - {mat: "802_11ax/...CBW80....mat", duration_ms: 20.0}
```
`duration_ms` sets the annotation's `core:sample_count` / `wfgt:length_samples`; the
original length is kept in `wfgt:original_length_samples`. A block whose waveforms don't
fit (sum BW + 5 MHz guards > usable) raises an error.

## Output
- `<name>.sigmf-meta` — TX ground truth (full annotations).
- `<name>.rx.sigmf-meta` — receiver params only, no annotations (input to `decode.py`).
