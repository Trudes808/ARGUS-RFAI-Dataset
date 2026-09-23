# Decoding ground truth from a received capture

`decode.py` produces the ground-truth annotations in the published attenuation-sweep and
LTE-holdout captures (`attenuation_dB_*.sigmf-meta`, `lte_attenuation_dB_*.sigmf-meta`).
The labels are **recovered from the received IQ itself**, not copied from transmit-side
bookkeeping. So anyone with a `.sigmf-data` file and this directory can regenerate them.

Run everything below from `composition/` in the `holoscan_compose` env
(`conda env create -f environment.yml`).

## How the labels are encoded and recovered

The transmitted composite is a sequence of **time groups**. Each group holds up to three
50 MHz frequency blocks centered at −60 / 0 / +60 MHz (a wide waveform can merge 2–3
blocks, which is what the ±30 MHz candidate centers cover). Each block is laid out in time
as:

```
[ Zadoff-Chu sync ] → [ OFDM metadata burst ] → guard → [ data: the waveforms ] → guard
```

The metadata burst (QPSK OFDM with repetition FEC and a CRC-32) encodes, for every waveform
in the block, its class, variation name, bandwidth, power, and position **relative to the
ZC**. `decode.py` does the following:

1. Energy-gates the capture and runs a frequency-shifted ZC matched filter at each candidate
   block center (`geometry.RX_CANDIDATE_CENTERS` = −60, −30, 0, +30, +60 MHz), keeping
   peaks above a normalized-correlation threshold of 0.35.
2. Decodes the metadata burst after each ZC. The CRC rejects false alarms.
3. Converts the ZC-relative positions to absolute sample indices and writes one annotation
   per waveform, plus one `ZC` and one `METADATA` annotation per block.
4. De-duplicates, **replaces** the meta's `annotations` array, and writes the JSON.

Because positions are ZC-relative, bulk channel delay cancels out. Frequencies are
**complex baseband** (±122.88 MHz); the decoder does not read `core:frequency`.

Decoder modules: `decode.py` (CLI + `decode_composite()` / `validate()`),
`fine_comb_decode.py` (low-SNR alternative), `zc_sync.py`, `protocol.py`,
`metadata_modem.py`, `geometry.py`. ZC replica: `../generated_sync_sequences/
ZadoffChu_bw50MHz_R25_N601.mat` (root 25, length 601, 50 Mchip/s resampled to 245.76 MSps =
2955 samples).

Memory: the whole capture is loaded as `complex64`, so a ~14 GB attenuation capture needs
~14 GB+ of free RAM. It decodes in a few minutes, dominated by disk read.

## Usage

**The CLI overwrites the `.sigmf-meta` you pass it, in place.** Its `annotations` array is
replaced, and everything else (`global`, `captures`) is kept. Work on a copy if you want to
keep the original:

```bash
cp attenuation_dB_30.sigmf-meta attenuation_dB_30.decoded.sigmf-meta
python decode.py attenuation_dB_30.sigmf-data attenuation_dB_30.decoded.sigmf-meta \
                 ../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat
```

The input meta only needs `global.core:sample_rate` (a bare receiver meta with no
annotations is fine; any existing annotations are discarded). To write to a separate file
without copying, use the Python API:

```python
import decode
decode.decode_composite("cap.sigmf-data", "cap.sigmf-meta",
                        "../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat",
                        out_meta_path="cap.decoded.sigmf-meta")
```

Options (defaults are what the published labels used; see below):

| Option | Default | Meaning |
|---|---|---|
| `profile` (4th positional) | `fast` | Metadata modem profile: `fast`, `robust`, or `auto` (try each, let the CRC pick). |
| `--centers MHz,...` | all 5 | Block centers to scan. `-60,0,60` is ~40% faster if no merged 2-block waveforms are present. |
| `--active-factor X` | 1.5 | Energy-gate sensitivity. Lower means more search windows (finds weaker ZCs, slower). |

Low-SNR alternative (positional args: data, meta, zc, [pfa=1e-7], [profile=auto]). It
combs the entire capture with a CFAR threshold instead of energy gating; it is slower:

```bash
python fine_comb_decode.py cap.sigmf-data cap.decoded.sigmf-meta \
       ../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat
```

### Reading the output

The decoder prints the blocks/time groups decoded and the annotation count. If time-group
indices are missing, it prints a `WARNING` classifying each gap:
- **decode issue**: the ZC was found but the metadata failed its CRC. This is an SNR or
  corruption problem; detection tuning won't help.
- **unfound ZC**: nothing cleared the detector. Try a lower `--active-factor` or
  `fine_comb_decode.py`.

## Annotation schema

All frequencies are baseband Hz. Three annotation kinds, keyed by `wfgt:kind`:

| `wfgt:kind` | `core:label` | Extent | Extra fields |
|---|---|---|---|
| `waveform` | waveform class: `BPSK`, `QPSK`, `16QAM`, `OFDM`, `5G_Downlink`, `802_11ax`, `Bluetooth`, `Broadband_FM`, `Narrowband_FM` (attenuation sweep); `LTE` (holdout) | the transmitted waveform's samples × occupied BW | `wfgt:class`, `wfgt:variation` (library waveform name), `wfgt:occupied_bw_hz`, `wfgt:power_db`, `wfgt:length_samples`, `wfgt:original_length_samples`, `wfgt:zc_sample`, `wfgt:zc_metric` |
| `zadoff_chu` | `ZC` | 2955 samples × block center ±25 MHz | `wfgt:zc_metric` |
| `metadata` | `METADATA` | metadata burst × block center ±13 MHz | — |

All kinds also carry `wfgt:block_center_hz` and `wfgt:time_group`. `wfgt:zc_metric` is the
normalized ZC correlation peak (0–1), a per-block reception-quality indicator. To get
absolute RF, add the capture's `core:frequency` to the edges.

## Reproducing the published labels

Every attenuation-sweep (`attenuation_dB_{0..85}`) and LTE-holdout (`lte_attenuation_dB_*`)
label file was produced by the plain default invocation:

```bash
python decode.py <capture>.sigmf-data <capture>.sigmf-meta \
                 ../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat
```

That is `profile=fast`, all 5 centers, threshold 0.35, `active_factor` 1.5. The expected
counts are 3,594 annotations for each 0–80 dB capture, 1,281 for 85 dB (the lowest SNR, so
fewer blocks decode), and 192 for each LTE capture.

Verified 2026-09-23: re-running `decode.py` on `attenuation_dB_0` and
`lte_attenuation_dB_0` reproduced the stored `annotations` arrays exactly (3,594 and 192
annotations). A check:

```python
import json
a = json.load(open("original.sigmf-meta"))["annotations"]
b = json.load(open("redecoded.sigmf-meta"))["annotations"]
assert a == b
```

The live OTA captures have no embedded ZC/metadata and so have no ground truth.
