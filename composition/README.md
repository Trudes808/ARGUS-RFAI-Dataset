# 200 MHz Composition Pipeline (SigMF)

Composes the generated 245.76 MSps waveforms into a 200 MHz baseband SigMF
recording, then recovers ground-truth annotations at the receiver by syncing the
Zadoff-Chu sequences and decoding an embedded OFDM metadata burst.

Run with the `holoscan_compose` conda env:
`/Users/bqn82/miniforge3/envs/holoscan_compose/bin/python`.

## Frequency plan
Three 50 MHz blocks on a **60 MHz pitch** (5 MHz guard each side), centered at
**−60 / 0 / +60 MHz**. A waveform wider than 50 MHz merges contiguous block
slots; merged usable bandwidth = `60·N − 10` MHz → **50 / 110 / 170 MHz** for
N = 1 / 2 / 3 slots (fits 80 and 160 MHz waveforms). Within one block, multiple
sub-50 MHz waveforms are stacked in frequency with 5 MHz guards.

## Per-block time layout (each block, same time window)
`[ ZC ] → gap → [ OFDM metadata burst ] → gap → [ data: stacked waveforms ]`
- **ZC**: the 50 MHz Zadoff-Chu sequence at the block center — timing reference.
- **Metadata**: OFDM burst (QPSK, reference-symbol channel estimate + pilots,
  repetition FEC + CRC-32) encoding each waveform's class, variation, position
  **relative to the ZC**, bandwidth, and power. Decodable after a channel.
- **Data**: the waveforms, peak-normalized then scaled by their `power_db`.
ZC and metadata are held at a fixed power (`ref_rms`) for reliable detection.

## Files
- `geometry.py` — frequency plan, slot/merge math, frequency stacking + fit checks.
- `metadata_modem.py` — standalone OFDM metadata modem (encode/decode bytes).
- `zc_sync.py` — ZC load + frequency-shifted matched-filter synchronization.
- `protocol.py` — shared TX/RX layout constants + metadata (de)serialization.
- `compose.py` — read config → build composite → write SigMF (TX truth + bare RX).
- `decode.py` — sync ZCs, decode metadata, rebuild annotations into the RX meta.
- `channel.py` — simple wireless channel (multipath + gain/phase + AWGN).
- `run_example.py` — composes the example configs and validates clean + channel.
- `example_*.yaml` — example configs; `example_bad.yaml` shows the fit error.

## Usage
```bash
python compose.py example_mixed.yaml          # -> composites/example_mixed.{sigmf-data,sigmf-meta} + .rx.*
python decode.py composites/<name>.rx.sigmf-data composites/<name>.rx.sigmf-meta \
                 ../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat
python run_example.py                          # full clean + channel validation
```

## Time multiplexing
The composition can be a sequence of **time groups** (each a full frequency-
multiplexed set of blocks). Within a group the time layout is
`ZC · metadata · [guard1] · waveform(s) · [guard2]`, where `guard1_ms` (default
2.5 ms) is the metadata→waveform guard and `guard2_ms` (default 5 ms) is the
guard before the next group's ZC. The receiver finds every group via energy-
gated multi-peak ZC detection. Use `time_groups:`; the single-group `blocks:`
form (below) is shorthand for one group.

## Length scaling & power
- Per-waveform `duration_ms` (or `duration_samples`) **tiles** (repeats) a short
  waveform or **cuts** a long one to that length. The annotation's
  `core:sample_count` / `wfgt:length_samples` is the new length; the original
  length is preserved in `wfgt:original_length_samples`.
- `power_mode: original` places each waveform at its as-stored amplitude (no
  peak-normalization, no `power_db`). Default `power_mode: peak_db` peak-
  normalizes then applies each waveform's `power_db`.

## Config schema
```yaml
name: my_composite
output_dir: composites
zc_file: ../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat
library_root: ../generated_waveforms_24576
guard1_ms: 2.5           # guard: metadata -> waveform
guard2_ms: 5             # guard: waveform -> next group's ZC
power_mode: peak_db      # or "original" (use as-stored amplitudes)
ref_rms: 0.15            # fixed ZC/metadata RMS
peak_normalize: 0.95     # final composite peak (preserves relative powers)
time_groups:
  - blocks:              # group 1 (at t=0)
      - slots: [0]       # frequency slots this block occupies (0,1,2)
        waveforms:       # multiple -> stacked in frequency with 5 MHz guards
          - {mat: "OFDM/....mat", power_db: -6, duration_ms: 10}  # tiled/cut to 10 ms
      - slots: [1, 2]    # contiguous slots -> one wide waveform (e.g. 80 MHz)
        waveforms:
          - {mat: "802_11ax/...CBW80....mat", power_db: -3}
  - blocks:              # group 2 (after a 25 ms guard)
      - slots: [0, 1, 2]
        waveforms:
          - {mat: "802_11ax/...CBW160....mat", power_db: 0}

# Single-group shorthand (one time group):
# blocks:
#   - {slots: [0], waveforms: [{mat: "...", power_db: 0}]}
```
A block whose waveforms don't fit (sum BW + 5 MHz guards > usable) raises an
error suggesting you split them into different blocks.

`build_comprehensive_config.py` auto-packs the **entire library** into a config
(varied power, one waveform each); `run_comprehensive.py` composes + decodes it
clean and through the channel. `build_comprehensive_ordered.py` + `run_ordered.py`
build the **ordered** comprehensive set (original powers; sectioned by class then
largest→smallest bandwidth, classes time-separated; repeated at 20/10/5/1/0.2/
0.04 ms via tile/cut) and clean-decode it.

## Performance
Decoding is **energy-gated**: it finds active-region rising edges (the start of
each time group) and runs the precise matched filter only in small windows
around them — a ~2 s / 4 GB capture decodes in ~10 s. Annotation positions are
ZC-relative, so a channel bulk delay self-cancels.

## Output
- `<name>.sigmf-meta` — TX ground truth (full annotations).
- `<name>.rx.sigmf-meta` — "received" capture: receiver params only, no annotations.
- `<name>.rx.decoded.sigmf-meta` — RX meta after the decoder rebuilds annotations.
- `.rxch.*` — the same through the wireless channel.
Annotations use `core:sample_start/sample_count/freq_lower_edge/freq_upper_edge/label`
plus `wfgt:` fields (class, variation, occupied_bw_hz, power_db, block_center_hz).
