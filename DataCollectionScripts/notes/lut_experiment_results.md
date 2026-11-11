# LUT classification experiment: results

Branch `lut_classification_experiment`, 2026-11-10. Filled in as the runs complete (see the plan document for the
design). Numbers come from `tests/results/lut_synthetic_eval.md`, `tests/results/lut_real_eval.md` and the
measurement unit test.

## 1. Measurement unit test (`tests/test_measure_synthetic.py`)

Seventeen synthetic emissions with known parameters (BLE advertising + scan requests, Wi-Fi OFDM with SIFS/ACK and
102.4 ms beacons, LTE downlink, NR TDD DDDSU, FM stereo, FM + HD sidebands, NBFM with CTCSS, AM voice, VOR,
P25 C4FM, POCSAG, DECT slots, ADS-B PPM, DME pairs, LoRa SF7, 8-VSB, Zigbee with ACK turnaround), each wrapped in
detector-like boxes and measured from a ci16 capture:

| in-band SNR | cases passing | what fails |
|---|---|---|
| 30 dB | 17 / 17 | - |
| 20 dB | 17 / 17 | - |
| 12 dB | 15 / 17 | P25 C4FM in a 12.5 kHz channel (3 samples per symbol in the tight band: no level histogram, no transition line); ADS-B pulses at 8 dB above the wide-band floor (too few pulses pass the fine threshold) |

Recovered at 20 dB: OFDM Tu / CP within 1 % / 1 %, symbol rates within 0.1 % (DECT 1.152 M, Zigbee 2 M chips, BLE 1 M,
POCSAG 1200, C4FM 4800 via its 2nd harmonic), chirp slope within 5 %, the 19 kHz pilot, VOR 30 / 9960 Hz, pulse
widths (0.5 us ADS-B, 3.5 us DME) and their 1 us / 12 us spacings, DECT 10 ms frames, NR 2.5 ms TDD period, Wi-Fi
10 us SIFS and 102.4 ms beacon interval, BLE 150 us T_IFS, Zigbee 192 us turnaround.

## 2. Synthetic end-to-end (`tests/test_lut_synthetic.py`)

189 captures: 63 library entries x 3 draws, each one emission generated from the entry's own descriptor and
recipe at a legal channel, a sweep sample rate (capped at 61.44 MSps) and a random in-band SNR of 10-30 dB on AWGN,
1 s long, classified from the ground-truth boxes (detector bypassed) with the full library in play. Per-entry
table: `tests/results/lut_synthetic_eval.md`.

| outcome | captures | share |
|---|---|---|
| exact label (variant suffixes allowed) | 166 | 87.8 % |
| parent label (AIS -> Marine VHF at 61 MSps, ACARS / FRS -> their catch-all) | 3 | 1.6 % |
| left as `signal` | 20 | 10.6 % |
| wrong label | 0 | 0 % |

168 of the 169 correct labels carry at least one verified descriptor field (modulation class 133, duty 94, burst
length 80, OFDM numerology 49, deviation 45, symbol rate 35, intervals 33, carrier line 26, periods 20, gaps 13,
pulse widths 8, AM tones 6, pulse pairs 3, FM pilot 3, chirp 3, HD sidebands 2). Exact-label share is 32 / 36 below
15 dB SNR and 134 / 153 above.

Left as `signal` by design or by a known limit: microwave ovens, TDWR and most radar draws (tier C, never labelled;
one S-band draw is labelled because the entry is tier B); MURS, which is physically identical to the VHF
narrowband catch-all and only distinguishable by its five channel centres, unresolvable at the sweep rates;
FRS / GMRS likewise (one draw got the UHF catch-all); 2 of 3 BLE draws that tie with Bluetooth BR/EDR when the y^2
line pair is too weak to fix the modulation index; one ACARS, one LAA, one 315 MHz remote, one L-band radar draw.
Compared with the v1 synthetic test (71.6 % exact on the same kind of positives), the descriptor evidence lifts the
exact-label share to 87.8 % while keeping zero wrong labels.

## 3. Real captures: TX/RX sweep of 2026-11-10 (M12-S detector) and the N320 stress set

Data: the 46-look default sweep plus 13 looks for library-only entries (UAT 978, 137 MHz satellites, amateur 2 m /
70 cm, ATSC 3.0, TDWR, WCS, Inmarsat, DECT, Iridium, WMTS, HF broadcasting), taken on the N320's TX/RX ports with
a VERT900 (< 1.5 GHz) and a VERT2450, 1 s per look, 0 overflows; the M12-S detector through Holohub's
`preprocess_dataset.py` (the 12 full-band downlink / TV captures the live config's flood guard emptied were re-run
with the guard off); plus nine stress-set captures with M12-S mask arrays. v1 = `common_waveforms_look_up.py` on
the same tracks. Full tables and the per-label parameter distributions: `tests/results/lut_real_eval.md`.

| set | tracks | v1 labels | v2 labels | v2 labels with >= 1 verified descriptor field | labels inconsistent with the capture span |
|---|---|---|---|---|---|
| sweep batch 1 (46 looks, live config) | 1084 | 312 | 379 | 334 | 0 |
| sweep batch 2 (12 looks re-run without the flood guard + 13 looks for library-only entries) | 287 | 30 | 37 | 30 | 0 |
| stress set (9 captures, M12-S mask arrays) | 1125 | 173 | 186 | 125 | 0 |
| total | 2496 | 515 | 602 | 489 | 0 |

Labels: Bluetooth BR/EDR 172, 850 MHz uplink 129, Bluetooth LE 56, 900 MHz ISM devices 44 (+ 10 `/RFID_reader`,
8 `/ERT_meter`), receiver spur combs 40, FM 36 (22 `/analog` with the pilot verified), 700 MHz uplink 20, ATSC 12,
700 MHz downlink 10, Wi-Fi 5 GHz 10 (9 `/OFDM`), AWS downlink 9, n41 TDD 9 (8 `/NR_30kHz`), C-band TDD 6, 850 MHz
downlink 5, pagers 5, Zigbee 4, amateur 2 m 3, ADS-B 2, DC spurs 2, 1900 MHz downlink 2, and one each of 600 MHz
uplink, VHF narrowband, UHF narrowband `/analog_FM`, Wi-Fi 2.4 GHz `/OFDM`.

What the descriptors verified on air:

- **Cellular downlinks**: every labelled 600 / 700 / 850 / 1900 / AWS block shows the OFDM cyclic-prefix peak at
  Tu = 66.6-66.7 us (15 kHz subcarriers) with CP 4.6-5.2 us. The n41 blocks show Tu = 33.3 us (30 kHz) and the
  5 ms TDD period, including seven fragments of the 2496-2550 MHz carrier inside the 200 MSps 2.4 GHz look that v1
  could not label; the C-band carriers verify the 2.5 ms TDD period.
- **Wi-Fi**: 312.5 kHz subcarriers, CP 0.80 us (synthetic 0.79), the SIFS before ACKs and the 102.4 ms beacon
  interval on 5 GHz channels (`/OFDM`, 7 more channels than v1).
- **Bluetooth**: 172 BR/EDR labels (v1 105): the 1 MHz rate line in |y|^2 or in the frequency transitions on 52
  of them, the band's hop statistics (about 64 000 channel changes per second over the tracks of one capture) on
  almost all, bursts 0.1-0.4 ms on the 1 MHz raster; 56 BLE labels with the 1 Msym/s transition line, the 2 MHz
  raster, T_IFS gaps and advertising-channel triplets. The y^2 line spacing (2 h Rs: 1.0 MHz for h = 0.5, 0.64 MHz
  for BR) separates the two where the lines are strong enough.
- **FM**: 36 stations, 22 with the 19 kHz stereo pilot in the demodulated baseband (up to 33 dB), 24 with the
  deviation in range (median 49 kHz); no HD Radio sidebands were found on air here.
- **ATSC**: 12 channels, 11 by the pilot carrier (34-43 dB above the in-band level). The 8-VSB symbol-rate line is
  not observable in |y|^2 on air (roll-off 0.0575 leaves almost no clock line).
- **900 MHz ISM**: 62 labels; 8 refined to `/ERT_meter` by the 32.768 kchip/s Manchester line, 10 to
  `/RFID_reader` by the carrier line plus 10-400 ms bursts, 1 to `/SUN-FSK` by a 150 kbps transition line; one
  `~LoRa_US915` from a measured chirp slope (BW 125 kHz, SF9).
- **Pulsed aviation**: the two ADS-B tracks and the receiver's DC spurs as in v1; the v1 `~DME_TACAN` label on a
  UAT 978 MHz track is gone (`requires_fields: [pulse_pair]`).
- **Uplinks**: v1's cellular uplink labels are kept (plus 27 more at 850 MHz); at 4-7 dB SNR the fragmented bursts
  verify nothing and contradict nothing (contradictions are scaled by SNR).

What v2 drops relative to v1: 16 BLE and 6 Zigbee tracks in the 2.4 GHz band whose 2.5-3.9 MHz boxes either measure
as OFDM (Wi-Fi fragments sharing the box) or carry a 1 MHz line where Zigbee needs 2 MHz (BLE 2M packets in a
Zigbee-width box), two `~LMR_800MHz_trunked` labels inside the 850 MHz uplink capture, two Wi-Fi 2.4 GHz labels and
one Iridium label with no verified field. Net: +87 labels, all consistent with their bands, 81 % of them with
measured physical-layer evidence.

## 4. Synthetic vs real: what differs and what to improve

The per-label parameter tables in `tests/results/lut_real_eval.md` put the measured real distributions next to the
synthetic ones. Where the physics is fixed by the standard the two agree closely; where the numbers come from
traffic, propagation or the detector they do not, and those are the levers.

**Agrees (standard-fixed parameters)**

| parameter | real (median) | synthetic (median) |
|---|---|---|
| Wi-Fi 5 GHz subcarrier spacing / CP | 312.5 kHz / 0.80 us | 311.9 kHz / 0.79 us |
| LTE 700 MHz downlink subcarrier spacing / CP | 15.0 kHz / 4.70 us | 15.0 kHz / 4.59 us |
| NR n41 subcarrier spacing / TDD period | 30 kHz / 5 ms | 30 kHz / 2.5-5 ms |
| Bluetooth / BLE rate line | 1.000 MHz | 1.000 MHz |
| BLE y^2 line spacing (2 h Rs) | 1.17 MHz | 1.00 MHz |
| Wi-Fi beacon interval | 102.4 ms | 102.4 ms |
| ERT meter chip line | 33.15 kHz | 32.77 kHz (library) |

**Differs**

- *Signal strength.* Real FM stations here sit at 6-15 dB SNR in their box (synthetic 23-27 dB): the pilot is still
  found on 22 of 36, but weak stations lose the deviation and HD tests. Real uplink bursts at 4-7 dB verify nothing.
  The synthetic SNR range (10-30 dB) is too optimistic for the lower half of what the detector flags; the next
  synthetic set should draw SNR from the real per-label distribution (median 8-16 dB).
- *Traffic.* Real Bluetooth bursts are short (median 0.14 ms: polls, NULLs, 1-slot packets) while the synthetic
  BT recipe sends 1/3/5-slot packets (median 1.6 ms); real BT deviation measures 250 kHz (many of those tracks are
  LE-like) against the synthetic BR 153 kHz. Real LTE downlinks carry a scheduling envelope (0.5 ms slot structure,
  0.2 ms fine intervals) that the uniformly loaded synthetic OFDM lacks, so `period 1 ms` verifies on air but the
  synthetic downlink verifies only the numerology. Real Wi-Fi ACKs are often missed by the detector, so the SIFS gap
  verifies through the band-level ACK-pair statistic rather than the track's own gaps.
- *Boxes.* The detector paints the -20 to -30 dB width, merges neighbours and fragments wide carriers: a real BLE
  box is 1.3-3.9 MHz where the truth box is 1.5 MHz, a 100 MHz n41 carrier becomes 60 fragments. Synthetic mode A
  (truth boxes) therefore overstates what the chain achieves; the generator now mimics painted widths for GFSK, but
  the honest test is mode B, the detector run over the synthetic captures (`preprocess_dataset.py` on the synthetic
  folder, then `lut_classify.py`), which this branch has not yet run.
- *Background.* AWGN has no spur comb, no DC line and no neighbours; the real 900 MHz band is a comb of 1.002 MHz
  lines plus meters, so narrow real tracks collide with lines the synthetic set never sees. `synth_lut_scenes.py
  --background` can use a real quiet-band recording; the quiet 1338-1362 MHz capture is the natural choice.
- *Shared bands.* On air the same box often holds two emitters over a second (BLE and Wi-Fi at 2.4 GHz): measuring a
  narrow bursty track inside its own regions fixed most of it, but a 2.5 MHz box over a Wi-Fi channel still measures
  OFDM. Synthetic scenes with one emitter per capture never exercise this; a two-emitter scene is the obvious addition.

**Improvements, in order of expected return**

1. Mode B synthetic evaluation (detector in the loop) with real-noise backgrounds and realistic SNR draws.
2. Traffic models in the recipes: short BT packets with polls, LTE scheduling envelopes, Wi-Fi ACKs that the detector
   may miss, BLE 2M in the BLE recipe.
3. Measurement: a BT/BLE modulation-index estimate that works below 10 dB line SNR (the y^2 spacing now needs two
   lines >= 10 dB); the ATSC data-segment line as pilot-independent evidence; beacon detection by duration class is
   fragile in heavy traffic.
4. Truth by decoding for the real captures (ADS-B via dump1090, FM station codes from RDS, BLE advertisements) so
   precision is measured against the air, not only against band plausibility.
