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
1 s long, classified from the ground-truth boxes (detector bypassed) with the full library in play and the decoders
on (`--attempt-decode`). Per-entry table: `tests/results/lut_synthetic_eval.md`. Final run (2026-11-11, after the
decode layer, the strong-field rule and the generator fix):

| outcome | captures |
|---|---|
| exact label (variant suffixes allowed) | 166 (87.8 %) |
| parent / child label | 3 (FRS/GMRS as the UHF catch-all, AIS as Marine VHF) |
| left as `signal` | 20 |
| wrong label | 0 |

All 166 exact labels carry verified descriptor evidence. Decoders: 47 of the 57 captures of entries with decoders
decoded and labelled exactly (all 15 LTE downlinks by physical cell identity; the misses are the analogue draws of the
amateur and LMR recipes and the NR TDD carrier, which has no LTE PSS). What stays `signal`: microwave ovens and the
TDWR / S-band radars (their entries need descriptors the recipes do not reproduce), MURS vs the VHF catch-all and
FRS/GMRS vs the UHF catch-all (identical physical layer, tie), LAA at 5 GHz vs the drone OFDM entry, HF shortwave only
when the AM carrier is weak, Bluetooth classic (the synthetic traffic is too sparse for a rate or slot line), and the
amateur 70 cm NBFM draws (a shared band with ISM remotes now needs a strong field). The one wrong label of the
intermediate run (an OOK remote at 433 MHz taken for amateur 70 cm on modulation class + deviation alone) is what the
strong-field rule removed.

## 3. Real captures: TX/RX sweep of 2026-11-10 (M12-S detector) and the N320 stress set

**Final run (2026-11-11, decoders on, strong-field and weak-signal rules):** 59 sweep looks (batches a + b) and the
whole 30-capture N320 stress set (`--masks M12S`), 6292 tracks: 1161 labelled (225 of them the receiver's comb
teeth, 213 at 1 MHz and 12 at 1.016 MHz), 772 with at least one verified descriptor field (a comb label is a
cross-track finding and carries none), 69 with a passing protocol decode, 0 inconsistent with the capture span. Per
capture: `tests/results/lut_real_eval.md` (all three batches from clean full runs on the final rules; a merged
stress report built while the last rerun was in flight gave the same 4921 rows and labels). The earlier figures below (2496 tracks over 59 looks + 9 stress captures, 602 labels)
are the pre-decode run kept for the v1 comparison.

**Receiver comb under band catch-alls (2026-11-11, from the user's review of the figures).** The N320 bench comb
(1.000 MHz spacing, teeth at x.27 MHz, visible from 839 to 932 MHz at 5-6 dB in the mask) was labelled `~spur_comb`
only in the ISM 900 look; in the 850 MHz uplink look its teeth carried `~cellular_uplink_850`, in the paging look
`~Pager_FLEX_POCSAG_929`, and in the LMR 800 look nothing. The comb pass exempted every track an entry explained at
>= 0.9 and the uplink / paging catch-alls explain every narrow persistent tooth in their band at 1.0; it also capped
teeth at 6 bins (M12-S paints a line 6-10 bins wide), rejected 6 dB teeth as "strong and wide" on a noise-inflated
occupied bandwidth and needed six teeth (a 5 MHz look holds five). Now only tracks with strong verified evidence are
exempt, teeth may be 300 kHz wide and must recur in half the frames, five teeth suffice when nearly gap-free, and the
comb must hold three times as many teeth as the densest off-grid residue of its spacing: the last rule is what keeps a
dense uplink band (144 narrow bursty tracks in the 700 MHz look) and a crowded FM band (stations on a 200 kHz raster
fall into five residue classes of a 1 MHz grid) from forming a comb by chance (`tests/test_comb_pass.py`, 901 / 901
over 300 seeds; the chance-count and the first residue formulations both failed those negatives). The grid tolerance
is 2.5 bins (1.5 lost teeth to the detector's box-centre jitter at 10-12.5 MSps), a finer grid replaces a coarser one
only with 1.5 x the lines (one stray line at a half step otherwise turned the narrow 850 uplink look's comb into a
0.5 MHz one), and the label carries the nominal spacing (`~spur_comb_1MHz`) instead of the per-capture fit
(1.001 / 1.002) unless the fit is more than 0.5 % off. The background of the residue test is taken among the lines
about as persistent as the comb's teeth (>= 0.8 x their median persistence): the narrow 850 MHz uplink look of the
stress set holds 78 narrow lines, most of them weak detections flickering in half the frames, and its comb (teeth in
every frame, 10 grid points, 25 tracks, 6 of them labelled uplink before) only stands out against lines of its own
persistence; the rule changes no other capture. The reruns also found the comb at 1.001 MHz in the UAT 978 look
(10 teeth) and a 1.016 MHz comb on the other receiver channel in the DECT look at 1925 MHz (12 teeth): the artefact
is on both channels, so a terminated-input reference recording per channel and gain would be the clean way to map it.

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

### 3c. 491.52 MSps X410 looks (2026-11-16): the cases the N320 could not record

Eleven 400 MHz-wide looks (`tests/x410_491_sweep.sh`; X410 3415B36 reflashed to CG_400 for the day, one antenna on
ch0 TX/RX0, auto gain, 1 s each, zero overflows; the 2450 MHz look stops at 0.77 s) at 250, 600, 900, 1200, 1500, 1950,
2450, 2650, 3650, 5300 and 5700 MHz. The detector was run twice on the same captures (`tests/x410_491_lut_run.sh`):

| detector geometry | tracks | labelled | verified field | decoded | inconsistent |
|---|---|---|---|---|---|
| M12-S native 1024-point (`preprocess_dataset.py --full-resolution`: 2.08 us rows, 40 tiles per frame) | 189 | 16 | 16 | 0 | 0 |
| M12-S real-time downsample front end (the trained geometry: 41.7 us rows, 2 tiles per frame) | 1217 | 62 | 59 | 8 | 0 |

**Native geometry is the wrong tool for the look-up.** M12-S was trained on the app's real-time front end (one wide
FFT per 41.7 us row, resized to 1024 bins), so at 2.08 us rows every burst is 20x longer than anything it saw. On
the strong VHF/UHF/700-1100 MHz looks it paints the whole 490 MHz band in every frame (one edge-to-edge track per
capture), on the others it yields a few large blobs; only the sparse 5 GHz Wi-Fi looks come out as per-packet boxes
(15 Wi-Fi labels). The real-time geometry gives 6x the tracks and the decodes: BLE advertising frames in the 2.4 GHz
and 2.5 GHz looks (8), Bluetooth BR/EDR and LE hops (24), Zigbee (2), Wi-Fi 5 GHz OFDM (15), 2.4 GHz Wi-Fi (2), LTE
band 13/14 uplink bursts at 777-794 MHz (7; 0.4-1.1 ms bursts, 1 ms interval), band 5 uplink (1), TDD-2500 downlink
bursts (5), PCS downlink carriers (1), two merged ATSC channel blocks with a pilot on the grid. Everything stays
inside its allocation (0 inconsistent). The one label I would not defend is the 2.4 MHz signal at 1710.24 MHz
(AWS uplink, band edge, only duty verified, seen in two looks); it may be an image or spur rather than a UE.

**A precision hole the native run exposed, now closed.** The 4-496 MHz whole-band fill (490 MHz wide, duty 1,
persistence 1) was labelled `~ATSC_TV` at 0.90 because two real TV pilots sit inside it and the pilot rule waived
the width check ("bandwidth is verified in step 6"). `common_waveforms_look_up.py` now refuses a box that runs edge to
edge of the capture, and a pilot-rescued box wider than four times the service's widest legal box, unless the
whole capture lies inside the allocation (a narrow look filled by one real emission). A first version that also
refused any box wider than 4x the service width cost five narrowband VHF labels (ACARS, AIS, marine: quantised boxes
at coarse bins) and was dropped. Synthetic set unchanged (166 / 3 / 20 / 0, 47 of 57 decoded); the N320 and sweep
results of section 3 are unaffected (no edge-to-edge tracks there); `tests/test_comb_pass.py` 61 / 61.

At 480 kHz bins the receiver's 1 MHz spur comb is below the comb pass's grid tolerance (2.5 bins = 1.2 MHz), so the
comb cannot be resolved in these looks; none of the cellular labels sit on persistent narrow lines (all bursty), so
the comb lesson of section 3b did not bite here. Figures: `figures/lut_spectrograms_x410_491_rt.jpg` (real-time
geometry, the one to look at) and `figures/lut_spectrograms_x410_491.jpg` (native geometry, the fills).

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

## 5. Attempt-to-decode layer (`decode_waveforms.py`, `lut_classify.py --attempt-decode`)

Added 2026-11-10 evening on the user's request ("make that an attempt-to-decode flag", "most important is that it does
not false positive"). Twelve protocol decoders, each ending in a check that random data cannot pass, plus the matching
frame encoders so the synthetic scenes carry valid frames. Round trip on modulated frames with a random carrier offset
(`tests/results/decode_synthetic_eval.md`): 46 / 48 cases at 10-30 dB; the two misses are RDS and P25 at 10 dB (below
their practical thresholds).

Real captures (sweep of 2026-11-10 batches a + b and the N320 stress set) decoded on air, final run: 69 tracks with
a passing decode (26 in batch a, 8 in batch b, 35 in the stress set):

| protocol | where | what came out |
|---|---|---|
| RDS | 90.5, 92.6, 93.7, 94.7, 95.5 MHz (two FM looks + the stress FM capture) | PI 2D2F = KLBJ (PS "KLBJFM"), 114F = KAMX, 2BA9 = KKMJ, 48A3, 9995 (PS "Austin's") - the Austin stations on those frequencies; 4-10 groups per second |
| BLE advertising | 2402 / 2426 / 2480 MHz in 11 captures | 120-1030 CRC-valid PDUs per second and channel, ~70-140 advertisers per channel, device names |
| LTE | 738.95-739.0, 751.1-751.2, 888.95, 1957.5, 1977.5, 2114.9-2119.1 MHz | PCI 208, 319, 340, 488, 476, 245 (all FDD); the same cells return the same identity in different looks (208 and 319 in two sweep captures, 245 in the sweep and the stress set) |
| P25 | 464.565 MHz (UHF) and 853.54 MHz (800 MHz trunked) | NAC 910 (PDU / TDU) and NAC 137 (LDU1 / LDU2), NIDs passing BCH(63,16) |
| DMR | 463.931 MHz | BS data sync x6, bursts every 30.0 ms (a base station) |
| FLEX | 929.616 MHz (and 931.062 in the smoke runs) | sync A = DEA0 (3200 sym/s, 4-level, as the SDW's 3200 Hz transition line says), FIW passes BCH: cycle / frame counters |

Not decoded here: ADS-B (the 1090 MHz replies at this indoor bench are at or below the noise floor of the 6 MHz box;
preambles are found, no frame passes the CRC even with single-bit correction), ERT meters (46 tracks carry the
32.768 kchip/s variant through the chip line, the envelope line sits 1.1 % high and is now measured per burst, but no
SCM / SCM+ checksum has passed on air yet: the next step is to check the Manchester polarity and the IDM packet
format against these bursts), UAT and DECT (no transmitter active in the 1 s looks: the 978 MHz tracks are narrow
spurs and DME-like pulses, the DECT band shows only weak slots), Zigbee (no 802.15.4 frames in the 2.4 GHz looks),
NR (the n41 / C-band carriers have no LTE PSS; an NR SSB decoder is the natural extension).

What the real signals taught the decoders (each fixed and re-verified on the unit test):

- Filter to the channel **before** burst detection: the extracted band is 2-4x the signal, and near the threshold the
  bursts fragment; then centre the carrier from the PSD centroid (a detector box is only bin-accurate: 24 kHz at
  25 MSps, which a 6 kHz C4FM low-pass cannot absorb), then the tight filter and the discriminator.
- Burst detection must fall back to the whole record for fluctuating continuous emissions (a paging carrier varies
  6-7 dB over 10 ms) and must smooth longer than a chip for OOK (Manchester chips otherwise fragment a 3 ms packet).
- Meters run off-nominal: measure the chip rate on the envelope per burst and resample.
- LTE boxes merge carriers (a 38 MHz box over three PCS carriers, a 25 MHz capture that is one box): carrier candidates
  come from the occupied blocks of the box PSD on the 100 kHz raster, and the SSS is decoded coherently with the PSS
  channel estimate using the even half's phase as reference (a residual offset rotates PSS and SSS differently).
- FLEX's Frame Information Word passes BCH(31,21) on air with the POCSAG bit convention; the nibble checksum passed on
  one of two stations, so BCH is the check and the checksum is reported.
- Sync correlation for P25 / DMR must be scale invariant (cosine similarity): when bursts merge with noise gaps the
  level normalisation drifts and a scale-dependent metric ranks a wrong 48-bit pattern above the right one.

## 6. The Holohub flood guard (why full-band captures came back empty)

The user asked whether the 30 % flood guard was ideal and what causes the flood. Traced in Holohub-Signal-Detection:
`signal_snipper.max_mask_occupancy: 0.30` was added by e6400821 (2026-09-08) to contain a feedback loop on the live
2.4 GHz band: dense traffic -> 700 snips per pack -> synchronous staging stalled the graph -> the CHDR converter's ring
wrapped and refilled IQ slots the FFT had not consumed -> torn frames painted as full-band "signals" (97 % lit) -> the
dynamic floor chased them. The ingest root causes were fixed afterwards (8-slot ring, 14b7ae67 per-packet valid-length
copy in the gather) and the fine-tuned detector got its own invalid-frame guard (now 0.98) that zeroes garbage masks
before the snipper sees them. The snipper cap stayed at 0.30 and, with M7-S/M12-S painting whole carriers, dropped
legitimate dense masks: the saved preprocess log of this sweep shows `FLOOD GUARD skipped mask frame` at 34-93 % lit in
28 of 46 captures, which is why every full-band downlink and TV capture came back with zero regions. Even the suggested
80 % would still drop real 850 / 700 MHz downlink frames (70-94 % occupancy on the stress captures).

Fix (Holohub, uncommitted): 0.98 in the fine-tuned live configs (the same number as the detector guard; only near-total
garbage trips it), 0.90 in the coherent-power and cuda_dino configs (no detector guard there; their fault episodes are
~97 %), and `run_cuda_dino_offline_file.disable_live_ingest_guards()` switches both guards off for file input, where
the faults cannot occur (`preprocess_dataset.py` inherits it). Validation with the real M12-S config on four captures:
LTE 850 downlink 67 regions and n41 TDD 57 regions (identical to the no-guard scratch run), ATSC lower 191 regions
(134 before: the dropped frames are back), FM 781 (unchanged, FM never floods). The scratch no-guard config is no longer
needed.
