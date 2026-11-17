# LUT classification experiment (branch `lut_classification_experiment`)

Started 2026-11-10. Goal: a systematic, ELINT-style identification of the common US waveforms: measure a standard
descriptor record per detected emission from the IQ (the comms analogue of a pulse descriptor word), match it
against a library whose parameters come from the open standards, and validate on synthetic and real captures.

The v1 look-up (`common_waveforms_look_up.py`, manifest `common_ota_waveforms_manifest_usa.yaml`) stays as it is on
`main`. Everything here is additive.

## Pieces

| file | role | status |
|---|---|---|
| `lut_library_usa.yaml` | descriptor layer: per manifest entry a `descriptor` block (modulation, symbol rate, OFDM structure, timing, spectral marks, hopping) with tolerances and the standard it comes from; new entries not in v1 are defined here in full | done: 76 entries (55 extended + 21 new), 63 with synth recipes |
| `manifest_io.load_library()` | merges manifest + descriptor layer, validates | done |
| `measure_descriptors.py` | the SDW: band extraction from the IQ, envelope/timing at two scales, periodicities (TOA-difference histogram + envelope autocorrelation), OFDM cyclic-prefix correlation, symbol-rate lines (|y|², y², y⁴, instantaneous-frequency transitions), modulation-class features, FM pilot/RDS, carrier line, band-level burst deinterleaving (hop/slot patterns, Wi-Fi SIFS+ACK pairs) | done; unit test `tests/test_measure_synthetic.py` 17/17 at 20 dB, 15/17 at 12 dB |
| `lut_classify.py` | v2 matcher: v1 band/bandwidth/temporal score x descriptor memberships with explicit "unmeasured" handling, evidence counts, ambiguity list; writes `argus:sdw` + `~labels` | done (first version) |
| `synth_modulators.py`, `synth_lut_scenes.py` | synthetic captures per library entry from its own descriptor (GFSK/CPM, FM stereo multiplex, AM, OFDM, 8-VSB, pulses/PPM, chirps, OOK, DSSS, PSK) on AWGN or a real quiet-band background, with ground-truth boxes in SigMF | done |
| `tests/test_lut_synthetic.py` | parameter-recovery and label accuracy on synthetic captures | done, running |
| `tests/evaluate_lut_on_captures.py` | real sweep: label plausibility, parameter distributions per label, synthetic-vs-real deltas, protocol decodes | done |
| `decode_waveforms.py` | attempt-to-decode layer: protocol encoders (for the synthetic scenes) and decoders (ADS-B CRC, RDS blocks, BLE advertising CRC, LTE PSS/SSS cell identity, POCSAG/FLEX BCH, ERT SCM/SCM+, P25 NID BCH, DMR sync, DECT R-CRC, 802.15.4 FCS, UAT sync); `lut_classify.py --attempt-decode` | done; `tests/test_decode_synthetic.py` 46/48 at 10-30 dB |
| `tests/test_comb_pass.py` | unit test of the shared cross-track spur-comb pass (`common_waveforms_look_up.comb_pass`) on synthetic track sets: bench comb under a band catch-all, five teeth in a 5 MHz capture, dense uplink band and crowded FM band as negatives | done, 901/901 over 300 seeds |
| `tests/plot_lut_spectrograms.py` | spectrograms of the test captures with the detector boxes, the labels and the decodes drawn on top | done |
| `notes/lut_experiment_results.md` | the findings | done, updated with the decode layer |

## Data

- Real: `~/Datasets/ARGUS/OOD_captures/lut_sweep_2026-11-10/` = the 46-look default sweep of the collector on the
  N320 (VERT900 ch1 below 1.5 GHz, VERT2450 ch0 above), 1 s per look. Extra looks for entries new in the library are
  added with `--entries`. Detector: Holohub `preprocess_dataset.py` in the container (no PyTorch on the host).
- Real reference: the 30-capture N320 stress set (`~/Datasets/ARGUS/OOD_captures/OTA`) with M7-S/M12-S/M13-S masks.
- Synthetic: generated here from the library.

## Measurement design (SDW fields)

Per track (detector boxes grouped across frames, as in v1):

- `band`: f_lo/f_hi/fc from the boxes; IQ extracted by mixing + cascaded FIR decimation to ~2.5x the box width.
- `envelope` at two scales: fine (0.25 us smoothing, 2 us merge gap: pulses, PPM messages, DME pairs) and coarse
  (5 us smoothing, 100 us merge gap: packets, slots, subframes). Per scale: burst count, duration p10/p50/p90,
  start-to-start interval histogram -> PRI candidates with strengths, duty.
- `periodicity`: autocorrelation of the on/off envelope (10 us grid, lags 100 us .. 250 ms) -> frame/beacon periods.
- `ofdm`: |R(tau)|/R(0) of the complex signal for tau >= 1 us -> Tu (peak lag), then the period of the CP metric
  -> Ts and CP; reports SCS = 1/Tu and the normalised peak.
- `rate_lines`: spectral lines of |y|² (linear modulations), y² and y⁴ (CPM/PSK), |d f_inst/dt| (FSK family):
  top-3 frequencies with line SNR each; aliasing allowed for in the matcher.
- `modulation`: envelope CV / kurtosis / PAPR over bursts, instantaneous-frequency std and number of levels,
  chirp slope, carrier line, band-edge steepness -> class guess with confidence.
- `fm`: after FM demodulation, 19 kHz pilot and 57 kHz RDS line levels, HD Radio sideband level.
- `pilots`: as v1 (ATSC grid).
- band-level: all coarse bursts of the capture sorted by time -> hop/slot statistics, SIFS+ACK pair fraction,
  TDD pattern.

Each field carries a value, an uncertainty or SNR, and `measured: true/false`. Unmeasured is neutral in the
matcher; measured-and-contradicting is a penalty; measured-and-matching counts as verified evidence.

## Status log

- 2026-11-10: branch created from main (6bb2170). Sweep started. Plan written.
- 2026-11-10 12:00: first sweep ran on the N320's default RX2 port (antennas are on TX/RX): sub-1.5 GHz captures were pure noise
  (2 dB spectral span). Kept as `lut_sweep_2026-11-10_rx2_noise/`. Re-swept with `--ant TX/RX` (46 looks, 0 overflows).
- 2026-11-10 13:00: measurement module validated on 17 synthetic cases; classifier end-to-end on synthetic smoke set;
  full synthetic evaluation and the M12-S detector run on the TX/RX sweep started.
- 2026-11-10 13:30: Holohub `preprocess_dataset.py` with the live M12-S config gives 0 regions on every full-band
  downlink / TV capture: `signal_snipper.max_mask_occupancy: 0.30` (flood guard) and the 0.98 invalid-frame guard. Those 12
  captures plus the 13 extra library looks are re-run with a config copy (`max_mask_occupancy: 1.0`,
  `invalid_frame_min_occupancy: 1.01`) into `lut_sweep_2026-11-10_b_preprocessed_m12s/`. The live config is untouched.
- Collector: `--library` option (plan from the merged library) and a flat-spectrum warning in the gain probe.
- Reader: `regions_from_preprocessed` now finds capture folders at any depth (the recursive preprocess mirrors the input tree).
- 2026-11-10 15:30: final real-capture run (`tests/results/lut_real_eval.md`): 2496 tracks over 59 looks + 9 stress captures,
  v2 512 labels (v1 515), 399 with a verified descriptor field, 0 inconsistent with the capture span. Measurement rules
  settled on the way: per-burst envelope statistics; instantaneous frequency on the 1.15x band; 50 % edge refinement before
  merging; consecutive intervals for PRI plus per-duration-class intervals (beacons); pulses = short consistent widths,
  flat tops, gaps at the noise floor; contradictions scaled by SNR and softened on cut boxes; narrow bursty tracks
  (<= 5 MHz) measured inside their own regions; band periodicity always ungated; `requires_fields` (DME needs pulse pairs);
  catch-all entries keep the v1 band label; variants need a rate / numerology / timing field, not the class alone.
- 2026-11-10 17:00: synthetic end-to-end (189 captures, 63 entries x 3 draws, 10-30 dB): 166 exact (87.8 %), 3 parent, 20 signal,
  0 wrong; 168/169 correct labels with verified evidence. Last matcher rules: evidence-based eligibility (strong verified
  weight >= 3 waives an unresolvable v1 raster requirement), thresholds 0.05 lower with verified evidence, ties broken by
  verified support, a 2 ms "frame" envelope scale for OOK packets. Final real re-runs started with these rules.
- 2026-11-10 20:30: user feedback: delete the RX2 noise sweep (done), question the 30 % flood guard, add an optional decode,
  no false positives before low-SNR sensitivity, spectrogram figures in tests. Flood guard traced in Holohub: the snipper's
  `max_mask_occupancy: 0.30` (e6400821) contained torn frames from the converter ring under backlog; the ingest root causes
  were fixed since (14b7ae67 per-packet copy) and the detector's invalid-frame guard (0.98) zeroes garbage masks, so the
  0.30 cap only dropped legitimate dense masks (saved preprocess log: skips at 34-93 % lit in 28 of 46 sweep captures).
  Fixed in Holohub: 0.98 in the fine-tuned live configs, 0.90 for coherent-power / cuda_dino, and both guards off for file
  input (`disable_live_ingest_guards`); validated on four full-band captures (67 / 191 / 57 / 781 regions, equal to the
  no-guard run). The scratch no-guard config is no longer needed.
- 2026-11-10 21:30: `decode_waveforms.py` (12 protocols, encoders + decoders) and `--attempt-decode`; a passing decode
  settles the label at 0.97 (0.86 for sync-only results) and names the variant; weak-signal rule: below 6 dB a label needs
  verified evidence or a decode (catch-alls exempt). Real-signal fixes found with the sweep captures: coarse channel
  filter before burst detection, carrier centring from the PSD centroid (boxes are bin-accurate only), ERT chip rate per
  burst (meters run 1.1 % high), LTE carrier candidates from the box PSD (merged boxes), FLEX FIW checksum advisory (BCH
  is the check), cosine-similarity sync correlation for P25 / DMR. On air: RDS PI 2D2F/114F/2BA9 = KLBJ / KAMX / KKMJ
  (Austin), BLE advertising on channel 38, LTE PCI 476 (1977.5 MHz) and 319 (751.2 MHz), P25 NAC 910 at 464.565 MHz,
  FLEX A=DEA0 (3200/4) on 929.616 and 931.062 MHz.
- 2026-11-10 23:00: two bugs found through the LTE decoder on the synthetic set. (1) `synth_lut_scenes.generate()` rounded
  the capture offset to a detector bin after computing the capture centre, so the written emission sat up to half a bin
  (20-30 kHz) off `truth.f_emission_hz`: harmless for the descriptor tests, fatal for a PSS/SSS decode at the truth centre
  (fixed: round first). (2) `measure_descriptors.CaptureIQ.extract()` mixed each overlap-save chunk relative to the chunk
  start, leaving a phase jump of 2 pi k_off hop / CHUNK at every chunk boundary (every 5-40 ms depending on the rate);
  fixed with a per-chunk phase rotation, verified on a pure tone (residual phase 1 mrad) and the measurement unit test
  (17/17 at 20 dB). All runs (synthetic set regenerated, sweep batches a / b, stress set) redone on the fixed code.
- 2026-11-10 23:30: the regenerated synthetic set gave one wrong label (an OOK remote at 433 MHz as `~Amateur_70cm`, verified
  by the weak fields modulation + deviation only). Rule added: `requires_descriptors` counts only strong verified fields
  (rate, OFDM, chirp, pulse, timing, tones, hop, decode), never modulation / deviation / duty / burst / carrier line.
  On the real sweep this drops ~20 band-plus-weak-field BLE labels on data channels (advertising channels keep theirs
  through the decode). Everything rerun once more.
- 2026-11-11 01:00: user review of the figures: the receiver comb's teeth inside the 850 MHz uplink look carried
  `~cellular_uplink_850` while the same comb in the ISM 900 look read `~spur_comb`. Cause: the comb pass exempted every
  track a library entry explained at >= 0.9, and the uplink entry is a band catch-all that explains every narrow
  persistent tooth at 1.0 (the ISM catch-all only reaches 0.89); the pass also capped teeth at 6 bins where M12-S paints
  6-10, excluded 6 dB teeth as "strong and wide" on a noise-inflated occupied bandwidth, and needed 6 teeth where a 5 MHz
  paging look holds 5. The same comb (1.000 MHz, teeth at x.27 MHz, 839-932 MHz) sat under `~Pager_FLEX_POCSAG_929`
  labels and went unlabelled in the LMR 800 look. Fixed in `comb_pass` + `lut_classify` (exempt only tracks with strong
  verified evidence; width <= max(6 bins, 300 kHz); persistence >= 0.5 instead of the temporal class; min 5 teeth, 80 %
  fill when short; teeth >= 3 x the densest off-grid residue; nominal spacing in the label) with
  `tests/test_comb_pass.py` (the dense-uplink and FM-raster negatives failed the first two formulations). Two more
  from the reruns: the grid tolerance of 1.5 bins lost teeth to box-centre jitter at 10-12.5 MSps (7 of 9 in the UAT
  978 look, 6 of 11 in the DECT look) -> 2.5 bins; a stray line at a half step let the 0.5 MHz grid beat the 1 MHz grid
  11 : 10 in the narrow 850 uplink look -> a finer grid needs 1.5 x the lines of the coarser one. The DECT look (ch0,
  1925 MHz) carries its own 1.016 MHz comb, so the comb is on both receiver channels. All real batches and the
  synthetic set rerun.
- 2026-11-11 12:30: one more comb rule from the stress set: the narrow 850 MHz uplink look holds 78 narrow lines (weak
  detections flickering in half the frames) and its 10-point comb failed the off-grid residue test against them -> the
  background is now taken among lines about as persistent as the teeth (`background_persistence_ratio: 0.8`); offline
  replay of the comb pass on every saved report (88 captures) shows that only this capture changes (25 tracks -> comb,
  6 of them labelled uplink before). Final real numbers: 6292 tracks, 1161 labelled (225 comb teeth), 772 verified,
  69 decoded, 0 inconsistent; synthetic unchanged (166 / 189 exact, 0 wrong, 47 / 57 decoded); `tests/test_comb_pass.py`
  901 / 901. Figures regenerated (41 per-capture PNGs; the real montage now shows the 850 uplink look in place of AWS).
- 2026-11-11 13:10: clean full rerun of the stress set on the final rules finished (no tracebacks): identical to the
  merged report row for row (4921 rows, 0 label differences, 709 labelled / 120 comb / 35 decoded / 443 verified);
  `tests/results/lut_real_eval.md` rebuilt from the three clean runs (same totals).
- 2026-11-16 20:40: X410 3415B36 reflashed to CG_400 (fixed 491.52 MSps, 100 GbE; the dual-X410 EndGen bench had both
  radios on X4_200 at 10 GbE) for the high-rate cases the N320 could not record: `tests/x410_491_sweep.sh` = eleven
  400 MHz-wide looks (250, 600, 900, 1200, 1500, 1950, 2450, 2650, 3650, 5300, 5700 MHz; 1 s each, auto gain, ch0
  TX/RX0, zero overflows) into `~/Datasets/ARGUS/OOD_captures/x410_491_2026-11-16/`. Collector fixes on the way: a
  fixed-gain `--survey` now measures (it only reported the gain-probe history before) and reports the spectral span;
  wide `--freqs` looks are named by the service overlapping the span most (the spur-comb entry, which covers every
  frequency, used to win). `tests/x410_491_lut_run.sh` runs the look-up chain with the detector at its native
  resolution (Holohub `preprocess_dataset.py --full-resolution`, new). Live-demo side recorded in Holohub
  `infocom_evals/signal_detection_experiments/x410_live_m12s_full_rate/`.
- 2026-11-16 22:30: look-up on the eleven 491.52 MSps looks, both detector geometries: native 1024-point 189 tracks /
  16 labels / 0 decodes (whole-band fills on the strong looks: wrong tool for M12-S, which was trained on the
  real-time front end); real-time geometry 1217 / 62 / 59 verified / 8 BLE decodes / 0 inconsistent. The native run
  exposed a precision hole (edge-to-edge fill with two real TV pilots labelled ATSC_TV): whole-band guard in
  `common_waveforms_look_up.py` (edge-to-edge boxes, and pilot-rescued boxes wider than 4x the widest legal box, unless
  the whole capture lies inside the allocation; a first 4x-width-for-everything version cost five narrowband VHF
  labels and was dropped). Synthetic set regenerated and rescored: unchanged 166 / 3 / 20 / 0, 47 of 57 decoded;
  `tests/test_comb_pass.py` 61 / 61. `tests/x410_491_lut_run.sh` has a `GEOMETRY=rt|native` switch. Doubtful label to
  revisit: a 2.4 MHz signal at 1710.24 MHz (AWS uplink band edge, duty only). Radio reflashed back to X4_200 afterwards.
