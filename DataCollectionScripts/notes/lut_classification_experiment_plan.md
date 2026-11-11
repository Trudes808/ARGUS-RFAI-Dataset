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
| `tests/evaluate_lut_on_captures.py` | real sweep: label plausibility, parameter distributions per label, synthetic-vs-real deltas | done, pending the detector run |
| `notes/lut_experiment_results.md` | the findings | todo |

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
