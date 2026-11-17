#!/usr/bin/env bash
# LUT stress run on the 491.52 MSps X410 captures (tests/x410_491_sweep.sh): detector at full native resolution
# (Holohub preprocess_dataset.py --full-resolution with the M12-S live config), then lut_classify.py with the decoders,
# the per-capture evaluation and the spectrogram figures.
#   CAPS=~/Datasets/ARGUS/OOD_captures/x410_491_2026-11-16 REPORT=/tmp/lut_x410_491 ./tests/x410_491_lut_run.sh [preprocess|classify|evaluate|plot|all]
# GEOMETRY=native (default: preprocess_dataset.py --full-resolution) or GEOMETRY=rt (the live real-time downsample
# front end, the trained geometry) for the A/B; outputs get a _rt suffix.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
HOLO=${HOLO:-$HOME/Documents/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection}
CAPS=${CAPS:-$HOME/Datasets/ARGUS/OOD_captures/x410_491_2026-11-16}
GEOMETRY=${GEOMETRY:-native}
if [[ "${GEOMETRY}" == "rt" ]]; then GEOM_FLAG=""; SUF="_rt"; else GEOM_FLAG="--full-resolution"; SUF=""; fi
PRE=${PRE:-${CAPS}_preprocessed_m12s_${GEOMETRY}}
REPORT=${REPORT:-/tmp/lut_x410_491${SUF}}
STEP=${1:-all}
cd "${HERE}"
if [[ "${STEP}" == "preprocess" || "${STEP}" == "all" ]]; then
  echo "== preprocess (M12-S, ${GEOMETRY} geometry) -> ${PRE}"
  sudo -n python3 "${HOLO}/preprocess_dataset.py" "${CAPS}" --recursive --detector cuda_dino_finetuned \
    --config "${HOLO}/config_live_v3_dino_ft_m12s.yaml" ${GEOM_FLAG} --out "${PRE}" || exit 1
  sudo -n chown -R "$(id -u):$(id -g)" "${PRE}" 2>/dev/null || true
fi
if [[ "${STEP}" == "classify" || "${STEP}" == "all" ]]; then
  echo "== classify (decoders on) -> ${REPORT}"
  mkdir -p "${REPORT}"
  python3 lut_classify.py "${PRE}" --no-write --attempt-decode --report-dir "${REPORT}" --quiet 2>&1 | tee "${REPORT}/classify.log" | tail -5
fi
if [[ "${STEP}" == "evaluate" || "${STEP}" == "all" ]]; then
  echo "== evaluate -> tests/results/lut_real_eval_x410_491${SUF}.md"
  python3 tests/evaluate_lut_on_captures.py --report "${REPORT}" --out "tests/results/lut_real_eval_x410_491${SUF}.md" \
    --title "LUT classification on 491.52 MSps X410 captures (400 MHz looks, M12-S at $([[ ${GEOMETRY} == rt ]] && echo 'the real-time downsample geometry' || echo 'native resolution'); decoders on)" 2>&1 | tail -3
fi
if [[ "${STEP}" == "plot" || "${STEP}" == "all" ]]; then
  echo "== spectrograms"
  python3 tests/plot_lut_spectrograms.py "${PRE}" --report "${REPORT}" --out "tests/results/spectrograms/x410_491${SUF}" --only-labelled 2>&1 | tail -2
  python3 tests/plot_lut_spectrograms.py "${PRE}" --report "${REPORT}" --out "/tmp/lut_x410_491${SUF}_montage" --montage "lut_spectrograms_x410_491${SUF}.jpg" --ncols 2 --no-single 2>&1 | tail -2
  cp "/tmp/lut_x410_491${SUF}_montage/lut_spectrograms_x410_491${SUF}.jpg" figures/ && ls -la "figures/lut_spectrograms_x410_491${SUF}.jpg"
fi
