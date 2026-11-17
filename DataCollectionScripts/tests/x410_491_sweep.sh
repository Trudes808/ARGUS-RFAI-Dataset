#!/usr/bin/env bash
# 400 MHz-wide looks at the X410's full 491.52 MSps (CG_400 image, fixed rate): the high-rate cases the N320 stress set
# could not record. Eleven centres cover FM/VHF/TV, UHF TV + 600/700 cellular, 700-1100 (850 cellular, LMR 800, ISM 900,
# paging, ADS-B), DME/L-band, GPS/Iridium, PCS/AWS/DECT, 2.4 GHz ISM + 2.5 GHz TDD, CBRS + C-band, and 5 GHz Wi-Fi.
# 1 s each (~2 GB ci16, staged in /dev/shm by the collector), auto gain, one antenna on ch0 TX/RX0.
#   OUT=~/Datasets/ARGUS/OOD_captures/x410_491_2026-11-16 ADDR=192.168.31.2 ./tests/x410_491_sweep.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
OUT=${OUT:-$HOME/Datasets/ARGUS/OOD_captures/x410_491_$(date +%Y-%m-%d)}
ADDR=${ADDR:-192.168.31.2}
CH=${CH:-0}; ANT=${ANT:-TX/RX0}
FREQS=${FREQS:-250e6,600e6,900e6,1200e6,1500e6,1950e6,2450e6,2650e6,3650e6,5300e6,5700e6}
cd "${HERE}"
python3 collect_common_waveforms.py --out "${OUT}" --args "addr=${ADDR}" --radio x410 \
  --freqs "${FREQS}" --rates 491.52e6 --dur 1 --max-gb 2.1 --channel "${CH}" --ant "${ANT}" --gain auto --gain-max 50 \
  --site "DGX Spark bench, Austin TX, indoor; X410 CG_400 (fixed 491.52 MSps); one antenna on ch${CH} ${ANT}" \
  --antenna "bench antenna on ch${CH} ${ANT} (model not recorded)" --pause 2 "$@"
