"""Frequency-plan geometry for the 200 MHz / 245.76 MSps composition.

Plan (confirmed): three 50 MHz blocks on a 60 MHz pitch (5 MHz guard each side),
centered at -60 / 0 / +60 MHz. A waveform wider than 50 MHz merges contiguous
block slots; merged usable bandwidth = 60*N - 10 MHz (50/110/170 for N=1/2/3).
Within a single block, multiple sub-50 MHz waveforms are stacked in frequency
with 5 MHz guards between them.
"""

from __future__ import annotations
import numpy as np

FS = 245.76e6
BLOCK_PITCH = 60e6
USABLE_BW_1 = 50e6          # usable signal bandwidth of one block
GUARD = 5e6                 # guard band each side / between stacked waveforms
BLOCK_CENTERS = (-60e6, 0.0, 60e6)   # slot index 0, 1, 2

# Candidate ZC centers a receiver must scan: single-block (+/-60, 0),
# 2-block merged (+/-30), 3-block merged (0).
RX_CANDIDATE_CENTERS = (-60e6, -30e6, 0.0, 30e6, 60e6)


def usable_bw(n_blocks: int) -> float:
    return BLOCK_PITCH * n_blocks - 2 * GUARD


def required_blocks(occ_bw_hz: float) -> int:
    for n in (1, 2, 3):
        if occ_bw_hz <= usable_bw(n) + 1.0:
            return n
    raise ValueError(
        f"occupied bandwidth {occ_bw_hz/1e6:.2f} MHz exceeds the 3-block "
        f"maximum of {usable_bw(3)/1e6:.1f} MHz")


def slot_center(idx: int) -> float:
    return BLOCK_CENTERS[idx]


def merged_center(slots) -> float:
    return float(np.mean([BLOCK_CENTERS[i] for i in slots]))


def contiguous(slots) -> bool:
    s = sorted(slots)
    return all(b - a == 1 for a, b in zip(s, s[1:]))


def stack_layout(bw_list, center_hz, usable):
    """Center a group of waveforms (bandwidths bw_list) within `usable`, with
    GUARD between them. Returns absolute center frequency for each. Raises if it
    does not fit."""
    bw_list = list(bw_list)
    total = sum(bw_list) + GUARD * (len(bw_list) - 1)
    if total > usable + 1.0:
        raise ValueError(
            f"waveforms in this block need {total/1e6:.2f} MHz "
            f"(sum BW + {GUARD/1e6:.0f} MHz guards) but only {usable/1e6:.1f} MHz "
            f"is usable; split them into different blocks")
    centers = []
    cursor = center_hz - total / 2.0
    for bw in bw_list:
        centers.append(cursor + bw / 2.0)
        cursor += bw + GUARD
    return centers
