"""Low-SNR ("fine comb") composite decoder.

decode.py is fast: it energy-gates the capture to a few windows and peak-picks a
normalized matched filter above a fixed cosine-similarity threshold. Both of
those break at very low SNR -- when the signal power approaches the noise floor,
the silence/signal split disappears (gating fails) and the cosine metric of a
buried-but-recoverable ZC drops below the threshold.

This module instead combs the ENTIRE capture (no gating) with the ZC matched
filter -- which gives the full ~10*log10(L) dB of processing gain (~34.7 dB for
the 2955-sample ZC) -- and detects peaks with a CFAR (constant-false-alarm-rate)
threshold set relative to a robust local noise estimate, so it adapts to the
actual noise level instead of a fixed cutoff. An optional small CFO search adds
robustness to a residual carrier offset (common OTA). The per-block CRC on the
decoded metadata remains the final gate, so a sensitive threshold is safe.

It is slower than decode.py (it correlates everything), but recovers ZCs far
deeper into the noise. Everything after detection (metadata decode, annotation
reconstruction) is reused from decode.py.

CLI:  python fine_comb_decode.py <data.sigmf-data> <rx.sigmf-meta> <zc.mat> [pfa]
"""

from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve, find_peaks

import geometry as geo
import decode as _decode
from zc_sync import load_zc, freq_shift

# 25th-percentile of a Rayleigh(sigma): x = sigma*sqrt(-2*ln(0.75))
_RAYLEIGH_P25 = float(np.sqrt(-2.0 * np.log(0.75)))   # ~0.7585


def _cfar_sweep(iq, mf, L, pfa, chunk):
    """Full matched-filter sweep with a CFAR threshold; returns (start, snr_sigma)."""
    N = iq.size
    M = N - L + 1
    thr_factor = float(np.sqrt(-2.0 * np.log(pfa)))    # Rayleigh tail for target Pfa
    out = []
    start = 0
    while start < M:
        stop = min(start + chunk, M)
        seg = iq[start:stop + L - 1]
        cc = np.abs(fftconvolve(seg, mf, mode="valid"))     # |corr|, index j -> ZC at start+j
        mlen = min(cc.size, stop - start)
        cc = cc[:mlen]
        # robust noise floor from a low percentile (quiet/guard samples dominate
        # the bottom quartile even when part of the chunk carries signal)
        sigma = float(np.percentile(cc, 25)) / _RAYLEIGH_P25 + 1e-20
        thr = sigma * thr_factor
        pk, _ = find_peaks(np.concatenate(([0.0], cc, [0.0])), height=thr, distance=L)
        pk = pk - 1
        pk = pk[(pk >= 0) & (pk < mlen)]
        for k in pk:
            out.append((int(start + k), float(cc[k] / sigma)))
        if stop >= M:
            break
        start += chunk - 2 * L                             # overlap so boundary peaks survive
    return out


def detect_fine(iq, zc, fs, centers, pfa=1e-7, cfo_grid_hz=(0.0,), chunk=1 << 26):
    """Low-SNR ZC detection: full-sweep CFAR matched filter, optional CFO search.

    Returns [{center_hz, start_sample, metric (peak/sigma), cfo_hz}]. The CRC on
    the decoded metadata rejects the (few) CFAR false alarms downstream.
    """
    iq = np.asarray(iq).ravel().astype(np.complex64)
    L = zc.size
    raw = []
    for fc in centers:
        for cfo in cfo_grid_hz:
            mf = np.conj(freq_shift(zc, fc + cfo, fs)[::-1])
            for s, m in _cfar_sweep(iq, mf, L, pfa, chunk):
                raw.append((float(fc), int(s), float(m), float(cfo)))
    # de-duplicate within L per center (overlap regions + CFO grid), keep strongest
    raw.sort(key=lambda r: (r[0], r[1]))
    dets = []
    for fc, s, m, cfo in raw:
        if dets and dets[-1]["center_hz"] == fc and s - dets[-1]["start_sample"] < L:
            if m > dets[-1]["metric"]:
                dets[-1] = {"center_hz": fc, "start_sample": s, "metric": m, "cfo_hz": cfo}
        else:
            dets.append({"center_hz": fc, "start_sample": s, "metric": m, "cfo_hz": cfo})
    return dets


def decode_composite_fine(data_path, rx_meta_path, zc_path, out_meta_path=None,
                          candidate_centers=geo.RX_CANDIDATE_CENTERS,
                          pfa=1e-7, cfo_grid_hz=(0.0,), profile="auto"):
    """Low-SNR decode: combs the whole capture (CFAR) then reuses decode.py's
    metadata decode + annotation reconstruction.

    `profile` is "auto" by default (self-describing: try fast then robust, lock on
    the first CRC pass). Pass "robust" if you know the capture used the robust
    metadata profile to skip the fast attempt on every detection."""
    det = lambda iq, zc, fs, c: detect_fine(iq, zc, fs, c,
                                            pfa=pfa, cfo_grid_hz=tuple(cfo_grid_hz))
    return _decode.decode_composite(data_path, rx_meta_path, zc_path,
                                    out_meta_path=out_meta_path,
                                    candidate_centers=candidate_centers, detector=det,
                                    profile=profile)


# convenience re-export
validate = _decode.validate


if __name__ == "__main__":
    import sys
    pfa = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-7
    profile = sys.argv[5] if len(sys.argv) > 5 else "auto"
    decode_composite_fine(sys.argv[1], sys.argv[2], sys.argv[3], pfa=pfa, profile=profile)
