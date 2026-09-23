"""Simple wireless channel for the composite (applied at 245.76 MSps)."""

from __future__ import annotations
import numpy as np


def apply_channel(x, fs, snr_db=np.inf, gain=1.0, phase_rad=0.0,
                  cfo_hz=0.0, taps_delay=None, taps_db=None, seed=0):
    """Multipath + flat gain/phase + (optional) CFO + AWGN.

    Stays in complex64 and adds noise in chunks so it scales to multi-GB
    captures. The metadata OFDM modem has pilots for residual phase but no full
    CFO tracking, so keep cfo_hz small/zero for reliable metadata recovery.
    """
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(x).ravel().astype(np.complex64, copy=False)
    N = x.size

    # multipath (causal tapped delay line), built without a full convolution copy
    if taps_delay is not None and len(taps_delay) > 0:
        g = (10 ** (np.asarray(taps_db, float) / 20.0)).astype(np.complex64)
        y = x * g[0]
        for d, gg in zip(taps_delay[1:], g[1:]):
            d = int(d)
            if d > 0:
                y[d:] += np.complex64(gg) * x[:N - d]
            else:
                y += np.complex64(gg) * x
    else:
        y = x.copy()

    if cfo_hz != 0.0:
        n = np.arange(N)
        y *= np.exp(2j * np.pi * cfo_hz / fs * n).astype(np.complex64)
    if gain != 1.0 or phase_rad != 0.0:
        y *= np.complex64(gain * np.exp(1j * phase_rad))

    if np.isfinite(snr_db):
        # signal power via a chunked sum (no 4 GB temporary)
        C = 1 << 24
        ssum, cnt = 0.0, 0
        for s in range(0, N, C):
            blk = y[s:s + C]
            ssum += float(np.sum((blk.real.astype(np.float64)) ** 2 + (blk.imag.astype(np.float64)) ** 2))
            cnt += blk.size
        sigma = np.sqrt((ssum / max(cnt, 1)) / (10 ** (snr_db / 10.0)) / 2.0)
        for s in range(0, N, C):
            m = min(C, N - s)
            nz = (rng.standard_normal(m).astype(np.float32)
                  + 1j * rng.standard_normal(m).astype(np.float32))
            y[s:s + m] += (np.float32(sigma) * nz).astype(np.complex64)

    return y


# A representative mild wireless profile for the composite.
WIRELESS = dict(snr_db=30.0, gain=0.7, phase_rad=0.4, cfo_hz=0.0,
                taps_delay=[0, 5, 13], taps_db=[0.0, -10.0, -16.0])
