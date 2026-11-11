#!/usr/bin/env python3
"""Baseband modulators for the synthetic library scenes (synth_lut_scenes.py) and the measurement unit tests.

Every function returns a complex64 baseband signal at the sample rate it is given (or the rate it reports), with
unit mean power over the "on" samples. Nothing here depends on the library; the scene generator maps the library
descriptors to these building blocks.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy import signal as sps

RNG = np.random.default_rng(1)


def set_seed(seed: int):
    global RNG
    RNG = np.random.default_rng(seed)


# ------------------------------------------------------------------------------------------- helpers
def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6 * fs_out:
        return x.astype(np.complex64)
    # a small rational ratio keeps the polyphase filter short; the rate error (< 0.5 %) is far below every tolerance
    fr = Fraction(fs_out / fs_in).limit_denominator(64)
    return sps.resample_poly(x, fr.numerator, fr.denominator).astype(np.complex64)


def normalise(x: np.ndarray, on: np.ndarray | None = None) -> np.ndarray:
    p = np.mean(np.abs(x[on] if on is not None and on.any() else x) ** 2)
    return (x / math.sqrt(max(p, 1e-30))).astype(np.complex64)


def rrc(beta: float, sps_: int, span: int = 10) -> np.ndarray:
    n = np.arange(-span * sps_, span * sps_ + 1, dtype=np.float64)
    t = n / sps_
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def gaussian_pulse(bt: float, sps_: int, span: int = 4) -> np.ndarray:
    t = np.arange(-span * sps_, span * sps_ + 1) / sps_
    alpha = np.sqrt(np.log(2) / 2) / bt
    g = np.sqrt(np.pi) / alpha * np.exp(-(np.pi * t / alpha) ** 2)
    return g / g.sum()


def audio_like(fs: float, dur: float, bw_hz: float, rng=None) -> np.ndarray:
    """Band-limited noise with a speech/music-like 1/f tilt and syllabic amplitude modulation, peak-normalised."""
    rng = rng or RNG
    n = int(dur * fs)
    w = rng.standard_normal(n).astype(np.float32)
    nf = 1 << int(math.ceil(math.log2(max(n, 2))))
    W = np.fft.rfft(w, nf)
    f = np.fft.rfftfreq(nf, 1 / fs)
    shape = 1.0 / np.sqrt(1 + (f / 300.0) ** 2) * (f <= bw_hz) * (f >= 40.0)
    a = np.fft.irfft(W * shape, nf)[:n]
    env = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * rng.uniform(2.0, 5.0) * np.arange(n) / fs + rng.uniform(0, 6.3)))
    a *= env
    return (a / max(np.max(np.abs(a)), 1e-9)).astype(np.float32)


# ------------------------------------------------------------------------------------------- analogue
def fm_mod(audio: np.ndarray, fs: float, dev_hz: float) -> np.ndarray:
    ph = 2 * np.pi * np.cumsum(dev_hz * audio) / fs
    return np.exp(1j * ph).astype(np.complex64)


def fm_stereo(fs: float, dur: float, audio_bw_hz: float = 15e3, pilot: bool = True, rds: bool = True, hd: bool = False,
              dev_hz: float = 75e3, rng=None) -> np.ndarray:
    """FM broadcast: L+R, 19 kHz pilot, 38 kHz DSB-SC L-R, 57 kHz RDS BPSK 1187.5 bps; optional NRSC-5 sidebands."""
    rng = rng or RNG
    n = int(dur * fs)
    t = np.arange(n) / fs
    lr = audio_like(fs, dur, audio_bw_hz, rng)
    lmr = 0.5 * audio_like(fs, dur, audio_bw_hz, rng)
    mpx = 0.9 * lr
    if pilot:
        mpx += 0.09 * np.sin(2 * np.pi * 19e3 * t)
        mpx += 0.45 * lmr * np.sin(2 * np.pi * 38e3 * t)
    if rds:
        bits = rng.integers(0, 2, int(dur * 1187.5) + 2) * 2 - 1
        bb = np.repeat(bits, int(round(fs / 1187.5)))[:n]
        bb = sps.lfilter(np.ones(int(fs / 2375)) / int(fs / 2375), 1, bb) if fs > 10e3 else bb
        mpx += 0.04 * bb * np.sin(2 * np.pi * 57e3 * t)
    mpx = mpx / max(np.max(np.abs(mpx)), 1e-9)
    x = fm_mod(mpx.astype(np.float32), fs, dev_hz)
    if hd and fs >= 500e3:
        # primary digital sidebands: OFDM 363.37 Hz spacing, ~191 subcarriers per sideband at +-(129.4..198.4) kHz, ~ -20 dBc
        sb = ofdm(fs, 363.372, 0.15e-3, n_used=190, dur=dur + 0.01, qam=4, rng=rng)
        sb = np.concatenate([sb, np.zeros(max(n - len(sb), 0), dtype=np.complex64)])[:n]
        side = sb * np.exp(2j * np.pi * 163.9e3 * t) + sb[::-1] * np.exp(-2j * np.pi * 163.9e3 * t)
        x = (x + 0.1 * side / np.sqrt(np.mean(np.abs(side) ** 2))).astype(np.complex64)
    return normalise(x)


def nbfm_voice(fs: float, dur: float, audio_bw_hz: float = 3e3, dev_hz: float = 2.5e3, ctcss_hz: float | None = None, rng=None) -> np.ndarray:
    rng = rng or RNG
    a = audio_like(fs, dur, audio_bw_hz, rng)
    if ctcss_hz:
        a = 0.85 * a + 0.15 * np.sin(2 * np.pi * ctcss_hz * np.arange(len(a)) / fs)
    return fm_mod(a.astype(np.float32), fs, dev_hz)


def am_mod(fs: float, dur: float, audio_bw_hz: float = 3e3, depth: float = 0.8, carrier: bool = True, rng=None) -> np.ndarray:
    rng = rng or RNG
    a = audio_like(fs, dur, audio_bw_hz, rng)
    x = (1.0 if carrier else 0.0) + depth * a
    return normalise(x.astype(np.complex64))


def am_tones(fs: float, dur: float, tones_hz: list[float], depths: list[float] | None = None, tick: bool = False, rng=None) -> np.ndarray:
    n = int(dur * fs)
    t = np.arange(n) / fs
    depths = depths or [0.3] * len(tones_hz)
    m = np.zeros(n, dtype=np.float64)
    for f, d in zip(tones_hz, depths):
        if f == 9960.0:                                   # VOR reference: 9960 Hz subcarrier FM'd +-480 Hz by 30 Hz
            m += d * np.cos(2 * np.pi * 9960 * t + (480 / 30) * np.sin(2 * np.pi * 30 * t))
        else:
            m += d * np.cos(2 * np.pi * f * t)
    if tick:                                              # 5 ms 1000 Hz tick every second
        for s in np.arange(0, dur, 1.0):
            i0, i1 = int(s * fs), int((s + 0.005) * fs)
            m[i0:i1] += 0.5 * np.cos(2 * np.pi * 1000 * t[i0:i1])
    return normalise((1.0 + m).astype(np.complex64))


def am_msk(fs: float, dur: float, bitrate: float = 2400.0, depth: float = 0.9, rng=None) -> np.ndarray:
    """ACARS: AM carrier modulated by an MSK audio signal (1200 / 2400 Hz tones)."""
    rng = rng or RNG
    nb = int(dur * bitrate) + 1
    bits = rng.integers(0, 2, nb)
    sps_ = int(round(fs / bitrate))
    f_tone = np.where(bits == 1, 1200.0, 2400.0)
    fi = np.repeat(f_tone, sps_)[: int(dur * fs)]
    audio = np.cos(2 * np.pi * np.cumsum(fi) / fs)
    return normalise((1.0 + depth * audio).astype(np.complex64))


# ------------------------------------------------------------------------------------------- digital single carrier
def cpm(symbols: np.ndarray, rs: float, fs: float, h: float, bt: float | None = None, levels: int = 2) -> np.ndarray:
    """Continuous-phase FSK: symbols in {-(L-1), .., L-1} step 2; h = 2 * peak_dev / Rs (per outer level pair);
    Gaussian (GFSK/GMSK) when bt is given, rectangular (CPFSK) otherwise."""
    sps_ = max(2, int(round(fs / rs)))
    fs_eff = rs * sps_
    a = np.repeat(symbols.astype(np.float64), sps_)
    if bt:
        a = np.convolve(a, gaussian_pulse(bt, sps_), mode="same")
    dev = 0.5 * h * rs                      # outer-level deviation = h Rs / 2 per unit symbol step
    scale = dev / max(levels - 1, 1)
    ph = 2 * np.pi * np.cumsum(scale * a) / fs_eff
    x = np.exp(1j * ph).astype(np.complex64)
    return resample_to(x, fs_eff, fs) if abs(fs_eff - fs) > 1e-6 * fs else x


def fsk_bits(n: int, levels: int = 2, rng=None) -> np.ndarray:
    rng = rng or RNG
    k = rng.integers(0, levels, n)
    return (2 * k - (levels - 1)).astype(np.float64)


def psk(fs: float, rs: float, n_sym: int, order: int = 4, rolloff: float = 0.35, differential: bool = False, rng=None) -> np.ndarray:
    rng = rng or RNG
    sps_ = max(2, int(round(fs / rs)))
    fs_eff = rs * sps_
    k = rng.integers(0, order, n_sym)
    if differential:
        k = np.cumsum(k) % order
    sym = np.exp(2j * np.pi * (k + (0.5 if order in (4, 8) else 0.0)) / order)
    up = np.zeros(n_sym * sps_, dtype=np.complex64)
    up[::sps_] = sym
    x = np.convolve(up, rrc(rolloff, sps_), mode="same").astype(np.complex64)
    return normalise(resample_to(x, fs_eff, fs))


def qam16(fs: float, rs: float, n_sym: int, rolloff: float = 0.35, rng=None) -> np.ndarray:
    rng = rng or RNG
    sps_ = max(2, int(round(fs / rs)))
    pts = np.array([-3, -1, 1, 3], dtype=np.float64)
    sym = pts[rng.integers(0, 4, n_sym)] + 1j * pts[rng.integers(0, 4, n_sym)]
    up = np.zeros(n_sym * sps_, dtype=np.complex64)
    up[::sps_] = sym
    x = np.convolve(up, rrc(rolloff, sps_), mode="same").astype(np.complex64)
    return normalise(resample_to(x, rs * sps_, fs))


def oqpsk_halfsine(fs: float, chip_rate: float, n_chips: int, rng=None) -> np.ndarray:
    """IEEE 802.15.4 O-QPSK with half-sine pulses (MSK-equivalent)."""
    rng = rng or RNG
    sps_ = max(2, int(round(fs / chip_rate)))
    i = rng.integers(0, 2, n_chips // 2 + 1) * 2 - 1
    q = rng.integers(0, 2, n_chips // 2 + 1) * 2 - 1
    hs = np.sin(np.pi * np.arange(2 * sps_) / (2 * sps_))
    ui = np.zeros((n_chips // 2 + 1) * 2 * sps_)
    uq = np.zeros_like(ui)
    ui[:: 2 * sps_] = i
    uq[:: 2 * sps_] = q
    xi = np.convolve(ui, hs, mode="full")[: len(ui)]
    xq = np.convolve(uq, hs, mode="full")[: len(uq)]
    xq = np.roll(xq, sps_)
    x = (xi + 1j * xq).astype(np.complex64)
    return normalise(resample_to(x, chip_rate * sps_, fs))


def dsss_barker(fs: float, n_bits: int, chip_rate: float = 11e6, rng=None) -> np.ndarray:
    """802.11b 1 Mbps DBPSK with the 11-chip Barker code."""
    rng = rng or RNG
    barker = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1], dtype=np.float64)
    bits = rng.integers(0, 2, n_bits) * 2 - 1
    d = np.cumprod(bits)
    chips = (d[:, None] * barker[None, :]).ravel()
    sps_ = max(2, int(round(fs / chip_rate)))
    up = np.repeat(chips, sps_)
    h = rrc(0.7, sps_, span=4)
    x = np.convolve(up, h, mode="same").astype(np.complex64)
    return normalise(resample_to(x, chip_rate * sps_, fs))


def vsb8(fs: float, dur: float, rs: float = 10.762238e6, pilot: float = 1.25, rng=None) -> np.ndarray:
    """ATSC 8-VSB: 8-level PAM + pilot, RRC 0.0575, upper sideband; the pilot lands at the lower edge + 309.441 kHz
    once the signal is centred on its 6 MHz channel (returned centred on the channel)."""
    rng = rng or RNG
    sps_ = 2
    fs_eff = rs * sps_
    n_sym = int(dur * rs) + 1
    lv = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)
    sym = lv[rng.integers(0, 8, n_sym)] + pilot
    up = np.zeros(n_sym * sps_)
    up[::sps_] = sym
    x = np.convolve(up, rrc(0.0575, sps_, span=24), mode="same")
    # analytic signal -> upper sideband; shift so that the channel is centred: pilot at -(3 MHz - 309.441 kHz)
    xa = sps.hilbert(x)
    t = np.arange(len(xa)) / fs_eff
    xc = xa * np.exp(-2j * np.pi * (3.0e6 - 309.441e3) * t)
    return normalise(resample_to(xc.astype(np.complex64), fs_eff, fs))


# ------------------------------------------------------------------------------------------- OFDM
def ofdm(fs: float, scs_hz: float, cp_s: float, n_used: int, dur: float, qam: int = 4, rng=None, dc_null: bool = True,
         symbol_on: np.ndarray | None = None) -> np.ndarray:
    """CP-OFDM at the sample rate fs: FFT size = next power of two >= n_used * 1.25 at fs_ofdm = N * scs, then
    resampled to fs. symbol_on (bool per OFDM symbol) blanks symbols (TDD patterns, LBT bursts)."""
    rng = rng or RNG
    N = 1 << int(math.ceil(math.log2(max(n_used * 1.25, 8))))
    fs_ofdm = N * scs_hz
    tu = 1.0 / scs_hz
    ncp = int(round(cp_s * fs_ofdm))
    n_sym = int(math.ceil(dur / (tu + cp_s))) + 1
    if qam == 4:
        pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    else:
        p = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        pts = (p[:, None] + 1j * p[None, :]).ravel()
    out = np.zeros(n_sym * (N + ncp), dtype=np.complex64)
    k_idx = np.concatenate([np.arange(1, n_used // 2 + 1), np.arange(N - n_used // 2, N)]) if dc_null else np.arange(-(n_used // 2), n_used // 2) % N
    for s in range(n_sym):
        if symbol_on is not None and not symbol_on[s % len(symbol_on)]:
            continue
        X = np.zeros(N, dtype=np.complex64)
        X[k_idx] = pts[rng.integers(0, len(pts), len(k_idx))]
        x = np.fft.ifft(X) * np.sqrt(N)
        out[s * (N + ncp): (s + 1) * (N + ncp)] = np.concatenate([x[-ncp:], x]) if ncp else x
    return resample_to(out, fs_ofdm, fs)


def tdd_pattern(pattern: str, slot_s: float, scs_hz: float, cp_s: float) -> np.ndarray:
    """OFDM-symbol on/off mask for a slot pattern like DDDSU (D = on, U = off, S = half on)."""
    sym_s = 1.0 / scs_hz + cp_s
    per_slot = max(1, int(round(slot_s / sym_s)))
    mask = []
    for ch in pattern:
        if ch == "D":
            mask += [True] * per_slot
        elif ch == "U":
            mask += [False] * per_slot
        else:
            mask += [True] * (per_slot * 2 // 3) + [False] * (per_slot - per_slot * 2 // 3)
    return np.array(mask, dtype=bool)


# ------------------------------------------------------------------------------------------- pulses / chirps / OOK
def pulse_train(fs: float, dur: float, times_s: np.ndarray, widths_s: np.ndarray, shape: str = "rect", amp: np.ndarray | None = None) -> np.ndarray:
    n = int(dur * fs)
    x = np.zeros(n, dtype=np.complex64)
    for k, (t0, w) in enumerate(zip(times_s, widths_s)):
        a = 1.0 if amp is None else amp[k]
        i0 = int(round(t0 * fs))
        if shape == "gauss":
            half = int(round(2.5 * w * fs))
            idx = np.arange(max(i0 - half, 0), min(i0 + half, n))
            x[idx] += a * np.exp(-0.5 * ((idx - i0) / (w * fs / 2.355)) ** 2)
        else:
            i1 = min(i0 + max(1, int(round(w * fs))), n)
            if i0 < n:
                x[i0:i1] += a
    return x


def adsb_message(fs: float, long: bool = True, rng=None) -> np.ndarray:
    """Mode S PPM: 8 us preamble (pulses at 0, 1, 3.5, 4.5 us) + 56 / 112 bits of 1 us (0.5 us pulse in the first / second half)."""
    rng = rng or RNG
    nbits = 112 if long else 56
    bits = rng.integers(0, 2, nbits)
    times = [0.0, 1.0, 3.5, 4.5] + [8.0 + i + (0.0 if b else 0.5) for i, b in enumerate(bits)]
    dur = 8.0 + nbits + 1
    return pulse_train(fs, dur * 1e-6, np.array(times) * 1e-6, np.full(len(times), 0.5e-6))


def dme_pair(fs: float, spacing_s: float = 12e-6, width_s: float = 3.5e-6) -> np.ndarray:
    return pulse_train(fs, spacing_s + 6 * width_s, np.array([2 * width_s, 2 * width_s + spacing_s]), np.array([width_s, width_s]), shape="gauss")


def lora_burst(fs: float, bw_hz: float, sf: int, n_payload: int, preamble: int = 8, rng=None) -> np.ndarray:
    """LoRa CSS: up-chirps of 2^sf / bw seconds sweeping bw, cyclically shifted by the symbol value; 2.25 down-chirps after the preamble."""
    rng = rng or RNG
    os_ = max(2, int(math.ceil(fs / bw_hz)))
    fs_eff = bw_hz * os_
    n_chip = 1 << sf
    n_s = n_chip * os_
    t = np.arange(n_s) / fs_eff
    base_rate = bw_hz / (n_chip / bw_hz)
    def chirp(sym: int, down: bool = False):
        f0 = -bw_hz / 2 + (sym / n_chip) * bw_hz
        f = f0 + base_rate * t
        f = np.where(f >= bw_hz / 2, f - bw_hz, f)
        if down:
            f = -f
        ph = 2 * np.pi * np.cumsum(f) / fs_eff
        return np.exp(1j * ph)
    parts = [chirp(0) for _ in range(preamble)] + [chirp(0, True), chirp(0, True), chirp(0, True)[: n_s // 4]]
    parts += [chirp(int(rng.integers(0, n_chip))) for _ in range(n_payload)]
    x = np.concatenate(parts).astype(np.complex64)
    return resample_to(x, fs_eff, fs)


def ook_burst(fs: float, bitrate: float, n_bits: int, manchester: bool = False, rng=None) -> np.ndarray:
    rng = rng or RNG
    bits = rng.integers(0, 2, n_bits)
    if manchester:
        bits = np.stack([bits, 1 - bits], axis=1).ravel()
        bitrate *= 2
    sps_ = max(2, int(round(fs / bitrate)))
    x = np.repeat(bits.astype(np.float32), sps_)
    x = ndimage_smooth(x, max(1, sps_ // 8))
    return resample_to(x.astype(np.complex64), bitrate * sps_, fs)


def ndimage_smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def magnetron(fs: float, dur: float, sweep_hz: float = 10e6, mains_hz: float = 60.0, rng=None) -> np.ndarray:
    """Microwave oven: noise-like carrier whose frequency sweeps with the half-wave rectified mains, on for ~half of each cycle."""
    rng = rng or RNG
    n = int(dur * fs)
    t = np.arange(n) / fs
    phase_m = 2 * np.pi * mains_hz * t
    on = np.sin(phase_m) > 0.3
    f = sweep_hz * (np.sin(phase_m) - 0.3) / 0.7
    ph = 2 * np.pi * np.cumsum(f) / fs
    x = np.exp(1j * ph) * (1 + 0.3 * rng.standard_normal(n)) * on
    return normalise(x.astype(np.complex64), on)
