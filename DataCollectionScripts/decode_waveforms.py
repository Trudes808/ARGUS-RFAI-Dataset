#!/usr/bin/env python3
"""Attempt-to-decode layer of the LUT classifier: protocol decoders that turn a plausible emission into a verified
identity, plus the matching frame encoders used by the synthetic scene generator and the unit test.

    python3 decode_waveforms.py <capture>.sigmf-meta --fc 1090e6 --bw 4e6 --protocol adsb
    python3 decode_waveforms.py <capture>.sigmf-meta --fc 98.1e6 --bw 240e3 --protocol rds

A decode succeeds only on a check that random data cannot pass (a CRC / BCH remainder, a 36-bit sync word with at
most two errors, a PSS/SSS cell identity that repeats every 5 ms), so a success is definitive evidence for the
library entry; a failure is neutral (weak signal, short capture, unsupported variant). Decoders:

    adsb      Mode S / ADS-B 1090ES: preamble, 112/56-bit PPM, CRC-24 (DF17/18 parity, DF11 all-call)
    rds       FM broadcast RDS: 57 kHz BPSK 1187.5 bps, biphase, block syndromes (offset words), PI / PTY / PS
    ble_adv   Bluetooth LE advertising channels 37/38/39: access address, de-whitening, CRC-24, AdvA / PDU type / name
    lte       LTE downlink: PSS (Zadoff-Chu) + SSS (m-sequences) -> physical cell identity, FDD/TDD, 5 ms repetition
    pocsag    POCSAG 512/1200/2400: sync word, BCH(31,21) + parity codewords, addresses (RIC) / functions
    flex      FLEX: 1600 bps sync (A word + 0xA6C6AAAA marker), Frame Information Word BCH + checksum, cycle / frame
    ert       Itron ERT (SCM BCH 0x6F63, SCM+ CRC-CCITT) OOK Manchester 32.768 kchip/s: meter id / type / reading
    p25       APCO P25 C4FM: 48-bit frame sync, NID BCH(63,16) -> NAC / DUID (status symbols skipped)
    dmr       ETSI DMR 4FSK: 48-bit sync patterns (BS / MS / direct mode), burst spacing
    dect      DECT: S-field sync (RFP / PP), A-field R-CRC -> header / RFPI
    zigbee    IEEE 802.15.4 O-QPSK 2 Mchip/s: preamble / SFD chip despreading, PHR, FCS CRC-16 -> addresses
    uat       UAT 978 MHz: 36-bit ADS-B / uplink sync words (<= 2 errors), frame class

Every decoder works on the complex baseband of one emission (``cap.extract``) and returns a ``Decode`` record.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VERSION = "decode_waveforms.py 0.1 (lut_classification_experiment)"


@dataclass
class Decode:
    protocol: str
    ok: bool = False
    n_ok: int = 0                 # frames / blocks that passed the check
    n_tried: int = 0              # candidates examined
    fields: dict = field(default_factory=dict)
    evidence: str = ""
    variant: str = ""             # library variant suffix the decode establishes (POCSAG, P25, LTE, ...)
    weak: bool = False            # sync-only result (no integrity check): evidence but not an identity
    seconds: float = 0.0

    def summary(self) -> str:
        return f"{self.protocol}: {self.evidence}" if self.evidence else self.protocol


# =========================================================================================== bit / CRC helpers
def int_to_bits(v: int, n: int) -> np.ndarray:
    """MSB first."""
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def bits_to_int(b) -> int:
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v


def bytes_to_bits_lsb(data: bytes) -> np.ndarray:
    """Transmission order of byte-oriented air interfaces (LSB of each byte first)."""
    out = np.zeros(8 * len(data), dtype=np.uint8)
    for i, by in enumerate(data):
        for j in range(8):
            out[8 * i + j] = (by >> j) & 1
    return out


def bits_lsb_to_bytes(bits) -> bytes:
    out = bytearray()
    for i in range(0, len(bits) - len(bits) % 8, 8):
        v = 0
        for j in range(8):
            v |= int(bits[i + j]) << j
        out.append(v)
    return bytes(out)


def poly_remainder(bits, gen: int, degree: int) -> int:
    """Remainder of the polynomial whose coefficients are `bits` (MSB first) modulo `gen` (degree `degree`)."""
    r = 0
    top = 1 << degree
    for b in bits:
        r = (r << 1) | int(b)
        if r & top:
            r ^= gen
    return r


def crc_msb(bits, gen: int, width: int, init: int = 0, xorout: int = 0) -> int:
    """Generic non-reflected CRC over a bit sequence (MSB first)."""
    r = init
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    for b in bits:
        fb = ((r >> (width - 1)) & 1) ^ int(b)
        r = (r << 1) & mask
        if fb:
            r ^= gen & mask
    return r ^ xorout


def crc16_kermit(data: bytes) -> int:
    """CRC-16/KERMIT (reflected 0x1021, init 0): the IEEE 802.15.4 FCS."""
    crc = 0
    for by in data:
        crc ^= by
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def crc16_ccitt_false(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for by in data:
        crc ^= by << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def crc16_poly(data: bytes, poly: int, init: int = 0) -> int:
    crc = init
    for by in data:
        crc ^= by << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def find_pattern(bits: np.ndarray, pattern: np.ndarray, max_err: int) -> list[int]:
    """Positions where `pattern` occurs in `bits` with at most max_err bit errors (sliding, vectorised)."""
    n, m = len(bits), len(pattern)
    if n < m:
        return []
    x = bits.astype(np.int8) * 2 - 1
    p = pattern.astype(np.int8) * 2 - 1
    c = np.correlate(x, p, mode="valid")           # m - 2 * errors
    return [int(i) for i in np.flatnonzero(c >= m - 2 * max_err)]


# =========================================================================================== signal helpers
def resample(y: np.ndarray, fs_in: float, fs_out: float, max_den: int = 4000) -> tuple[np.ndarray, float]:
    """Rational resampling; returns (y, achieved rate)."""
    if abs(fs_in - fs_out) <= 1e-9 * fs_out:
        return y, fs_in
    fr = Fraction(fs_out / fs_in).limit_denominator(max_den)
    out = sps.resample_poly(y, fr.numerator, fr.denominator)
    return out.astype(np.complex64) if np.iscomplexobj(y) else out.astype(np.float32), fs_in * fr.numerator / fr.denominator


def lowpass(y: np.ndarray, fs: float, cutoff: float, taps: int = 97) -> np.ndarray:
    """Channel filter before a frequency discriminator: the band extraction keeps the whole decimated band, so
    without it the noise bandwidth is 2-4x the signal's."""
    if cutoff >= 0.48 * fs:
        return y
    h = sps.firwin(taps, cutoff, fs=fs).astype(np.float32)
    return np.convolve(y, h, mode="same").astype(y.dtype)


def inst_freq(y: np.ndarray, fs: float) -> np.ndarray:
    """Instantaneous frequency (Hz), same length as y (first sample repeated)."""
    d = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * np.pi)
    return np.concatenate([d[:1], d]).astype(np.float32)


def find_bursts(y: np.ndarray, fs: float, smooth_s: float, min_len_s: float, margin_db: float = 8.0, gap_s: float = 0.0) -> list[tuple[int, int]]:
    """On-segments [start, end) of a bursty signal by power envelope against the quiet floor. For a continuous
    signal (floor not separable) the whole record is one segment."""
    p = (np.abs(y) ** 2).astype(np.float32)
    w = max(1, int(round(smooth_s * fs)))
    if w > 1:
        p = np.convolve(p, np.ones(w, dtype=np.float32) / w, mode="same")
    floor = float(np.percentile(p, 10))
    thr = floor * 10 ** (margin_db / 10)
    if float(np.percentile(p, 99.9)) <= thr:      # constant envelope: nothing stands above the quiet level
        return [(0, len(y))]
    on = p > thr
    frac = float(np.mean(on))
    if frac > 0.9:                                # nearly always on
        return [(0, len(y))]
    g = int(round(gap_s * fs))          # gaps shorter than this are bridged
    segs = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or (g and j + g < n and on[j: j + g].any())):
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    min_len = int(min_len_s * fs)
    segs = [(a, b) for a, b in segs if b - a >= min_len]
    if not segs:                                  # fluctuating continuous emission (or bursts below min_len): search the whole record
        return [(0, len(y))]
    return segs


def carrier_offset(y: np.ndarray, fs: float, segs=None, max_frac: float = 0.45) -> float:
    """Centroid of the spectral excess over the noise floor (over the on-segments if given): the carrier offset
    of a symmetric-spectrum emission inside the extracted band (detector boxes are only bin-accurate)."""
    z = np.concatenate([y[a:b] for a, b in segs]) if segs else y
    if len(z) < 256:
        return 0.0
    nseg = min(1024, 1 << int(math.floor(math.log2(len(z) / 4))))
    f, P = sps.welch(z, fs=fs, nperseg=nseg, return_onesided=False, scaling="density")
    f = np.fft.fftshift(f)
    P = np.fft.fftshift(P)
    keep = np.abs(f) <= max_frac * fs
    f, P = f[keep], P[keep]
    ex = P - np.median(P)
    ex[ex < 0.2 * ex.max()] = 0.0
    if ex.sum() <= 0:
        return 0.0
    return float(np.sum(f * ex) / np.sum(ex))


def center(y: np.ndarray, fs: float, segs=None) -> tuple[np.ndarray, float]:
    off = carrier_offset(y, fs, segs)
    if abs(off) < 1e-3 * fs:
        return y, off
    t = np.arange(len(y)) / fs
    return (y * np.exp(-2j * np.pi * off * t)).astype(np.complex64), off


def slice_bits(f: np.ndarray, sps_: int, phase: int, start: int, n_bits: int, invert: bool = False) -> np.ndarray:
    """Hard bits from a demodulated waveform: mean over the middle half of each bit period. bit 1 = positive."""
    idx0 = start + phase
    a, b = idx0 + sps_ // 4, idx0 + sps_ - sps_ // 4
    n = min(n_bits, (len(f) - idx0) // sps_)
    if n <= 0:
        return np.zeros(0, dtype=np.uint8)
    m = f[idx0: idx0 + n * sps_].reshape(n, sps_)[:, sps_ // 4: sps_ - sps_ // 4].mean(axis=1)
    bits = (m > 0).astype(np.uint8)
    return 1 - bits if invert else bits


def remove_dc(f: np.ndarray, levels: int = 2) -> np.ndarray:
    """Centre a demodulated FSK waveform: midpoint of the outer level estimates (robust to unbalanced data)."""
    lo, hi = np.percentile(f, 8), np.percentile(f, 92)
    return f - 0.5 * (lo + hi)


# =========================================================================================== ADS-B / Mode S
MODES_GEN = 0x1FFF409          # x^24 + ... (24-bit CRC generator with the leading 1)
MODES_CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"


def modes_remainder(bits) -> int:
    return poly_remainder(bits, MODES_GEN, 24)


def _modes_syndromes(n_bits: int) -> dict:
    out = {}
    for i in range(n_bits):
        e = np.zeros(n_bits, dtype=np.uint8)
        e[i] = 1
        out[modes_remainder(e)] = i
    return out


MODES_SYND_112 = _modes_syndromes(112)


def modes_encode_df17(icao: int, me_bits: np.ndarray, ca: int = 5) -> np.ndarray:
    head = np.concatenate([int_to_bits(17, 5), int_to_bits(ca, 3), int_to_bits(icao, 24), me_bits])
    par = modes_remainder(np.concatenate([head, np.zeros(24, dtype=np.uint8)]))
    return np.concatenate([head, int_to_bits(par, 24)])


def modes_encode_df11(icao: int, ca: int = 5, ii: int = 0) -> np.ndarray:
    head = np.concatenate([int_to_bits(11, 5), int_to_bits(ca, 3), int_to_bits(icao, 24)])
    par = modes_remainder(np.concatenate([head, np.zeros(24, dtype=np.uint8)])) ^ ii
    return np.concatenate([head, int_to_bits(par, 24)])


def modes_me_identification(callsign: str, category: int = 3) -> np.ndarray:
    cs = (callsign.upper() + "        ")[:8]
    out = [int_to_bits(4, 5), int_to_bits(category, 3)]
    for ch in cs:
        out.append(int_to_bits(MODES_CHARSET.index(ch) if ch in MODES_CHARSET else 32, 6))
    return np.concatenate(out)


def modes_me_airborne_position(rng) -> np.ndarray:
    tc = int(rng.integers(9, 19))
    return np.concatenate([int_to_bits(tc, 5), rng.integers(0, 2, 51).astype(np.uint8)])


def modes_random_message(rng, icao: int | None = None, long: bool = True) -> np.ndarray:
    icao = int(rng.integers(0xA00000, 0xAFFFFF)) if icao is None else icao      # US registry block
    if not long:
        return modes_encode_df11(icao)
    if rng.random() < 0.35:
        cs = "N" + "".join(str(int(rng.integers(0, 10))) for _ in range(int(rng.integers(3, 6))))
        return modes_encode_df17(icao, modes_me_identification(cs))
    return modes_encode_df17(icao, modes_me_airborne_position(rng))


def modes_ppm_waveform(fs: float, bits: np.ndarray) -> np.ndarray:
    """Preamble (pulses at 0, 1, 3.5, 4.5 us) + PPM data: a 1 is a 0.5 us pulse in the first half of its 1 us chip."""
    n = int(round((8.0 + len(bits) + 1) * 1e-6 * fs))
    x = np.zeros(n, dtype=np.float32)
    w = max(1, int(round(0.5e-6 * fs)))
    for t in (0.0, 1.0, 3.5, 4.5):
        i = int(round(t * 1e-6 * fs))
        x[i: i + w] = 1.0
    for k, b in enumerate(bits):
        i = int(round((8.0 + k + (0.0 if b else 0.5)) * 1e-6 * fs))
        x[i: i + w] = 1.0
    return x.astype(np.complex64)


def _modes_decode_callsign(me: np.ndarray) -> str:
    s = "".join(MODES_CHARSET[bits_to_int(me[8 + 6 * i: 14 + 6 * i])] for i in range(8))
    return s.replace("#", "").strip()


def decode_adsb(y: np.ndarray, fs: float, max_msgs: int = 400) -> Decode:
    """Mode S preamble + PPM decoder on |y|^2 at 8 MSps (4 samples per half chip)."""
    t0 = time.time()
    d = Decode("adsb")
    y8, fs8 = resample(y, fs, 8e6)
    y8 = lowpass(y8, fs8, 2.6e6, 65)
    m = (np.abs(y8) ** 2).astype(np.float32)
    n = len(m)
    if n < 2000:
        return d
    # half-chip energies on a 4-sample grid for every offset: e[i] = sum m[i:i+4]
    e = np.convolve(m, np.ones(4, dtype=np.float32), mode="full")[3:]        # e[i] = m[i-3..i] -> shift
    e = np.concatenate([e[3:], np.zeros(3, dtype=np.float32)])                # e[i] = sum m[i:i+4]
    floor = float(np.percentile(m, 50)) * 4
    pre = e[0: n - 40] + e[8: n - 32] + e[28: n - 12] + e[36: n - 4]          # the four preamble pulses
    quiet = e[4: n - 36] + e[12: n - 28] + e[20: n - 20] + e[24: n - 16]      # between them
    score = pre - quiet
    thr = max(8.0 * floor, 0.25 * float(np.percentile(pre, 99.9)))
    cand = np.flatnonzero((pre > 4 * quiet + 4 * floor) & (pre > thr))
    if len(cand) == 0:
        d.seconds = time.time() - t0
        return d
    # local maxima of the preamble score among candidates
    keep = []
    for i in cand:
        lo, hi = max(0, i - 6), min(len(score), i + 7)
        if score[i] >= score[lo:hi].max():
            keep.append(int(i))
    icaos, dfs, calls = {}, {}, set()
    n_ok = 0
    n_corr = 0
    last = -1000
    for i in keep[:20000]:
        if i - last < 60:          # one message per preamble
            continue
        start = i + 64              # data begins 8 us after the preamble
        if start + 112 * 8 > n:
            break
        d.n_tried += 1
        first = e[start: start + 112 * 8: 8]
        second = e[start + 4: start + 112 * 8: 8]
        bits = (first > second).astype(np.uint8)
        df = bits_to_int(bits[:5])
        ok = False
        if df in (17, 18):
            rem = modes_remainder(bits[:112])
            if rem != 0 and rem in MODES_SYND_112:           # single-bit correction (dump1090 style)
                bits = bits.copy()
                bits[MODES_SYND_112[rem]] ^= 1
                df = bits_to_int(bits[:5])
                rem = modes_remainder(bits[:112]) if df in (17, 18) else 1
                n_corr += 1 if rem == 0 else 0
        else:
            rem = 1
        if df in (17, 18) and rem == 0:
            ok = True
            icao = bits_to_int(bits[8:32])
            icaos[icao] = icaos.get(icao, 0) + 1
            tc = bits_to_int(bits[32:37])
            if 1 <= tc <= 4:
                calls.add(_modes_decode_callsign(bits[32:88]))
            dfs[df] = dfs.get(df, 0) + 1
            n_ok += 1
        elif df == 11 and modes_remainder(bits[:56]) < 128:
            ok = True
            icao = bits_to_int(bits[8:32])
            icaos[icao] = icaos.get(icao, 0) + 1
            dfs[11] = dfs.get(11, 0) + 1
            n_ok += 1
        if ok:
            last = i
        if n_ok >= max_msgs:
            break
    d.n_ok = n_ok
    d.ok = any(k in (17, 18) for k in dfs) or dfs.get(11, 0) >= 2
    d.fields = {"icao": [f"{k:06X}" for k, _ in sorted(icaos.items(), key=lambda kv: -kv[1])[:12]], "df_counts": dfs, "callsigns": sorted(calls)[:12],
                "n_preambles": len(keep), "corrected": n_corr}
    if d.ok:
        d.evidence = f"{n_ok} Mode S frames with valid parity (" + ", ".join(f"DF{k} x{v}" for k, v in sorted(dfs.items())) + \
            f"), {len(icaos)} aircraft" + (f" incl. {', '.join(sorted(calls)[:3])}" if calls else "")
        d.variant = "ModeS"
    d.seconds = time.time() - t0
    return d


# =========================================================================================== RDS
RDS_GEN = 0x5B9                                                 # x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1
RDS_OFFSETS = {"A": 0x0FC, "B": 0x198, "C": 0x168, "C'": 0x350, "D": 0x1B4}
RDS_SEQ = {"A": ("B",), "B": ("C", "C'"), "C": ("D",), "C'": ("D",), "D": ("A",)}
RDS_BIT_RATE = 1187.5


def rds_block(info16: int, offset: str) -> np.ndarray:
    info = int_to_bits(info16, 16)
    crc = poly_remainder(np.concatenate([info, np.zeros(10, dtype=np.uint8)]), RDS_GEN, 10)
    return np.concatenate([info, int_to_bits(crc ^ RDS_OFFSETS[offset], 10)])


def rds_groups(pi: int, ps: str = "ARGUS FM", pty: int = 10, n_groups: int = 60, rng=None) -> np.ndarray:
    """Alternating 0A (PS name) and 2A (RadioText) groups: the bits of n_groups groups (26 x 4 each)."""
    rng = rng or np.random.default_rng(0)
    ps = (ps + "        ")[:8]
    rt = ("ARGUS LUT CLASSIFIER " * 4)[:64]
    out = []
    seg0 = 0
    seg2 = 0
    for g in range(n_groups):
        if g % 2 == 0:
            b = (0 << 12) | (0 << 11) | (1 << 10) | (pty << 5) | (1 << 4) | (1 << 2) | seg0       # 0A, TP, PTY, TA=0, MS=1, DI, segment
            c = 0xE2C9                                                                     # AF: one AF + filler
            dchars = ps[2 * seg0: 2 * seg0 + 2]
            dd = (ord(dchars[0]) << 8) | ord(dchars[1])
            out += [rds_block(pi, "A"), rds_block(b, "B"), rds_block(c, "C"), rds_block(dd, "D")]
            seg0 = (seg0 + 1) % 4
        else:
            b = (2 << 12) | (0 << 11) | (1 << 10) | (pty << 5) | (0 << 4) | seg2                    # 2A, segment address 0..15
            chars = rt[4 * seg2: 4 * seg2 + 4]
            c = (ord(chars[0]) << 8) | ord(chars[1])
            dd = (ord(chars[2]) << 8) | ord(chars[3])
            out += [rds_block(pi, "A"), rds_block(b, "B"), rds_block(c, "C"), rds_block(dd, "D")]
            seg2 = (seg2 + 1) % 16
    return np.concatenate(out)


def rds_baseband(fs: float, dur: float, bits: np.ndarray, t_offset_bits: int = 0) -> np.ndarray:
    """Differential + biphase coded RDS data stream (real, unit peak) at fs, 1187.5 bps, band-limited to 2.4 kHz."""
    fs_r = 228e3
    n_bits = int(dur * RDS_BIT_RATE) + 4
    reps = int(math.ceil((n_bits + t_offset_bits) / max(len(bits), 1)))
    b = np.tile(bits, reps)[t_offset_bits: t_offset_bits + n_bits]
    # differential encoding: t_k = m_k xor t_{k-1}
    t = np.zeros(len(b), dtype=np.uint8)
    prev = 0
    for i, m in enumerate(b):
        prev ^= int(m)
        t[i] = prev
    sym = (t.astype(np.float32) * 2 - 1)
    half = 96
    bb = np.zeros(len(sym) * 192, dtype=np.float32)
    bb[0::192] = sym
    bb[half::192] = -sym
    # hold each impulse for a half bit, then band-limit (the spec's 100 % cosine roll-off, 2.4 kHz) with a window
    bb = np.convolve(bb, np.ones(half, dtype=np.float32), mode="full")[: len(sym) * 192]
    h = sps.firwin(401, 2.4e3, fs=fs_r)
    bb = np.convolve(bb, h, mode="same")
    out, _ = resample(bb.astype(np.float32), fs_r, fs)
    n = int(dur * fs)
    out = np.concatenate([out, np.zeros(max(n - len(out), 0), dtype=np.float32)])[:n]
    return out / max(np.max(np.abs(out)), 1e-9)


def _lowpass_complex(x: np.ndarray, fs: float, f_shift: float, bw: float, decim: int) -> np.ndarray:
    t = np.arange(len(x)) / fs
    z = x * np.exp(-2j * np.pi * f_shift * t)
    h = sps.firwin(int(4 * fs / bw) | 1, bw, fs=fs)
    z = sps.lfilter(h, 1.0, z)
    return z[::decim]


def rds_pi_to_callsign(pi: int) -> str:
    """US RBDS PI code -> 4-letter call sign (K... 0x1000-0x54A7, W... 0x54A8-0x994F); '' otherwise."""
    if 0x1000 <= pi <= 0x54A7:
        n, first = pi - 0x1000, "K"
    elif 0x54A8 <= pi <= 0x994F:
        n, first = pi - 0x54A8, "W"
    else:
        return ""
    return first + chr(65 + n // 676) + chr(65 + (n // 26) % 26) + chr(65 + n % 26)


def decode_rds(y: np.ndarray, fs: float) -> Decode:
    """FM demodulate the station, recover the 57 kHz BPSK stream, biphase + differential decode, block sync by
    syndromes. Returns PI, PTY, group types and the PS name when all four segments were received."""
    t0 = time.time()
    d = Decode("rds")
    if fs < 300e3 or len(y) < int(0.2 * fs):
        return d
    mpx = inst_freq(lowpass(y, fs, 110e3), fs)
    mpx -= float(np.mean(mpx))
    m228, fsr = resample(mpx, fs, 228e3)
    fsr = 228e3
    t = np.arange(len(m228)) / fsr
    # RDS subcarrier to baseband (complex), 4.8 kHz wide
    r = m228 * np.exp(-2j * np.pi * 57e3 * t)
    h = sps.firwin(301, 3.0e3, fs=fsr)
    r = sps.lfilter(h, 1.0, r)[150:]
    # pilot-locked carrier: the 57 kHz subcarrier is the third harmonic of the 19 kHz pilot
    p = m228 * np.exp(-2j * np.pi * 19e3 * t)
    hp = sps.firwin(801, 400.0, fs=fsr)
    p = sps.lfilter(hp, 1.0, p)[150:]
    n = min(len(r), len(p))
    r, p = r[:n], p[:n]
    pilot_db = 10 * np.log10(np.mean(np.abs(p) ** 2) / max(np.mean(np.abs(m228) ** 2), 1e-12) + 1e-12)
    if np.mean(np.abs(p) ** 2) > 1e-6 * np.mean(np.abs(m228) ** 2):
        ph = np.angle(p)
        z = r * np.exp(-3j * ph)
    else:
        z = r
    # residual constant frequency / phase: squaring estimate
    z2 = z * z
    nfft = 1 << int(math.ceil(math.log2(len(z2))))
    Z = np.abs(np.fft.fft(z2, nfft))
    k = int(np.argmax(Z))
    f2 = k / nfft * fsr
    if f2 > fsr / 2:
        f2 -= fsr
    if abs(f2) > 50.0:                    # pilot lock failed: correct the residual frequency
        z = z * np.exp(-1j * np.pi * f2 * np.arange(len(z)) / fsr)
    theta = 0.5 * np.angle(np.mean(z * z))
    b = np.real(z * np.exp(-1j * theta)).astype(np.float32)
    # biphase: bit k = sign(sum first half - sum second half) at the best clock phase
    L = 192
    nb = (len(b) - L) // L
    best = None
    for off in range(0, L, 6):
        seg = b[off: off + nb * L].reshape(nb, L)
        v = seg[:, :96].sum(axis=1) - seg[:, 96:].sum(axis=1)
        q = float(np.mean(np.abs(v))) / (float(np.std(v)) + 1e-9)
        eye = float(np.percentile(np.abs(v), 20)) / (float(np.percentile(np.abs(v), 80)) + 1e-9)
        score = eye
        if best is None or score > best[0]:
            best = (score, off, v)
    _, off, v = best
    raw = (v > 0).astype(np.uint8)
    bits = raw[1:] ^ raw[:-1]
    # block sync: syndrome of every 26-bit window equals the offset word of a valid block
    offs_by_val = {v_: k_ for k_, v_ in RDS_OFFSETS.items()}
    synd = np.full(len(bits), -1, dtype=np.int32)
    for i in range(0, len(bits) - 26):
        synd[i] = poly_remainder(bits[i: i + 26], RDS_GEN, 10)
    names = [offs_by_val.get(int(s), "") for s in synd]
    groups, pi_votes, ptys, gtypes, ps_segs = 0, {}, {}, {}, {}
    i = 0
    n_blocks_ok = 0
    while i < len(bits) - 26 * 4:
        if names[i] == "A" and names[i + 26] == "B" and names[i + 52] in ("C", "C'") and names[i + 78] == "D":
            A = bits_to_int(bits[i: i + 16])
            B = bits_to_int(bits[i + 26: i + 42])
            C = bits_to_int(bits[i + 52: i + 68])
            D = bits_to_int(bits[i + 78: i + 94])
            groups += 1
            n_blocks_ok += 4
            pi_votes[A] = pi_votes.get(A, 0) + 1
            gt = f"{B >> 12}{'B' if (B >> 11) & 1 else 'A'}"
            gtypes[gt] = gtypes.get(gt, 0) + 1
            pty = (B >> 5) & 0x1F
            ptys[pty] = ptys.get(pty, 0) + 1
            if (B >> 12) == 0:
                seg = B & 0x3
                ps_segs[seg] = chr((D >> 8) & 0x7F) + chr(D & 0x7F)
            i += 104
        else:
            if names[i] in RDS_OFFSETS:
                n_blocks_ok += 1
            i += 1
    d.n_tried = max(nb // 104, 1)
    d.n_ok = groups
    d.fields = {"pilot_db": round(float(pilot_db), 1), "groups": groups, "blocks_ok": n_blocks_ok}
    if groups >= 2:
        pi = max(pi_votes.items(), key=lambda kv: kv[1])[0]
        pty = max(ptys.items(), key=lambda kv: kv[1])[0]
        d.ok = True
        call = rds_pi_to_callsign(pi)
        d.fields.update({"pi": f"{pi:04X}", "callsign": call, "pty": pty, "group_types": gtypes})
        ps = "".join(ps_segs.get(s, "??") for s in range(4)) if len(ps_segs) >= 2 else ""
        if ps:
            d.fields["ps"] = ps
        d.evidence = f"RDS PI {pi:04X}" + (f" ({call})" if call else "") + f", {groups} groups ({', '.join(f'{k} x{v}' for k, v in sorted(gtypes.items()))})" + (f", PS '{ps.strip()}'" if ps else "")
        d.variant = ""
    d.seconds = time.time() - t0
    return d


# =========================================================================================== Bluetooth LE advertising
BLE_ADV_AA = 0x8E89BED6
BLE_ADV_CHANNELS = {2402e6: 37, 2426e6: 38, 2480e6: 39}
BLE_PDU_TYPES = {0: "ADV_IND", 1: "ADV_DIRECT_IND", 2: "ADV_NONCONN_IND", 3: "SCAN_REQ", 4: "SCAN_RSP", 5: "CONNECT_IND", 6: "ADV_SCAN_IND", 7: "ADV_EXT_IND"}


def ble_whitening_seq(channel: int, n: int) -> np.ndarray:
    """Core spec Vol 6 Part B 3.2: x^7 + x^4 + 1, position 0 = 1, positions 1..6 = channel index MSB..LSB,
    output taken from position 6."""
    p = [1] + [(channel >> (5 - i)) & 1 for i in range(6)]
    out = np.zeros(n, dtype=np.uint8)
    for k in range(n):
        o = p[6]
        out[k] = o
        p = [o, p[0], p[1], p[2], p[3] ^ o, p[4], p[5]]
    return out


def ble_crc24(bits: np.ndarray, init: int = 0x555555) -> np.ndarray:
    """Core spec Vol 6 Part B 3.1.1 LFSR; returns the 24 transmitted CRC bits (position 23 first)."""
    state = init
    for b in bits:
        fb = ((state >> 23) & 1) ^ int(b)
        state = ((state << 1) & 0xFFFFFF)
        if fb:
            state ^= 0x00065B
    return int_to_bits(state, 24)


def ble_adv_pdu(adv_addr: bytes, pdu_type: int = 0, name: str = "", rng=None, extra_bytes: int = 0) -> bytes:
    """Legacy advertising PDU bytes (header + payload) for ADV_IND / ADV_NONCONN_IND / SCAN_RSP."""
    rng = rng or np.random.default_rng(0)
    payload = bytearray(adv_addr[::-1])           # AdvA transmitted LSB first (reverse of the printed MAC)
    ad = bytearray([2, 0x01, 0x06])               # flags
    if name:
        nb = name.encode()[:20]
        ad += bytes([len(nb) + 1, 0x09]) + nb
    if extra_bytes:
        ad += bytes([extra_bytes + 1, 0xFF]) + bytes(int(x) for x in rng.integers(0, 256, extra_bytes))
    payload += ad
    payload = payload[:37]
    tx_add = 1 if adv_addr[0] & 0xC0 == 0xC0 else 0
    header = bytes([pdu_type & 0xF | (tx_add << 6), len(payload)])
    return header + bytes(payload)


def ble_adv_bits(pdu: bytes, channel: int) -> np.ndarray:
    """Preamble + access address + whitened (PDU + CRC) in transmission order."""
    pdu_bits = bytes_to_bits_lsb(pdu)
    crc = ble_crc24(pdu_bits)
    body = np.concatenate([pdu_bits, crc])
    w = ble_whitening_seq(channel, len(body))
    aa = bytes_to_bits_lsb(BLE_ADV_AA.to_bytes(4, "little"))
    pre = np.array([(i + 1 + int(aa[0])) % 2 for i in range(8)], dtype=np.uint8)   # alternating, ends opposite to AA bit 0
    return np.concatenate([pre, aa, body ^ w])


def decode_ble_adv(y: np.ndarray, fs: float, channel: int) -> Decode:
    """GFSK 1 Mbps on an advertising channel: access address search per bit phase, de-whitening, CRC-24."""
    t0 = time.time()
    d = Decode("ble_adv")
    y8, fs8 = resample(y, fs, 8e6)
    y8 = lowpass(y8, fs8, 1.1e6)
    segs = find_bursts(y8, fs8, 4e-6, 40e-6, gap_s=2e-6)
    y8, f_off = center(y8, fs8, segs)
    y8 = lowpass(y8, fs8, 0.75e6)
    sps_ = 8
    f = inst_freq(y8, fs8)
    aa = bytes_to_bits_lsb(BLE_ADV_AA.to_bytes(4, "little"))
    d.fields["carrier_offset_hz"] = round(f_off)
    addrs, types, names = {}, {}, set()
    n_ok = 0
    for a, b in segs[:3000]:
        if b - a > int(20e-3 * fs8):
            b = a + int(20e-3 * fs8)
        fseg = remove_dc(f[a:b])
        hits = []                                   # (bit position, phase) of access addresses, all phases
        for ph in range(sps_):
            bits = slice_bits(fseg, sps_, ph, 0, (b - a) // sps_)
            for pos in find_pattern(bits, aa, 1):
                hits.append((pos * sps_ + ph, ph, bits, pos + 32))
        hits.sort()
        done_until = -1
        for samp, ph, bits, start in hits:
            if samp < done_until:
                continue
            d.n_tried += 1
            body = bits[start:]
            if len(body) < 16 + 24:
                continue
            w = ble_whitening_seq(channel, len(body))
            dw = body ^ w
            hdr = bits_lsb_to_bytes(dw[:16])
            pdu_type, length = hdr[0] & 0xF, hdr[1]
            if length > 255 or 16 + 8 * length + 24 > len(dw):
                continue
            pdu_bits = dw[: 16 + 8 * length]
            crc_rx = dw[16 + 8 * length: 16 + 8 * length + 24]
            if hamming(ble_crc24(pdu_bits), crc_rx) != 0:
                continue
            done_until = samp + (16 + 8 * length + 24) * sps_
            n_ok += 1
            payload = bits_lsb_to_bytes(pdu_bits[16:])
            types[BLE_PDU_TYPES.get(pdu_type, str(pdu_type))] = types.get(BLE_PDU_TYPES.get(pdu_type, str(pdu_type)), 0) + 1
            if pdu_type in (0, 1, 2, 4, 6) and len(payload) >= 6:
                adv = ":".join(f"{x:02X}" for x in payload[5::-1])
                addrs[adv] = addrs.get(adv, 0) + 1
                i = 6
                while i + 1 < len(payload):
                    ln, typ = payload[i], payload[i + 1]
                    if ln == 0 or i + 1 + ln > len(payload):
                        break
                    if typ in (0x08, 0x09):
                        try:
                            names.add(payload[i + 2: i + 1 + ln].decode("utf-8", "replace"))
                        except Exception:  # noqa: BLE001
                            pass
                    i += 1 + ln
    d.n_ok = n_ok
    d.ok = n_ok >= 1
    d.fields.update({"channel": channel, "adv_addresses": [k for k, _ in sorted(addrs.items(), key=lambda kv: -kv[1])[:12]], "pdu_types": types,
                "names": sorted(names)[:8], "bursts": len(segs)})
    if d.ok:
        d.evidence = f"{n_ok} LE advertising PDUs with valid CRC on ch {channel} (" + ", ".join(f"{k} x{v}" for k, v in sorted(types.items())) + \
            f"), {len(addrs)} advertisers" + (f", names {', '.join(sorted(names)[:3])}" if names else "")
        d.variant = "1M"
    d.seconds = time.time() - t0
    return d


# =========================================================================================== LTE PSS / SSS
LTE_FS = 1.92e6
LTE_N = 128
LTE_CP = 9                     # normal CP at 1.92 MSps (symbols 1-6 of a slot); symbol 0 has 10
LTE_SYM = LTE_N + LTE_CP
LTE_HALF_FRAME = 9600          # 5 ms


def lte_pss_freq(nid2: int) -> np.ndarray:
    u = (25, 29, 34)[nid2]
    n = np.arange(62)
    d = np.where(n <= 30, np.exp(-1j * np.pi * u * n * (n + 1) / 63), np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / 63))
    return d.astype(np.complex128)


def lte_place_center(seq62: np.ndarray, N: int) -> np.ndarray:
    """Place 62 values on subcarriers -31..-1, 1..31 of an N-point grid (bin 0 = DC)."""
    X = np.zeros(N, dtype=np.complex128)
    k = np.concatenate([np.arange(-31, 0), np.arange(1, 32)])
    X[k % N] = seq62
    return X


def lte_pss_time(nid2: int) -> np.ndarray:
    return np.fft.ifft(lte_place_center(lte_pss_freq(nid2), LTE_N)) * math.sqrt(LTE_N)


def _mseq(taps_fn) -> np.ndarray:
    x = [0, 0, 0, 0, 1]
    for i in range(26):
        x.append(taps_fn(x, i))
    return 1 - 2 * np.array(x, dtype=np.float64)


_S = _mseq(lambda x, i: (x[i + 2] + x[i]) % 2)
_C = _mseq(lambda x, i: (x[i + 3] + x[i]) % 2)
_Z = _mseq(lambda x, i: (x[i + 4] + x[i + 2] + x[i + 1] + x[i]) % 2)


def lte_sss_m0m1(nid1: int) -> tuple[int, int]:
    qp = nid1 // 30
    q = (nid1 + qp * (qp + 1) // 2) // 30
    mp = nid1 + q * (q + 1) // 2
    m0 = mp % 31
    m1 = (m0 + mp // 31 + 1) % 31
    return m0, m1


LTE_M_TO_NID1 = {lte_sss_m0m1(n): n for n in range(168)}


def lte_sss_freq(nid1: int, nid2: int, subframe: int) -> np.ndarray:
    m0, m1 = lte_sss_m0m1(nid1)
    n = np.arange(31)
    s0 = _S[(n + m0) % 31]
    s1 = _S[(n + m1) % 31]
    c0 = _C[(n + nid2) % 31]
    c1 = _C[(n + nid2 + 3) % 31]
    z0 = _Z[(n + (m0 % 8)) % 31]
    z1 = _Z[(n + (m1 % 8)) % 31]
    d = np.zeros(62, dtype=np.complex128)
    if subframe == 0:
        d[0::2] = s0 * c0
        d[1::2] = s1 * c1 * z0
    else:
        d[0::2] = s1 * c0
        d[1::2] = s0 * c1 * z1
    return d


def lte_sync_overwrite(nid1: int, nid2: int, n_fft: int, scs_hz: float, cp_s: float):
    """For synth_modulators.ofdm(sym_fn=...): a function (symbol index -> {bin: value}) that writes PSS / SSS into
    symbols 6 / 5 of subframes 0 and 5 (FDD, normal CP: 14 symbols per 1 ms subframe, 140 per frame)."""
    pss = lte_pss_freq(nid2)
    sss = {0: lte_sss_freq(nid1, nid2, 0), 5: lte_sss_freq(nid1, nid2, 5)}
    k = np.concatenate([np.arange(-31, 0), np.arange(1, 32)]) % n_fft

    def fn(s: int) -> dict:
        r = s % 140
        if r % 70 == 6:
            return {int(kk): v for kk, v in zip(k, pss)}
        if r % 70 == 5:
            return {int(kk): v for kk, v in zip(k, sss[0 if r < 70 else 5])}
        return {}
    return fn


def decode_lte(y: np.ndarray, fs: float, min_peaks: int = 2) -> Decode:
    """PSS correlation at 1.92 MSps (3 roots), 5 ms repetition check, SSS decode (FDD: symbol before PSS; TDD: three
    symbols before) with the PSS channel estimate -> NID1 / NID2 / PCI."""
    t0 = time.time()
    d = Decode("lte")
    x, fsl = resample(y, fs, LTE_FS)
    n = len(x)
    if n < 3 * LTE_HALF_FRAME:
        d.seconds = time.time() - t0
        return d
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    pss_t = [lte_pss_time(k) for k in range(3)]
    # normalised correlation via FFT
    nfft = 1 << int(math.ceil(math.log2(n + LTE_N)))
    X = np.fft.fft(x, nfft)
    pw = np.convolve(np.abs(x) ** 2, np.ones(LTE_N), mode="full")[LTE_N - 1: LTE_N - 1 + n - LTE_N + 1]
    best = None
    corr = []
    for k in range(3):
        P = np.fft.fft(np.conj(pss_t[k][::-1]), nfft)
        c = np.fft.ifft(X * P)[LTE_N - 1: n]                   # c[i] = sum_j x[i+j] conj(p[j])
        c = np.abs(c[: len(pw)]) ** 2 / (pw * LTE_N + 1e-12)   # 1 for a perfect match
        corr.append(c)
        i = int(np.argmax(c))
        if best is None or c[i] > best[0]:
            best = (float(c[i]), k, i)
    peak, nid2, i0 = best
    d.fields = {"pss_peak": round(peak, 3), "nid2": nid2}
    if peak < 0.08:
        d.seconds = time.time() - t0
        return d
    c = corr[nid2]
    # all occurrences of this PSS: local maxima above half the best peak
    cand = np.flatnonzero(c > 0.5 * peak)
    peaks = []
    for i in cand:
        lo, hi = max(0, i - 20), min(len(c), i + 21)
        if c[i] >= c[lo:hi].max():
            peaks.append(int(i))
    # 5 ms repetition: positions congruent to i0 modulo the half-frame. The period is measured from the peak spacings
    # (a sample-rate error of 0.5 % moves it by 48 samples) and must lie within 1 % of 9600 samples.
    period = float(LTE_HALF_FRAME)
    if len(peaks) >= 3:
        dd = np.diff(np.sort(peaks)).astype(float)
        near = dd[(dd > 0.99 * LTE_HALF_FRAME) & (dd < 1.01 * LTE_HALF_FRAME)]
        if len(near) >= 2:
            period = float(np.median(near))
    rep = [p for p in peaks if abs(((p - i0 + period / 2) % period) - period / 2) <= 6]
    d.fields["half_frame_samples"] = round(period, 1)
    d.n_tried = len(peaks)
    d.fields["pss_repeats"] = len(rep)
    if len(rep) < min_peaks:
        d.seconds = time.time() - t0
        return d
    # SSS: coherent with the PSS channel estimate, two duplex hypotheses
    kbins = np.concatenate([np.arange(-31, 0), np.arange(1, 32)]) % LTE_N
    pss_f = lte_pss_freq(nid2)
    votes = {}
    for p in rep[:12]:
        if p < 3 * LTE_SYM + 2 or p + LTE_N > n:
            continue
        H = np.fft.fft(x[p: p + LTE_N])[kbins] / pss_f
        for duplex, off in (("FDD", LTE_SYM), ("TDD", 3 * LTE_SYM + 1)):
            q = p - off
            if q < 0:
                continue
            S = np.fft.fft(x[q: q + LTE_N])[kbins] / (H + 1e-9)
            even, odd = S[0::2], S[1::2]
            nn = np.arange(31)
            c0 = _C[(nn + nid2) % 31]
            c1 = _C[(nn + nid2 + 3) % 31]
            # a residual carrier offset rotates the whole SSS symbol against the PSS channel estimate by one common
            # phase: correlate the even half complex, take its phase as the reference for the odd half
            sc = np.array([np.sum(even * c0 * _S[(nn + m) % 31]) for m in range(31)])
            m_e = int(np.argmax(np.abs(sc)))
            ref = np.exp(-1j * np.angle(sc[m_e]))
            o = np.real(odd * c1 * _Z[(nn + (m_e % 8)) % 31] * ref)
            so = np.array([np.sum(o * _S[(nn + m) % 31]) for m in range(31)])
            m_o = int(np.argmax(so))
            sca = np.abs(sc)
            margin = min(sca[m_e] / (np.sort(sca)[-2] + 1e-9), so[m_o] / (max(np.sort(so)[-2], 1e-9)))
            for sf, (m0, m1) in ((0, (m_e, m_o)), (5, (m_o, m_e))):
                nid1 = LTE_M_TO_NID1.get((m0, m1))
                if nid1 is None:
                    continue
                key = (nid1, duplex)
                votes.setdefault(key, []).append(float(margin))
    if not votes:
        d.seconds = time.time() - t0
        return d
    (nid1, duplex), margins = max(votes.items(), key=lambda kv: (len(kv[1]), np.mean(kv[1])))
    n_votes = len(margins)
    pci = 3 * nid1 + nid2
    d.n_ok = n_votes
    d.fields.update({"nid1": nid1, "pci": pci, "duplex": duplex, "sss_margin": round(float(np.median(margins)), 2),
                     "votes": {f"{k[0]}/{k[1]}": len(v) for k, v in votes.items()}})
    total = sum(len(v) for v in votes.values())
    d.ok = n_votes >= 2 and n_votes >= 0.6 * total and float(np.median(margins)) >= 1.6
    if d.ok:
        d.evidence = f"LTE PCI {pci} (NID1 {nid1}, NID2 {nid2}, {duplex}), PSS peak {peak:.2f} repeating every 5 ms x{len(rep)}, SSS agreed {n_votes}/{total}"
        d.variant = "LTE"
    d.seconds = time.time() - t0
    return d


# =========================================================================================== POCSAG / FLEX
POCSAG_GEN = 0x769                 # BCH(31,21) x^10 + x^9 + x^8 + x^6 + x^5 + x^3 + 1
POCSAG_SYNC = 0x7CD215D8
POCSAG_IDLE = 0x7A89C197
POCSAG_RATES = (1200.0, 2400.0, 512.0)
PAGING_FS = 76800.0                # 64 / 32 / 150 samples per bit at 1200 / 2400 / 512; 48 at 1600 (FLEX)
FLEX_MARKER = 0xA6C6AAAA
FLEX_MODES = {0x870C: ("1600/2", 1600, 2), 0xB068: ("1600/4", 1600, 4), 0x7B18: ("3200/2", 3200, 2), 0xDEA0: ("3200/4", 3200, 4), 0x4C7C: ("3200/4", 3200, 4)}


def bch3121_codeword(msg21: int) -> int:
    """21 info bits -> 32-bit codeword: BCH(31,21) check bits + even parity (POCSAG / FLEX)."""
    bits = int_to_bits(msg21, 21)
    rem = poly_remainder(np.concatenate([bits, np.zeros(10, dtype=np.uint8)]), POCSAG_GEN, 10)
    cw31 = (msg21 << 10) | rem
    parity = bin(cw31).count("1") & 1
    return (cw31 << 1) | parity


def bch3121_check(cw32: int) -> bool:
    bits = int_to_bits(cw32 >> 1, 31)
    return poly_remainder(bits, POCSAG_GEN, 10) == 0 and (bin(cw32).count("1") & 1) == 0


def pocsag_address_codeword(ric: int, func: int) -> int:
    return bch3121_codeword(((ric >> 3) << 2) | (func & 3))      # bit 0 of the 21 = 0 (address) is the MSB here


def pocsag_message_codeword(data20: int) -> int:
    return bch3121_codeword((1 << 20) | (data20 & 0xFFFFF))


def pocsag_stream(rng, n_batches: int = 4, preamble_bits: int = 576) -> tuple[np.ndarray, list[int]]:
    """Preamble + n batches (sync + 8 frames x 2 codewords) with a few pages; returns (bits, RICs)."""
    rics = []
    words = []
    for _ in range(n_batches):
        words.append(POCSAG_SYNC)
        frame_words = [POCSAG_IDLE] * 16
        n_pages = int(rng.integers(1, 4))
        for _ in range(n_pages):
            ric = int(rng.integers(1000, 2 ** 21 - 1))
            fr = ric & 7
            if frame_words[2 * fr] != POCSAG_IDLE:
                continue
            frame_words[2 * fr] = pocsag_address_codeword(ric, int(rng.integers(0, 4)))
            rics.append(ric)
            k = 2 * fr + 1
            for _ in range(int(rng.integers(1, 5))):
                if k >= 16:
                    break
                frame_words[k] = pocsag_message_codeword(int(rng.integers(0, 1 << 20)))
                k += 1
        words += frame_words
    bits = [np.array([(i + 1) % 2 for i in range(preamble_bits)], dtype=np.uint8)] + [int_to_bits(w, 32) for w in words]
    return np.concatenate(bits), rics


def flex_fiw(cycle: int, frame: int) -> int:
    """Frame Information Word: 4-bit checksum (nibble sum = 0xF) | cycle 4 | frame 7 | fix3 6 (21 bits) -> BCH 32."""
    fiw = (cycle << 4) | (frame << 8)          # fix3 = 0
    s = 0
    for sh in (4, 8, 12, 16):
        s += (fiw >> sh) & 0xF
    s += (fiw >> 20) & 1
    chk = (0xF - s) & 0xF
    fiw |= chk
    return bch3121_codeword(fiw & 0x1FFFFF)


def flex_frame_bits(cycle: int, frame: int, rng, mode_word: int = 0x870C) -> np.ndarray:
    """Sync 1 (bit sync + A + marker + ~A) + FIW (dotting + word) + the rest of the 1.875 s frame as random 1600 bps bits."""
    bitsync = np.array([(i % 2) for i in range(32)], dtype=np.uint8)
    A = int_to_bits(mode_word, 16)
    sync = np.concatenate([bitsync, A, int_to_bits(FLEX_MARKER, 32), 1 - A])
    dotting = np.array([(i % 2) for i in range(16)], dtype=np.uint8)
    fiw = int_to_bits(flex_fiw(cycle, frame), 32)
    head = np.concatenate([sync, dotting, fiw])
    n_total = 3000                                 # 1.875 s x 1600
    rest = rng.integers(0, 2, n_total - len(head)).astype(np.uint8)
    return np.concatenate([head, rest])


def _paging_bits_all_phases(f: np.ndarray, sps_: int):
    """Bit streams for every bit phase (step sps_/8) and both polarities."""
    out = []
    for ph in range(0, sps_, max(1, sps_ // 8)):
        b = slice_bits(f, sps_, ph, 0, (len(f) - ph) // sps_)
        out.append((ph, False, b))
        out.append((ph, True, 1 - b))
    return out


def decode_pocsag(y: np.ndarray, fs: float) -> Decode:
    t0 = time.time()
    d = Decode("pocsag")
    yr, fsr = resample(y, fs, PAGING_FS)
    yr = lowpass(yr, fsr, 30e3)
    segs = find_bursts(yr, fsr, 2e-3, 0.08)
    yr, f_off = center(yr, fsr, segs)
    f = inst_freq(lowpass(yr, fsr, 10e3), fsr)
    d.fields["carrier_offset_hz"] = round(f_off)
    sync = int_to_bits(POCSAG_SYNC, 32)
    best = None
    for a, b in segs[:20]:
        fseg = remove_dc(f[a:b])
        for rate in POCSAG_RATES:
            sps_ = int(round(PAGING_FS / rate))
            for ph, inv, bits in _paging_bits_all_phases(fseg, sps_):
                pos = find_pattern(bits, sync, 2)
                if not pos:
                    continue
                # decode batches from every sync position
                n_ok, n_idle, addrs, n_cw = 0, 0, {}, 0
                i = 0
                starts = [p for p in pos]
                used = set()
                for p in starts:
                    if any(abs(p - u) < 544 for u in used):
                        continue
                    used.add(p)
                    q = p + 32
                    for fr in range(8):
                        for k in range(2):
                            if q + 32 > len(bits):
                                break
                            cw = bits_to_int(bits[q: q + 32])
                            q += 32
                            n_cw += 1
                            if bch3121_check(cw):
                                n_ok += 1
                                if cw == POCSAG_IDLE:
                                    n_idle += 1
                                elif not (cw >> 31) & 1:
                                    ric = (((cw >> 13) & 0x3FFFF) << 3) | fr
                                    func = (cw >> 11) & 3
                                    addrs[(ric, func)] = addrs.get((ric, func), 0) + 1
                if best is None or n_ok > best[0]:
                    best = (n_ok, rate, n_cw, n_idle, addrs, len(used), inv)
    if best is None:
        d.seconds = time.time() - t0
        return d
    n_ok, rate, n_cw, n_idle, addrs, n_sync, inv = best
    d.n_ok, d.n_tried = n_ok, n_cw
    d.fields.update({"baud": rate, "syncs": n_sync, "codewords_ok": n_ok, "codewords": n_cw, "idle": n_idle,
                "addresses": [{"ric": r, "func": fn, "n": n} for (r, fn), n in sorted(addrs.items(), key=lambda kv: -kv[1])[:12]], "inverted": inv})
    d.ok = n_sync >= 1 and n_ok >= 6
    if d.ok:
        d.evidence = f"POCSAG {rate:g} bps: {n_sync} sync, {n_ok}/{n_cw} codewords pass BCH ({n_idle} idle), {len(addrs)} addresses" + \
            (f" e.g. RIC {list(addrs)[0][0]}" if addrs else "")
        d.variant = "POCSAG"
    d.seconds = time.time() - t0
    return d


def decode_flex(y: np.ndarray, fs: float) -> Decode:
    t0 = time.time()
    d = Decode("flex")
    yr, fsr = resample(y, fs, PAGING_FS)
    yr = lowpass(yr, fsr, 30e3)
    segs = find_bursts(yr, fsr, 2e-3, 0.08)
    yr, f_off = center(yr, fsr, segs)
    f = inst_freq(lowpass(yr, fsr, 10e3), fsr)
    d.fields["carrier_offset_hz"] = round(f_off)
    marker = int_to_bits(FLEX_MARKER, 32)
    sps_ = 48
    found = []
    for a, b in segs[:20]:
        fseg = remove_dc(f[a:b])
        for ph, inv, bits in _paging_bits_all_phases(fseg, sps_):
            for p in find_pattern(bits, marker, 3):
                if p < 16 or p + 32 + 16 + 16 + 32 > len(bits):
                    continue
                A = bits[p - 16: p]
                Abar = bits[p + 32: p + 48]
                if hamming(A, 1 - Abar) > 3:
                    continue
                aw = bits_to_int(A)
                mode = min(FLEX_MODES.items(), key=lambda kv: bin(kv[0] ^ aw).count("1"))
                if bin(mode[0] ^ aw).count("1") > 3:
                    continue
                fiw = bits_to_int(bits[p + 48 + 16: p + 48 + 48])
                if not bch3121_check(fiw):
                    continue
                w = fiw >> 11
                chk = (sum((w >> sh) & 0xF for sh in (0, 4, 8, 12, 16)) + ((w >> 20) & 1)) & 0xF
                cyc, frm = (w >> 4) & 0xF, (w >> 8) & 0x7F
                if cyc > 14:                       # BCH(31,21) + parity is the integrity check; the nibble checksum is reported
                    continue
                found.append({"mode": mode[1][0], "a_word": f"{aw:04X}", "cycle": cyc, "frame": frm, "checksum_ok": chk == 0xF, "inverted": inv,
                              "t_s": round((a + (p + ph) * sps_) / fsr, 4)})
                break
            if found:
                break
        if found:
            break
    d.n_tried = len(segs)
    d.n_ok = len(found)
    d.ok = bool(found)
    d.fields["frames"] = found[:4]
    if d.ok:
        fr = found[0]
        d.evidence = f"FLEX sync A={fr['a_word']} ({fr['mode']}), FIW passes BCH{' + checksum' if fr['checksum_ok'] else ''}: cycle {fr['cycle']}, frame {fr['frame']}"
        d.variant = "FLEX"
    d.seconds = time.time() - t0
    return d


# =========================================================================================== Itron ERT (SCM / SCM+)
ERT_CHIP_RATE = 32768.0
ERT_FS = 524288.0                  # 16 samples per chip, 32 per bit
SCM_PREAMBLE = int("111110010101001100000", 2)
SCMPLUS_PREAMBLE = 0x16A3


def scm_packet(ert_id: int, ert_type: int, consumption: int, tamper_phy: int = 0, tamper_enc: int = 0) -> np.ndarray:
    """96 bits: preamble 21 | id[25:24] 2 | reserved 1 | tamper_phy 2 | type 4 | tamper_enc 2 | consumption 24 | id[23:0] 24 | BCH 16."""
    bits = np.concatenate([int_to_bits(SCM_PREAMBLE, 21), int_to_bits((ert_id >> 24) & 3, 2), np.zeros(1, dtype=np.uint8), int_to_bits(tamper_phy, 2),
                           int_to_bits(ert_type, 4), int_to_bits(tamper_enc, 2), int_to_bits(consumption & 0xFFFFFF, 24), int_to_bits(ert_id & 0xFFFFFF, 24)])
    data = bits_to_int(bits).to_bytes(10, "big")                    # 80 bits
    crc = crc16_poly(data[2:10], 0x6F63, 0)
    return np.concatenate([bits, int_to_bits(crc, 16)])


def scm_parse(bits96: np.ndarray) -> dict | None:
    data = bits_to_int(bits96).to_bytes(12, "big")
    if crc16_poly(data[2:12], 0x6F63, 0) != 0:
        return None
    b = bits96
    ert_id = (bits_to_int(b[21:23]) << 24) | bits_to_int(b[56:80])
    return {"kind": "SCM", "id": ert_id, "type": bits_to_int(b[26:30]), "tamper_phy": bits_to_int(b[24:26]), "tamper_enc": bits_to_int(b[30:32]),
            "consumption": bits_to_int(b[32:56])}


def scmplus_packet(endpoint_id: int, endpoint_type: int, consumption: int, tamper: int = 0, xorout: int = 0x1D0F) -> np.ndarray:
    body = bytes([0x16, 0xA3, 0x1E, endpoint_type & 0xFF]) + endpoint_id.to_bytes(4, "big") + consumption.to_bytes(4, "big") + tamper.to_bytes(2, "big")
    crc = crc16_ccitt_false(body[2:], 0xFFFF) ^ xorout
    return np.concatenate([bytes_to_bits_msb(body), int_to_bits(crc, 16)])


def bytes_to_bits_msb(data: bytes) -> np.ndarray:
    return np.concatenate([int_to_bits(b, 8) for b in data]).astype(np.uint8)


def scmplus_parse(bits128: np.ndarray) -> dict | None:
    data = bits_to_int(bits128).to_bytes(16, "big")
    if data[2] != 0x1E:
        return None
    crc_rx = int.from_bytes(data[14:16], "big")
    c = crc16_ccitt_false(data[2:14], 0xFFFF)
    if c == crc_rx:
        form = "ccitt"
    elif (c ^ 0x1D0F) == crc_rx:
        form = "ccitt^1D0F"
    else:
        return None
    eid = int.from_bytes(data[4:8], "big")
    if eid == 0:
        return None
    return {"kind": "SCM+", "id": eid, "type": data[3], "consumption": int.from_bytes(data[8:12], "big"), "tamper": int.from_bytes(data[12:14], "big"), "crc_form": form}


def ert_manchester_chips(bits: np.ndarray) -> np.ndarray:
    """bit 1 -> (1, 0), bit 0 -> (0, 1) chips (the rtlamr matched filter convention)."""
    out = np.zeros(2 * len(bits), dtype=np.float32)
    out[0::2] = bits
    out[1::2] = 1 - bits
    return out


def decode_ert(y: np.ndarray, fs: float) -> Decode:
    t0 = time.time()
    d = Decode("ert")
    yr, fsr = resample(y, fs, ERT_FS)
    yr = lowpass(yr, fsr, 90e3)
    segs = find_bursts(yr, fsr, 0.4e-3, 2.5e-3, gap_s=1e-3)      # smoothing longer than a chip: Manchester must not fragment the packet
    yr, f_off = center(yr, fsr, segs)
    yr = lowpass(yr, fsr, 60e3)
    m = np.abs(yr).astype(np.float32)
    d.fields["carrier_offset_hz"] = round(f_off)
    pre_scm = int_to_bits(SCM_PREAMBLE, 21)
    pre_plus = int_to_bits(SCMPLUS_PREAMBLE, 16)
    pkts = []
    spb = 32
    rates = []
    for a, b in segs[:200]:
        a = max(0, a - 2 * spb)
        seg = m[a: b + 2 * spb]
        # meters run a few tenths of a percent off 32.768 kchip/s: measure the chip rate on the envelope and resample
        e = seg - seg.mean()
        nfft = 1 << int(math.ceil(math.log2(max(len(e), 1024) * 4)))
        E = np.abs(np.fft.rfft(e, nfft))
        fr = np.fft.rfftfreq(nfft, 1 / fsr)
        band = (fr >= 0.94 * ERT_CHIP_RATE) & (fr <= 1.06 * ERT_CHIP_RATE)
        if band.any():
            k = int(np.argmax(E[band]))
            cr = float(fr[band][k])
            if E[band][k] > 3 * np.median(E[(fr > 10e3) & (fr < 60e3)]):
                seg = resample(seg.astype(np.float32), fsr, fsr * ERT_CHIP_RATE / cr)[0]
                rates.append(cr)
        n_bits = len(seg) // spb - 1
        if n_bits < 60:
            continue
        d.n_tried += 1
        got = False
        for ph in range(0, spb, 2):
            x = seg[ph: ph + n_bits * spb].reshape(n_bits, spb)
            v = x[:, :16].mean(axis=1) - x[:, 16:].mean(axis=1)
            for inv in (False, True):
                bits = (v < 0).astype(np.uint8) if inv else (v > 0).astype(np.uint8)
                for p in find_pattern(bits, pre_scm, 1):
                    if p + 96 <= len(bits):
                        r = scm_parse(bits[p: p + 96])
                        if r:
                            pkts.append(r)
                            got = True
                            break
                if not got:
                    for p in find_pattern(bits, pre_plus, 0):
                        if p + 128 <= len(bits):
                            r = scmplus_parse(bits[p: p + 128])
                            if r:
                                pkts.append(r)
                                got = True
                                break
                if got:
                    break
            if got:
                break
    d.n_ok = len(pkts)
    d.ok = bool(pkts)
    ids = {}
    for p in pkts:
        ids[(p["kind"], p["id"])] = ids.get((p["kind"], p["id"]), 0) + 1
    d.fields.update({"packets": pkts[:12], "bursts": len(segs), "chip_rate_hz": (round(float(np.median(rates))) if rates else None)})
    if d.ok:
        kinds = sorted({p["kind"] for p in pkts})
        d.evidence = f"{len(pkts)} ERT {'/'.join(kinds)} packets pass the checksum: " + ", ".join(f"{k} id {i} (type {next(p['type'] for p in pkts if p['id'] == i)})" for (k, i), _ in list(ids.items())[:3])
        d.variant = "ERT_meter"
    d.seconds = time.time() - t0
    return d


# =========================================================================================== P25 / DMR (4FSK 4800)
LMR_FS = 48000.0                   # 10 samples per symbol
P25_FS = 0x5575F5FF77FF
P25_BCH_GEN = 0xCD930BDD3B2B       # BCH(63,16,23) generator (degree 47), primitive x^6 + x + 1
DIBIT_TO_SYM = {0b01: 3, 0b00: 1, 0b10: -1, 0b11: -3}
SYM_TO_DIBIT = {3: 0b01, 1: 0b00, -1: 0b10, -3: 0b11}
DMR_SYNCS = {"BS_voice": 0x755FD7DF75F7, "BS_data": 0xDFF57D75DF5D, "MS_voice": 0x7F7D5DD57DFD, "MS_data": 0xD5D7F77FD757, "MS_RC": 0x77D55F7DFD77,
             "DM_TS1_voice": 0x5D577F7757FF, "DM_TS1_data": 0xF7FDD5DDFD55, "DM_TS2_voice": 0x7DFFD5F55D5F, "DM_TS2_data": 0xD7557F5FF7F5}
P25_DUID = {0: "HDU", 3: "TDU", 5: "LDU1", 7: "TSDU", 10: "LDU2", 12: "PDU", 15: "TDULC"}


def dibits_to_symbols(bits: np.ndarray) -> np.ndarray:
    return np.array([DIBIT_TO_SYM[(int(bits[i]) << 1) | int(bits[i + 1])] for i in range(0, len(bits) - 1, 2)], dtype=np.float64)


def symbols_to_bits(sym: np.ndarray) -> np.ndarray:
    out = np.zeros(2 * len(sym), dtype=np.uint8)
    for i, s in enumerate(sym):
        db = SYM_TO_DIBIT[int(s)]
        out[2 * i], out[2 * i + 1] = db >> 1, db & 1
    return out


def p25_nid_codeword(nac: int, duid: int) -> np.ndarray:
    """64 bits: BCH(63,16) codeword of NAC (12) | DUID (4) plus one parity bit."""
    info = int_to_bits(((nac & 0xFFF) << 4) | (duid & 0xF), 16)
    rem = poly_remainder(np.concatenate([info, np.zeros(47, dtype=np.uint8)]), P25_BCH_GEN, 47)
    cw = np.concatenate([info, int_to_bits(rem, 47)])
    parity = int(cw.sum()) & 1
    return np.concatenate([cw, np.array([parity], dtype=np.uint8)])


def p25_frame_symbols(nac: int, duid: int, n_payload_dibits: int, rng) -> np.ndarray:
    """FS + NID + random payload with status dibits inserted after every 35 dibits (from the frame start)."""
    data = np.concatenate([dibits_to_symbols(int_to_bits(P25_FS, 48)), dibits_to_symbols(p25_nid_codeword(nac, duid)),
                           dibits_to_symbols(rng.integers(0, 2, 2 * n_payload_dibits).astype(np.uint8))])
    out = []
    for i, s in enumerate(data):
        out.append(s)
        if (len(out) + 1) % 36 == 0:                 # position 35, 71, ... (0-based) is a status symbol
            out.append(3.0)                          # status 01 = inbound / idle
    return np.array(out, dtype=np.float64)


def p25_strip_status(sym: np.ndarray) -> np.ndarray:
    """Remove status symbols (every 36th symbol counting from the frame sync start)."""
    return np.array([s for i, s in enumerate(sym) if (i + 1) % 36 != 0])


def dmr_burst_symbols(sync_name: str, rng) -> np.ndarray:
    """One 264-bit burst: 108 payload bits + 48 sync bits + 108 payload bits (132 dibits, 27.5 ms)."""
    sync = dibits_to_symbols(int_to_bits(DMR_SYNCS[sync_name], 48))
    pay = lambda: dibits_to_symbols(rng.integers(0, 2, 108).astype(np.uint8))  # noqa: E731
    return np.concatenate([pay(), sync, pay()])


def _lmr_symbols_all_phases(f: np.ndarray, sps_: int):
    """Soft 4-level symbols for each symbol phase, normalised so the outer levels sit at +-3."""
    out = []
    n = (len(f) - sps_) // sps_
    if n < 40:
        return out
    for ph in range(sps_):
        s = f[ph: ph + n * sps_].reshape(n, sps_)[:, sps_ // 4: sps_ - sps_ // 4].mean(axis=1)
        outer = np.percentile(np.abs(s), 92)
        if outer <= 0:
            continue
        out.append((ph, s * (3.0 / outer)))
    return out


def _sync_corr(sym: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Cosine similarity (1 = exact) of a +-3 sync pattern against every window of the soft symbol stream: scale
    invariant, so a wrong level normalisation (bursts merged with noise gaps) cannot push a wrong pattern above the
    right one."""
    n = len(pattern)
    if len(sym) < n:
        return np.zeros(0)
    c = np.correlate(sym, pattern, mode="valid")
    e = np.convolve(sym * sym, np.ones(n), mode="valid")
    return c / (np.sqrt(e * float(np.sum(pattern * pattern))) + 1e-9)


def decode_p25(y: np.ndarray, fs: float) -> Decode:
    t0 = time.time()
    d = Decode("p25")
    yr, fsr = resample(y, fs, LMR_FS)
    yr = lowpass(yr, fsr, 20e3)
    segs = find_bursts(yr, fsr, 2e-3, 0.05)
    yr, f_off = center(yr, fsr, segs)
    f = inst_freq(lowpass(yr, fsr, 7.5e3), fsr)
    d.fields["carrier_offset_hz"] = round(f_off)
    sps_ = 10
    fs_pat = dibits_to_symbols(int_to_bits(P25_FS, 48))
    nids, n_fs, n_ok = {}, 0, 0
    for a, b in segs[:30]:
        fseg = f[a:b] - float(np.mean(f[a:b]))
        best = None
        for ph, s in _lmr_symbols_all_phases(fseg, sps_):
            c = _sync_corr(s, fs_pat)
            if len(c) == 0:
                continue
            i = int(np.argmax(c))
            if best is None or c[i] > best[0]:
                best = (float(c[i]), ph, s, c)
        if best is None or best[0] < 0.8:
            continue
        _, ph, s, c = best
        d.n_tried += 1
        peaks = [int(i) for i in np.flatnonzero(c >= 0.8) if c[i] >= c[max(0, i - 5): i + 6].max()]
        for p in peaks:
            n_fs += 1
            seg = s[p: p + 24 + 33]
            if len(seg) < 57:
                continue
            seg = p25_strip_status(seg)[24: 56]
            if len(seg) < 32:
                continue
            levels = np.array([3, 1, -1, -3], dtype=np.float64)
            hard = levels[np.argmin(np.abs(seg[:, None] - levels[None, :]), axis=1)]
            bits = symbols_to_bits(hard)
            if poly_remainder(bits[:63], P25_BCH_GEN, 47) == 0 and (int(bits[:63].sum()) & 1) == int(bits[63]):
                nac, duid = bits_to_int(bits[:12]), bits_to_int(bits[12:16])
                nids[(nac, duid)] = nids.get((nac, duid), 0) + 1
                n_ok += 1
    d.n_ok = n_ok
    d.fields.update({"frame_syncs": n_fs, "nid_ok": n_ok, "nids": [{"nac": f"{k[0]:03X}", "duid": P25_DUID.get(k[1], str(k[1])), "n": v} for k, v in sorted(nids.items(), key=lambda kv: -kv[1])[:6]]})
    if n_ok >= 1:
        d.ok = True
        nac = max(nids.items(), key=lambda kv: kv[1])[0][0]
        d.evidence = f"P25 frame sync x{n_fs}, {n_ok} NID pass BCH(63,16): NAC {nac:03X} (" + ", ".join(f"{P25_DUID.get(k[1], k[1])} x{v}" for k, v in nids.items()) + ")"
        d.variant = "P25"
    elif n_fs >= 2:
        # frame syncs without one passing NID are reported but do not count as a decode (a real P25 channel at this
        # level passes the BCH within a few frames; 18 syncs with 0 NIDs on air turned out to be something else)
        d.ok, d.weak = False, True
        d.evidence = f"P25 frame sync x{n_fs} but no NID passes BCH: not accepted"
    d.seconds = time.time() - t0
    return d


def decode_dmr(y: np.ndarray, fs: float) -> Decode:
    t0 = time.time()
    d = Decode("dmr")
    yr, fsr = resample(y, fs, LMR_FS)
    yr = lowpass(yr, fsr, 20e3)
    segs = find_bursts(yr, fsr, 0.5e-3, 0.02, gap_s=1e-3)
    yr, f_off = center(yr, fsr, segs)
    f = inst_freq(lowpass(yr, fsr, 7.5e3), fsr)
    d.fields["carrier_offset_hz"] = round(f_off)
    sps_ = 10
    pats = {k: dibits_to_symbols(int_to_bits(v, 48)) for k, v in DMR_SYNCS.items()}
    hits, times = {}, []
    for a, b in segs[:200]:
        fseg = f[a:b] - float(np.mean(f[a:b]))
        d.n_tried += 1
        found = []                                  # (time, score, name) over all phases and patterns
        for ph, s in _lmr_symbols_all_phases(fseg, sps_):
            for name, pat in pats.items():
                c = _sync_corr(s, pat)
                if len(c) == 0:
                    continue
                for i in np.flatnonzero(c >= 0.9):
                    if c[i] >= c[max(0, i - 3): i + 4].max():
                        found.append(((a + i * sps_ + ph) / fsr, float(c[i]), name))
        found.sort()
        merged = []                                 # one sync per burst (27.5 ms): the strongest candidate wins
        for tt, sc, name in found:
            if merged and tt - merged[-1][0] < 0.020:
                if sc > merged[-1][1]:
                    merged[-1] = (tt, sc, name)
                continue
            merged.append((tt, sc, name))
        for tt, sc, name in merged:
            hits[name] = hits.get(name, 0) + 1
            times.append(tt)
    n = sum(hits.values())
    d.n_ok = n
    spacing = ""
    if len(times) >= 3:
        dt = np.diff(sorted(times))
        med = float(np.median(dt))
        if abs(med - 0.030) < 0.002 or abs(med - 0.060) < 0.003:
            spacing = f", bursts every {med * 1e3:.1f} ms"
    d.fields.update({"syncs": hits, "burst_spacing_s": (round(float(np.median(np.diff(sorted(times)))), 4) if len(times) >= 3 else None)})
    if n >= 2:                                      # one 48-bit sync alone is never enough
        d.ok = True
        d.weak = n < 3 and not spacing
        d.evidence = f"DMR sync x{n} (" + ", ".join(f"{k} x{v}" for k, v in hits.items()) + ")" + spacing
        d.variant = "DMR"
    d.seconds = time.time() - t0
    return d


# =========================================================================================== DECT
DECT_FS = 4.608e6                  # 4 samples per bit at 1.152 Mbps
DECT_SYNC_RFP = 0xAAAAE98A
DECT_SYNC_PP = 0x55551675
DECT_RCRC_GEN = 0x0589             # x^16 + x^10 + x^8 + x^7 + x^3 + 1
DECT_P32_BITS = 420                # S 32 + A 64 + B 320 + X 4


def dect_rcrc(bits48: np.ndarray) -> int:
    return crc_msb(bits48, DECT_RCRC_GEN, 16, 0, 0) ^ 0x0001


def dect_packet_bits(rfp: bool, rfpi: bytes | None, rng) -> np.ndarray:
    """P32 packet: S-field + A-field (header TA=3 identity with RFPI, or TA=0) + R-CRC + random B-field + X-CRC (random)."""
    s = int_to_bits(DECT_SYNC_RFP if rfp else DECT_SYNC_PP, 32)
    if rfpi is not None:
        header = (3 << 5) | (0 << 4) | (0 << 1)        # TA=3 (Nt), Q1=0, BA=000, Q2=0
        tail = bytes_to_bits_msb(rfpi)
    else:
        header = (0 << 5) | (0 << 1)                   # TA=0 Ct
        tail = rng.integers(0, 2, 40).astype(np.uint8)
    a48 = np.concatenate([int_to_bits(header, 8), tail])
    a = np.concatenate([a48, int_to_bits(dect_rcrc(a48), 16)])
    b = rng.integers(0, 2, 320 + 4).astype(np.uint8)
    return np.concatenate([s, a, b])


def decode_dect(y: np.ndarray, fs: float) -> Decode:
    t0 = time.time()
    d = Decode("dect")
    yr, fsr = resample(y, fs, DECT_FS)
    yr = lowpass(yr, fsr, 1.3e6)
    segs = find_bursts(yr, fsr, 2e-6, 150e-6, gap_s=5e-6)
    yr, f_off = center(yr, fsr, segs)
    f = inst_freq(lowpass(yr, fsr, 0.9e6), fsr)
    d.fields["carrier_offset_hz"] = round(f_off)
    sps_ = 4
    pats = {"RFP": int_to_bits(DECT_SYNC_RFP, 32), "PP": int_to_bits(DECT_SYNC_PP, 32)}
    kinds, rfpis, tas, n_sync = {}, {}, {}, 0
    n_ok = 0
    for a, b in segs[:400]:
        fseg = remove_dc(f[a:b])
        done = []                                   # sample positions of decoded packets (one per 420 bits)
        for ph in range(sps_):
            bits = slice_bits(fseg, sps_, ph, 0, (b - a) // sps_)
            for inv in (False, True):
                bb = 1 - bits if inv else bits
                for kind, pat in pats.items():
                    for p in find_pattern(bb, pat, 1):
                        samp = p * sps_ + ph
                        if p + 32 + 64 > len(bb) or any(abs(samp - q) < 400 * sps_ for q in done):
                            continue
                        n_sync += 1
                        a48 = bb[p + 32: p + 80]
                        crc_rx = bits_to_int(bb[p + 80: p + 96])
                        if dect_rcrc(a48) == crc_rx:
                            n_ok += 1
                            done.append(samp)
                            kinds[kind] = kinds.get(kind, 0) + 1
                            ta = bits_to_int(a48[:3])
                            tas[ta] = tas.get(ta, 0) + 1
                            if ta == 3:
                                rfpi = bits_to_int(a48[8:48])
                                rfpis[f"{rfpi:010X}"] = rfpis.get(f"{rfpi:010X}", 0) + 1
        d.n_tried += 1
    d.n_ok = n_ok
    d.fields.update({"bursts": len(segs), "syncs": n_sync, "kinds": kinds, "ta": {str(k): v for k, v in tas.items()}, "rfpi": list(rfpis)[:4]})
    if n_ok >= 1:
        d.ok = True
        d.evidence = f"DECT {n_ok} A-fields pass the R-CRC (" + ", ".join(f"{k} x{v}" for k, v in kinds.items()) + ")" + (f", RFPI {list(rfpis)[0]}" if rfpis else "")
        d.variant = ""
    d.seconds = time.time() - t0
    return d


# =========================================================================================== IEEE 802.15.4 (Zigbee)
ZB_CHIP_RATE = 2e6
ZB_FS = 8e6                        # 4 samples per chip
ZB_C0 = "11011001110000110101001000101110"


def zigbee_chip_table() -> np.ndarray:
    """16 x 32 chips: symbols 1-7 = symbol 0 rotated right by 4 chips each; 8-15 = 0-7 with odd chips inverted."""
    c0 = np.array([int(ch) for ch in ZB_C0], dtype=np.uint8)
    rows = [np.roll(c0, 4 * s) for s in range(8)]
    odd = np.array([i % 2 for i in range(32)], dtype=np.uint8)
    rows += [r ^ odd for r in rows]
    return np.array(rows, dtype=np.uint8)


ZB_CHIPS = zigbee_chip_table()


def zigbee_frame_bytes(payload: bytes, seq: int, pan: int, dst: int, src: int, frame_type: int = 1) -> bytes:
    """MAC frame (FCF data frame, short addresses, PAN compression) + FCS, as PSDU bytes."""
    fcf = frame_type | (1 << 6) | (2 << 10) | (2 << 14)        # PAN id compression, dst/src short addresses
    mhr = fcf.to_bytes(2, "little") + bytes([seq & 0xFF]) + pan.to_bytes(2, "little") + dst.to_bytes(2, "little") + src.to_bytes(2, "little")
    body = mhr + payload
    fcs = crc16_kermit(body)
    return body + fcs.to_bytes(2, "little")


def zigbee_ppdu_symbols(psdu: bytes) -> np.ndarray:
    """Preamble (8 x 0) + SFD 0xA7 + PHR (length) + PSDU as 4-bit symbols (low nibble first)."""
    out = [0] * 8 + [7, 10]
    for by in bytes([len(psdu) & 0x7F]) + psdu:
        out += [by & 0xF, by >> 4]
    return np.array(out, dtype=np.int64)


def zigbee_chips(symbols: np.ndarray) -> np.ndarray:
    return ZB_CHIPS[symbols].ravel()


def oqpsk_chips_waveform(fs: float, chips01: np.ndarray, chip_rate: float = ZB_CHIP_RATE) -> np.ndarray:
    """O-QPSK half-sine: even chips on I, odd chips on Q delayed by one chip (IEEE 802.15.4-2006 6.5.2)."""
    sps_ = max(2, int(round(fs / chip_rate)))
    c = chips01.astype(np.float64) * 2 - 1
    i_ch, q_ch = c[0::2], c[1::2]
    n = len(i_ch)
    hs = np.sin(np.pi * np.arange(2 * sps_) / (2 * sps_))
    ui = np.zeros(n * 2 * sps_ + 2 * sps_)
    uq = np.zeros_like(ui)
    ui[: n * 2 * sps_: 2 * sps_] = i_ch
    uq[: len(q_ch) * 2 * sps_: 2 * sps_] = q_ch[: n]
    xi = np.convolve(ui, hs, mode="full")[: len(ui)]
    xq = np.convolve(uq, hs, mode="full")[: len(uq)]
    xq = np.roll(xq, sps_)
    x = (xi + 1j * xq).astype(np.complex64)
    x = x[: n * 2 * sps_ + sps_]
    return resample(x, chip_rate * sps_, fs)[0] if abs(chip_rate * sps_ - fs) > 1e-6 * fs else x


def _zb_diff_chips(y: np.ndarray, sps_: int, ph: int) -> np.ndarray:
    """MSK-view differential chips: sign of the phase advance over one chip period, at chip phase ph."""
    a = y[ph + sps_:: sps_]
    b = y[ph: len(y) - sps_: sps_]
    n = min(len(a), len(b))
    return (np.angle(a[:n] * np.conj(b[:n])) > 0).astype(np.uint8)


def _zb_diff_table() -> np.ndarray:
    """The 16 symbols as the decoder sees them (differential chips), derived by running the ideal modulator."""
    rows = []
    sps_ = 4
    for s in range(16):
        chips = np.concatenate([ZB_CHIPS[s], ZB_CHIPS[s], ZB_CHIPS[s]])
        x = oqpsk_chips_waveform(ZB_FS, chips)
        dc = _zb_diff_chips(x, sps_, 0)
        rows.append(dc[32: 64])                      # the middle repetition: steady state
    return np.array(rows, dtype=np.uint8)


ZB_DIFF = None


def zb_diff_table() -> np.ndarray:
    global ZB_DIFF
    if ZB_DIFF is None:
        ZB_DIFF = _zb_diff_table()
    return ZB_DIFF


def _zb_symbols_from_diff(dc: np.ndarray, start: int, n_sym: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-symbol decisions (ignoring the first chip of every symbol, which depends on the previous symbol)."""
    T = zb_diff_table()
    out, errs = [], []
    for k in range(n_sym):
        seg = dc[start + 32 * k: start + 32 * k + 32]
        if len(seg) < 32:
            break
        e = np.count_nonzero(T[:, 1:] != seg[None, 1:], axis=1)
        out.append(int(np.argmin(e)))
        errs.append(int(e.min()))
    return np.array(out, dtype=np.int64), np.array(errs, dtype=np.int64)


def decode_zigbee(y: np.ndarray, fs: float) -> Decode:
    t0 = time.time()
    d = Decode("zigbee")
    yr, fsr = resample(y, fs, ZB_FS)
    yr = lowpass(yr, fsr, 1.5e6)
    sps_ = 4
    segs = find_bursts(yr, fsr, 4e-6, 200e-6, gap_s=4e-6)
    T = zb_diff_table()
    pre = np.tile(T[0], 2)
    frames, n_ok = [], 0
    for a, b in segs[:400]:
        d.n_tried += 1
        done = []
        for ph in range(sps_):
            dc = _zb_diff_chips(yr[a:b], sps_, ph)
            for inv in (False, True):
                dd = 1 - dc if inv else dc
                for p in find_pattern(dd, pre, 4):
                    # walk the preamble to the SFD
                    syms, errs = _zb_symbols_from_diff(dd, p, 12)
                    k = 0
                    while k < len(syms) and syms[k] == 0:
                        k += 1
                    if k + 2 > len(syms) or syms[k] != 7 or syms[k + 1] != 10:
                        continue
                    start = p + 32 * (k + 2)
                    if any(abs(start * sps_ + ph - q) < 16 * sps_ for q in done):      # the same frame seen from an earlier preamble symbol / phase
                        continue
                    hdr, _ = _zb_symbols_from_diff(dd, start, 2)
                    if len(hdr) < 2:
                        continue
                    length = (hdr[0] | (hdr[1] << 4)) & 0x7F
                    if length < 5:
                        continue
                    body, errs_b = _zb_symbols_from_diff(dd, start + 64, 2 * length)
                    if len(body) < 2 * length:
                        continue
                    psdu = bytes(int(body[2 * i] | (body[2 * i + 1] << 4)) for i in range(length))
                    if crc16_kermit(psdu[:-2]) != int.from_bytes(psdu[-2:], "little"):
                        continue
                    done.append(start * sps_ + ph)
                    n_ok += 1
                    fcf = int.from_bytes(psdu[:2], "little")
                    fr = {"len": length, "type": ("beacon", "data", "ack", "cmd")[fcf & 7] if (fcf & 7) < 4 else str(fcf & 7), "seq": psdu[2] if length > 2 else None,
                          "chip_err_mean": round(float(np.mean(errs_b)), 2)}
                    dam, sam = (fcf >> 10) & 3, (fcf >> 14) & 3
                    i = 3
                    if dam:
                        fr["pan"] = f"{int.from_bytes(psdu[i: i + 2], 'little'):04X}"
                        i += 2
                        fr["dst"] = psdu[i: i + (2 if dam == 2 else 8)][::-1].hex().upper()
                        i += 2 if dam == 2 else 8
                    if sam:
                        if not (fcf >> 6) & 1:
                            i += 2
                        fr["src"] = psdu[i: i + (2 if sam == 2 else 8)][::-1].hex().upper()
                    frames.append(fr)
    d.n_ok = n_ok
    d.fields = {"bursts": len(segs), "frames": frames[:12]}
    if n_ok >= 1:
        d.ok = True
        kinds = {}
        for fr in frames:
            kinds[fr["type"]] = kinds.get(fr["type"], 0) + 1
        pans = sorted({fr.get("pan", "") for fr in frames if fr.get("pan")})
        d.evidence = f"{n_ok} 802.15.4 frames pass the FCS (" + ", ".join(f"{k} x{v}" for k, v in kinds.items()) + ")" + (f", PAN {', '.join(pans[:2])}" if pans else "")
    d.seconds = time.time() - t0
    return d


# =========================================================================================== UAT 978
UAT_RATE = 1041666.67
UAT_FS = 4 * UAT_RATE
UAT_SYNC_ADSB = 0xEACDDA4E2
UAT_SYNC_UPLINK = 0x153225B1D


def uat_frame_bits(uplink: bool, n_bits: int, rng) -> np.ndarray:
    sync = int_to_bits(UAT_SYNC_UPLINK if uplink else UAT_SYNC_ADSB, 36)
    return np.concatenate([sync, rng.integers(0, 2, max(n_bits - 36, 0)).astype(np.uint8)])


def decode_uat(y: np.ndarray, fs: float) -> Decode:
    t0 = time.time()
    d = Decode("uat")
    yr, fsr = resample(y, fs, UAT_FS)
    yr = lowpass(yr, fsr, 1.1e6)
    segs = find_bursts(yr, fsr, 2e-6, 150e-6, gap_s=10e-6)
    yr, f_off = center(yr, fsr, segs)
    f = inst_freq(lowpass(yr, fsr, 0.8e6), fsr)
    d.fields["carrier_offset_hz"] = round(f_off)
    sps_ = 4
    pats = {"ADS-B": int_to_bits(UAT_SYNC_ADSB, 36), "uplink": int_to_bits(UAT_SYNC_UPLINK, 36)}
    hits, lengths = {}, []
    for a, b in segs[:400]:
        fseg = remove_dc(f[a:b])
        d.n_tried += 1
        found = []
        for ph in range(sps_):
            bits = slice_bits(fseg, sps_, ph, 0, (b - a) // sps_)
            for kind, pat in pats.items():
                for pos in find_pattern(bits, pat, 2):
                    found.append((pos * sps_ + ph, kind, len(bits) - pos - 36))
        found.sort()
        last = -10 ** 9
        for samp, kind, rest in found:
            if samp - last < 200 * sps_:
                continue
            last = samp
            hits[kind] = hits.get(kind, 0) + 1
            lengths.append(rest)
    n = sum(hits.values())
    d.n_ok = n
    d.fields.update({"bursts": len(segs), "syncs": hits, "payload_bits_p50": (int(np.median(lengths)) if lengths else None)})
    if n >= 1:
        d.ok = True
        d.weak = n < 2
        cls = ""
        if lengths:
            m = int(np.median(lengths))
            cls = " (short ADS-B)" if 200 <= m <= 300 else (" (long ADS-B)" if 340 <= m <= 450 else (" (uplink)" if m > 4000 else ""))
        d.evidence = f"UAT sync x{n} (" + ", ".join(f"{k} x{v}" for k, v in hits.items()) + ")" + cls
    d.seconds = time.time() - t0
    return d


# =========================================================================================== registry / attempt
# protocol -> (library entry ids that may carry it, band to extract (Hz), oversample, centre rule)
REGISTRY = {
    "adsb": {"entries": ["ssr_adsb_1090"], "bw": 4e6, "oversample": 2.0, "centre": "fixed:1090e6"},
    "rds": {"entries": ["fm_broadcast"], "bw": 240e3, "oversample": 1.6, "centre": "raster:200e3:87.9e6"},
    "ble_adv": {"entries": ["bluetooth_le"], "bw": 2e6, "oversample": 2.0, "centre": "set:2402e6,2426e6,2480e6:1.3e6"},
    "lte": {"entries": ["cell_600_dl", "cell_700_dl", "cell_850_dl", "cell_1900_dl", "cell_aws_dl", "cell_2500_tdd"], "bw": 1.08e6, "oversample": 1.8, "centre": "lte"},
    "pocsag": {"entries": ["paging_929"], "bw": 25e3, "oversample": 2.0, "centre": "box"},
    "flex": {"entries": ["paging_929"], "bw": 25e3, "oversample": 2.0, "centre": "box"},
    "ert": {"entries": ["ism_900_generic"], "bw": 150e3, "oversample": 2.0, "centre": "box"},
    "p25": {"entries": ["lmr_vhf_high", "lmr_uhf_450", "lmr_800_trunked", "amateur_70cm", "amateur_2m", "murs_151"], "bw": 15e3, "oversample": 3.0, "centre": "box"},
    "dmr": {"entries": ["lmr_vhf_high", "lmr_uhf_450", "lmr_800_trunked", "amateur_70cm", "amateur_2m"], "bw": 15e3, "oversample": 3.0, "centre": "box"},
    "dect": {"entries": ["dect_6_0"], "bw": 1.728e6, "oversample": 2.0, "centre": "set:1921.536e6,1923.264e6,1924.992e6,1926.72e6,1928.448e6:0.9e6"},
    "zigbee": {"entries": ["zigbee_802154"], "bw": 2.5e6, "oversample": 2.0, "centre": "raster:5e6:2405e6:1.6e6"},
    "uat": {"entries": ["uat_978"], "bw": 1.3e6, "oversample": 2.0, "centre": "fixed:978e6"},
}
DECODER_FN = {"adsb": lambda y, fs, c: decode_adsb(y, fs), "rds": lambda y, fs, c: decode_rds(y, fs), "ble_adv": lambda y, fs, c: decode_ble_adv(y, fs, c["channel"]),
              "lte": lambda y, fs, c: decode_lte(y, fs), "pocsag": lambda y, fs, c: decode_pocsag(y, fs), "flex": lambda y, fs, c: decode_flex(y, fs),
              "ert": lambda y, fs, c: decode_ert(y, fs), "p25": lambda y, fs, c: decode_p25(y, fs), "dmr": lambda y, fs, c: decode_dmr(y, fs),
              "dect": lambda y, fs, c: decode_dect(y, fs), "zigbee": lambda y, fs, c: decode_zigbee(y, fs), "uat": lambda y, fs, c: decode_uat(y, fs)}
# minimum in-band SNR (dB) at which a protocol is worth attempting (below it the decode cannot succeed; saves time)
MIN_SNR_DB = {"adsb": 6.0, "rds": 8.0, "ble_adv": 6.0, "lte": 0.0, "pocsag": 6.0, "flex": 6.0, "ert": 6.0, "p25": 6.0, "dmr": 6.0, "dect": 6.0, "zigbee": 4.0, "uat": 6.0}


def protocols_for_entry(entry_id: str, entry: dict | None = None) -> list[str]:
    if entry and entry.get("decode"):
        return [p for p in entry["decode"] if p in REGISTRY]
    return [p for p, spec in REGISTRY.items() if entry_id in spec["entries"]]


def centre_for(protocol: str, fc_box: float, bw_box: float, rate: float) -> list[float]:
    """Candidate carrier centres for the decoder from the detector box (fixed channel, raster snap, channel set)."""
    rule = REGISTRY[protocol]["centre"]
    if rule.startswith("fixed:"):
        f = float(rule.split(":")[1])
        return [f] if abs(f - fc_box) <= max(0.6 * bw_box, 1.5e6) else []
    if rule.startswith("raster:"):
        parts = rule.split(":")
        sp, off = float(parts[1]), float(parts[2])
        tol = float(parts[3]) if len(parts) > 3 else 0.5 * sp
        f = off + round((fc_box - off) / sp) * sp
        return [f] if abs(f - fc_box) <= tol else []
    if rule.startswith("set:"):
        parts = rule.split(":")
        chans = [float(x) for x in parts[1].split(",")]
        tol = float(parts[2])
        f = min(chans, key=lambda c: abs(c - fc_box))
        return [f] if abs(f - fc_box) <= tol else []
    if rule == "lte":
        # 100 kHz raster around the box centre, widened by the detector's frequency resolution
        bin_hz = rate / 1024
        span = 2 * bin_hz + 60e3
        base = round(fc_box / 100e3) * 100e3
        k = int(math.ceil(span / 100e3))
        return [base + 100e3 * i for i in range(-k, k + 1)]
    return [fc_box]


def lte_candidates(cap, fc_box: float, bw_box: float, max_cands: int = 15) -> list[float]:
    """Carrier centres inside a (possibly merged or cut) box: occupied blocks >= 1 MHz in the box's PSD, each
    snapped to the 100 kHz raster with its neighbours; falls back to the raster around the box centre."""
    bw = max(bw_box, 1.4e6)
    y, fs, _ = cap.extract(fc_box, bw, oversample=1.3)
    if len(y) < 4096:
        return centre_for("lte", fc_box, bw_box, cap.rate)
    nseg = 1 << int(round(math.log2(max(fs / 15e3, 64))))
    f, P = sps.welch(y[: min(len(y), 1 << 21)], fs=fs, nperseg=nseg, return_onesided=False)
    f, P = np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(P) + 1e-30)
    keep = np.abs(f) <= 0.5 * bw * 1.1
    f, P = f[keep], P[keep]
    floor = float(np.percentile(P, 10))
    occ = P > floor + 6.0
    bin_hz = fs / nseg
    cands: list[float] = []
    i, n = 0, len(occ)
    while i < n:
        if occ[i]:
            j = i
            gap = 0
            while j < n and (occ[j] or gap < 10):
                gap = 0 if occ[j] else gap + 1
                j += 1
            j -= gap
            width = (j - i) * bin_hz
            if width >= 1.0e6:
                lo_edge, hi_edge = fc_box + f[i], fc_box + f[j - 1]
                centres = [0.5 * (lo_edge + hi_edge)]
                # a block wider than one carrier may be several merged carriers: standard widths from either edge
                for occ_w in (4.5e6, 9.0e6, 13.5e6, 18.0e6):
                    if width >= occ_w + 2.0e6:
                        centres += [lo_edge + occ_w / 2, hi_edge - occ_w / 2]
                for centre in centres:
                    base = round(centre / 100e3) * 100e3
                    cands += [base - 100e3, base, base + 100e3]
            i = j
        else:
            i += 1
    if not cands:
        return centre_for("lte", fc_box, bw_box, cap.rate)
    out = []
    for c in cands:
        if all(abs(c - o) > 1.0 for o in out):
            out.append(c)
    return out[:max(max_cands, 30)]


def attempt(protocol: str, cap, fc_box: float, bw_box: float, snr_db: float | None = None, t0: float | None = None, t1: float | None = None) -> Decode:
    """Extract the band from a measure_descriptors.CaptureIQ and run one decoder (LTE tries its raster candidates)."""
    spec = REGISTRY[protocol]
    t_start = time.time()
    if snr_db is not None and snr_db < MIN_SNR_DB[protocol]:
        d = Decode(protocol)
        d.evidence = f"skipped (SNR {snr_db:.0f} dB)"
        d.seconds = time.time() - t_start
        return d
    cands = lte_candidates(cap, fc_box, bw_box) if protocol == "lte" else centre_for(protocol, fc_box, bw_box, cap.rate)
    if not cands:
        d = Decode(protocol)
        d.evidence = "box centre not on a decodable channel"
        return d
    best = None
    for fc in cands:
        y, fs, _ = cap.extract(fc, spec["bw"], oversample=spec["oversample"], t0=t0 or 0.0, t1=t1)
        if len(y) < 256:
            continue
        ctx = {"channel": BLE_ADV_CHANNELS.get(fc, 37), "fc": fc}
        try:
            d = DECODER_FN[protocol](y, fs, ctx)
        except Exception as ex:  # noqa: BLE001
            d = Decode(protocol)
            d.evidence = f"error {type(ex).__name__}: {str(ex)[:80]}"
        d.fields["fc_hz"] = fc
        if best is None or (d.ok and not best.ok) or (d.ok == best.ok and d.n_ok > best.n_ok):
            best = d
        if d.ok and not d.weak:
            break
    best.seconds = time.time() - t_start
    return best


# =========================================================================================== CLI
def main(argv=None) -> int:
    import measure_descriptors as md
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta")
    ap.add_argument("--protocol", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--fc", type=float, required=True, help="emission centre (Hz)")
    ap.add_argument("--bw", type=float, default=None, help="emission width (Hz, default: the protocol's band)")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    a = ap.parse_args(argv)
    meta = json.loads(Path(a.meta).read_text())
    g = meta["global"]
    data = Path(a.meta).with_name(Path(a.meta).name[: -len(".sigmf-meta")] + ".sigmf-data")
    cap = md.CaptureIQ(data, g.get("core:datatype", "cf32_le"), float(g["core:sample_rate"]), float(meta["captures"][0]["core:frequency"]))
    d = attempt(a.protocol, cap, a.fc, a.bw or REGISTRY[a.protocol]["bw"], None, a.t0, a.t1)
    print(json.dumps({"protocol": d.protocol, "ok": d.ok, "weak": d.weak, "n_ok": d.n_ok, "n_tried": d.n_tried, "variant": d.variant, "evidence": d.evidence,
                      "fields": d.fields, "seconds": round(d.seconds, 2)}, indent=1, default=str))
    return 0 if d.ok else 1


if __name__ == "__main__":
    sys.exit(main())
