#!/usr/bin/env python3
"""Signal descriptor words (SDW): measure the physical-layer parameters of a detected emission from the IQ.

The comms analogue of an ELINT pulse descriptor word. For every tracked emission (detector boxes grouped across
frames, as in common_waveforms_look_up.py) the band is extracted from the capture IQ and measured without any
knowledge of the library: envelope and timing at two scales, periodicities, OFDM cyclic-prefix structure,
symbol-rate lines, modulation-class features, FM/AM demodulated tones. lut_classify.py then matches the SDW
against the library descriptors.

    python3 measure_descriptors.py <capture>.sigmf-meta            # SDW of every annotated region group
    python3 measure_descriptors.py <capture folder> --masks M12S
    python3 measure_descriptors.py <dir>_preprocessed --select fm_broadcast --out sdw.json

Every SDW field says whether it was measured; unmeasured fields are neutral in the matcher.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_waveforms_look_up as lk  # noqa: E402
import manifest_io as mio  # noqa: E402

VERSION = "measure_descriptors.py 0.1 (lut_classification_experiment)"
MAX_SAMPLES = 64 << 20          # samples of a capture analysed per track (0.26 s at 250 MSps, the whole second below 64 MSps)
CHUNK = 1 << 20                 # overlap-save FFT chunk
GUARD = 1 << 13                 # samples discarded at both ends of every chunk (brick-wall filter wrap-around)
ENV_GRID_S = 10e-6              # on/off envelope grid for periodicity analysis


# =========================================================================================== capture + band extraction
class CaptureIQ:
    """Memory-mapped access to a SigMF recording, plus cached chunk FFTs for band extraction."""

    def __init__(self, path: Path, dtype: str, rate: float, fc: float, max_samples: int = MAX_SAMPLES):
        self.path, self.dtype, self.rate, self.fc = Path(path), dtype, float(rate), float(fc)
        np_t, per, bps = lk._DTYPES.get(dtype, lk._DTYPES["cf32_le"])
        self.np_t, self.per, self.bps = np_t, per, bps
        self.n_total = self.path.stat().st_size // bps
        self.n = int(min(self.n_total, max_samples))
        self._mm = np.memmap(self.path, dtype=np_t, mode="r")
        self._ffts: dict[int, np.ndarray] = {}
        self.hop = CHUNK - 2 * GUARD
        self.n_chunks = max(1, int(math.ceil(self.n / self.hop)))

    def read(self, start: int, count: int) -> np.ndarray:
        start = max(0, min(start, self.n_total))
        count = max(0, min(count, self.n_total - start))
        if self.per == 2:
            raw = self._mm[2 * start: 2 * (start + count)]
            return raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        return np.asarray(self._mm[start: start + count]).astype(np.complex64)

    def chunk_fft(self, k: int) -> np.ndarray:
        """FFT of chunk k: input samples [k*hop - GUARD, k*hop - GUARD + CHUNK), zero padded outside the window."""
        if k in self._ffts:
            return self._ffts[k]
        s = k * self.hop - GUARD
        buf = np.zeros(CHUNK, dtype=np.complex64)
        a, b = max(s, 0), min(s + CHUNK, self.n)
        if b > a:
            buf[a - s: b - s] = self.read(a, b - a)
        X = np.fft.fft(buf).astype(np.complex64)
        if len(self._ffts) > 96:
            self._ffts.pop(next(iter(self._ffts)))
        self._ffts[k] = X
        return X

    def extract(self, f_abs_center: float, bw: float, oversample: float = 2.5, t0: float = 0.0, t1: float | None = None):
        """Band-pass + mix + decimate by overlap-save FFT: returns (y complex64, fs, t_start_s)."""
        q = 1
        target = oversample * bw
        while self.rate / (q * 2) >= target and q * 2 <= CHUNK // 32:
            q *= 2
        fs = self.rate / q
        M = CHUNK // q
        k_off = int(round((f_abs_center - self.fc) / self.rate * CHUNK))
        n0 = int(t0 * self.rate)
        n1 = self.n if t1 is None else int(min(t1 * self.rate, self.n))
        k_first, k_last = n0 // self.hop, max(n0, n1 - 1) // self.hop
        out = []
        for k in range(k_first, k_last + 1):
            X = np.roll(self.chunk_fft(k), -k_off)
            Y = np.concatenate([X[: M // 2], X[-M // 2:]])
            y = (np.fft.ifft(Y) / q).astype(np.complex64)
            # the bin shift mixes relative to the chunk start; rotate so the mixer phase is continuous in absolute time
            # (otherwise every chunk boundary carries a phase jump of 2 pi k_off hop / CHUNK)
            s = k * self.hop - GUARD
            y *= np.complex64(np.exp(-2j * np.pi * k_off * s / CHUNK))
            g = GUARD // q
            out.append(y[g: M - g] if M > 2 * g else y)
        y = np.concatenate(out) if out else np.zeros(0, dtype=np.complex64)
        start_sample = k_first * self.hop
        # trim to [n0, n1)
        a = (n0 - start_sample) // q
        b = a + (n1 - n0) // q
        return y[max(a, 0): max(b, 0)], fs, n0 / self.rate


# =========================================================================================== small helpers
def _db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def _welch(z: np.ndarray, fs: float, nseg: int | None = None, complex_in: bool | None = None):
    """Averaged periodogram (Hann, 50 % overlap). Returns (f, P_db) with f shifted for complex input."""
    n = len(z)
    if n < 64:
        return np.zeros(0), np.zeros(0)
    if nseg is None:
        nseg = 1 << min(18, int(math.floor(math.log2(max(n // 4, 2)))))
        nseg = max(nseg, 256) if n >= 256 else n
    nseg = min(nseg, n)
    if complex_in is None:
        complex_in = np.iscomplexobj(z)
    f, P = sps.welch(z, fs=fs, window="hann", nperseg=nseg, noverlap=nseg // 2, return_onesided=not complex_in,
                     detrend=False, scaling="density")
    if complex_in:
        f = np.fft.fftshift(f)
        P = np.fft.fftshift(P)
    return f, _db(P)


def _baseline(P: np.ndarray, w: int) -> np.ndarray:
    """Block-median baseline of a spectrum (cheap running median)."""
    w = max(8, int(w))
    nb = max(1, len(P) // w)
    cut = nb * w
    med = np.median(P[:cut].reshape(nb, w), axis=1)
    xs = (np.arange(nb) + 0.5) * w
    return np.interp(np.arange(len(P)), xs, med)


def _lines(f: np.ndarray, P_db: np.ndarray, fmin: float, fmax: float, min_snr_db: float = 8.0, max_lines: int = 4,
           exclude_dc_hz: float = 0.0, min_sep_hz: float | None = None) -> list[list[float]]:
    """Spectral lines in [fmin, fmax] (and [-fmax, -fmin] for two-sided spectra): [[f_hz, snr_db], ...] strongest
    first. Peaks closer than min_sep_hz to a stronger one are the same lobe and are dropped."""
    if len(f) < 32:
        return []
    base = _baseline(P_db, max(8, len(P_db) // 300))
    snr = P_db - base
    sel = (np.abs(f) >= fmin) & (np.abs(f) <= fmax) & (np.abs(f) > exclude_dc_hz)
    if not np.any(sel):
        return []
    idx = np.where(sel)[0]
    pk, props = sps.find_peaks(snr[idx], height=min_snr_db, distance=3)
    if len(pk) == 0:
        return []
    df = abs(f[1] - f[0]) if len(f) > 1 else 1.0
    if min_sep_hz is None:
        min_sep_hz = max(25 * df, 0.004 * fmax)
    order = np.argsort(props["peak_heights"])[::-1]
    out = []
    for i in order:
        fi = float(f[idx[pk[i]]])
        if any(abs(fi - o[0]) < min_sep_hz for o in out):
            continue
        out.append([fi, float(round(props["peak_heights"][i], 1))])
        if len(out) >= max_lines:
            break
    return out


def _log_hist_peaks(values: np.ndarray, lo: float, hi: float, n_ref: int, per_decade: int = 60, min_count: int = 3,
                    min_share: float = 0.03, max_peaks: int = 6) -> list[dict]:
    """Peaks of a log-spaced histogram of positive values: [{value, count, share}] by count."""
    v = values[(values > lo) & (values < hi)]
    if len(v) < min_count:
        return []
    edges = np.logspace(math.log10(lo), math.log10(hi), int(per_decade * math.log10(hi / lo)) + 1)
    h, _ = np.histogram(v, bins=edges)
    need = max(min_count, int(min_share * n_ref))
    peaks = []
    for i in np.argsort(h)[::-1]:
        if h[i] < need:
            break
        if any(abs(i - j) <= 2 for j, _ in peaks):
            continue
        peaks.append((i, h[i]))
        if len(peaks) >= max_peaks:
            break
    out = []
    for i, c in peaks:
        sel = (v >= edges[max(i - 1, 0)]) & (v < edges[min(i + 2, len(edges) - 1)])
        out.append({"value": float(np.median(v[sel])) if np.any(sel) else float(edges[i]), "count": int(c), "share": round(float(c) / max(n_ref, 1), 3)})
    return out


# =========================================================================================== envelope / timing
def detect_bursts(p: np.ndarray, fs: float, noise_lin: float, smooth_s: float, merge_gap_s: float, min_len_s: float, t_offset: float,
                  min_margin_db: float = 6.0, min_len_samples: int = 1, peak_margin_db: float = 0.0):
    """On/off segmentation of a power series against an absolute noise reference. Each run's edges are refined to
    the 50 % level of its smoothed plateau (so lengths and gaps are not widened by the smoothing window), then
    runs closer than merge_gap_s are merged. Returns (bursts [[start_s, dur_s, mean_db, peak_db]], duty, thr_db, on_mask)."""
    w = max(1, int(round(smooth_s * fs)))
    ps = ndimage.uniform_filter1d(p, w, mode="nearest") if w > 1 else p
    noise_db = _db(noise_lin) if noise_lin > 0 else _db(np.percentile(ps, 20))
    p99 = _db(np.percentile(ps, 99.5))
    thr_db = noise_db + max(min_margin_db, min(12.0, 0.4 * (p99 - noise_db)))
    thr = 10 ** (thr_db / 10)
    on = ps > thr
    off_thr = 10 ** ((thr_db - 3) / 10)
    on = _hysteresis(on, ps > off_thr)
    if not on.any():
        return [], 0.0, float(thr_db), on
    d = np.diff(on.astype(np.int8), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    n = len(ps)
    peak_lin = 10 ** ((noise_db + peak_margin_db) / 10) if peak_margin_db > 0 else 0.0
    runs = []
    for s_, e_ in zip(starts, ends):
        seg = ps[s_:e_]
        pk = float(seg.max())
        if pk < peak_lin:
            continue
        half = 0.5 * float(np.median(seg)) if e_ - s_ >= 3 else 0.5 * pk
        a = s_
        while a > max(s_ - w, 0) and ps[a - 1] >= half:
            a -= 1
        while a < e_ - 1 and ps[a] < half:
            a += 1
        b = e_
        while b < min(e_ + w, n) and ps[b] >= half:
            b += 1
        while b > a + 1 and ps[b - 1] < half:
            b -= 1
        runs.append([a, b, pk])
    if not runs:
        return [], 0.0, float(thr_db), np.zeros_like(on)
    gap = int(round(merge_gap_s * fs))
    merged = [runs[0]]
    for a, b, pk in runs[1:]:
        if a - merged[-1][1] <= gap:
            merged[-1][1] = b
            merged[-1][2] = max(merged[-1][2], pk)
        else:
            merged.append([a, b, pk])
    min_len = max(min_len_samples, int(round(min_len_s * fs)))
    bursts = []
    keep = np.zeros_like(on)
    for a, b, pk in merged:
        if b - a < min_len:
            continue
        keep[a:b] = True
        bursts.append([round(t_offset + a / fs, 7), round((b - a) / fs, 7), round(float(_db(ps[a:b].mean())), 1), round(float(_db(pk)), 1)])
    return bursts, float(np.mean(keep)), float(thr_db), keep


def _hysteresis(on_high: np.ndarray, above_low: np.ndarray) -> np.ndarray:
    """Pixels above the high threshold seed runs that extend over everything above the low threshold."""
    lab, n = ndimage.label(above_low)
    if n == 0:
        return on_high
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(lab[on_high])] = True
    keep[0] = False
    return keep[lab]


def timing_stats(bursts: list, n_ref: int | None = None, max_lag: float = 0.5) -> dict:
    """Burst-length percentiles; start-to-start interval peaks of consecutive bursts (pri) and over up to 30
    neighbours (pri_multi, for structures the detector only catches in part); end-to-start gap peaks; and the
    interval peaks within each burst-duration class (a beacon every 102.4 ms among random packets)."""
    out = {"n": len(bursts)}
    if not bursts:
        return out
    b = np.array(bursts, dtype=np.float64)
    starts, durs = b[:, 0], b[:, 1]
    ends = starts + durs
    out["dur_p10"], out["dur_p50"], out["dur_p90"] = [float(x) for x in np.percentile(durs, [10, 50, 90])]
    out["dur_min"], out["dur_max"] = float(durs.min()), float(durs.max())
    n = len(starts)
    if n >= 3:
        d1 = starts[1:] - starts[:-1]
        out["pri"] = _log_hist_peaks(d1, 2e-7, max_lag, n_ref or n)
        diffs = np.concatenate([starts[k:] - starts[:-k] for k in range(1, min(30, n))])
        out["pri_multi"] = _log_hist_peaks(diffs, 2e-7, max_lag, len(diffs), min_share=0.02)
        gaps = starts[1:] - ends[:-1]
        out["gap"] = _log_hist_peaks(gaps[gaps > 0], 2e-7, max_lag, n_ref or n, min_count=3)
        order = np.argsort(durs)
        cls = np.zeros(n, dtype=int)
        c0, ref = 0, durs[order[0]]
        for i in order:
            if durs[i] > 1.3 * ref:
                c0 += 1
                ref = durs[i]
            cls[i] = c0
        by_class = []
        for c in np.unique(cls):
            st = starts[cls == c]
            if len(st) < 4:
                continue
            dd = np.concatenate([st[k:] - st[:-k] for k in range(1, min(8, len(st)))])
            for pk_ in _log_hist_peaks(dd, 1e-4, max_lag, len(st), min_count=3, min_share=0.3, max_peaks=2):
                by_class.append({"dur": float(np.median(durs[cls == c])), "n": int(len(st)), **pk_})
        out["pri_by_dur"] = sorted(by_class, key=lambda r: -r["count"])[:8]
    return out


def envelope_periods(on: np.ndarray, fs: float, p: np.ndarray | None = None, min_lag: float = 50e-6, max_lag: float = 0.5) -> list[list[float]]:
    """Autocorrelation peaks of the on/off envelope (and of the power envelope) on a 10 us grid: [[period_s, height], ...]."""
    step = max(1, int(round(ENV_GRID_S * fs)))
    env = on.astype(np.float32)[::step]
    if p is not None:
        pw = _db(p)[::step]
        env = 0.5 * env + 0.5 * (pw - pw.min()) / max(pw.max() - pw.min(), 1e-6)
    fs_e = fs / step
    env = env - env.mean()
    n = len(env)
    if n < 64 or np.all(env == 0):
        return []
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    R = np.fft.irfft(np.abs(np.fft.rfft(env, nfft)) ** 2)[:n]
    if R[0] <= 0:
        return []
    r = R / R[0]
    lo, hi = int(min_lag * fs_e), int(min(max_lag, n / fs_e / 3) * fs_e)
    if hi <= lo + 4:
        return []
    seg = r[lo:hi]
    pk, props = sps.find_peaks(seg, height=0.12, prominence=0.08, distance=max(2, int(0.5 * lo)))
    if len(pk) == 0:
        return []
    order = np.argsort(props["peak_heights"])[::-1][:6]
    return [[round(float((lo + pk[i]) / fs_e), 7), round(float(props["peak_heights"][i]), 3)] for i in order]


# =========================================================================================== OFDM cyclic prefix
def ofdm_cp(y: np.ndarray, fs: float, bw: float, max_tu_s: float = 6e-3, max_n: int = 1 << 22) -> dict:
    """Peaks of |R(tau)|/R(0) for tau >= 3/bw: an OFDM signal correlates at tau = Tu (useful symbol length).
    For the strongest peak the symbol period Ts (and so the CP) follows from the periodicity of the CP metric."""
    n = len(y)
    if n < 4096:
        return {"measured": False}
    if n > max_n:
        s = (n - max_n) // 2
        y = y[s: s + max_n]
        n = max_n
    yf = y - y.mean()
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    R = np.fft.ifft(np.abs(np.fft.fft(yf, nfft)) ** 2)[: n // 2]
    if abs(R[0]) <= 0:
        return {"measured": False}
    r = np.abs(R) / abs(R[0])
    lo = int(max(3.0 / max(bw, 1.0), 1e-6) * fs) + 1
    hi = int(min(max_tu_s * fs, n // 2 - 1))
    if hi <= lo + 8:
        return {"measured": False}
    seg = r[lo:hi]
    base = _baseline(seg, max(8, len(seg) // 200))
    excess = seg - base
    pk, props = sps.find_peaks(excess, height=0.012, distance=max(3, lo // 2))
    out = {"measured": True, "peaks": []}
    if len(pk) == 0:
        return out
    order = np.argsort(props["peak_heights"])[::-1][:4]
    for i in order:
        lag = lo + pk[i]
        out["peaks"].append({"tu_s": lag / fs, "scs_hz": fs / lag, "height": round(float(seg[pk[i]]), 4), "excess": round(float(excess[pk[i]]), 4),
                             "ratio": round(float(seg[pk[i]] / max(base[pk[i]], 1e-6)), 1)})
    best = out["peaks"][0]
    if best["ratio"] < 4.0:
        return out
    # CP from the period of the CP metric c[n] = sum_W y[n+k] conj(y[n+k+L])
    L = int(round(best["tu_s"] * fs))
    W = max(4, L // 16)
    prod = y[:-L] * np.conj(y[L:])
    c = np.abs(np.convolve(prod, np.ones(W, dtype=np.float32), mode="valid"))
    c = c - c.mean()
    fsym_lo, fsym_hi = 1.0 / (1.6 * best["tu_s"]), 1.0 / (1.0005 * best["tu_s"])
    f, P = _welch(c.astype(np.float32), fs, complex_in=False)
    if len(f):
        lines = _lines(f, P, fsym_lo, fsym_hi, min_snr_db=6.0, max_lines=1)
        if lines:
            ts = 1.0 / lines[0][0]
            out.update({"tu_s": best["tu_s"], "scs_hz": best["scs_hz"], "ts_s": ts, "cp_s": ts - best["tu_s"], "cp_snr_db": lines[0][1]})
            return out
    out.update({"tu_s": best["tu_s"], "scs_hz": best["scs_hz"]})
    return out


# =========================================================================================== rate lines
def rate_lines(y: np.ndarray, fs: float, bw: float, on: np.ndarray | None = None) -> dict:
    """Spectral lines of |y|^2 (linear modulations: line at the symbol rate), y^2 and y^4 (CPM / PSK: lines at
    2 f_c +- h Rs, 4 f_c). The frequency-transition line of the FSK family is measured in finst_features()."""
    n = len(y)
    out = {"measured": n >= 4096}
    if n < 4096:
        return out
    fmin = max(50.0, bw / 50.0)
    fmax = fs / 2
    yy = y if on is None else np.where(on, y, 0).astype(np.complex64)
    z1 = (np.abs(yy) ** 2).astype(np.float32)
    z1 -= z1.mean()
    f, P = _welch(z1, fs, complex_in=False)
    out["abs2"] = _lines(f, P, fmin, fmax)
    z2 = (yy * yy).astype(np.complex64)
    f, P = _welch(z2, fs)
    out["sq"] = _lines(f, P, 0.0, fmax, max_lines=6)
    z4 = (z2 * z2).astype(np.complex64)
    f, P = _welch(z4, fs)
    out["quad"] = _lines(f, P, 0.0, fmax, max_lines=4)
    if len(out["sq"]) >= 2:
        out["sq_spacing_hz"] = round(abs(out["sq"][0][0] - out["sq"][1][0]), 1)
    if len(out["quad"]) >= 2:
        out["quad_spacing_hz"] = round(abs(out["quad"][0][0] - out["quad"][1][0]), 1)
    return out


def finst_features(yt: np.ndarray, fst: float, bw: float, on: np.ndarray | None) -> dict:
    """Instantaneous-frequency statistics on the tight band (box width x 1.15, so the noise outside the box does
    not dominate): deviation, discrete levels, chirp slope, and the symbol-rate line of the frequency transitions."""
    out = {"measured": False}
    sel = on if on is not None and on.any() else np.ones(len(yt), dtype=bool)
    sel_idx = np.where(sel)[0]
    if len(sel_idx) < 256:
        return out
    if len(sel_idx) > (1 << 21):
        sel_idx = sel_idx[: 1 << 21]
    runs = np.split(sel_idx, np.where(np.diff(sel_idx) > 1)[0] + 1)
    smooth = max(1, int(round(fst / max(bw, 1.0) * 2)))
    fis, fsms, slopes, cons = [], [], [], []
    d_full = np.zeros(max(len(yt) - 2, 1), dtype=np.float32)
    for r in runs:
        if len(r) < 16:
            continue
        seg = yt[r[0]: r[-1] + 1]
        fi = np.angle(seg[1:] * np.conj(seg[:-1])).astype(np.float32) * (fst / (2 * np.pi))
        fi -= np.median(fi)
        fis.append(fi)
        fsm = ndimage.uniform_filter1d(fi, smooth, mode="nearest") if smooth > 1 else fi
        fsms.append(fsm)
        if len(fi) >= 4:
            dd = np.abs(np.diff(fi))
            d_full[r[0]: r[0] + len(dd)] = dd
        if len(fi) >= 8 * smooth + 16:
            # chirp slope: median frequency step over a stride; the stride grows until the step beats the noise
            best = None
            for k in (1, 2, 4, 8, 16):
                st = smooth * k
                if len(fsm) < 10 * st:
                    break
                dfi = np.diff(fsm[::st]) * (fst / st)
                if len(dfi) < 64:
                    break
                med = float(np.median(dfi))
                if med == 0:
                    continue
                c = float(np.mean(np.sign(dfi) == np.sign(med)))
                sel_ = dfi[(dfi > 0.4 * med) & (dfi < 1.6 * med)] if med > 0 else dfi[(dfi < 0.4 * med) & (dfi > 1.6 * med)]
                if len(sel_) >= 16:
                    h, e = np.histogram(sel_, bins=48)
                    hs = ndimage.uniform_filter1d(h.astype(np.float32), 5)
                    med = float(0.5 * (e[int(np.argmax(hs))] + e[int(np.argmax(hs)) + 1]))
                if best is None or c >= best[1]:
                    best = (med, c)
            if best:
                slopes.append(best[0])
                cons.append(best[1])
    if not fis:
        return out
    out["measured"] = True
    fi = np.concatenate(fis)
    fsm = np.concatenate(fsms)
    out["dev_std_hz"] = round(float(fsm.std()), 1)
    out["dev_p90_hz"] = round(float(np.percentile(np.abs(fsm), 90)), 1)
    out["dev_p99_hz"] = round(float(np.percentile(np.abs(fsm), 99)), 1)
    lim = max(out["dev_p99_hz"] * 1.3, 1.0)
    h, e = np.histogram(fsm[np.abs(fsm) <= lim], bins=81, range=(-lim, lim))
    hs = ndimage.uniform_filter1d(h.astype(np.float32), 3)
    pk, pr = sps.find_peaks(hs, prominence=0.08 * hs.max(), distance=4)
    out["n_levels"] = int(len(pk))
    if len(pk) >= 2:
        centers = np.sort(0.5 * (e[pk] + e[pk + 1]))
        out["level_spacing_hz"] = round(float(np.median(np.diff(centers))), 1)
        out["dev_levels_hz"] = round(float(0.5 * (centers[-1] - centers[0])), 1)
    out["dev_est_hz"] = out.get("dev_levels_hz", out["dev_p90_hz"])
    if slopes:
        out["chirp_rate_hz_s"] = float(np.median(slopes))
        out["chirp_consistency"] = round(float(np.median(cons)), 3)
    # symbol-rate line of the frequency transitions
    nz = d_full > 0
    if nz.sum() >= 1024:
        d_full[nz] -= d_full[nz].mean()
        f, P = _welch(d_full, fst, complex_in=False)
        out["finst_lines"] = _lines(f, P, max(50.0, bw / 50.0), fst / 2)
    return out


# =========================================================================================== modulation features
def _gap_depth(pw: np.ndarray, fs: float, fine_b: list, t0: float, max_gap_s: float = 50e-6, noise_lin: float = 0.0) -> tuple[float, float]:
    """Over the gaps between consecutive fine bursts closer than max_gap_s: median (burst level - gap level) in dB
    and median (gap level - noise level) in dB. Deep gaps at the noise floor mean real pulses; shallow gaps mean
    the fine scale only cut a continuous signal."""
    if len(fine_b) < 3:
        return 0.0, 99.0
    b = np.array(fine_b[:3000], dtype=np.float64)
    starts = ((b[:, 0] - t0) * fs).astype(np.int64)
    ends = starts + np.maximum((b[:, 1] * fs).astype(np.int64), 1)
    depths, above = [], []
    for i in range(len(b) - 1):
        g0, g1 = ends[i], starts[i + 1]
        if g1 - g0 < 1 or (g1 - g0) / fs > max_gap_s or g1 > len(pw):
            continue
        burst = float(np.mean(pw[starts[i]: ends[i]]))
        gap = float(np.mean(pw[g0:g1]))
        gap_floor = float(np.percentile(pw[g0:g1], 10)) if g1 - g0 >= 3 else float(np.min(pw[g0:g1]))
        if burst > 0 and gap > 0:
            depths.append(10.0 * math.log10(burst / gap))
            if noise_lin > 0:
                above.append(10.0 * math.log10(max(gap_floor, 1e-30) / noise_lin))
    if not depths:
        return 0.0, 99.0
    return round(float(np.median(depths)), 1), (round(float(np.median(above)), 1) if above else 99.0)


def _on_cv(pw: np.ndarray, fs: float, fine_b: list, t0: float) -> float:
    """Median amplitude CV inside the fine bursts: flat pulse tops are low, envelope fades are high."""
    cvs = []
    for b in fine_b[:2000]:
        i0 = int(round((b[0] - t0) * fs))
        i1 = i0 + max(int(round(b[1] * fs)), 1)
        if i1 - i0 < 3 or i1 > len(pw):
            continue
        a = np.sqrt(pw[i0:i1])
        cvs.append(float(a.std() / max(a.mean(), 1e-9)))
    return round(float(np.median(cvs)), 3) if cvs else 99.0


def envelope_stats(yt: np.ndarray, fst: float, on: np.ndarray | None, noise_lin: float) -> dict:
    """Envelope CV per burst on the tight band (bursts of different devices differ by tens of dB, so the
    statistic is taken inside each burst and pooled as a median), corrected for the noise contribution
    (a constant-envelope signal at SNR rho shows cv ~ 1 / sqrt(2 rho))."""
    sel = on if on is not None and on.any() else np.ones(len(yt), dtype=bool)
    idx = np.where(sel)[0]
    if len(idx) < 64:
        return {}
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    cvs, cvc, rhos, kurts, paprs = [], [], [], [], []
    for r in runs:
        if len(r) < 16:
            continue
        a = np.abs(yt[r[0]: r[-1] + 1]).astype(np.float32)
        ma = float(a.mean())
        p_on = float(np.mean(a ** 2))
        rho = (p_on - noise_lin) / max(noise_lin, 1e-30)
        if rho < 1.0:
            continue
        cv = float(a.std() / max(ma, 1e-9))
        cvs.append(cv)
        cvc.append(math.sqrt(max(cv * cv - 1.0 / (2.0 * rho), 0.0)))
        rhos.append(rho)
        kurts.append(float(((a - ma) ** 4).mean() / max(a.var() ** 2, 1e-18) - 3.0))
        paprs.append(float(_db(np.percentile(a ** 2, 99.9)) - _db(p_on)))
        if len(cvs) >= 400:
            break
    if not cvs:
        return {}
    return {"env_cv_tight": round(float(np.median(cvs)), 4), "env_cv_corr": round(float(np.median(cvc)), 4),
            "env_kurt_burst": round(float(np.median(kurts)), 2), "papr_burst_db": round(float(np.median(paprs)), 2),
            "snr_tight_db": round(float(10 * math.log10(np.median(rhos))), 1), "snr_tight_max_db": round(float(10 * math.log10(max(rhos))), 1),
            "n_bursts_stats": len(cvs)}


def modulation_features(y: np.ndarray, fs: float, bw: float, on: np.ndarray, noise_psd_lin: float) -> dict:
    """Envelope and spectral-shape statistics over the on-samples of the wide band."""
    out = {"measured": False}
    sel = on if on is not None and on.any() else np.ones(len(y), dtype=bool)
    sel_idx = np.where(sel)[0]
    if len(sel_idx) < 256:
        return out
    if len(sel_idx) > (1 << 21):
        sel_idx = sel_idx[: 1 << 21]
    yo = y[sel_idx]
    a = np.abs(yo).astype(np.float32)
    ma = float(a.mean())
    out.update({"measured": True, "env_cv": round(float(a.std() / max(ma, 1e-9)), 4),
                "env_kurt": round(float(((a - ma) ** 4).mean() / max(a.var() ** 2, 1e-18) - 3.0), 2),
                "papr_db": round(float(_db(np.percentile(a ** 2, 99.9)) - _db((a ** 2).mean())), 2)})
    f, P = _welch(yo, fs)
    if len(f):
        inb = np.abs(f) <= 0.5 * bw
        edge_in = (np.abs(f) >= 0.3 * bw) & (np.abs(f) <= 0.42 * bw)
        edge_out = (np.abs(f) >= 0.56 * bw) & (np.abs(f) <= 0.68 * bw)
        if inb.sum() > 8 and edge_in.sum() > 2 and edge_out.sum() > 2:
            med_in = float(np.median(P[inb]))
            out["edge_drop_db"] = round(float(np.mean(P[edge_in]) - np.mean(P[edge_out])), 1)
            k = int(np.argmax(np.where(inb, P, -1e9)))
            base = _baseline(P, max(8, len(P) // 100))
            out["carrier_line_db"] = round(float(P[k] - base[k]), 1)
            out["carrier_offset_hz"] = round(float(f[k]), 1)
            out["psd_flatness_db"] = round(float(np.std(P[(np.abs(f) <= 0.4 * bw)])), 2)
            if noise_psd_lin > 0:
                out["snr_median_db"] = round(float(med_in - _db(noise_psd_lin)), 1)
                mean_in = float(np.mean(10 ** (P[inb] / 10)))
                out["snr_db"] = round(float(_db(max(mean_in - noise_psd_lin, 1e-30)) - _db(noise_psd_lin)), 1)
            win = np.abs(f) <= 1.25 * bw
            lin = 10 ** (P[win] / 10) - (noise_psd_lin if noise_psd_lin > 0 else 0)
            lin = np.maximum(lin, 0)
            if lin.sum() > 0:
                c = np.cumsum(lin) / lin.sum()
                fw = f[win]
                out["obw99_hz"] = round(float(fw[min(np.searchsorted(c, 0.995), len(fw) - 1)] - fw[min(np.searchsorted(c, 0.005), len(fw) - 1)]), 1)
                out["centroid_hz"] = round(float(np.sum(fw * lin) / lin.sum()), 1)
    return out


def classify(mod: dict, ofdm: dict, lines: dict, fine: dict, coarse: dict) -> tuple[str, float]:
    """Coarse modulation class from the measured features (one descriptor among the others in the matcher)."""
    if not mod.get("measured"):
        return "unknown", 0.0
    cv = mod.get("env_cv_corr", mod.get("env_cv", 1.0))
    snr = max(mod.get("snr_db", 0.0), mod.get("snr_tight_db", 0.0), mod.get("snr_tight_max_db", 0.0))
    if snr < 3:
        return "unknown", 0.0
    if fine.get("is_pulse_structure"):
        return "pulse", 0.8
    if mod.get("chirp_consistency", 0) >= 0.7 and abs(mod.get("chirp_rate_hz_s", 0)) > 1e6 and cv < 0.35:
        return "css", 0.8
    best = ofdm.get("peaks", [{}])[0] if ofdm.get("peaks") else {}
    if best.get("ratio", 0) >= 6 and best.get("excess", 0) >= 0.02 and cv >= 0.3:
        return "ofdm", min(1.0, (0.7 if ofdm.get("cp_s") else 0.5) + best["excess"] * 3)
    if mod.get("carrier_line_db", 0) >= 8 and mod.get("carrier_offset_hz", 0) < -0.35 * max(mod.get("obw99_hz", 0), 1) and cv >= 0.25 \
            and lines.get("sq_spacing_hz", 0) and 5e6 <= lines["sq_spacing_hz"] <= 12e6:
        return "vsb", 0.7
    if mod.get("carrier_line_db", 0) >= 12 and cv >= 0.12:
        return "am", 0.7
    if mod.get("carrier_line_db", 0) >= 12 and cv < 0.12 and mod.get("dev_std_hz", 1e9) < 0.01 * max(mod.get("obw99_hz", 1), 1):
        return "cw", 0.7
    if cv < 0.16:
        if mod.get("n_levels", 1) >= 2 and (lines.get("finst") or lines.get("sq_spacing_hz")):
            return "fsk", 0.8
        if mod.get("dev_std_hz", 0) > 0.05 * max(mod.get("obw99_hz", 1), 1):
            return "fm", 0.7
        if mod.get("dev_std_hz", 0) < 0.01 * max(mod.get("obw99_hz", 1), 1):
            return "cw", 0.6
        return "fm", 0.5
    if cv < 0.45:
        if lines.get("abs2"):
            return "psk", 0.7
        if mod.get("n_levels", 1) >= 2:
            return "fsk", 0.5
        return "psk", 0.4
    if abs(mod.get("env_kurt", 0)) < 1.0 and 0.45 <= cv <= 0.62:
        return "noise", 0.5
    return "unknown", 0.2


# =========================================================================================== demodulated tones
def fm_demod_lines(yt: np.ndarray, fst: float, on: np.ndarray | None) -> dict:
    """Lines in the FM-demodulated baseband of the tight band (19 kHz pilot, CTCSS, APT subcarrier, ...)."""
    fi = np.angle(yt[1:] * np.conj(yt[:-1])).astype(np.float32) * (fst / (2 * np.pi))
    if on is not None and on.any() and not on.all():
        fi = np.where(on[1:], fi, 0.0).astype(np.float32)
        nz = fi != 0
        fi[nz] -= fi[nz].mean()
    else:
        fi -= fi.mean()
    f, P = _welch(fi, fst, complex_in=False)
    out = {"measured": len(f) > 0}
    if len(f):
        out["lines"] = _lines(f, P, 20.0, fst / 2, min_snr_db=10.0, max_lines=10)
    return out


def am_env_lines(y: np.ndarray, fs: float) -> dict:
    a = np.abs(y).astype(np.float32)
    a -= a.mean()
    f, P = _welch(a, fs, complex_in=False)
    out = {"measured": len(f) > 0}
    if len(f):
        out["lines"] = _lines(f, P, 20.0, fs / 2, min_snr_db=10.0, max_lines=10)
    return out


def hd_sidebands(y: np.ndarray, fs: float) -> dict:
    """NRSC-5 FM hybrid: digital sidebands at +-(129..198) kHz against the floor beyond +-230 kHz."""
    if fs < 560e3:
        return {"measured": False}
    f, P = _welch(y, fs)
    sb = (np.abs(f) >= 135e3) & (np.abs(f) <= 195e3)
    ref = (np.abs(f) >= 235e3) & (np.abs(f) <= 270e3)
    ana = np.abs(f) <= 60e3
    if sb.sum() < 4 or ref.sum() < 4 or ana.sum() < 4:
        return {"measured": False}
    return {"measured": True, "sideband_db": round(float(np.median(P[sb]) - np.median(P[ref])), 1),
            "sideband_flatness_db": round(float(np.std(P[sb])), 2), "analog_db": round(float(np.median(P[ana]) - np.median(P[ref])), 1)}


# =========================================================================================== per track
def _region_intervals(track: lk.Track, margin_s: float) -> np.ndarray:
    iv = np.array(sorted((r.t0 - margin_s, r.t0 + r.dur + margin_s) for r in track.regions), dtype=np.float64)
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.array(merged)


def _interval_mask(iv: np.ndarray, fs: float, t0: float, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in iv:
        i0, i1 = int(max((a - t0) * fs, 0)), int(min((b - t0) * fs, n))
        if i1 > i0:
            m[i0:i1] = True
    return m


def _bursts_in(bursts: list, iv: np.ndarray) -> list:
    if not bursts or len(iv) == 0:
        return bursts
    starts = iv[:, 0]
    out = []
    for b in bursts:
        k = np.searchsorted(starts, b[0], side="right") - 1
        if k >= 0 and b[0] + b[1] > iv[k, 0] and b[0] < iv[k, 1]:
            out.append(b)
        elif k + 1 < len(iv) and b[0] + b[1] > iv[k + 1, 0] and b[0] < iv[k + 1, 1]:
            out.append(b)
    return out


def measure_track(track: lk.Track, cap: CaptureIQ, want_fm: bool = True) -> dict:
    t_start = time.time()
    bw = max(track.bw, 2 * track.bin_hz)
    fc = track.fc
    if track.cut == "lo":
        fc = track.f_hi - 0.5 * bw
    elif track.cut == "hi":
        fc = track.f_lo + 0.5 * bw
    # wide band (2.5 x box) for envelope fine structure, OFDM, rate lines, spectral shape; tight band (1.15 x box)
    # for the coarse envelope and everything derived from the instantaneous frequency
    y, fs, t0 = cap.extract(fc, bw, oversample=2.5)
    sdw = {"version": VERSION, "fc_hz": fc, "bw_box_hz": track.bw, "fs_hz": fs, "t0_s": t0, "n_samples": int(len(y))}
    if len(y) < 1024:
        sdw["measured"] = False
        return sdw
    sdw["measured"] = True
    sdw["t1_s"] = t0 + len(y) / fs
    f, P = _welch(y, fs)
    outer = (np.abs(f) >= 0.62 * fs / 2 * 0.95) & (np.abs(f) <= 0.95 * fs / 2)
    if bw >= 0.7 * cap.rate or outer.sum() <= 8:
        # the box fills the capture: no clean band beside it, take the quietest 5 % of the band as the floor
        noise_psd_db = float(np.percentile(P, 5))
        sdw["noise_from"] = "p5"
    else:
        noise_psd_db = float(np.percentile(P[outer], 30))
        sdw["noise_from"] = "outer"
    noise_psd_lin = 10 ** (noise_psd_db / 10)
    sdw["noise_psd_db"] = round(noise_psd_db, 1)
    sdw["fc_cap_hz"] = cap.fc
    sdw["rate_hz"] = cap.rate
    yt, fst, _ = cap.extract(fc, bw, oversample=1.15)
    p = (np.abs(yt) ** 2).astype(np.float32)
    pw = (np.abs(y) ** 2).astype(np.float32)
    fine_b, duty_f, thr_f, on_f = detect_bursts(pw, fs, noise_psd_lin * fs, 0.1e-6, 0.2e-6, 0.2e-6, t0, min_margin_db=8.0, min_len_samples=2, peak_margin_db=11.0)
    coarse_b, duty_c, thr_c, on_c = detect_bursts(p, fst, noise_psd_lin * fst, max(4e-6, 3.0 / fst), 6e-6, 2e-6, t0, min_len_samples=4, peak_margin_db=7.0)
    # a narrow bursty track (packet radio) is measured inside its own detector regions only: other emitters share
    # the band at other times. Wide carriers that the detector paints in fragments keep the whole window, and the
    # band's envelope periodicity is always taken ungated (a TDD pattern survives even in a fragment).
    on_band = on_c.copy()
    own = (track.persistence < 0.75 or track.duty < 0.5) and bw <= 5e6
    sdw["own_regions_only"] = bool(own)
    if own:
        iv = _region_intervals(track, margin_s=2.0 * track.row_s)
        on_c &= _interval_mask(iv, fst, t0, len(on_c))
        on_f &= _interval_mask(iv, fs, t0, len(on_f))
        fine_b = _bursts_in(fine_b, iv)
        coarse_b = _bursts_in(coarse_b, iv)
        duty_c = float(np.mean(on_c)) if len(on_c) else 0.0
    sdw["fine"] = {"duty": round(duty_f, 4), "thr_db": round(thr_f, 1), "fs_hz": fs, "on_s": round(float(sum(b[1] for b in fine_b)), 6), **timing_stats(fine_b, max_lag=0.05)}
    sdw["coarse"] = {"duty": round(duty_c, 4), "thr_db": round(thr_c, 1), "fs_hz": fst, "on_s": round(float(sum(b[1] for b in coarse_b)), 6), **timing_stats(coarse_b)}
    packet_b, duty_p, thr_p, _ = detect_bursts(p, fst, noise_psd_lin * fst, max(8e-6, 3.0 / fst), 100e-6, 20e-6, t0, min_len_samples=4, peak_margin_db=7.0)
    frame_b, duty_fr, _, _ = detect_bursts(p, fst, noise_psd_lin * fst, max(8e-6, 3.0 / fst), 2e-3, 0.5e-3, t0, min_len_samples=4, peak_margin_db=7.0)
    if own:
        packet_b = _bursts_in(packet_b, iv)
        frame_b = _bursts_in(frame_b, iv)
    sdw["frame"] = {"duty": round(duty_fr, 4), "on_s": round(float(sum(b[1] for b in frame_b)), 6), **timing_stats(frame_b)}
    sdw["packet"] = {"duty": round(duty_p, 4), "on_s": round(float(sum(b[1] for b in packet_b)), 6), **timing_stats(packet_b)}
    sdw["fine"]["gap_depth_db"], sdw["fine"]["gap_above_noise_db"] = _gap_depth(pw, fs, fine_b, t0, noise_lin=noise_psd_lin * fs)
    # real pulses: the gaps between them sit at the noise floor; a fluctuating continuous envelope has gaps well above it
    fw = sdw["fine"]
    width_ratio = (fw["dur_p90"] / max(fw["dur_p10"], 1e-9)) if fw.get("dur_p10") else 99.0
    fw["width_ratio"] = round(float(width_ratio), 2)
    fw["on_cv"] = _on_cv(pw, fs, fine_b, t0)
    fw["is_pulse_structure"] = len(fine_b) >= 3 and fw.get("dur_p50", 1.0) < 20e-6 and fw["gap_depth_db"] >= 5.0 and \
        fw["gap_above_noise_db"] <= 5.0 and width_ratio <= 3.5 and fw["on_cv"] <= 0.3
    sdw["cut"] = track.cut
    sdw["fine"]["bursts"] = fine_b[:4000]
    sdw["coarse"]["bursts"] = coarse_b[:4000]
    sdw["periods"] = envelope_periods(on_band, fst, p)
    idx = np.minimum((np.arange(len(y)) * (fst / fs)).astype(np.int64), len(on_c) - 1)
    on_w = on_c[idx] if len(on_c) else np.ones(len(y), dtype=bool)
    if on_w.sum() < 256:
        on_w = np.ones(len(y), dtype=bool)
    on_t = on_c if on_c.sum() >= 128 else np.ones(len(yt), dtype=bool)
    sdw["ofdm"] = ofdm_cp(np.where(on_w, y, 0).astype(np.complex64) if own else y, fs, bw)
    sdw["lines"] = rate_lines(y, fs, bw, on_w)
    sdw["mod"] = modulation_features(y, fs, bw, on_w, noise_psd_lin)
    sdw["mod"].update(envelope_stats(yt, fst, on_t, noise_psd_lin * fst))
    fi_feats = finst_features(yt, fst, bw, on_t)
    sdw["lines"]["finst"] = fi_feats.pop("finst_lines", [])
    sdw["mod"].update({k: v for k, v in fi_feats.items() if k != "measured"})
    cls, conf = classify(sdw["mod"], sdw["ofdm"], sdw["lines"], sdw["fine"], sdw["coarse"])
    sdw["mod"]["class"], sdw["mod"]["class_conf"] = cls, round(conf, 2)
    if want_fm and sdw["mod"].get("env_cv", 1.0) < 0.5:
        sdw["fm"] = fm_demod_lines(yt, fst, on_t)
    sdw["am"] = am_env_lines(y, fs)
    sdw["hd"] = hd_sidebands(y, fs) if bw <= 600e3 else {"measured": False}
    sdw["seconds"] = round(time.time() - t_start, 2)
    return sdw


# =========================================================================================== band level
def measure_band(track_sdws: list[tuple[lk.Track, dict]], span_s: float) -> dict:
    """Deinterleave the coarse bursts of every bursty track of a capture: hop statistics, SIFS+ACK pairs,
    advertising-channel triplets."""
    rows = []
    for t, s in track_sdws:
        if not s.get("measured") or s["coarse"].get("duty", 1.0) > 0.9:
            continue
        for b in s["coarse"].get("bursts", []):
            rows.append((b[0], b[1], t.fc, t.bw, t.tid))
    out = {"n_bursts": len(rows)}
    if len(rows) < 6:
        return out
    rows.sort()
    a = np.array([[r[0], r[1], r[2], r[3]] for r in rows])
    starts, durs, fcs = a[:, 0], a[:, 1], a[:, 2]
    ends = starts + durs
    d_start = np.diff(starts)
    out["pri"] = _log_hist_peaks(d_start, 2e-7, 0.5, len(rows))
    gaps = starts[1:] - ends[:-1]
    out["gap"] = _log_hist_peaks(gaps[gaps > 0], 2e-7, 0.5, len(rows))
    # SIFS + ACK pairs: a burst >= 100 us followed within 6..24 us by a burst of 20..70 us on the same channel
    same = np.abs(np.diff(fcs)) < 0.5 * np.maximum(a[1:, 3], a[:-1, 3])
    pair = (gaps >= 6e-6) & (gaps <= 24e-6) & (durs[1:] >= 20e-6) & (durs[1:] <= 70e-6) & (durs[:-1] >= 100e-6) & same
    long_bursts = max(int(np.sum(durs[:-1] >= 100e-6)), 1)
    out["ack_pairs"] = int(pair.sum())
    out["ack_pair_fraction"] = round(float(pair.sum()) / long_bursts, 3)
    # channel changes between consecutive bursts
    ch_change = np.abs(np.diff(fcs)) > 0.5e6
    out["hop_fraction"] = round(float(np.mean(ch_change)), 3)
    out["hops_per_s"] = round(float(ch_change.sum()) / max(span_s, 1e-3), 1)
    if ch_change.any():
        out["hop_step_hz_p50"] = round(float(np.median(np.abs(np.diff(fcs))[ch_change])), 1)
    # BLE advertising triplets: bursts within 1.2 MHz of 2402 / 2426 / 2480 MHz in sequences of <= 10 ms
    adv = np.array([2402e6, 2426e6, 2480e6])
    near = np.min(np.abs(fcs[:, None] - adv[None, :]), axis=1) < 1.2e6
    trip = 0
    i = 0
    while i < len(rows) - 2:
        if near[i] and near[i + 1] and near[i + 2] and starts[i + 2] - starts[i] <= 10e-3 and len({int(np.argmin(np.abs(fcs[j] - adv))) for j in (i, i + 1, i + 2)}) == 3:
            trip += 1
            i += 3
        else:
            i += 1
    out["ble_adv_triplets"] = trip
    return out


# =========================================================================================== CLI
def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--select", nargs="*", default=None, help="only tracks whose id is listed, or captures matching these substrings")
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON output (default: sdw.json next to the input)")
    a = ap.parse_args(argv)
    inp = Path(a.input).expanduser()
    if a.masks:
        caps = lk.regions_from_masks(inp, a.masks)
    elif inp.is_file():
        caps = lk.regions_from_sigmf(inp, a.cols)
    else:
        caps = lk.regions_from_preprocessed(inp, a.cols)
    result = {}
    for stem, cap in caps.items():
        if a.select and not any(s in stem for s in a.select):
            continue
        if not cap.get("data_path"):
            print(f"skip {stem}: no IQ", file=sys.stderr)
            continue
        tracks = lk.build_tracks(cap["regions"], cap["n_frames"])
        ciq = CaptureIQ(cap["data_path"], cap["dtype"], cap["rate"], cap["fc"])
        done = []
        for t in tracks[: a.max_tracks or None]:
            s = measure_track(t, ciq)
            done.append((t, s))
            m = s.get("mod", {})
            print(f"{stem} track {t.tid}: {t.fc / 1e6:.4f} MHz bw {t.bw / 1e3:.1f} kHz {t.temporal}: class {m.get('class')} ({m.get('class_conf')}), "
                  f"snr {m.get('snr_db')} dB, cv {m.get('env_cv')}, dev {m.get('dev_p99_hz')} Hz, ofdm {s['ofdm'].get('scs_hz', '')} lines {s['lines'].get('abs2', [])[:1]} "
                  f"{s['lines'].get('finst', [])[:1]} coarse n {s['coarse'].get('n')} p50 {s['coarse'].get('dur_p50')} pri {s['coarse'].get('pri', [])[:2]} ({s['seconds']} s)", flush=True)
        band = measure_band(done, ciq.n / ciq.rate)
        result[stem] = {"tracks": {t.tid: s for t, s in done}, "band": band}
        print(f"{stem}: band {band}")
    out = Path(a.out) if a.out else (inp.parent if inp.is_file() else inp) / "sdw.json"
    out.write_text(json.dumps(result, default=_json_default))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
