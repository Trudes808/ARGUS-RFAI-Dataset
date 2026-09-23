"""Zadoff-Chu load + frequency-shifted matched-filter synchronization.

The composer places the 50 MHz ZC (generated at 245.76 MSps) at each block's
center frequency and at the start of each block in time. The receiver scans the
known candidate block centers, frequency-shifts a ZC replica to each, and
matched-filters against the wideband capture to find the ZC start sample for
every present block.
"""

from __future__ import annotations
import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, oaconvolve, find_peaks


def load_zc(mat_path):
    """Load the ZC IQ (complex64, 245.76 MSps) from the generator's .mat."""
    d = loadmat(mat_path)
    zc = np.asarray(d["f_sig"]).ravel().astype(np.complex64)
    return zc


def freq_shift(x, fc, fs):
    n = np.arange(x.size)
    return (x * np.exp(2j * np.pi * fc / fs * n)).astype(np.complex64)


def matched_filter(rx, replica):
    """Return correlation magnitude vs ZC-start sample (length len(rx))."""
    h = np.conj(replica[::-1])
    y = fftconvolve(rx, h, mode="full")
    # peak at index k corresponds to ZC starting at k - (len(replica)-1)
    start = y[len(replica) - 1:len(replica) - 1 + len(rx)]
    return np.abs(start)


def active_windows(rx, guard_samples, coarse_win=1 << 14, margin_win=3,
                   active_factor=1.5):
    """Energy-gate the capture: return (start, end) sample windows around each
    rising edge (silence -> signal), i.e. the start of every time group. The
    25 ms guards make this bimodal and robust; falls back to the whole capture
    if no clear silence/signal split is found.

    `active_factor` sets how far above the noise floor a coarse window must sit
    to count as active. It must stay BELOW the weakest real ZC/metadata burst:
    a single-block time group emits only one ZC (vs three for a 3-block group),
    so its wideband power over a coarse window is ~1/3 as high and has been seen
    as low as ~2.3x the floor (25 dB attenuation). The noise floor itself is
    very flat (windows sit within a few percent of it up to ~p90), so there is a
    wide clean gap; 1.5 is the log-midpoint of that gap (~1.5x above the noise
    top, ~1.5x below the single-block level) for the most consistent margin. The
    earlier 4.0 (~6 dB) silently dropped every single-block group. This gate is
    still fundamentally power-based, so heavier attenuation (e.g. 30 dB) can
    push a single ZC below any safe threshold -- there, use fine_comb_decode.py
    (a full matched-filter sweep with no energy gate)."""
    rx = np.asarray(rx)
    N = rx.size
    W = int(coarse_win)
    nwin = N // W
    if nwin < 8:
        return [(0, N)]
    env = np.empty(nwin, dtype=np.float32)            # coarse power per window (chunked)
    B = 4096
    for i in range(0, nwin, B):
        j = min(i + B, nwin)
        blk = rx[i * W:j * W]
        pw = blk.real.astype(np.float32) ** 2 + blk.imag.astype(np.float32) ** 2
        env[i:j] = pw.reshape(j - i, W).mean(axis=1)
    lo = float(np.percentile(env, 20))                # guards dominate -> noise floor
    hi = float(np.percentile(env, 99))                # strongest active windows
    if hi <= max(lo, 1e-30) * active_factor:
        return [(0, N)]                               # no clear signal -> search all
    # Threshold a few dB above the NOISE floor (not the signal level) so a
    # low-power ZC/metadata island stays flagged active even when the data
    # waveforms in the block are much stronger or a channel has attenuated it.
    thr = max(lo, 1e-30) * active_factor
    active = env > thr
    # Only treat a rising edge as a group start if it follows >= ~1 ms of silence,
    # so short low-energy dips *inside* a waveform don't each spawn a search window.
    min_sil = max(1, int(round(245760.0 / W)))
    rise = []
    sil = min_sil                                     # capture start counts as "after silence"
    for wi in range(nwin):
        if active[wi]:
            if sil >= min_sil:
                rise.append(wi)
            sil = 0
        else:
            sil += 1
    wins = []
    for e in rise:
        s = max(0, (int(e) - margin_win) * W)
        en = min(N, (int(e) + margin_win) * W)
        wins.append([s, en])
    wins.sort()
    merged = []
    for w in wins:
        if merged and w[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], w[1])
        else:
            merged.append(w)
    return [(int(s), int(e)) for s, e in merged]


def _correlate_segment(seg, abs_offset, zc, fs, centers, Ezc, thresh, L):
    """Normalized matched-filter ZC detection inside one segment (chunked if big)."""
    n = seg.size
    if n < L + 1:
        return []
    p = (np.abs(seg).astype(np.float64)) ** 2
    ce = np.concatenate(([0.0], np.cumsum(p)))
    denom = np.sqrt(np.maximum(ce[L:n + 1] - ce[0:n - L + 1], 1e-20) * Ezc)
    dets = []
    for fc in centers:
        mf = np.conj(freq_shift(zc, fc, fs)[::-1])
        cc = np.abs(fftconvolve(seg, mf, mode="valid"))   # index j = ZC start at abs_offset+j
        mlen = min(cc.size, denom.size)
        metric = cc[:mlen] / denom[:mlen]
        padded = np.concatenate(([0.0], metric, [0.0]))
        pk, _ = find_peaks(padded, height=thresh, distance=L)
        pk = pk - 1
        pk = pk[(pk >= 0) & (pk < mlen)]
        for k in pk:
            dets.append({"center_hz": float(fc), "start_sample": int(abs_offset + k),
                         "metric": float(metric[k])})
    return dets


def detect_multi(rx, zc, fs, centers, thresh=0.35, guard_samples=None,
                 active_factor=1.5):
    """Detect ALL ZC bursts (across time groups) at the candidate centers.

    Energy-gates the capture to a handful of small search windows (one per time
    group) and runs the precise normalized matched filter only there -- orders
    of magnitude faster than correlating the whole capture, and robust to where
    the signal sits within an over-captured recording. CRC on the decoded
    metadata is the final gate, so a moderate `thresh` is fine.
    """
    rx = np.asarray(rx).ravel().astype(np.complex64)
    L = zc.size
    Ezc = float(np.sum(np.abs(zc) ** 2))
    if guard_samples is None:
        guard_samples = L
    wins = active_windows(rx, guard_samples, active_factor=active_factor)
    BIG = 1 << 25                                     # chunk windows larger than this
    raw = []
    for ws, we_ in wins:
        if we_ - ws <= BIG:
            raw += _correlate_segment(rx[ws:we_], ws, zc, fs, centers, Ezc, thresh, L)
        else:
            start = ws
            while start < we_ - L:
                stop = min(start + BIG, we_)
                raw += _correlate_segment(rx[start:stop], start, zc, fs, centers, Ezc, thresh, L)
                start += BIG - 2 * L
    # de-duplicate peaks within L samples per center, keep the strongest
    raw.sort(key=lambda d: (d["center_hz"], d["start_sample"]))
    dets = []
    for d in raw:
        if dets and dets[-1]["center_hz"] == d["center_hz"] \
                and d["start_sample"] - dets[-1]["start_sample"] < L:
            if d["metric"] > dets[-1]["metric"]:
                dets[-1] = d
        else:
            dets.append(d)
    return dets


def detect(rx, zc, fs, centers, rel_threshold=0.5, min_sep=None):
    """Detect ZC bursts at the given candidate center frequencies.

    Returns a list of dicts: {center_hz, start_sample, metric}. `metric` is the
    matched-filter peak normalized by the local signal energy (so a clean ZC
    scores ~1). Detections below rel_threshold of the global best are dropped.
    """
    rx = np.asarray(rx).ravel().astype(np.complex64)
    L = zc.size
    norm = np.sqrt(np.mean(np.abs(rx) ** 2) * L) + 1e-12
    if min_sep is None:
        min_sep = L
    dets = []
    for fc in centers:
        rep = freq_shift(zc, fc, fs)
        corr = matched_filter(rx, rep) / norm
        k = int(np.argmax(corr))
        dets.append({"center_hz": float(fc), "start_sample": k,
                     "metric": float(corr[k])})
    if not dets:
        return []
    best = max(d["metric"] for d in dets)
    keep = [d for d in dets if d["metric"] >= rel_threshold * best]
    keep.sort(key=lambda d: -d["metric"])
    return keep
