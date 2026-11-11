#!/usr/bin/env python3
"""Unit test of measure_descriptors.py: synthetic emissions with known parameters are written as ci16 SigMF
captures, wrapped in detector-like boxes, measured, and the recovered parameters compared with the truth.

    python3 tests/test_measure_synthetic.py [--snr 20] [--keep] [--only ble wifi]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common_waveforms_look_up as lk  # noqa: E402
import manifest_io as mio  # noqa: E402
import measure_descriptors as md  # noqa: E402
import synth_modulators as sm  # noqa: E402


def write_capture(x: np.ndarray, fs: float, fc: float, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = 0.3 * 32767 / max(np.max(np.abs(x)), 1e-9)
    iq = np.empty(2 * len(x), dtype="<i2")
    iq[0::2] = np.clip(np.round(x.real * scale), -32768, 32767)
    iq[1::2] = np.clip(np.round(x.imag * scale), -32768, 32767)
    (out_dir / f"{stem}.sigmf-data").write_bytes(iq.tobytes())
    meta = {"global": {"core:datatype": "ci16_le", "core:sample_rate": fs, "core:version": "1.0.0"},
            "captures": [{"core:sample_start": 0, "core:frequency": fc}], "annotations": []}
    (out_dir / f"{stem}.sigmf-meta").write_text(json.dumps(meta))
    return out_dir / f"{stem}.sigmf-data"


def place(buf: np.ndarray, sig: np.ndarray, fs: float, f_off: float, t0: float, amp: float):
    n0 = int(t0 * fs)
    n = min(len(sig), len(buf) - n0)
    if n <= 0:
        return
    t = (np.arange(n) + n0) / fs
    buf[n0: n0 + n] += (amp * sig[:n] * np.exp(2j * np.pi * f_off * t)).astype(np.complex64)


def make_track(fs: float, fc: float, f_lo: float, f_hi: float, dur: float, on_times: list[tuple[float, float]] | None, cols: int = 1024) -> lk.Track:
    """Detector-like regions: one per frame (continuous) or one per burst, quantised to the frame geometry."""
    geo = mio.frame_geometry(fs, cols)
    regs = []
    n_frames = int(dur / geo["frame_s"])
    if on_times is None:
        for f in range(n_frames):
            regs.append(lk.Region("synth", f_lo, f_hi, f * geo["frame_s"], geo["frame_s"], f, fs, fc, geo["bin_hz"], geo["row_s"], geo["frame_s"]))
    else:
        for t0, d in on_times:
            f = int(t0 // geo["frame_s"])
            regs.append(lk.Region("synth", f_lo, f_hi, t0, max(d, geo["row_s"]), f, fs, fc, geo["bin_hz"], geo["row_s"], geo["frame_s"]))
    tracks = lk.build_tracks(regs, max(n_frames, 1))
    return max(tracks, key=lambda t: len(t.regions))


def burst_times(dur: float, period: float, burst: float, jitter: float = 0.0, rng=None) -> list[tuple[float, float]]:
    rng = rng or sm.RNG
    out, t = [], 0.002
    while t + burst < dur:
        out.append((t, burst))
        t += period + (rng.uniform(-jitter, jitter) if jitter else 0.0)
    return out


def noise(n: int, rng) -> np.ndarray:
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)).astype(np.complex64)


# ----------------------------------------------------------------------------------------------- cases
def case_ble(fs, dur, rng):
    """BLE 1M advertising-like bursts of 300 us every 20 ms, each followed after 150 us by a 176 us scan request."""
    rs, h, bt = 1e6, 0.5, 0.5
    on = []
    buf = np.zeros(int(dur * fs), dtype=np.complex64)
    t = 0.003
    while t < dur - 0.002:
        for d in (300e-6, 176e-6):
            sig = sm.cpm(sm.fsk_bits(int(d * rs), rng=rng), rs, fs, h, bt)
            place(buf, sig, fs, 0.0, t, 1.0)
            on.append((t, d))
            t += d + 150e-6
        t += 20e-3 - 626e-6 + rng.uniform(0, 5e-3)
    return buf, on, {"class": "fsk", "finst_line": rs, "sq_spacing": rs, "coarse_p50": (300e-6, 176e-6), "gap": 150e-6, "dev_p99": (200e3, 320e3)}


def case_wifi(fs, dur, rng):
    """20 MHz OFDM packets (312.5 kHz, CP 0.8 us) + SIFS 10 us + 44 us ACK; beacons every 102.4 ms."""
    buf = np.zeros(int(dur * fs), dtype=np.complex64)
    on = []
    t = 0.001
    beacon_next = 0.0
    while t < dur - 0.004:
        if t >= beacon_next:
            d = 1.0e-3
            beacon_next += 102.4e-3
        else:
            d = rng.uniform(150e-6, 800e-6)
        pkt = sm.ofdm(fs, 312.5e3, 0.8e-6, 52, d, rng=rng)
        place(buf, pkt[: int(d * fs)], fs, 0.0, t, 1.0)
        on.append((t, d))
        t += d + 10e-6
        ack = sm.ofdm(fs, 312.5e3, 0.8e-6, 52, 44e-6, rng=rng)
        place(buf, ack[: int(44e-6 * fs)], fs, 0.0, t, 1.0)
        on.append((t, 44e-6))
        t += 44e-6 + rng.uniform(0.3e-3, 3e-3)
    return buf, on, {"class": "ofdm", "tu": 3.2e-6, "cp": 0.8e-6, "gap": 10e-6, "period": 102.4e-3}


def case_lte_dl(fs, dur, rng):
    sig = sm.ofdm(fs, 15e3, 4.69e-6, 600, dur, rng=rng)
    buf = sig[: int(dur * fs)].copy()
    return buf, None, {"class": "ofdm", "tu": 66.67e-6, "cp": 4.69e-6}


def case_nr_tdd(fs, dur, rng):
    mask = sm.tdd_pattern("DDDSU", 0.5e-3, 30e3, 2.34e-6)
    sig = sm.ofdm(fs, 30e3, 2.34e-6, 600, dur, rng=rng, symbol_on=mask)
    buf = sig[: int(dur * fs)].copy()
    return buf, None, {"class": "ofdm", "tu": 33.33e-6, "cp": 2.34e-6, "period": 2.5e-3}


def case_fm(fs, dur, rng):
    sig = sm.fm_stereo(fs, dur, pilot=True, rds=True, hd=False, rng=rng)
    return sig[: int(dur * fs)].copy(), None, {"class": "fm", "fm_lines": (19e3,), "dev_p99": (40e3, 90e3)}


def case_fm_hd(fs, dur, rng):
    sig = sm.fm_stereo(fs, dur, pilot=True, rds=True, hd=True, rng=rng)
    return sig[: int(dur * fs)].copy(), None, {"class": "fm", "fm_lines": (19e3,), "hd": True}


def case_nbfm(fs, dur, rng):
    sig = sm.nbfm_voice(fs, dur, dev_hz=2.5e3, ctcss_hz=100.0, rng=rng)
    return sig[: int(dur * fs)].copy(), None, {"class": "fm", "dev_p99": (1.5e3, 4e3)}


def case_am(fs, dur, rng):
    sig = sm.am_mod(fs, dur, audio_bw_hz=2.5e3, depth=0.8, rng=rng)
    return sig[: int(dur * fs)].copy(), None, {"class": "am", "carrier_line": 12}


def case_vor(fs, dur, rng):
    sig = sm.am_tones(fs, dur, [30.0, 9960.0, 1020.0], [0.3, 0.3, 0.1], rng=rng)
    return sig[: int(dur * fs)].copy(), None, {"class": "am", "am_lines": (30.0, 9960.0)}


def case_c4fm(fs, dur, rng):
    rs = 4800.0
    sig = sm.cpm(sm.fsk_bits(int(dur * rs) + 1, levels=4, rng=rng), rs, fs, 0.75, bt=None, levels=4)
    # C4FM uses raised-cosine shaping; a 4-level CPFSK with the same deviation is close enough for the rate line
    return sig[: int(dur * fs)].copy(), None, {"class": "fsk", "finst_line": rs, "n_levels": 4, "dev_p99": (1.5e3, 2.2e3)}


def case_dect(fs, dur, rng):
    rs = 1.152e6
    on = burst_times(dur, 10e-3, 364.6e-6)
    buf = np.zeros(int(dur * fs), dtype=np.complex64)
    for t, d in on:
        sig = sm.cpm(sm.fsk_bits(420, rng=rng), rs, fs, 0.5, 0.5)
        place(buf, sig, fs, 0.0, t, 1.0)
    return buf, on, {"class": "fsk", "finst_line": rs, "pri": 10e-3, "coarse_p50": (364.6e-6,)}


def case_adsb(fs, dur, rng):
    buf = np.zeros(int(dur * fs), dtype=np.complex64)
    on = []
    t = 0.001
    while t < dur - 0.001:
        msg = sm.adsb_message(fs, long=True, rng=rng)
        place(buf, msg, fs, 0.0, t, 1.0)
        on.append((t, 120e-6))
        t += rng.uniform(2e-3, 12e-3)
    return buf, on, {"class": "pulse", "fine_p50": (0.3e-6, 0.75e-6), "coarse_p50": (120e-6,), "pri_fine": 1e-6, "abs2_line": 2e6}


def case_dme(fs, dur, rng):
    buf = np.zeros(int(dur * fs), dtype=np.complex64)
    on = []
    t = 0.001
    pair = sm.dme_pair(fs)
    while t < dur - 0.001:
        place(buf, pair, fs, 0.0, t, 1.0)
        on.append((t, 20e-6))
        t += rng.uniform(0.3e-3, 1.0e-3)
    return buf, on, {"class": "pulse", "fine_p50": (2.5e-6, 4.8e-6), "pri_fine": 12e-6}


def case_lora(fs, dur, rng):
    buf = np.zeros(int(dur * fs), dtype=np.complex64)
    on = []
    t = 0.005
    while t < dur - 0.1:
        b = sm.lora_burst(fs, 125e3, 7, 30, rng=rng)
        place(buf, b, fs, 0.0, t, 1.0)
        on.append((t, len(b) / fs))
        t += len(b) / fs + rng.uniform(0.05, 0.15)
    return buf, on, {"class": "css", "chirp_rate": 125e3 ** 2 / 128}


def case_vsb(fs, dur, rng):
    sig = sm.vsb8(fs, dur, rng=rng)
    return sig[: int(dur * fs)].copy(), None, {"class": "vsb", "abs2_line": 10.762238e6, "carrier_offset": -(3e6 - 309.441e3)}


def case_zigbee(fs, dur, rng):
    buf = np.zeros(int(dur * fs), dtype=np.complex64)
    on = []
    t = 0.002
    while t < dur - 0.01:
        d = rng.uniform(0.5e-3, 3e-3)
        sig = sm.oqpsk_halfsine(fs, 2e6, int(d * 2e6), rng=rng)
        place(buf, sig, fs, 0.0, t, 1.0)
        on.append((t, d))
        t += d + 192e-6
        ack = sm.oqpsk_halfsine(fs, 2e6, 704, rng=rng)
        place(buf, ack, fs, 0.0, t, 1.0)
        on.append((t, 352e-6))
        t += 352e-6 + rng.uniform(5e-3, 30e-3)
    return buf, on, {"class": "fsk", "sq_spacing": 2e6, "gap": 192e-6, "coarse_p50": (352e-6,)}


def case_pocsag(fs, dur, rng):
    rs = 1200.0
    sig = sm.cpm(sm.fsk_bits(int(dur * rs) + 1, rng=rng), rs, fs, 7.5, bt=None)   # +-4.5 kHz at 1200 bps: h = 7.5
    return sig[: int(dur * fs)].copy(), None, {"class": "fsk", "finst_line": rs, "sq_spacing": 18000.0, "dev_p99": (3.5e3, 6e3), "n_levels": 2}


CASES = {
    "ble": (case_ble, 10e6, 2.0e6, 0.6),
    "wifi": (case_wifi, 50e6, 20e6, 0.45),
    "lte_dl": (case_lte_dl, 25e6, 9.0e6, 0.3),
    "nr_tdd": (case_nr_tdd, 50e6, 18.0e6, 0.3),
    "fm": (case_fm, 5e6, 0.2e6, 0.6),
    "fm_hd": (case_fm_hd, 5e6, 0.4e6, 0.6, 10.0),
    "nbfm": (case_nbfm, 2e6, 12.5e3, 0.6),
    "am": (case_am, 2e6, 6e3, 0.6),
    "vor": (case_vor, 2e6, 22e3, 0.6),
    "c4fm": (case_c4fm, 2e6, 12.5e3, 0.6),
    "pocsag": (case_pocsag, 2e6, 12.5e3, 0.6),
    "dect": (case_dect, 10e6, 1.5e6, 0.6),
    "adsb": (case_adsb, 25e6, 3e6, 0.4),
    "dme": (case_dme, 10e6, 0.6e6, 0.4),
    "lora": (case_lora, 2e6, 125e3, 1.0),
    "vsb": (case_vsb, 25e6, 6e6, 0.3),
    "zigbee": (case_zigbee, 10e6, 2.5e6, 0.6),
}


def run_case(name: str, snr_db: float, out_dir: Path, rng) -> tuple[dict, dict, dict]:
    fn, fs, bw, dur = CASES[name][:4]
    snr_db += CASES[name][4] if len(CASES[name]) > 4 else 0.0
    sig, on, truth = fn(fs, dur, rng)
    n = len(sig)
    sig_p = np.mean(np.abs(sig[np.abs(sig) > 0]) ** 2) if np.any(sig) else 1.0
    # noise so that the in-band SNR is snr_db: noise power in bw = sig_p / 10^(snr/10); white over fs
    noise_p_total = sig_p / 10 ** (snr_db / 10) * (fs / bw)
    x = sig + noise(n, rng) * math.sqrt(noise_p_total)
    f_off = rng.uniform(-0.3, 0.3) * fs
    f_off = round(f_off / (fs / 1024)) * (fs / 1024)                    # on a detector bin
    t = np.arange(n) / fs
    x = (x * np.exp(2j * np.pi * f_off * t)).astype(np.complex64)
    fc = 1.0e9
    path = write_capture(x, fs, fc, out_dir, f"synth_{name}")
    bin_hz = fs / 1024
    f_lo = fc + f_off - 0.5 * bw
    f_hi = fc + f_off + 0.5 * bw
    f_lo = math.floor((f_lo - (fc - fs / 2)) / bin_hz) * bin_hz + (fc - fs / 2)
    f_hi = math.ceil((f_hi - (fc - fs / 2)) / bin_hz) * bin_hz + (fc - fs / 2)
    track = make_track(fs, fc, f_lo, f_hi, dur, on)
    cap = md.CaptureIQ(path, "ci16_le", fs, fc)
    sdw = md.measure_track(track, cap)
    return truth, sdw, {"fs": fs, "bw": bw, "f_off": f_off, "temporal": track.temporal, "n_regions": len(track.regions)}


def near(x, v, tol=0.03):
    return x is not None and v and abs(x - v) <= tol * abs(v)


def check(name: str, truth: dict, s: dict) -> list[str]:
    """Return the list of failed expectations."""
    fails = []
    m, L, o, fine, co = s.get("mod", {}), s.get("lines", {}), s.get("ofdm", {}), s.get("fine", {}), s.get("coarse", {})
    if truth.get("class") and m.get("class") != truth["class"]:
        if not (truth["class"] == "vsb" and m.get("class") in ("psk", "am")):
            fails.append(f"class {m.get('class')} != {truth['class']}")
    if "finst_line" in truth:
        r = truth["finst_line"]
        ok = any(near(f, k * r, 0.03) for f, _ in L.get("finst", [])[:3] for k in (1, 2, 3))
        ok = ok or (truth.get("sq_spacing") and near(L.get("sq_spacing_hz"), truth["sq_spacing"], 0.03))
        if not ok:
            fails.append(f"finst line {r} (or harmonic / y^2 spacing) not in {L.get('finst')} / {L.get('sq_spacing_hz')}")
    if "sq_spacing" in truth and L.get("sq_spacing_hz") is not None:
        sp = L["sq_spacing_hz"]
        if not (near(sp, truth["sq_spacing"], 0.05) or near(sp, 2 * truth["sq_spacing"], 0.05)):
            fails.append(f"y^2 spacing {sp} vs {truth['sq_spacing']}")
    if "abs2_line" in truth:
        fs = s["fs_hz"]
        v = truth["abs2_line"]
        alias = [abs(v - k * fs) for k in range(-3, 4)]
        sq_ok = truth.get("class") == "vsb" and near(L.get("sq_spacing_hz"), v, 0.01)
        if not sq_ok and not any(any(near(f, a, 0.03) for a in alias if a > 0) for f, _ in L.get("abs2", [])[:3]):
            fails.append(f"|y|^2 line {v} (aliases {[round(a) for a in alias if 0 < a < fs/2]}) not in {L.get('abs2')}")
    if "tu" in truth:
        if not near(o.get("tu_s"), truth["tu"], 0.03):
            fails.append(f"ofdm Tu {o.get('tu_s')} != {truth['tu']} (peaks {o.get('peaks', [])[:2]})")
        elif "cp" in truth and not near(o.get("cp_s"), truth["cp"], 0.25):
            fails.append(f"ofdm CP {o.get('cp_s')} != {truth['cp']}")
    if "coarse_p50" in truth:
        p = co.get("dur_p50")
        if not any(near(p, v, 0.35) for v in truth["coarse_p50"]):
            fails.append(f"coarse burst p50 {p} not in {truth['coarse_p50']} (n {co.get('n')})")
    if "fine_p50" in truth:
        p = fine.get("dur_p50")
        lo, hi = truth["fine_p50"]
        if p is None or not (lo <= p <= hi):
            fails.append(f"fine burst p50 {p} not in [{lo}, {hi}] (n {fine.get('n')})")
    if "gap" in truth:
        if not any(near(g["value"], truth["gap"], 0.2) for g in co.get("gap", [])[:4]):
            fails.append(f"gap {truth['gap']} not in {co.get('gap')}")
    if "pri" in truth:
        if not any(near(g["value"], truth["pri"], 0.05) for g in co.get("pri", [])[:4]):
            fails.append(f"pri {truth['pri']} not in {co.get('pri')}")
    if "pri_fine" in truth:
        if not any(near(g["value"], truth["pri_fine"], 0.1) for g in fine.get("pri", [])[:5]):
            fails.append(f"fine pri {truth['pri_fine']} not in {fine.get('pri')}")
    if "period" in truth:
        ok = any(near(p, truth["period"], 0.03) for p, _ in s.get("periods", [])[:6])
        ok = ok or any(near(r["value"], truth["period"], 0.03) for r in co.get("pri_by_dur", []))
        if not ok:
            fails.append(f"period {truth['period']} not in {s.get('periods')} nor {co.get('pri_by_dur')}")
    if "dev_p99" in truth:
        lo, hi = truth["dev_p99"]
        d = m.get("dev_est_hz")
        if d is None or not (0.5 * lo <= d <= 1.3 * hi):
            fails.append(f"dev est {d} not in [{0.5 * lo}, {1.3 * hi}] (p90 {m.get('dev_p90_hz')}, levels {m.get('dev_levels_hz')})")
    if "n_levels" in truth and m.get("n_levels") != truth["n_levels"]:
        fails.append(f"levels {m.get('n_levels')} != {truth['n_levels']}")
    if "fm_lines" in truth:
        fl = [f for f, _ in s.get("fm", {}).get("lines", [])]
        for v in truth["fm_lines"]:
            if not any(near(f, v, 0.01) for f in fl):
                fails.append(f"fm tone {v} not in {fl[:6]}")
    if "am_lines" in truth:
        al = [f for f, _ in s.get("am", {}).get("lines", [])]
        for v in truth["am_lines"]:
            if not any(near(f, v, 0.02) for f in al):
                fails.append(f"am tone {v} not in {al[:6]}")
    if "carrier_line" in truth and (m.get("carrier_line_db") or 0) < truth["carrier_line"]:
        fails.append(f"carrier line {m.get('carrier_line_db')} < {truth['carrier_line']}")
    if "carrier_offset" in truth and not near(m.get("carrier_offset_hz"), truth["carrier_offset"], 0.05):
        fails.append(f"carrier offset {m.get('carrier_offset_hz')} != {truth['carrier_offset']}")
    if "chirp_rate" in truth and not near(m.get("chirp_rate_hz_s"), truth["chirp_rate"], 0.1):
        fails.append(f"chirp rate {m.get('chirp_rate_hz_s')} != {truth['chirp_rate']} (consistency {m.get('chirp_consistency')})")
    if truth.get("hd") and (s.get("hd", {}).get("sideband_db") or -99) < 6:
        fails.append(f"hd sidebands {s.get('hd')}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    sm.set_seed(a.seed)
    out_dir = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="lut_measure_"))
    names = a.only or list(CASES)
    n_fail = 0
    for name in names:
        truth, sdw, info = run_case(name, a.snr, out_dir, rng)
        fails = check(name, truth, sdw)
        n_fail += bool(fails)
        m = sdw.get("mod", {})
        print(f"[{'FAIL' if fails else ' ok '}] {name:8s} fs {info['fs']/1e6:g} MSps bw {info['bw']/1e3:g} kHz {info['temporal']:11s} -> class {m.get('class')} ({m.get('class_conf')}) "
              f"snr {m.get('snr_db')} cv {m.get('env_cv')} dev99 {m.get('dev_p99_hz')} lvl {m.get('n_levels')} | ofdm {sdw['ofdm'].get('tu_s')}/{sdw['ofdm'].get('cp_s')} | "
              f"abs2 {sdw['lines'].get('abs2', [])[:1]} finst {sdw['lines'].get('finst', [])[:1]} sq {sdw['lines'].get('sq_spacing_hz')} | "
              f"fine n{sdw['fine'].get('n')} p50 {sdw['fine'].get('dur_p50')} | coarse n{sdw['coarse'].get('n')} p50 {sdw['coarse'].get('dur_p50')} pri {[(round(p['value'],7), p['share']) for p in sdw['coarse'].get('pri', [])[:2]]} "
              f"gap {[(round(p['value'],7), p['share']) for p in sdw['coarse'].get('gap', [])[:2]]} | periods {[(p, h) for p, h in sdw.get('periods', [])[:2]]} "
              f"bydur {[(round(r['dur'],6), round(r['value'],5), r['n']) for r in sdw['coarse'].get('pri_by_dur', [])[:2]]} | {sdw.get('seconds')} s")
        for f in fails:
            print(f"        - {f}")
    print(f"{len(names) - n_fail}/{len(names)} cases pass at {a.snr} dB")
    if not a.keep:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
