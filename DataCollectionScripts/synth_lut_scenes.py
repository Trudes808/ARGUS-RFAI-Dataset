#!/usr/bin/env python3
"""Synthetic captures for every library entry, generated from its own descriptor and `synth` recipe.

    python3 synth_lut_scenes.py --out /tmp/lut_synth --draws 3 --snr 10 30
    python3 synth_lut_scenes.py --out /tmp/lut_synth --entries fm_broadcast bluetooth_le --draws 5 --dur 0.5

Each draw is one SigMF capture (ci16) holding one emission of the entry at a random legal channel, a random capture
rate the entry's sweep would use, a random in-band SNR, on AWGN (or a real quiet-band recording with --background),
plus the ground-truth boxes written as annotations in the detector's frame geometry (so lut_classify.py can read
the capture directly, bypassing the detector) and a `<stem>.truth.json`. The emissions follow the standard's
numbers (symbol rates, numerology, timing) from the descriptor; the recipe only adds what the descriptor does not
say (audio model, packet lengths, traffic).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decode_waveforms as dw  # noqa: E402
import manifest_io as mio  # noqa: E402
import synth_modulators as sm  # noqa: E402

DEFAULT_RATES = [5e6, 10e6, 12.5e6, 20e6, 25e6, 31.25e6, 50e6]


def _rng_choice(rng, x):
    return x[int(rng.integers(0, len(x)))]


def _pick(rng, spec, default):
    """A scalar from a scalar / [lo, hi] / list spec."""
    if spec is None:
        return default
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec) and spec[0] < spec[1] and not isinstance(spec[0], bool):
            return float(rng.uniform(spec[0], spec[1]))
        return _rng_choice(rng, spec)
    return spec


def burst_schedule(rng, dur, rate_per_s, burst_len_fn, t_start=0.003):
    """Poisson-ish bursts: (t0, length) pairs."""
    out, t = [], t_start
    while t < dur - 0.002:
        d = min(burst_len_fn(), dur - t - 0.001)
        if d <= 0:
            break
        out.append((t, d))
        t += d + float(rng.exponential(1.0 / max(rate_per_s, 1e-3)))
    return out


# ------------------------------------------------------------------------------------------- recipes
# each returns (baseband complex64 at fs_e, fs_e, bw_occupied_hz, on_times or None, params dict)
def r_fm_stereo(e, d, syn, rng, dur):
    fs_e = 1.0e6
    hd = rng.random() < float(syn.get("hd_prob", 0.3))
    pi = int(rng.integers(0x1000, 0xFFFF))
    ps = "K" + "".join(chr(int(c)) for c in rng.integers(65, 91, 3)) + " FM "
    rds_bits = dw.rds_groups(pi, ps, int(rng.integers(1, 30)), 40, rng) if syn.get("rds", True) else None
    x = sm.fm_stereo(fs_e, dur, float(syn.get("audio_bw_hz", 15e3)), bool(syn.get("pilot", True)), bool(syn.get("rds", True)), hd, rng=rng, rds_bits=rds_bits)
    return x, fs_e, (400e3 if hd else 190e3), None, {"hd": hd, "rds_pi": f"{pi:04X}", "rds_ps": ps.strip()}


def r_nbfm_voice(e, d, syn, rng, dur):
    fs_e = 100e3
    dev = float(syn.get("deviation_hz", 2.5e3))
    on = burst_schedule(rng, dur, 0.6, lambda: float(rng.uniform(0.3, 1.5))) if syn.get("ptt") or syn.get("continuous") is False else None
    x = sm.nbfm_voice(fs_e, dur, float(syn.get("audio_bw_hz", 3e3)), dev, syn.get("ctcss_hz"), rng=rng)
    if on:
        x = _gate(x, fs_e, on)
    return x, fs_e, 2 * (dev + float(syn.get("audio_bw_hz", 3e3))), on, {"dev_hz": dev}


def r_am_voice(e, d, syn, rng, dur):
    fs_e = 50e3
    bwa = float(syn.get("audio_bw_hz", 3e3))
    x = sm.am_mod(fs_e, dur, bwa, float(syn.get("mod_depth", 0.8)), rng=rng)
    on = burst_schedule(rng, dur, 0.6, lambda: float(rng.uniform(0.3, 1.5))) if syn.get("ptt") else None
    if on:
        x = _gate(x, fs_e, on)
    return x, fs_e, 2 * bwa, on, {}


def r_am_tones(e, d, syn, rng, dur):
    fs_e = 50e3
    tones = [float(t) for t in syn.get("tones_hz", [1000.0])]
    x = sm.am_tones(fs_e, dur, tones, syn.get("depth"), bool(syn.get("tick", False)), rng=rng)
    return x, fs_e, 2 * max(tones) + 2e3, None, {}


def r_am_msk(e, d, syn, rng, dur):
    fs_e = 50e3
    x = sm.am_msk(fs_e, dur, float(syn.get("bitrate", 2400.0)), rng=rng)
    on = burst_schedule(rng, dur, 1.0, lambda: float(rng.uniform(*syn.get("burst_s", [0.1, 0.8]))))
    return _gate(x, fs_e, on), fs_e, 6e3, on, {}


def r_vsb8(e, d, syn, rng, dur):
    fs_e = 2 * 10.762238e6
    return sm.vsb8(fs_e, dur, rng=rng), fs_e, 5.9e6, None, {}


def r_ofdm_continuous(e, d, syn, rng, dur):
    scs = float(syn["scs_hz"])
    cp = float(syn["cp_s"])
    bw = min(float(_pick(rng, syn.get("bw_hz"), 9e6)), 0.75 * MAX_RATE)
    n_used = int(bw / scs)
    n_used -= n_used % 2
    N = sm.ofdm_fft_size(n_used)
    fs_e = N * scs
    params = {"scs_hz": scs, "cp_s": cp, "bw_hz": bw}
    sym_fn = None
    if syn.get("lte_sync"):
        nid1, nid2 = int(rng.integers(0, 168)), int(rng.integers(0, 3))
        sym_fn = dw.lte_sync_overwrite(nid1, nid2, N, scs, cp)
        params.update({"pci": 3 * nid1 + nid2, "nid1": nid1, "nid2": nid2})
    if syn.get("raster_hz"):
        params["raster_hz"] = float(syn["raster_hz"])
    cp_pat = [5.21e-6] + [4.69e-6] * 6 if syn.get("lte_sync") else None        # LTE normal CP: 160 / 144 samples per slot
    x = sm.ofdm(fs_e, scs, cp, n_used, dur, rng=rng, sym_fn=sym_fn, cp_pattern_s=cp_pat)
    return x[: int(dur * fs_e)], fs_e, n_used * scs, None, params


def r_ofdm_tdd(e, d, syn, rng, dur):
    scs, cp = float(syn["scs_hz"]), float(syn["cp_s"])
    bw = min(float(_pick(rng, syn.get("bw_hz"), 19e6)), 0.75 * MAX_RATE)
    n_used = int(bw / scs)
    n_used -= n_used % 2
    N = 1 << int(math.ceil(math.log2(n_used * 1.25)))
    fs_e = N * scs
    mask = sm.tdd_pattern(str(syn.get("pattern", "DDDSU")), float(syn.get("slot_s", 0.5e-3)), scs, cp)
    x = sm.ofdm(fs_e, scs, cp, n_used, dur, rng=rng, symbol_on=mask)
    x = x[: int(dur * fs_e)]
    sym_s = 1 / scs + cp
    on, t, k = [], 0.0, 0
    while t < dur:
        if mask[k % len(mask)]:
            if on and abs(on[-1][0] + on[-1][1] - t) < 1e-9:
                on[-1] = (on[-1][0], on[-1][1] + sym_s)
            else:
                on.append((t, sym_s))
        t += sym_s
        k += 1
    return x, fs_e, n_used * scs, on, {"scs_hz": scs, "cp_s": cp, "pattern": syn.get("pattern")}


def r_scfdma_bursts(e, d, syn, rng, dur):
    scs, cp = float(syn["scs_hz"]), float(syn["cp_s"])
    bw = float(_pick(rng, syn.get("bw_hz"), 1.08e6))
    n_used = max(12, int(bw / scs))
    n_used -= n_used % 2
    N = 1 << int(math.ceil(math.log2(n_used * 1.25)))
    fs_e = N * scs
    sub = float(syn.get("subframe_s", 1e-3))
    n_sub = int(dur / sub)
    active = rng.random(n_sub) < float(rng.uniform(0.1, 0.5))
    sym_s = 1 / scs + cp
    per_sub = int(round(sub / sym_s))
    mask = np.repeat(active, per_sub)
    x = sm.ofdm(fs_e, scs, cp, n_used, dur, rng=rng, symbol_on=mask)[: int(dur * fs_e)]
    on = []
    for i, a in enumerate(active):
        if a:
            if on and abs(on[-1][0] + on[-1][1] - i * sub) < 1e-9:
                on[-1] = (on[-1][0], on[-1][1] + sub)
            else:
                on.append((i * sub, sub))
    return x, fs_e, n_used * scs, on, {"scs_hz": scs, "cp_s": cp, "bw_hz": bw}


def r_wifi_ofdm(e, d, syn, rng, dur):
    bw = min(float(_pick(rng, syn.get("bw_hz"), 20e6)), 0.6 * MAX_RATE)
    scs = float(syn.get("scs_hz", 312.5e3))
    n_used = int(bw / scs * 0.9)
    n_used -= n_used % 2
    N = 1 << int(math.ceil(math.log2(n_used * 1.25)))
    fs_e = N * scs
    cp = 0.8e-6
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    on = []
    beacon_s, sifs, load = float(syn.get("beacon_s", 102.4e-3)), float(syn.get("sifs_s", 10e-6)), float(_pick(rng, syn.get("load"), 0.2))
    ack_rng = syn.get("ack_s", [28e-6, 44e-6])
    pk_rng = syn.get("packet_s", [0.1e-3, 3e-3])
    t, next_beacon = 0.001, 0.001
    while t < dur - 0.004:
        if t >= next_beacon:
            dpk = float(rng.uniform(0.4e-3, 1.5e-3))
            next_beacon += beacon_s
            with_ack = False
        else:
            dpk = float(rng.uniform(pk_rng[0], pk_rng[1]))
            with_ack = True
        _place(buf, sm.ofdm(fs_e, scs, cp, n_used, dpk + 2 * (1 / scs + cp), rng=rng)[: int(dpk * fs_e)], fs_e, t)
        on.append((t, dpk))
        t += dpk
        if with_ack:
            t += sifs
            dack = float(rng.uniform(ack_rng[0], ack_rng[1]))
            _place(buf, sm.ofdm(fs_e, scs, cp, n_used, dack + 2 * (1 / scs + cp), rng=rng)[: int(dack * fs_e)], fs_e, t)
            on.append((t, dack))
            t += dack
        t += float(rng.exponential(dpk / max(load, 0.02)))
    return buf, fs_e, n_used * scs, on, {"bw_hz": bw, "scs_hz": scs, "load": load}


def r_fsk_burst(e, d, syn, rng, dur):
    rs = float(syn["symbol_rate_hz"])
    h = float(syn.get("h", 0.5))
    bt = syn.get("bt")
    levels = int(syn.get("levels", 2))
    fs_e = max(8 * rs, 4 * (h * rs + rs))
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    on = []
    codec = syn.get("codec")
    if syn.get("frame_s"):                       # TDMA slots (DECT): active slots of every frame
        frame, nslots = float(syn["frame_s"]), int(syn.get("slots_per_frame", 24))
        slot = frame / nslots
        n_active = int(_pick(rng, syn.get("active_slots", [1, 2]), 1))
        slots = sorted(rng.choice(nslots, n_active, replace=False))
        bits = int(_pick(rng, syn.get("burst_bits", [420]), 420))
        rfpi = bytes(int(v) for v in rng.integers(0, 256, 5))
        t = 0.001
        while t + frame < dur:
            for sl in slots:
                t0 = t + sl * slot
                if codec == "dect":
                    b = dw.dect_packet_bits(sl < nslots // 2, rfpi if rng.random() < 0.5 else None, rng).astype(np.float64) * 2 - 1
                else:
                    b = sm.fsk_bits(bits, levels, rng=rng)
                x = sm.cpm(b, rs, fs_e, h, bt, levels)
                _place(buf, x, fs_e, t0)
                on.append((t0, len(b) / rs))
            t += frame
    elif syn.get("slot_s"):                      # slotted (AIS): bursts start on slot boundaries
        slot = float(syn["slot_s"])
        bits = int(_pick(rng, syn.get("burst_bits", [256]), 256))
        rate = float(_pick(rng, syn.get("rate_per_s"), 5))
        t = 0.002
        while t + bits / rs < dur:
            if rng.random() < rate * slot:
                x = sm.cpm(sm.fsk_bits(bits, levels, rng=rng), rs, fs_e, h, bt, levels)
                _place(buf, x, fs_e, t)
                on.append((t, bits / rs))
            t += slot
    else:
        rate = float(_pick(rng, syn.get("rate_per_s"), 5))
        bits_spec = syn.get("burst_bits")
        bs = syn.get("burst_s")
        def blen():
            if bits_spec:
                return int(_pick(rng, bits_spec, 256)) / rs
            return float(_pick(rng, bs, 0.01))
        on = burst_schedule(rng, dur, rate, blen)
        for t0, dl in on:
            nb = int(dl * rs) + 1
            if codec == "uat":
                b = dw.uat_frame_bits(nb > 3000, nb, rng).astype(np.float64) * 2 - 1
            else:
                b = sm.fsk_bits(nb, levels, rng=rng)
            x = sm.cpm(b, rs, fs_e, h, bt, levels)
            _place(buf, x[: int(dl * fs_e)], fs_e, t0)
    bw = 2 * (0.5 * h * rs) + rs * (1.0 if bt else 1.5)
    return buf, fs_e, bw, on, {"rs": rs, "h": h, "bt": bt, "levels": levels}


def r_lmr_mix(e, d, syn, rng, dur):
    kind = _rng_choice(rng, syn.get("kinds", ["nbfm_voice"]))
    ch = float(syn.get("channel_hz", 12.5e3))
    if kind == "nbfm_voice":
        return r_nbfm_voice(e, d, {"deviation_hz": 2.5e3 if ch <= 12.5e3 else 5e3, "audio_bw_hz": 3e3, "ptt": True}, rng, dur)
    if kind == "c4fm":                           # P25 phase 1 voice: HDU + LDU1/LDU2 frames (180 ms each) with the NAC
        rs, fs_e = 4800.0, 48e3
        buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
        nac = int(rng.integers(0, 0x1000))
        on = burst_schedule(rng, dur, 0.4, lambda: float(rng.uniform(0.5, 3.0)))
        for t0, dl in on:
            n_frames = max(1, int(dl / 0.180))
            duids = [0] + [5, 10] * n_frames
            sym = np.concatenate([dw.p25_frame_symbols(nac, du, 800, rng) for du in duids[:n_frames]])
            x = sm.cpm(sym, rs, fs_e, 0.75, None, 4)
            _place(buf, x[: int(dl * fs_e)], fs_e, t0)
        return buf, fs_e, 12.5e3, on, {"kind": "c4fm", "nac": f"{nac:03X}"}
    if kind in ("dmr_sub", "dmr_bs"):            # DMR: 27.5 ms bursts, every 60 ms (one slot, subscriber) or 30 ms (base, both slots)
        rs, fs_e = 4800.0, 48e3
        buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
        on, t = [], 0.002
        step = 0.060 if kind == "dmr_sub" else 0.030
        k = 0
        while t + 0.0275 < dur:
            name = "MS_voice" if kind == "dmr_sub" else ("BS_voice" if k % 2 == 0 else "BS_data")
            x = sm.cpm(dw.dmr_burst_symbols(name, rng), rs, fs_e, 0.81, None, 4)
            _place(buf, x[: int(0.0275 * fs_e)], fs_e, t)
            on.append((t, 0.0275))
            t += step
            k += 1
        return buf, fs_e, 12.5e3, on, {"kind": kind}
    if kind == "fsk2_control":                   # trunked control channel: continuous 2-FSK
        rs = float(_rng_choice(rng, [9600.0, 3600.0]))
        fs_e = 100e3
        x = sm.cpm(sm.fsk_bits(int(dur * rs) + 1, rng=rng), rs, fs_e, 0.6, None, 2)
        return x[: int(dur * fs_e)], fs_e, 25e3, None, {"kind": "fsk2_control", "rs": rs}
    raise ValueError(kind)


def r_psk_burst(e, d, syn, rng, dur):
    rs = float(syn["symbol_rate_hz"])
    fs_e = 8 * rs
    order = int(syn.get("order", 4))
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    if syn.get("frame_s"):
        frame = float(syn["frame_s"])
        bs = syn.get("burst_s", [0.008, 0.02])
        on, t = [], 0.002
        while t + frame < dur:
            dl = float(_pick(rng, bs, 0.01))
            x = sm.psk(fs_e, rs, int(dl * rs) + 2, order, float(syn.get("rolloff", 0.35)), bool(syn.get("differential", False)), rng=rng)
            _place(buf, x[: int(dl * fs_e)], fs_e, t)
            on.append((t, dl))
            t += frame
    else:
        on = burst_schedule(rng, dur, 1.0, lambda: float(_pick(rng, syn.get("burst_s", [0.01, 0.1]), 0.05)))
        for t0, dl in on:
            x = sm.psk(fs_e, rs, int(dl * rs) + 2, order, float(syn.get("rolloff", 0.35)), bool(syn.get("differential", False)), rng=rng)
            _place(buf, x[: int(dl * fs_e)], fs_e, t0)
    return buf, fs_e, rs * (1 + float(syn.get("rolloff", 0.35))), on, {"rs": rs, "order": order}


def r_psk_continuous(e, d, syn, rng, dur):
    rs = float(syn["symbol_rate_hz"])
    fs_e = 4 * rs
    x = sm.psk(fs_e, rs, int(dur * rs) + 2, int(syn.get("order", 4)), float(syn.get("rolloff", 0.35)), rng=rng)
    return x[: int(dur * fs_e)], fs_e, rs * (1 + float(syn.get("rolloff", 0.35))), None, {"rs": rs}


def r_adsb_ppm(e, d, syn, rng, dur):
    fs_e = 20e6
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    rate = float(_pick(rng, syn.get("rate_per_s"), 100))
    on = burst_schedule(rng, dur, rate, lambda: 120e-6 if rng.random() < 0.7 else 64e-6)
    fleet = [int(v) for v in rng.integers(0xA00000, 0xAFFFFF, int(rng.integers(2, 8)))]
    for t0, dl in on:
        bits = dw.modes_random_message(rng, _rng_choice(rng, fleet), long=dl > 100e-6)
        msg = dw.modes_ppm_waveform(fs_e, bits)
        _place(buf, msg * float(rng.uniform(0.3, 1.0)), fs_e, t0)
    return buf, fs_e, 4e6, on, {"rate_per_s": rate, "icao": [f"{v:06X}" for v in fleet]}


def r_ssr_interrogation(e, d, syn, rng, dur):
    fs_e = 20e6
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    rate = float(_pick(rng, syn.get("rate_per_s"), 200))
    on = []
    t = 0.001
    while t < dur - 0.001:
        mode_c = rng.random() < 0.5
        sp = 21e-6 if mode_c else 8e-6
        p = sm.pulse_train(fs_e, sp + 3e-6, np.array([0.0, 2e-6, sp]), np.full(3, 0.8e-6))
        _place(buf, p, fs_e, t)
        on.append((t, sp + 0.8e-6))
        t += float(rng.exponential(1.0 / rate))
    return buf, fs_e, 3e6, on, {}


def r_dme_pairs(e, d, syn, rng, dur):
    fs_e = 10e6
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    rate = float(_pick(rng, syn.get("pairs_per_s"), 1000))
    pair = sm.dme_pair(fs_e, float(syn.get("spacing_s", 12e-6)))
    on, t = [], 0.001
    while t < dur - 0.001:
        _place(buf, pair, fs_e, t)
        on.append((t, 30e-6))
        t += float(rng.exponential(1.0 / rate))
    return buf, fs_e, 1e6, on, {}


def r_radar_pulses(e, d, syn, rng, dur):
    fs_e = 20e6
    pw, pri = float(syn["pw_s"]), float(syn["pri_s"])
    scan, beam = float(syn.get("scan_s", 4.8)), float(syn.get("beam_s", 0.02))
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    on = []
    t_ill = float(rng.uniform(0.05, max(dur - beam - 0.05, 0.06)))   # one illumination inside the capture
    t = t_ill
    while t < t_ill + beam:
        amp = math.exp(-((t - t_ill - beam / 2) / (beam / 3)) ** 2)
        _place(buf, sm.pulse_train(fs_e, pw * 3, np.array([pw]), np.array([pw])) * amp, fs_e, t)
        on.append((t + pw, pw))
        t += pri * float(rng.uniform(0.97, 1.03))
    return buf, fs_e, 2.0 / pw, on, {"pw_s": pw, "pri_s": pri}


def r_lora(e, d, syn, rng, dur):
    bw = float(_rng_choice(rng, syn.get("bw_hz", [125e3])))
    sf = int(_rng_choice(rng, syn.get("sf", [7])))
    fs_e = 4 * bw
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    on, t = [], 0.005
    while t < dur - 0.05:
        b = sm.lora_burst(fs_e, bw, sf, int(_pick(rng, syn.get("payload_symbols", [20, 80]), 30)), rng=rng)
        if t + len(b) / fs_e > dur:
            break
        _place(buf, b, fs_e, t)
        on.append((t, len(b) / fs_e))
        t += len(b) / fs_e + float(rng.uniform(0.05, 0.3))
    return buf, fs_e, bw, on, {"bw_hz": bw, "sf": sf}


def r_oqpsk_burst(e, d, syn, rng, dur):
    cr = float(syn.get("chip_rate_hz", 2e6))
    fs_e = 4 * cr
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    rate = float(_pick(rng, syn.get("rate_per_s"), 10))
    fb = syn.get("frame_bytes", [11, 127])
    on = []
    t = 0.002
    pan = int(rng.integers(1, 0xFFFE))
    seq = int(rng.integers(0, 256))
    while t < dur - 0.01:
        nb = int(rng.integers(fb[0], fb[1] + 1))
        psdu = dw.zigbee_frame_bytes(bytes(int(v) for v in rng.integers(0, 256, max(nb - 11, 0))), seq, pan, int(rng.integers(0, 0xFFF0)), int(rng.integers(1, 0xFFF0)))
        x = dw.oqpsk_chips_waveform(fs_e, dw.zigbee_chips(dw.zigbee_ppdu_symbols(psdu)), cr)
        dl = len(x) / fs_e
        _place(buf, x, fs_e, t)
        on.append((t, dl))
        t += dl
        if rng.random() < float(syn.get("ack_prob", 0.5)):
            t += 192e-6
            ack = bytes([0x02, 0x00, seq & 0xFF])
            ack = ack + dw.crc16_kermit(ack).to_bytes(2, "little")
            xa = dw.oqpsk_chips_waveform(fs_e, dw.zigbee_chips(dw.zigbee_ppdu_symbols(ack)), cr)
            _place(buf, xa, fs_e, t)
            on.append((t, len(xa) / fs_e))
            t += len(xa) / fs_e
        seq = (seq + 1) & 0xFF
        t += float(rng.exponential(1.0 / rate))
    return buf, fs_e, 2.5e6, on, {"pan": f"{pan:04X}"}


def r_bt_classic(e, d, syn, rng, dur):
    """One channel of a hopping link: packets land here every ~79 slots (plus retries), 1 Msym/s GFSK."""
    rs, fs_e = 1e6, 8e6
    slot = float(syn.get("slot_s", 625e-6))
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    on, k = [], 0
    n_slots = int(dur / slot)
    while k < n_slots:
        if rng.random() < 1 / 40:
            ns = int(_rng_choice(rng, syn.get("packet_slots", [1, 3, 5])))
            dl = min(366e-6 + (ns - 1) * slot, ns * slot - 259e-6)
            t0 = k * slot + 0.0
            if t0 + dl < dur:
                _place(buf, sm.cpm(sm.fsk_bits(int(dl * rs), rng=rng), rs, fs_e, float(syn.get("h", 0.32)), float(syn.get("bt", 0.5))), fs_e, t0)
                on.append((t0, dl))
            k += ns
        else:
            k += 1
    return buf, fs_e, 1.0e6, on, {}


def r_ble_adv(e, d, syn, rng, dur):
    """One advertising channel: ADV_IND every advInterval (+0..10 ms), sometimes answered after T_IFS by a SCAN_REQ."""
    rs, fs_e = 1e6, 8e6
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    interval = float(_pick(rng, syn.get("adv_interval_s", [0.02, 1.0]), 0.1))
    pdu = int(_pick(rng, syn.get("pdu_bytes", [10, 47]), 30))
    fc_adv = float(_rng_choice(rng, [2402e6, 2426e6, 2480e6]))
    channel = dw.BLE_ADV_CHANNELS[fc_adv]
    mac = bytes(int(v) for v in rng.integers(0, 256, 6))
    mac = bytes([mac[0] | 0xC0]) + mac[1:]
    name = _rng_choice(rng, ["", "", "ARGUS-" + str(int(rng.integers(100, 999))), "Tile", "LE-Headset"])
    on, t = [], 0.003
    while t + 1e-3 < dur:
        pdu_bytes = dw.ble_adv_pdu(mac, 0 if rng.random() < 0.7 else 2, name, rng, max(0, min(pdu - 9 - (len(name) + 2 if name else 0), 20)))
        bits = dw.ble_adv_bits(pdu_bytes, channel).astype(np.float64) * 2 - 1
        dl = len(bits) / rs
        _place(buf, sm.cpm(bits, rs, fs_e, float(syn.get("h", 0.5)), float(syn.get("bt", 0.5))), fs_e, t)
        on.append((t, dl))
        t2 = t + dl
        if rng.random() < float(syn.get("scan_req_prob", 0.3)):
            t2 += 150e-6
            _place(buf, sm.cpm(sm.fsk_bits(176, rng=rng), rs, fs_e, 0.5, 0.5), fs_e, t2)
            on.append((t2, 176e-6))
        t += interval + float(rng.uniform(0, 0.01))
    # the detector paints the -20 dB width of a GFSK channel (~1.5 MHz), not the 99 % width
    return buf, fs_e, 1.5e6, on, {"adv_interval_s": interval, "f_center_hz": fc_adv, "channel": channel, "adv_addr": ":".join(f"{x:02X}" for x in mac)}


def r_ook_burst(e, d, syn, rng, dur):
    br = float(_pick(rng, syn.get("bitrate", [2000.0]), 2000.0))
    fs_e = max(50e3, 20 * br)
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    nb = int(_pick(rng, syn.get("bits", [24, 100]), 48))
    reps = int(_pick(rng, syn.get("repeats", [3, 8]), 4))
    gap = float(_pick(rng, syn.get("repeat_gap_s", [0.01, 0.1]), 0.03))
    on, t = [], float(rng.uniform(0.01, 0.1))
    while t < dur - 0.05:
        for _ in range(reps):
            x = sm.ook_burst(fs_e, br, nb, rng=rng)
            if t + len(x) / fs_e > dur:
                break
            _place(buf, x, fs_e, t)
            on.append((t, len(x) / fs_e))
            t += len(x) / fs_e + gap
        t += float(rng.uniform(0.2, 0.6))
    return buf, fs_e, 4 * br, on, {"bitrate": br}


def r_paging(e, d, syn, rng, dur):
    """POCSAG batches (preamble + sync + codewords, 512/1200/2400 bps) or a continuous FLEX stream (1600 bps)."""
    codec = str(_rng_choice(rng, syn.get("codecs", ["pocsag1200", "flex1600"])))
    fs_e = 76800.0
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    if codec.startswith("flex"):
        rs, h = 1600.0, 2 * 4.8e3 / 1600.0
        cyc, frm = int(rng.integers(0, 15)), int(rng.integers(0, 128))
        bits = np.concatenate([dw.flex_frame_bits(cyc, (frm + k) % 128, rng) for k in range(int(math.ceil(dur / 1.875)) + 1)])
        off = int(rng.integers(0, 3000))
        bits = bits[off: off + int(dur * rs) + 2]
        x = sm.cpm(bits.astype(np.float64) * 2 - 1, rs, fs_e, h, None, 2)
        return x[: int(dur * fs_e)], fs_e, 20e3, None, {"codec": codec, "cycle": cyc, "frame": frm}
    rs = float(codec[6:])
    h = 2 * 4.5e3 / rs
    rate = float(_pick(rng, syn.get("rate_per_s"), 1.0))
    on = burst_schedule(rng, dur, rate, lambda: float(_pick(rng, syn.get("burst_s", [0.4, 3.0]), 1.0)))
    rics = []
    for t0, dl in on:
        n_batches = max(1, int((dl * rs - 576) / 544))
        bits, r = dw.pocsag_stream(rng, n_batches)
        rics += r
        x = sm.cpm(bits.astype(np.float64) * 2 - 1, rs, fs_e, h, None, 2)
        _place(buf, x[: int(dl * fs_e)], fs_e, t0)
    return buf, fs_e, 20e3, on, {"codec": codec, "rics": rics[:12]}


def r_ism900_mix(e, d, syn, rng, dur):
    """902-928 MHz devices: a SUN-FSK sensor (random GFSK packets) or an ERT meter (OOK Manchester SCM / SCM+ packets)."""
    kind = str(_rng_choice(rng, syn.get("kinds", ["sun_fsk", "ert"])))
    if kind == "sun_fsk":
        x, fs_e, bw, on, p = r_fsk_burst(e, d, {k: v for k, v in syn.items() if k not in ("kind", "kinds")}, rng, dur)
        return x, fs_e, bw, on, {**p, "kind": kind}
    fs_e = 524288.0
    buf = np.zeros(int(dur * fs_e), dtype=np.complex64)
    on, ids = [], []
    t = float(rng.uniform(0.005, 0.2))
    while t < dur - 0.02:
        eid = int(rng.integers(1, 2 ** 26))
        bits = dw.scm_packet(eid, int(rng.integers(0, 16)), int(rng.integers(0, 2 ** 24))) if rng.random() < 0.6 else dw.scmplus_packet(eid, 0x9C, int(rng.integers(0, 2 ** 24)))
        ids.append(eid)
        x = np.repeat(dw.ert_manchester_chips(bits), 16).astype(np.complex64)
        _place(buf, x, fs_e, t)
        on.append((t, len(x) / fs_e))
        t += len(x) / fs_e + float(rng.uniform(0.05, 0.4))
    return buf, fs_e, 120e3, on, {"kind": "ert", "ert_ids": ids[:8]}


def r_magnetron(e, d, syn, rng, dur):
    fs_e = 40e6
    x = sm.magnetron(fs_e, dur, float(syn.get("sweep_hz", 10e6)), float(syn.get("mains_hz", 60.0)), rng=rng)
    return x, fs_e, 2 * float(syn.get("sweep_hz", 10e6)), None, {}


RECIPES = {"paging": r_paging, "ism900_mix": r_ism900_mix, "fm_stereo": r_fm_stereo, "nbfm_voice": r_nbfm_voice, "am_voice": r_am_voice, "am_tones": r_am_tones, "am_msk": r_am_msk,
           "vsb8": r_vsb8, "ofdm_continuous": r_ofdm_continuous, "ofdm_tdd": r_ofdm_tdd, "scfdma_bursts": r_scfdma_bursts, "wifi_ofdm": r_wifi_ofdm,
           "fsk_burst": r_fsk_burst, "lmr_mix": r_lmr_mix, "psk_burst": r_psk_burst, "psk_continuous": r_psk_continuous, "adsb_ppm": r_adsb_ppm,
           "ssr_interrogation": r_ssr_interrogation, "dme_pairs": r_dme_pairs, "radar_pulses": r_radar_pulses, "lora": r_lora,
           "oqpsk_burst": r_oqpsk_burst, "bt_classic": r_bt_classic, "ble_adv": r_ble_adv, "ook_burst": r_ook_burst, "magnetron": r_magnetron}


# ------------------------------------------------------------------------------------------- helpers
def _place(buf, x, fs, t0):
    n0 = int(round(t0 * fs))
    n = min(len(x), len(buf) - n0)
    if n > 0:
        buf[n0: n0 + n] += x[:n]


def _gate(x, fs, on):
    g = np.zeros(len(x), dtype=np.float32)
    for t0, dl in on:
        g[int(t0 * fs): int((t0 + dl) * fs)] = 1.0
    return (x * g).astype(np.complex64)


def pick_center(e: dict, rng, bw: float) -> float:
    """A legal emission centre for the entry: channel list, raster, or anywhere inside a band."""
    chans = [float(c) for c in e.get("channel_centers_hz", [])]
    if chans:
        return _rng_choice(rng, chans)
    ranges = [(float(lo), float(hi)) for lo, hi in e["freq_ranges_hz"]]
    lo, hi = _rng_choice(rng, ranges)
    ras = e.get("raster")
    if ras:
        sp, off = float(ras["spacing_hz"]), float(ras.get("offset_hz", 0.0))
        k0, k1 = math.ceil((lo + bw / 2 - off) / sp), math.floor((hi - bw / 2 - off) / sp)
        if k1 >= k0:
            return off + sp * int(rng.integers(k0, k1 + 1))
    return float(rng.uniform(lo + bw / 2, max(lo + bw / 2, hi - bw / 2)))


MAX_RATE = 61.44e6      # synthetic captures stay below this: generating and resampling 1 s at 100-200 MSps is too slow


def capture_rate(e: dict, rng, bw: float, rates_arg) -> float:
    rates = [float(r) for r in (rates_arg or [])]
    if not rates:
        sw = e.get("sweep", {})
        rates = [float(r) for r in sw.get("rates_hz", [])] + [float(l["rate_hz"]) for l in sw.get("looks", [])]
    rates = [min(r, MAX_RATE) for r in rates]
    rates = [r for r in rates if r >= 1.3 * bw] or [r for r in DEFAULT_RATES + [MAX_RATE] if r >= 1.3 * bw] or [MAX_RATE]
    r = float(_rng_choice(rng, rates))
    return mio.snap_rate(r, "n320")[0]


def write_sigmf(x: np.ndarray, fs: float, fc: float, path_stem: Path, annotations: list, extra: dict):
    scale = 0.25 * 32767 / max(float(np.percentile(np.abs(x), 99.99)), 1e-9)
    iq = np.empty(2 * len(x), dtype="<i2")
    iq[0::2] = np.clip(np.round(x.real * scale), -32768, 32767)
    iq[1::2] = np.clip(np.round(x.imag * scale), -32768, 32767)
    Path(str(path_stem) + ".sigmf-data").write_bytes(iq.tobytes())      # not with_suffix: the stem holds "c131.845M"
    meta = {"global": {"core:datatype": "ci16_le", "core:sample_rate": fs, "core:version": "1.0.0", "core:description": "synthetic (synth_lut_scenes.py)", **extra},
            "captures": [{"core:sample_start": 0, "core:frequency": fc}], "annotations": annotations}
    Path(str(path_stem) + ".sigmf-meta").write_text(json.dumps(meta, indent=1))


def truth_annotations(fs: float, fc: float, f_lo: float, f_hi: float, dur: float, on, cols: int = 1024) -> list:
    geo = mio.frame_geometry(fs, cols)
    bin_hz = geo["bin_hz"]
    cap_lo = fc - fs / 2
    lo = cap_lo + math.floor((f_lo - cap_lo) / bin_hz) * bin_hz
    hi = cap_lo + math.ceil((f_hi - cap_lo) / bin_hz) * bin_hz
    anns = []
    if on is None:
        n_frames = int(dur / geo["frame_s"])
        for k in range(n_frames):
            anns.append({"core:sample_start": int(k * geo["frame_samples"]), "core:sample_count": int(geo["frame_samples"]), "core:freq_lower_edge": lo,
                         "core:freq_upper_edge": hi, "core:label": "signal", "wfgt:frame_number": k})
    else:
        for t0, dl in on:
            s0 = int(round(t0 / geo["row_s"])) * int(geo["fft_size"])
            cnt = max(int(math.ceil(dl / geo["row_s"])), 1) * int(geo["fft_size"])
            if s0 + cnt > dur * fs:
                continue
            anns.append({"core:sample_start": s0, "core:sample_count": cnt, "core:freq_lower_edge": lo, "core:freq_upper_edge": hi, "core:label": "signal",
                         "wfgt:frame_number": int(s0 // geo["frame_samples"])})
    return anns


def generate(e: dict, rng, out: Path, draw: int, dur: float, snr_range, rates_arg, background: np.ndarray | None, bg_rate: float | None) -> dict | None:
    syn = e.get("synth")
    if not syn or syn.get("kind") not in RECIPES:
        return None
    x, fs_e, bw, on, params = RECIPES[syn["kind"]](e, e.get("descriptor", {}), syn, rng, dur)
    if len(x) < 100:
        return None
    if background is not None and bg_rate:
        rate = bg_rate
    else:
        rate = capture_rate(e, rng, bw, rates_arg)
    f_em = float(params["f_center_hz"]) if params.get("f_center_hz") else pick_center(e, rng, bw)
    if params.get("raster_hz"):
        f_em = round(f_em / float(params["raster_hz"])) * float(params["raster_hz"])
    # capture centre such that the emission is inside, away from DC by at least 5 % of the rate
    while True:
        off = float(rng.uniform(-0.35, 0.35)) * rate
        if abs(off) > 0.05 * rate and abs(off) + bw / 2 < 0.47 * rate:
            break
    bin_hz = rate / 1024
    off = round(off / bin_hz) * bin_hz          # rounded first: the capture centre must put the emission exactly at f_em
    fc = f_em - off
    xs = sm.resample_to(x, fs_e, rate)
    n = int(dur * rate)
    xs = np.concatenate([xs, np.zeros(max(n - len(xs), 0), dtype=np.complex64)])[:n]
    on_mask = np.abs(xs) > 0
    p_sig = float(np.mean(np.abs(xs[on_mask]) ** 2)) if on_mask.any() else 1.0
    snr_db = float(rng.uniform(snr_range[0], snr_range[1]))
    noise_in_band = p_sig / 10 ** (snr_db / 10)
    t = np.arange(n) / rate
    sig = (xs * np.exp(2j * np.pi * off * t)).astype(np.complex64)
    if background is not None:
        bg = background[:n] if len(background) >= n else np.tile(background, int(math.ceil(n / len(background))))[:n]
        p_bg_band = float(np.mean(np.abs(bg) ** 2)) * bw / rate          # white-ish floor assumption
        cap = (bg * math.sqrt(noise_in_band / max(p_bg_band, 1e-30)) + sig).astype(np.complex64)
    else:
        nz = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) / math.sqrt(2)
        cap = (sig + nz * math.sqrt(noise_in_band * rate / bw)).astype(np.complex64)
    stem = f"synth_{e['id']}_{draw:02d}_c{fc / 1e6:.3f}M_r{rate / 1e6:g}M_snr{snr_db:.0f}"
    anns = truth_annotations(rate, fc, f_em - bw / 2, f_em + bw / 2, dur, on)
    truth = {"entry": e["id"], "label": e["label"], "fc_capture_hz": fc, "rate_hz": rate, "f_emission_hz": f_em, "bw_hz": bw, "snr_db": snr_db,
             "kind": syn["kind"], "params": params, "n_bursts": len(on) if on else 0, "continuous": on is None, "dur_s": dur}
    write_sigmf(cap, rate, fc, out / stem, anns, {"argus:synthetic_entry": e["id"], "argus:synthetic_snr_db": round(snr_db, 1)})
    (out / f"{stem}.truth.json").write_text(json.dumps(truth, indent=1, default=float))
    return truth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", default=str(mio.DEFAULT_LIBRARY))
    ap.add_argument("--entries", nargs="*", default=None)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--dur", type=float, default=1.0)
    ap.add_argument("--snr", nargs=2, type=float, default=[10.0, 30.0])
    ap.add_argument("--rates", nargs="*", default=None, help="capture rates in Hz (default: the entry's sweep rates)")
    ap.add_argument("--background", default=None, help="a real recording (.sigmf-data) used as the noise floor at its own rate")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    sm.set_seed(a.seed)
    lib = mio.load_library(None, a.library)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    background, bg_rate = None, None
    if a.background:
        bp = Path(a.background).expanduser()
        meta = json.loads(bp.with_suffix(".sigmf-meta").read_text())
        bg_rate = float(meta["global"]["core:sample_rate"])
        raw = np.fromfile(bp, dtype="<i2", count=2 * int(min(a.dur + 0.1, 1.0) * bg_rate))
        background = (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)).astype(np.complex64)
        background /= max(float(np.sqrt(np.mean(np.abs(background) ** 2))), 1e-9)
    entries = [e for e in lib["entries"] if e.get("synth") and (not a.entries or e["id"] in a.entries)]
    index = []
    for e in entries:
        for k in range(a.draws):
            tr = generate(e, rng, out, k, a.dur, a.snr, a.rates, background, bg_rate)
            if tr:
                index.append(tr)
                print(f"{e['id']:24s} draw {k}: {tr['rate_hz'] / 1e6:g} MSps, {tr['f_emission_hz'] / 1e6:.3f} MHz, bw {tr['bw_hz'] / 1e3:.4g} kHz, SNR {tr['snr_db']:.0f} dB, {tr['n_bursts']} bursts", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1, default=float))
    print(f"{len(index)} captures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
