#!/usr/bin/env python3
"""LUT classification: match the measured descriptor of every detected emission against the library.

    python3 lut_classify.py <dir>_preprocessed                 # labels written into the snippets (like v1)
    python3 lut_classify.py <capture>.sigmf-meta --no-write
    python3 lut_classify.py <capture folder> --masks M12S
    python3 lut_classify.py <folder of capture folders> --masks M12S --select fm_broadcast

Per track: the v1 score (band, channel raster, box width, temporal class, burst length, PSD shape, pilots) from
common_waveforms_look_up.py is multiplied by the descriptor memberships of measure_descriptors.py's SDW: modulation
class, symbol-rate lines, OFDM numerology, chirp slope, deviation, pulse width, burst length, intervals, gaps,
periods, pulse pairs, duty, carrier line, demodulated tones, HD sidebands, band-level hopping. Each field is
measured-and-matching (verified evidence), measured-and-contradicting (penalty) or unmeasured (neutral). A label
needs the tier threshold, the manifest `requires` features, `requires_descriptors` verified descriptor fields and
a margin over the runner-up. Variants (P25 vs DMR, LTE vs NR, DSSS vs OFDM Wi-Fi) refine the label when exactly
one of them is verified: `~Narrowband_VHF_150-174/P25`.

Outputs: lut_regions.csv (one row per track with the SDW summary, label, confidence, verified fields, evidence),
lut_summary.json, sdw_<capture>.json (the full SDWs) and, unless --no-write, the labels in the SigMF annotations
(`core:label`, `argus:estimate_*`, `argus:sdw`).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_waveforms_look_up as lk  # noqa: E402
import manifest_io as mio  # noqa: E402
import decode_waveforms as dw  # noqa: E402
import measure_descriptors as md  # noqa: E402

VERSION = "lut_classify.py 0.2 (lut_classification_experiment)"
LOW_SNR_DB = 6.0        # below this a label needs verified descriptor evidence (or a decode): no band-only guesses on weak signals
DECODE_CONF, DECODE_WEAK_CONF = 0.97, 0.86
THRESH = lk.THRESH
PREFIX = lk.ESTIMATE_PREFIX

# measured class -> membership of each library class
MOD_COMPAT = {
    "fm": {"fm": 1.0, "fsk": 0.7, "cw": 0.5, "ssb": 0.4, "css": 0.3, "am": 0.3, "pulse": 0.3, "noise": 0.3, "psk": 0.25, "ofdm": 0.2, "vsb": 0.2},
    "fsk": {"fsk": 1.0, "fm": 0.7, "css": 0.4, "psk": 0.4, "cw": 0.3, "am": 0.3, "pulse": 0.3, "noise": 0.3, "ssb": 0.3, "ofdm": 0.2, "vsb": 0.2},
    "psk": {"psk": 1.0, "vsb": 0.8, "ssb": 0.6, "ofdm": 0.5, "fsk": 0.5, "am": 0.5, "noise": 0.5, "pulse": 0.4, "fm": 0.3, "cw": 0.3, "css": 0.3},
    "ofdm": {"ofdm": 1.0, "noise": 0.5, "psk": 0.3, "vsb": 0.3, "fsk": 0.15, "fm": 0.15, "am": 0.15, "cw": 0.15, "css": 0.15, "pulse": 0.2, "ssb": 0.15},
    "vsb": {"vsb": 1.0, "psk": 0.8, "am": 0.7, "ofdm": 0.3, "fsk": 0.2, "fm": 0.2, "cw": 0.3, "css": 0.2, "pulse": 0.2, "noise": 0.3, "ssb": 0.3},
    "am": {"am": 1.0, "vsb": 0.8, "pulse": 0.7, "cw": 0.6, "ssb": 0.5, "psk": 0.4, "fm": 0.3, "fsk": 0.3, "noise": 0.3, "ofdm": 0.2, "css": 0.2},
    "pulse": {"pulse": 1.0, "am": 0.6, "css": 0.4, "fsk": 0.3, "psk": 0.3, "noise": 0.3, "fm": 0.2, "ofdm": 0.2, "vsb": 0.2, "cw": 0.2, "ssb": 0.2},
    "css": {"css": 1.0, "fm": 0.4, "fsk": 0.4, "pulse": 0.3, "psk": 0.2, "ofdm": 0.2, "am": 0.2, "vsb": 0.2, "cw": 0.2, "noise": 0.2, "ssb": 0.2},
    "cw": {"cw": 1.0, "am": 0.6, "fm": 0.4, "fsk": 0.3, "noise": 0.2, "psk": 0.3, "vsb": 0.3, "ofdm": 0.2, "css": 0.2, "pulse": 0.3, "ssb": 0.3},
    "noise": {"noise": 1.0, "ofdm": 0.7, "psk": 0.5, "fsk": 0.3, "fm": 0.3, "am": 0.3, "cw": 0.2, "vsb": 0.4, "css": 0.3, "pulse": 0.3, "ssb": 0.3},
}


def _gauss(err: float, tol: float, floor: float = 0.0) -> float:
    return max(floor, math.exp(-(err / max(tol, 1e-9)) ** 2))


def _fold(f: float, fs: float) -> float:
    """Where a real-spectrum line at f lands after sampling at fs."""
    return abs(((f + fs / 2) % fs) - fs / 2)


def _vals(items) -> list[tuple[float, float, float, str]]:
    """Library timing items -> [(lo, hi, tol, note)]."""
    out = []
    for it in items or []:
        if isinstance(it, dict):
            if "value" in it:
                v, tol = float(it["value"]), float(it.get("tol", 0.03))
                out.append((v, v, tol, str(it.get("note", ""))))
            elif "min" in it and "max" in it:
                out.append((float(it["min"]), float(it["max"]), float(it.get("tol", 0.0)), str(it.get("note", ""))))
        else:
            v = float(it)
            out.append((v, v, 0.03, ""))
    return out


def _match_value(x: float, lo: float, hi: float, tol: float) -> float:
    """Membership of x in [lo, hi] with relative tolerance tol outside the interval."""
    if lo <= x <= hi:
        return 1.0
    ref = lo if x < lo else hi
    err = abs(x - ref) / max(ref, 1e-12)
    return _gauss(err, max(tol, 1e-3))


@dataclass
class FieldResult:
    name: str
    weight: float
    membership: float = 1.0
    verified: bool = False
    evidence: str = ""


# =========================================================================================== field tests
def t_modulation(d: dict, s: dict) -> FieldResult:
    allowed = [str(x) for x in d.get("modulation", [])]
    m = s["mod"]
    cls, conf = m.get("class", "unknown"), float(m.get("class_conf", 0.0))
    r = FieldResult("modulation", 0)
    if not allowed or cls == "unknown" or conf <= 0:
        return r
    comp = MOD_COMPAT.get(cls, {})
    best = max(comp.get(c, 0.2) for c in allowed)
    r.membership = 1.0 - conf * (1.0 - best)
    if best >= 0.99 and conf >= 0.6:
        r.verified = True
        r.evidence = f"class {cls} ({conf:.1f})"
    elif best < 0.5:
        r.evidence = f"class {cls} ({conf:.1f}) vs {'/'.join(allowed)}"
    return r


def t_symbol_rate(d: dict, s: dict) -> FieldResult:
    r = FieldResult("symbol_rate", 0)
    rates = _vals(d.get("symbol_rate_hz"))
    if not rates:
        return r
    L, fs = s.get("lines", {}), float(s["fs_hz"])
    mods = set(str(x) for x in d.get("modulation", []))
    cands: list[tuple[float, float, str, float]] = []
    if mods & {"psk", "vsb", "pulse", "am", "ofdm", "ssb", "noise"}:
        cands += [(f, snr, "|y|^2 line", 1.0) for f, snr in L.get("abs2", [])]
    if mods & {"fsk", "fm", "css"}:
        cands += [(f, snr, "f_inst transitions line", 1.0) for f, snr in L.get("finst", [])]
    sq, sql = L.get("sq_spacing_hz"), L.get("sq", [])
    if sq and len(sql) >= 2:
        snr_sq = min(sql[0][1], sql[1][1])
        if mods & {"fsk"}:
            hs = [float(h) for h in d.get("mod_index", [])] or [0.5, 1.0]
            for h in hs:
                cands.append((sq / (2 * h), snr_sq, f"y^2 line spacing / 2h (h={h:g})", 0.9 if d.get("mod_index") else 0.7))
        if mods & {"vsb", "psk"}:
            cands.append((sq, snr_sq, "y^2 line spacing", 0.9))
    if not cands:
        return r
    best = None
    for lo, hi, tol, note in rates:
        v = lo
        for f, snr, kind, credit in cands:
            for k, kc in ((1.0, 1.0), (2.0, 0.8), (3.0, 0.7), (0.5, 0.7)):
                target = v * k
                tf = _fold(target, fs) if kind.startswith("|y|^2") else target
                if tf <= 0:
                    continue
                err = abs(f - tf) / tf
                if err <= 4 * tol:
                    mem = _gauss(err, tol) * credit * kc
                    cand = (mem, snr, kind, v, f, k, note)
                    if best is None or cand[0] > best[0]:
                        best = cand
    if best and best[0] >= 0.75 and best[1] >= 8.0:          # a half-rate or weak harmonic match never verifies on its own
        mem, snr, kind, v, f, k, note = best
        r.membership = 1.0 if mem >= 0.8 else mem
        r.verified = True
        r.evidence = f"rate {v / 1e3:g} kHz{(' ' + note) if note else ''} via {kind} at {f / 1e3:.4g} kHz" + (f" (x{k:g})" if k != 1 else "") + f", {snr:.0f} dB"
        return r
    if sq and len(sql) >= 2 and d.get("mod_index") and "fsk" in mods and min(sql[0][1], sql[1][1]) >= 10.0:
        # a measured y^2 line pair fixes 2 h Rs: wrong for this entry's modulation index
        exp = [2 * float(h) * lo for lo, _, _, _ in rates for h in d["mod_index"]]
        if not any(abs(sq - x) / x <= 0.15 for x in exp):
            r.membership = 0.7
            r.evidence = f"y^2 line spacing {sq / 1e3:.4g} kHz = 2 h Rs does not fit h {d['mod_index']}"
            return r
    strong = [c for c in cands if c[1] >= 15.0]
    if strong and not (mods & {"ofdm", "noise", "pulse"}):
        r.membership = 0.7
        r.evidence = f"strong {strong[0][2]} at {strong[0][0] / 1e3:.4g} kHz matches no library rate"
    return r


def t_ofdm(d: dict, s: dict) -> FieldResult:
    r = FieldResult("ofdm", 0)
    o = s.get("ofdm", {})
    scs_list = [float(x) for x in d.get("ofdm", {}).get("scs_hz", [])]
    if not scs_list:
        return r
    peaks = o.get("peaks", [])
    measured = bool(peaks) and peaks[0].get("ratio", 0) >= 6 and peaks[0].get("excess", 0) >= 0.02
    if measured:
        best = None
        for pk in peaks[:3]:
            if pk.get("ratio", 0) < 4:
                continue
            for scs in scs_list:
                err = abs(pk["scs_hz"] - scs) / scs
                mem = _gauss(err, 0.03)
                if best is None or mem > best[0]:
                    best = (mem, pk, scs)
        if best and best[0] >= 0.6:
            mem, pk, scs = best
            ev = f"OFDM Tu {pk['tu_s'] * 1e6:.2f} us = {pk['scs_hz'] / 1e3:.4g} kHz subcarriers (library {scs / 1e3:g} kHz)"
            cps = [float(x) for x in d.get("ofdm", {}).get("cp_s", [])]
            if o.get("cp_s") and cps:
                cp_mem = max(_gauss(abs(o["cp_s"] - c) / c, 0.3) for c in cps)
                ev += f", CP {o['cp_s'] * 1e6:.2f} us" + ("" if cp_mem >= 0.4 else f" (library {', '.join(f'{c * 1e6:.2f}' for c in cps)} us)")
                if cp_mem < 0.3:
                    mem *= 0.8
            r.membership = 1.0 if mem >= 0.8 else mem
            r.verified = True
            r.evidence = ev
            return r
        r.membership = 0.25
        r.evidence = f"OFDM with {peaks[0]['scs_hz'] / 1e3:.4g} kHz subcarriers, library expects {', '.join(f'{x / 1e3:g}' for x in scs_list)} kHz"
        return r
    m = s["mod"]
    if m.get("snr_db", 0) >= 12 and m.get("class") in ("fsk", "fm", "am", "css", "pulse", "cw") and m.get("class_conf", 0) >= 0.7:
        r.membership = 0.6
        r.evidence = f"no OFDM structure (class {m.get('class')}) at {m.get('snr_db')} dB"
    return r


def t_chirp(d: dict, s: dict) -> FieldResult:
    r = FieldResult("chirp", 0)
    spec = d.get("chirp") or []
    if not spec:
        return r
    expected = []
    for c in spec:
        bw = float(c["bw_hz"])
        for sf in c.get("sf", []):
            expected.append((bw * bw / (2 ** int(sf)), f"BW {bw / 1e3:g} kHz SF{sf}"))
    m = s["mod"]
    rate, cons = m.get("chirp_rate_hz_s"), m.get("chirp_consistency", 0)
    if rate and cons >= 0.7 and abs(rate) > 1e6:
        best = max(((_gauss(abs(abs(rate) - e) / e, 0.15), e, note) for e, note in expected), key=lambda x: x[0])
        if best[0] >= 0.5:
            r.membership = 1.0 if best[0] >= 0.8 else best[0]
            r.verified = True
            r.evidence = f"chirp {abs(rate) / 1e6:.3g} MHz/s = {best[2]}"
        else:
            r.membership = 0.3
            r.evidence = f"chirp {abs(rate) / 1e6:.3g} MHz/s matches no SF/BW"
        return r
    if m.get("snr_db", 0) >= 12 and m.get("class_conf", 0) >= 0.7 and m.get("class") != "css":
        r.membership = 0.6
        r.evidence = "no chirp structure at high SNR"
    return r


def t_deviation(d: dict, s: dict) -> FieldResult:
    r = FieldResult("deviation", 0)
    dv = d.get("deviation_hz")
    m = s["mod"]
    if not dv or m.get("class") not in ("fm", "fsk", "css") or m.get("snr_db", 0) < 6 or not m.get("dev_est_hz"):
        return r
    lo, hi = float(dv.get("peak_min", 0)), float(dv.get("peak_max", 1e12))
    dev = float(m["dev_est_hz"])
    if 0.5 * lo <= dev <= 1.3 * hi:
        r.verified = True
        r.evidence = f"deviation {dev / 1e3:.3g} kHz in {lo / 1e3:g}-{hi / 1e3:g}"
    else:
        r.membership = 0.7
        r.evidence = f"deviation {dev / 1e3:.3g} kHz outside {lo / 1e3:g}-{hi / 1e3:g}"
    return r


def t_pulse(d: dict, s: dict) -> FieldResult:
    r = FieldResult("pulse", 0)
    rng = d.get("timing", {}).get("pulse_s")
    if not rng:
        return r
    f = s["fine"]
    m = s["mod"]
    if f.get("is_pulse_structure"):
        p50 = f.get("dur_p50", 0)
        if 0.7 * float(rng["min"]) <= p50 <= 1.4 * float(rng["max"]):
            r.verified = True
            r.evidence = f"pulses {p50 * 1e6:.2f} us"
        else:
            r.membership = 0.5
            r.evidence = f"pulses {p50 * 1e6:.2f} us outside {float(rng['min']) * 1e6:g}-{float(rng['max']) * 1e6:g} us"
    elif m.get("snr_db", 0) >= 10 and m.get("class_conf", 0) >= 0.6 and m.get("class") != "pulse":
        r.membership = 0.6
        r.evidence = "not pulsed"
    return r


def t_burst(d: dict, s: dict) -> FieldResult:
    r = FieldResult("burst", 0)
    rng = d.get("timing", {}).get("burst_s")
    if not rng:
        return r
    lo_lib = float(rng["min"])
    c = s["frame"] if (lo_lib >= 2e-3 and s.get("frame")) else (s["packet"] if (lo_lib >= 300e-6 and s.get("packet")) else s["coarse"])
    if c.get("n", 0) < 2:
        return r
    lo, hi = 0.7 * float(rng["min"]), 1.4 * float(rng["max"])
    p50 = c["dur_p50"]
    if lo <= p50 <= hi:
        r.verified = True
        r.evidence = f"bursts {p50 * 1e3:.3g} ms"
    elif c["dur_p10"] <= hi and c["dur_p90"] >= lo:
        r.membership = 0.8
        r.evidence = f"bursts p50 {p50 * 1e3:.3g} ms, p10-p90 overlaps {float(rng['min']) * 1e3:g}-{float(rng['max']) * 1e3:g} ms"
    else:
        r.membership = 0.5
        r.evidence = f"bursts {p50 * 1e3:.3g} ms outside {float(rng['min']) * 1e3:g}-{float(rng['max']) * 1e3:g} ms"
    return r


def _interval_candidates(s: dict, fine_too: bool) -> list[tuple[float, float, str]]:
    """(value_s, share, source) from the coarse PRI peaks, duration-class PRIs, envelope periods and fine PRIs."""
    out = []
    c, f = s["coarse"], s["fine"]
    out += [(p["value"], p["share"], "interval") for p in c.get("pri", [])]
    out += [(p["value"], p["share"], "interval (packets)") for p in s.get("packet", {}).get("pri", [])]
    out += [(p["value"], p["share"], f"interval of {p['dur'] * 1e3:.3g} ms packets") for p in s.get("packet", {}).get("pri_by_dur", [])]
    out += [(p["value"], p["share"], "interval (frames)") for p in s.get("frame", {}).get("pri", [])]
    out += [(p["value"], 0.8 * p["share"], "interval (multi)") for p in c.get("pri_multi", [])]
    out += [(p["value"], p["share"], f"interval of {p['dur'] * 1e3:.3g} ms bursts") for p in c.get("pri_by_dur", [])]
    out += [(p, h, "envelope period") for p, h in s.get("periods", [])]
    if fine_too:
        out += [(p["value"], p["share"], "pulse interval") for p in f.get("pri", [])]
    return out


def _test_intervals(name: str, items, cands, min_share: float = 0.08) -> FieldResult:
    r = FieldResult(name, 0)
    vals = _vals(items)
    if not vals or not cands:
        return r
    best = None
    for lo, hi, tol, note in vals:
        for v, share, src in cands:
            for k, kc in ((1.0, 1.0), (2.0, 0.8), (3.0, 0.7), (0.5, 0.75)):
                mem = _match_value(v / k, lo, hi, max(tol, 0.03)) * kc
                if best is None or mem * (0.5 + 0.5 * min(share, 1.0)) > best[0]:
                    best = (mem * (0.5 + 0.5 * min(share, 1.0)), mem, v, share, src, lo, hi, k, note)
    if best and best[1] >= 0.6 and best[3] >= min_share:
        _, mem, v, share, src, lo, hi, k, note = best
        r.membership = 1.0 if mem >= 0.8 else mem
        r.verified = True
        unit = (f"{v * 1e6:.3g} us" if v < 1e-3 else f"{v * 1e3:.4g} ms")
        r.evidence = f"{name} {unit} ({src}, share {share:.2f})" + (f" = {k:g}x {note or 'library'}" if k != 1 else (f" {note}" if note else ""))
        return r
    dominant = [c for c in cands if c[1] >= 0.5 and c[2] == "interval"]
    if dominant:
        v = dominant[0][0]
        if not any(_match_value(v / k, lo, hi, max(tol, 0.03) * 3) >= 0.5 for lo, hi, tol, _ in vals for k in (1.0, 2.0, 3.0, 0.5)):
            r.membership = 0.7
            r.evidence = f"dominant {name} {v * 1e3:.4g} ms matches none"
    return r


def t_pri(d: dict, s: dict) -> FieldResult:
    items = d.get("timing", {}).get("pri_s")
    fine_too = s["fine"].get("is_pulse_structure", False) or any(_vals(items) and lo < 50e-6 for lo, _, _, _ in _vals(items))
    return _test_intervals("pri", items, _interval_candidates(s, fine_too))


def t_period(d: dict, s: dict) -> FieldResult:
    items = d.get("timing", {}).get("period_s")
    return _test_intervals("period", items, _interval_candidates(s, False), min_share=0.05)


def t_pulse_pair(d: dict, s: dict) -> FieldResult:
    items = d.get("timing", {}).get("pulse_pair_s")
    r = FieldResult("pulse_pair", 0)
    if not items:
        return r
    f = s["fine"]
    cands = [(p["value"], p["share"], "pulse pair") for p in f.get("pri", [])]
    res = _test_intervals("pulse_pair", items, cands, min_share=0.1)
    if not res.verified and f.get("is_pulse_structure") and f.get("n", 0) >= 10:
        res.membership = min(res.membership, 0.6)
        res.evidence = res.evidence or "pulsed but no pair spacing"
    return res


def t_gap(d: dict, s: dict, band: dict) -> FieldResult:
    r = FieldResult("gap", 0)
    vals = _vals(d.get("timing", {}).get("gap_s"))
    if not vals:
        return r
    cands = [(g["value"], g["share"], "gap") for g in s["coarse"].get("gap", [])]
    best = None
    for lo, hi, tol, note in vals:
        for v, share, _ in cands:
            mem = _match_value(v, lo, hi, max(tol, 0.1))
            if best is None or mem > best[0]:
                best = (mem, v, share, note)
    if best and best[0] >= 0.6 and best[2] >= 0.08:
        r.verified = True
        r.evidence = f"gap {best[1] * 1e6:.3g} us (share {best[2]:.2f}){(' ' + best[3]) if best[3] else ''}"
        return r
    if any(hi <= 30e-6 for _, hi, _, _ in vals) and band.get("ack_pair_fraction", 0) >= 0.15:
        r.verified = True
        r.evidence = f"SIFS+ACK pairs in band ({band['ack_pair_fraction']:.2f} of long bursts)"
    return r


def t_duty(d: dict, s: dict) -> FieldResult:
    r = FieldResult("duty", 0)
    rng = d.get("timing", {}).get("duty")
    c = s["coarse"]
    if not rng or c.get("n", 0) < 1:
        return r
    duty = c.get("duty", 0)
    lo, hi = float(rng["min"]), float(rng["max"])
    if lo - 0.15 <= duty <= hi + 0.15:
        r.verified = True
        r.evidence = f"duty {duty:.2f}"
    else:
        r.membership = 0.75
        r.evidence = f"duty {duty:.2f} outside {lo:g}-{hi:g}"
    return r


def t_carrier_line(d: dict, s: dict) -> FieldResult:
    r = FieldResult("carrier_line", 0)
    sp = d.get("spectral", {})
    if "carrier_line" not in sp:
        return r
    cl = s["mod"].get("carrier_line_db", 0)
    snr = s["mod"].get("snr_db", 0)
    if sp["carrier_line"]:
        if cl >= 8:
            r.verified = True
            r.evidence = f"carrier line {cl:.0f} dB"
        elif cl < 4 and snr >= 10:
            r.membership = 0.6
            r.evidence = f"no carrier line ({cl:.0f} dB) at {snr:.0f} dB"
    elif cl >= 15:
        f_line = float(s.get("fc_hz", 0)) + float(s["mod"].get("carrier_offset_hz", 0))
        if s.get("fc_cap_hz") and abs(f_line - float(s["fc_cap_hz"])) <= 3.0 * float(s.get("rate_hz", 1e9)) / 1024:
            r.evidence = f"line at the capture centre ({cl:.0f} dB): receiver DC, ignored"
        else:
            r.membership = 0.8
            r.evidence = f"unexpected carrier line {cl:.0f} dB"
    return r


def _tone_test(name: str, tones, lines, tol_rel: float, tol_abs: float) -> FieldResult:
    r = FieldResult(name, 0)
    tones = [float(x) for x in tones or []]
    if not tones or not lines:
        return r
    hits = []
    for t in tones:
        for f, snr in lines:
            if abs(f - t) <= max(tol_rel * t, tol_abs) and snr >= 10:
                hits.append(f"{t:g} Hz ({snr:.0f} dB)")
                break
    if hits:
        r.verified = True
        r.evidence = f"{name.replace('_', ' ')} " + ", ".join(hits)
    return r


def t_fm_tones(d: dict, s: dict) -> FieldResult:
    return _tone_test("fm_tones", d.get("spectral", {}).get("fm_tones_hz"), s.get("fm", {}).get("lines", []), 0.01, 20.0)


def t_am_tones(d: dict, s: dict) -> FieldResult:
    return _tone_test("am_tones", d.get("spectral", {}).get("am_tones_hz"), s.get("am", {}).get("lines", []), 0.02, 5.0)


def t_hd(d: dict, s: dict) -> FieldResult:
    r = FieldResult("hd_sidebands", 0)
    if not d.get("spectral", {}).get("hd_sidebands"):
        return r
    h = s.get("hd", {})
    if h.get("measured") and h.get("sideband_db", -99) >= 6 and h.get("sideband_flatness_db", 99) <= 4:
        r.verified = True
        r.evidence = f"HD sidebands {h['sideband_db']:.0f} dB above the floor"
    return r


def t_hop(d: dict, s: dict, band: dict) -> FieldResult:
    r = FieldResult("hop", 0)
    hop = d.get("hop")
    if not hop or not band:
        return r
    if hop.get("rate_hz") and band.get("hops_per_s", 0) >= 0.25 * float(hop["rate_hz"]) and band.get("hop_fraction", 0) >= 0.4:
        r.verified = True
        r.evidence = f"band hops {band['hops_per_s']:.0f}/s"
    elif hop.get("set_hz") and band.get("ble_adv_triplets", 0) >= 1:
        r.verified = True
        r.evidence = f"{band['ble_adv_triplets']} advertising-channel triplets in band"
    return r


WEAK_FIELDS = {"modulation", "duty", "burst", "deviation", "carrier_line"}
TESTS = [t_modulation, t_symbol_rate, t_ofdm, t_chirp, t_deviation, t_pulse, t_burst, t_pri, t_period, t_pulse_pair, t_duty,
         t_carrier_line, t_fm_tones, t_am_tones, t_hd]


def descriptor_score(d: dict, s: dict, band: dict, weights: dict) -> tuple[float, float, list[FieldResult], list[FieldResult]]:
    """-> (factor, support, verified results, contradicting results)."""
    if not d or not s.get("measured"):
        return 1.0, 0.0, [], []
    results = [t(d, s) for t in TESTS] + [t_gap(d, s, band), t_hop(d, s, band)]
    # contradictions from a weak or cut emission are unreliable: scale them towards neutral
    m = s.get("mod", {})
    snr = max(m.get("snr_db", 0.0) or 0.0, m.get("snr_tight_db", 0.0) or 0.0)
    rel = max(0.0, min(1.0, (snr - 6.0) / 8.0))
    if s.get("cut"):
        rel = min(rel, 0.3)
    for r in results:
        if not r.verified and r.membership < 0.999:
            r.membership = 1.0 - rel * (1.0 - r.membership)
    fields = set(mio.descriptor_fields({"descriptor": d}))
    tested, verified, contra = [], [], []
    w_all = 0.0
    for r in results:
        r.weight = float(weights.get(r.name, mio.DESCRIPTOR_WEIGHTS.get(r.name, 1.0)))
        if r.name in fields:
            w_all += r.weight
        if r.verified:
            verified.append(r)
            tested.append(r)
        elif r.membership < 0.999:
            contra.append(r)
            tested.append(r)
    if tested:
        wsum = sum(r.weight for r in tested)
        penalty = math.exp(sum(r.weight * math.log(max(r.membership, 1e-3)) for r in tested) / wsum)
    else:
        penalty = 1.0
    support = sum(r.weight for r in verified) / w_all if w_all > 0 else 0.0
    return penalty * (1.0 + 0.5 * support), support, verified, contra


# =========================================================================================== decision
def combine(v1_conf: float, factor: float, support: float) -> float:
    """v1 confidence x descriptor penalty, then lifted towards 1 by the verified share of the descriptor:
    conf = 1 - (1 - v1 * penalty) * (1 - support). A fully verified descriptor nearly settles it; weak fields
    (duty, burst length) carry little weight so they lift little."""
    penalty = factor / (1.0 + 0.5 * support) if support > 0 else factor
    base = max(0.0, min(1.0, v1_conf * penalty))
    return max(0.0, min(1.0, 1.0 - (1.0 - base) * (1.0 - min(support, 0.85))))


@dataclass
class Match2:
    entry: dict
    v1: lk.Match
    conf: float
    factor: float
    support: float
    verified: list
    contradictions: list
    variant: str = ""
    variant_evidence: str = ""
    decode: object = None

    @property
    def label(self) -> str:
        return self.entry["label"] + (f"/{self.variant}" if self.variant else "")


def pick_variant(e: dict, s: dict, band: dict) -> tuple[str, str]:
    cands = []
    for v in e.get("variants", []):
        f, support, ver, contra = descriptor_score(v.get("descriptor", {}), s, band, e.get("weights", {}))
        # a variant needs more than the coarse modulation class: a rate, numerology, tone, timing field or two weak ones
        if ver and sum(r.weight for r in ver) >= 2.5 and not any(c.membership < 0.5 for c in contra):
            cands.append((support, v.get("suffix", v.get("id", "")), "; ".join(r.evidence for r in ver)))
    if not cands:
        return "", ""
    cands.sort(reverse=True)
    if len(cands) == 1 or cands[0][0] >= cands[1][0] + 0.2:
        return cands[0][1], cands[0][2]
    return "", ""


def decide_v2(t: lk.Track, s: dict, band: dict, library: dict, margin: float) -> tuple[Match2 | None, list[Match2], str]:
    if t.forced_label:
        e = next(x for x in library["entries"] if x.get("kind") == "cross_track")
        fake = dict(e)
        fake["label"] = t.forced_label
        m1 = lk.Match(fake, 0.95, {"comb"}, [t.forced_evidence], True)
        return Match2(fake, m1, 0.95, 1.0, 0.0, [], []), [], "ok"
    matches = []
    for e in library["entries"]:
        m1 = lk.score_entry(t, e)
        if m1 is None:
            continue
        f, support, ver, contra = descriptor_score(e.get("descriptor", {}), s, band, e.get("weights", {}))
        matches.append(Match2(e, m1, combine(m1.conf, f, support), f, support, ver, contra))
    matches.sort(key=lambda m: -m.conf)
    def _fields_ok(m):
        need = set(m.entry.get("requires_fields", []))
        return need <= {r.name for r in m.verified}

    def _strong(m):
        # physical-layer evidence that can stand in for an unresolvable channel raster: rates, numerology, timing, tones
        return sum(r.weight for r in m.verified if r.name not in WEAK_FIELDS)

    mm = s.get("mod", {}) if s.get("measured") else {}
    snr = max(mm.get("snr_db", 0.0) or 0.0, mm.get("snr_tight_db", 0.0) or 0.0) if mm else None
    def _weak_unverified(m):
        # a weak emission with nothing but band / width evidence is left as "signal": false labels cost more than misses
        return snr is not None and snr < LOW_SNR_DB and not m.verified and not m.entry.get("catch_all") and m.entry.get("kind") != "cross_track"

    def _n_strong(m):
        # the verified descriptor fields an entry in a shared band must show: rates, numerology, timing, tones, chirp,
        # hop or a decode. Modulation class, deviation, duty, burst length and a carrier line are too easy to share
        # (an OOK remote looks like narrow FM with a small deviation to the class test)
        return sum(1 for r in m.verified if r.name not in WEAK_FIELDS)

    def _eligible(m):
        if m.entry["tier"] not in THRESH or _weak_unverified(m):
            return False
        if _n_strong(m) == 0 and m.v1.conf < 0.5:
            return False                    # weak fields may confirm a plausible band score, never rescue a poor one
        thr = THRESH[m.entry["tier"]] - (0.05 if m.verified else 0.0)
        return (m.v1.requires_ok or (_strong(m) >= 3.0 and not any(r.membership < 0.6 for r in m.contradictions))) \
            and m.conf >= thr and _n_strong(m) >= int(m.entry.get("requires_descriptors", 0)) and _fields_ok(m)
    eligible = [m for m in matches if _eligible(m)]
    if not eligible:
        if not matches:
            return None, matches, "no entry passes"
        b = matches[0]
        why = f"best {b.entry['label']} {b.conf:.2f}"
        if _weak_unverified(b):
            why += f" (weak signal {snr:.0f} dB, no verified descriptor)"
        elif not b.v1.requires_ok:
            why += " (required features unverified)"
        elif _n_strong(b) < int(b.entry.get("requires_descriptors", 0)):
            why += " (no strong verified descriptor)"
        elif not _fields_ok(b):
            why += f" (needs {', '.join(b.entry.get('requires_fields', []))})"
        return None, matches, why
    eligible.sort(key=lambda m: (m.entry.get("catch_all", False), -m.conf))
    best = eligible[0]
    for m in matches:
        if m.entry["id"] == best.entry["id"]:
            continue
        if best.entry.get("catch_all") or not m.entry.get("catch_all"):
            if m.conf >= best.conf - margin and m.entry["tier"] in THRESH:
                # verified evidence breaks ties: clearly more verified descriptor weight wins
                if best.support >= m.support + 0.15:
                    continue
                if m.support >= best.support + 0.15 and m in eligible:
                    best = m
                    continue
                return None, matches, f"ambiguous: {best.entry['label']} {best.conf:.2f} vs {m.entry['label']} {m.conf:.2f}"
    best.variant, best.variant_evidence = pick_variant(best.entry, s, band)
    return best, matches, "ok"


# =========================================================================================== decode
def apply_decode(t: lk.Track, s: dict, best, matches: list, capiq, max_protocols: int = 4, cache: dict | None = None) -> tuple:
    """Try the protocol decoders of the plausible entries: the chosen one first, then the runners-up, then any
    entry whose decodable channel the track sits on (a BLE advertising channel, 1090 MHz, a DECT carrier) whatever
    its band score. A passing decode settles the label (confidence 0.97; 0.86 for a sync-only result) and names the
    variant. Results are cached per (protocol, channel) within a capture: neighbouring tracks of one emitter share them."""
    if t.forced_label or capiq is None or not s.get("measured"):
        return best, []
    mm = s.get("mod", {})
    snr = max(mm.get("snr_db", 0.0) or 0.0, mm.get("snr_tight_db", 0.0) or 0.0)
    order = ([best.entry] if best else []) + [m.entry for m in matches if m.conf >= 0.2 and (not best or m.entry["id"] != best.entry["id"])]
    for m in matches:
        if m.entry not in order and m.entry.get("decode") and any(
                dw.REGISTRY[pr]["centre"].split(":")[0] in ("fixed", "set", "raster") and dw.centre_for(pr, t.fc, t.bw, capiq.rate) for pr in dw.protocols_for_entry(m.entry["id"], m.entry)):
            order.append(m.entry)
    protos = []
    for e in order:
        for pr in dw.protocols_for_entry(e["id"], e):
            if all(pr != q for q, _ in protos):
                protos.append((pr, e))
    decodes = []
    for pr, e in protos[:max_protocols]:
        key = None
        if cache is not None and dw.REGISTRY[pr]["centre"].split(":")[0] in ("fixed", "set", "raster"):
            cands = dw.centre_for(pr, t.fc, t.bw, capiq.rate)
            key = (pr, cands[0]) if cands else None
        if key is not None and key in cache:
            d = cache[key]
        else:
            d = dw.attempt(pr, capiq, t.fc, t.bw, snr)
            if key is not None and (d.ok or "skipped" not in d.evidence):
                cache[key] = d
        decodes.append(d)
        if not d.ok:
            continue
        m = next((x for x in matches if x.entry["id"] == e["id"]), None)
        if m is None:
            m = Match2(e, lk.Match(e, 0.5, set(), [], True), 0.5, 1.0, 0.0, [], [])
            matches.append(m)
        m.verified.append(FieldResult("decode", 3.0, 1.0, True, "DECODED " + d.evidence))
        m.conf = max(m.conf, DECODE_WEAK_CONF if d.weak else DECODE_CONF)
        m.decode = d
        if d.variant:
            m.variant, m.variant_evidence = d.variant, d.evidence
        best = m
        if not d.weak:
            break
    return best, decodes


# =========================================================================================== per capture
def sdw_summary(s: dict) -> dict:
    if not s.get("measured"):
        return {"sdw_measured": False}
    m, o, L, f, c = s["mod"], s["ofdm"], s["lines"], s["fine"], s["coarse"]
    return {"sdw_measured": True, "sdw_class": m.get("class"), "sdw_class_conf": m.get("class_conf"), "sdw_snr_db": m.get("snr_db"),
            "sdw_env_cv": m.get("env_cv_corr", m.get("env_cv")), "sdw_dev_hz": m.get("dev_est_hz"), "sdw_levels": m.get("n_levels"),
            "sdw_carrier_db": m.get("carrier_line_db"), "sdw_edge_db": m.get("edge_drop_db"), "sdw_obw99_hz": m.get("obw99_hz"),
            "sdw_chirp_hz_s": m.get("chirp_rate_hz_s") if m.get("chirp_consistency", 0) >= 0.6 else "",
            "sdw_ofdm_scs_hz": (round(o["scs_hz"], 1) if o.get("scs_hz") and o.get("peaks") and o["peaks"][0].get("ratio", 0) >= 6 else ""),
            "sdw_ofdm_cp_us": (round(o["cp_s"] * 1e6, 3) if o.get("cp_s") else ""),
            "sdw_abs2_line": (round(L["abs2"][0][0], 1) if L.get("abs2") else ""), "sdw_finst_line": (round(L["finst"][0][0], 1) if L.get("finst") else ""),
            "sdw_sq_spacing": L.get("sq_spacing_hz", ""),
            "sdw_pulsed": f.get("is_pulse_structure", False), "sdw_pulse_us": (round(f["dur_p50"] * 1e6, 3) if f.get("is_pulse_structure") and f.get("dur_p50") else ""),
            "sdw_fine_pri_us": (round(f["pri"][0]["value"] * 1e6, 3) if f.get("pri") else ""),
            "sdw_bursts": c.get("n", 0), "sdw_burst_ms": (round(c["dur_p50"] * 1e3, 4) if c.get("dur_p50") else ""), "sdw_duty": c.get("duty"),
            "sdw_pri_ms": (round(c["pri"][0]["value"] * 1e3, 4) if c.get("pri") else ""), "sdw_gap_us": (round(c["gap"][0]["value"] * 1e6, 2) if c.get("gap") else ""),
            "sdw_period_ms": (round(s["periods"][0][0] * 1e3, 3) if s.get("periods") else ""),
            "sdw_fm_lines": ";".join(f"{x[0]:.0f}" for x in s.get("fm", {}).get("lines", [])[:4]),
            "sdw_hd_db": s.get("hd", {}).get("sideband_db", ""), "sdw_seconds": s.get("seconds")}


def run_capture(stem: str, cap: dict, library: dict, margin: float, measure: bool = True, max_tracks: int = 600, quiet: bool = False,
                attempt_decode: bool = False, max_decode_tracks: int = 400):
    t0 = time.time()
    tracks = lk.build_tracks(cap["regions"], cap["n_frames"])
    capiq = None
    if cap.get("data_path"):
        capiq = md.CaptureIQ(cap["data_path"], cap["dtype"], cap["rate"], cap["fc"])
        grids = lk.pilot_grids_from_manifest(library)
        for t in tracks[:max_tracks]:
            try:
                t.iq = lk.iq_features(t, cap["data_path"], cap["dtype"], grids)
            except Exception as ex:  # noqa: BLE001
                t.iq = {"error": f"{type(ex).__name__}: {str(ex)[:80]}"}
    sdws: dict[int, dict] = {}
    if measure and capiq is not None:
        for i, t in enumerate(tracks[:max_tracks]):
            try:
                sdws[t.tid] = md.measure_track(t, capiq)
            except Exception as ex:  # noqa: BLE001
                sdws[t.tid] = {"measured": False, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}
            if not quiet and (i + 1) % 50 == 0:
                print(f"   {stem}: measured {i + 1}/{min(len(tracks), max_tracks)} tracks ({time.time() - t0:.0f} s)", flush=True)
    band = md.measure_band([(t, sdws[t.tid]) for t in tracks if t.tid in sdws], capiq.n / capiq.rate if capiq else 1.0) if sdws else {}
    empty = {"measured": False}
    # first pass: which tracks are explained on physical-layer evidence (never comb candidates). A band label alone
    # does not protect a track: the bench comb runs through the 850 MHz uplink and paging bands, whose catch-all
    # entries label every narrow persistent tooth at 1.0, as much as through ISM 900
    explained = set()
    for t in tracks:
        best, _, _ = decide_v2(t, sdws.get(t.tid, empty), band, library, margin)
        if best and best.conf >= 0.9 and any(r.name not in WEAK_FIELDS for r in best.verified):
            explained.add(t.tid)
    lk.comb_pass(tracks, library, explained)
    rows = []
    n_dec = 0
    t_dec = 0.0
    dec_cache: dict = {}
    for t in tracks:
        s = sdws.get(t.tid, empty)
        best, matches, why = decide_v2(t, s, band, library, margin)
        decodes = []
        if attempt_decode and n_dec < max_decode_tracks:
            t1 = time.time()
            best, decodes = apply_decode(t, s, best, matches, capiq, cache=dec_cache)
            t_dec += time.time() - t1
            n_dec += 1 if decodes else 0
            if best is not None and best.decode is not None:
                why = "ok"
                matches.sort(key=lambda m: -m.conf)
        top = matches[0] if matches else None
        row = {"capture": stem, "track": t.tid, "n_regions": len(t.regions), "frames_present": len({r.frame for r in t.regions}), "frames_total": t.n_frames_total,
               "fc_hz": round(t.fc, 1), "bw_hz": round(t.bw, 1), "f_lo_hz": round(t.f_lo, 1), "f_hi_hz": round(t.f_hi, 1), "duty_mask": round(t.duty, 3),
               "persistence": round(t.persistence, 3), "temporal": t.temporal, "cut": t.cut, **sdw_summary(s),
               "estimate": (PREFIX + best.label) if best else "signal", "entry": best.entry["id"] if best else "", "variant": best.variant if best else "",
               "confidence": round(best.conf, 3) if best else "", "v1_confidence": round(best.v1.conf, 3) if best else (round(top.v1.conf, 3) if top else ""),
               "descriptor_factor": round(best.factor, 3) if best else (round(top.factor, 3) if top else ""),
               "support": round(best.support, 3) if best else "", "verified": ";".join(r.name for r in best.verified) if best else "",
               "contradictions": ";".join(f"{r.name}({r.membership:.2f})" for r in best.contradictions) if best else "",
               "evidence": ("; ".join(best.v1.evidence) + " | SDW: " + "; ".join(r.evidence for r in best.verified + best.contradictions if r.evidence)
                            + (f" | variant: {best.variant_evidence}" if best.variant else "")) if best else why,
               "decoded": (best.decode.protocol if (best and best.decode is not None) else ""),
               "decode_summary": (best.decode.evidence if (best and best.decode is not None) else ""),
               "decode_fields": (json.dumps(best.decode.fields, default=_json_default)[:600] if (best and best.decode is not None) else ""),
               "decode_attempts": ";".join(f"{d.protocol}:{'ok' if d.ok else 'no'}" for d in decodes),
               "runner_up": (f"{top.label} {top.conf:.2f}" if (top and (not best or top.entry['id'] != best.entry['id'])) else
                             (f"{matches[1].label} {matches[1].conf:.2f}" if best and len(matches) > 1 else "")),
               "_best": best, "_track": t, "_sdw": s}
        rows.append(row)
    if attempt_decode and not quiet and n_dec:
        print(f"   {stem}: decode attempted on {n_dec} tracks ({t_dec:.0f} s), {sum(1 for r in rows if r['decoded'])} decoded", flush=True)
    return tracks, rows, sdws, band


def write_labels(rows: list[dict], root_for_manifest: Path | None) -> int:
    by_meta: dict[Path, list] = defaultdict(list)
    for row in rows:
        for r in row["_track"].regions:
            if r.meta_path is None or r.ann_index is None:
                continue
            by_meta[r.meta_path].append((r.ann_index, row))
    n = 0
    label_by_key = {}
    for mp, items in by_meta.items():
        meta = json.loads(mp.read_text())
        anns = meta.get("annotations", [])
        for idx, row in items:
            if idx >= len(anns):
                continue
            a = anns[idx]
            best = row["_best"]
            orig = a.get("argus:detector_label", a.get("core:label", ""))
            a["argus:detector_label"] = orig
            for k in list(a.keys()):
                if k.startswith("argus:estimate_") or k in ("argus:sdw", "argus:decode"):
                    del a[k]
            if best:
                a["core:label"] = PREFIX + best.label
                a["argus:estimate_entry"] = best.entry["id"]
                if best.variant:
                    a["argus:estimate_variant"] = best.variant
                a["argus:estimate_confidence"] = round(best.conf, 3)
                a["argus:estimate_verified"] = [r.name for r in best.verified]
                a["argus:estimate_evidence"] = row["evidence"][:600]
                if best.decode is not None:
                    a["argus:decode"] = {"protocol": best.decode.protocol, "evidence": best.decode.evidence[:400],
                                         "fields": json.loads(json.dumps(best.decode.fields, default=_json_default))}
            else:
                a["core:label"] = "signal"
            s = row["_sdw"]
            if s.get("measured"):
                summ = {k[4:]: v for k, v in sdw_summary(s).items() if v not in ("", None, False) and k not in ("sdw_measured", "sdw_seconds")}
                a["argus:sdw"] = summ
            a["argus:estimate_source"] = VERSION
            label_by_key[(mp.name[: -len(".sigmf-meta")], str(a.get("core:sample_start", 0)), str(a.get("core:freq_lower_edge")))] = a["core:label"]
            n += 1
        mp.write_text(json.dumps(meta, indent=2) + "\n")
    if root_for_manifest:
        for mcsv in [root_for_manifest / "manifest.csv"] + sorted(root_for_manifest.glob("*/manifest.csv")):
            if not mcsv.exists():
                continue
            with open(mcsv, newline="") as f:
                rd = csv.DictReader(f)
                fields = rd.fieldnames or []
                rws = list(rd)
            if "label" not in fields:
                continue
            ch = 0
            for rw in rws:
                stem = Path(rw.get("file", "")).name
                stem = stem[: -len(".sigmf-data")] if stem.endswith(".sigmf-data") else stem
                lab = None
                for (s2, ss, fl), v in label_by_key.items():
                    if s2 == stem and ss == str(rw.get("sample_start", "0")) and lk._close(fl, rw.get("freq_lower_edge_hz")):
                        lab = v
                        break
                if lab is not None and rw["label"] != lab:
                    rw["label"] = lab
                    ch += 1
            if ch:
                with open(mcsv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rws)
    return n


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def collect_inputs(inp: Path, masks: str | None, cols: int | None) -> tuple[dict, Path, bool]:
    if masks:
        if (inp / "masks" / f"{masks}.npz").exists():
            return lk.regions_from_masks(inp, masks), inp, False
        caps = {}
        for d in sorted(q for q in inp.iterdir() if q.is_dir() and (q / "masks" / f"{masks}.npz").exists()):
            caps.update(lk.regions_from_masks(d, masks))
        return caps, inp, False
    if inp.is_file() and inp.name.endswith(".sigmf-meta"):
        return lk.regions_from_sigmf(inp, cols), inp.parent, True
    if inp.is_dir():
        if (inp / "preprocess_summary.json").exists() or any((q / "snippets").is_dir() for q in inp.iterdir() if q.is_dir()):
            return lk.regions_from_preprocessed(inp, cols), inp, True
        caps = {}
        for mp in sorted(q for q in inp.glob("*.sigmf-meta") if not q.name.endswith(".source.sigmf-meta")):
            caps.update(lk.regions_from_sigmf(mp, cols))
        return caps, inp, True
    raise SystemExit(f"cannot read {inp}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--library", default=str(mio.DEFAULT_LIBRARY))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--select", nargs="*", default=None, help="only captures whose name contains one of these substrings")
    ap.add_argument("--no-measure", action="store_true", help="v1 scoring only (no SDW)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--max-tracks", type=int, default=600, help="measure at most this many tracks per capture")
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--attempt-decode", action="store_true", help="run the protocol decoders (decode_waveforms.py) on plausible tracks; a pass settles the label")
    ap.add_argument("--max-decode-tracks", type=int, default=400, help="decode at most this many tracks per capture")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    library = mio.load_library(a.manifest, a.library)
    inp = Path(a.input).expanduser()
    caps, default_report, can_write = collect_inputs(inp, a.masks, a.cols)
    report_dir = Path(a.report_dir) if a.report_dir else default_report
    report_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    t_all = time.time()
    for stem, cap in caps.items():
        if a.select and not any(s in stem for s in a.select):
            continue
        t0 = time.time()
        tracks, rows, sdws, band = run_capture(stem, cap, library, a.margin, not a.no_measure, a.max_tracks, a.quiet, a.attempt_decode, a.max_decode_tracks)
        all_rows.extend(rows)
        (report_dir / f"sdw_{stem}.json").write_text(json.dumps({"tracks": sdws, "band": band}, default=_json_default))
        if not a.quiet:
            c = Counter(r["estimate"] for r in rows)
            print(f"{stem}: {len(cap['regions'])} regions -> {len(tracks)} tracks in {time.time() - t0:.0f} s; " + ", ".join(f"{k} {v}" for k, v in c.most_common()), flush=True)
    n_written = write_labels(all_rows, inp if inp.is_dir() else None) if (can_write and not a.no_write) else 0
    fields = [k for k in all_rows[0].keys() if not k.startswith("_")] if all_rows else ["capture"]
    with open(report_dir / "lut_regions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    labels = Counter(r["estimate"] for r in all_rows)
    summ = {"tool": VERSION, "library": library.get("_library_path"), "input": str(inp), "tracks": len(all_rows), "attempt_decode": a.attempt_decode,
            "decoded": Counter(r["decoded"] for r in all_rows if r.get("decoded")).most_common(),
            "labelled": sum(v for k, v in labels.items() if k != "signal"), "labels": dict(labels.most_common()),
            "measured": sum(1 for r in all_rows if r.get("sdw_measured")), "annotations_written": n_written, "seconds": round(time.time() - t_all, 1)}
    (report_dir / "lut_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    if not a.quiet:
        print(f"tracks {summ['tracks']}, labelled {summ['labelled']}: {summ['labels']} -> {report_dir / 'lut_regions.csv'}" + (f"; {n_written} annotations updated" if n_written else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
