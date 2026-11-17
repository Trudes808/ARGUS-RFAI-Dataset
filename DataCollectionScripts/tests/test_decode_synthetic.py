#!/usr/bin/env python3
"""Unit test of decode_waveforms.py: every protocol's frames are encoded, modulated (synth_modulators), buried in
noise at several SNRs with a random carrier offset, and decoded; the recovered identity fields must match.

    python3 tests/test_decode_synthetic.py [--snr 10 15 20 30] [--only adsb ble_adv] [--seed 1]

Writes tests/results/decode_synthetic_eval.md.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import decode_waveforms as dw  # noqa: E402
import synth_modulators as sm  # noqa: E402


def add_noise(x: np.ndarray, fs: float, bw: float, snr_db: float, f_off: float, rng) -> np.ndarray:
    """In-band SNR over `bw`; the emission is offset by f_off (the box-centre error a decoder must absorb)."""
    on = np.abs(x) > 0
    p = float(np.mean(np.abs(x[on]) ** 2)) if on.any() else 1.0
    n0 = p / 10 ** (snr_db / 10) * fs / bw
    t = np.arange(len(x)) / fs
    nz = (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))).astype(np.complex64) / math.sqrt(2)
    return (x * np.exp(2j * np.pi * f_off * t) + nz * math.sqrt(n0)).astype(np.complex64)


def place(buf: np.ndarray, x: np.ndarray, fs: float, t0: float):
    n0 = int(t0 * fs)
    n = min(len(x), len(buf) - n0)
    if n > 0:
        buf[n0: n0 + n] += x[:n]


# ----------------------------------------------------------------------------------------- cases
def case_adsb(rng, snr):
    fs = 8e6
    buf = np.zeros(int(0.3 * fs), dtype=np.complex64)
    icaos = set()
    t = 0.002
    while t < 0.29:
        icao = int(rng.integers(0xA00000, 0xA000FF))
        icaos.add(icao)
        bits = dw.modes_random_message(rng, icao, long=rng.random() < 0.8)
        place(buf, dw.modes_ppm_waveform(fs, bits) * float(rng.uniform(0.4, 1.0)), fs, t)
        t += float(rng.uniform(0.002, 0.006))
    y = add_noise(buf, fs, 3e6, snr, float(rng.uniform(-100e3, 100e3)), rng)
    d = dw.decode_adsb(y, fs)
    got = {int(h, 16) for h in d.fields.get("icao", [])}
    return d, d.ok and len(got & icaos) >= 1 and len(got - icaos) == 0, f"{len(got & icaos)}/{len(icaos)} aircraft"


def case_rds(rng, snr):
    fs = 480e3
    pi = int(rng.integers(0x1000, 0xFFFF))
    bits = dw.rds_groups(pi, "KARG", 10, 40, rng)
    x = sm.fm_stereo(fs, 1.0, 15e3, True, True, False, rng=rng, rds_bits=bits)
    y = add_noise(x, fs, 200e3, snr, float(rng.uniform(-3e3, 3e3)), rng)
    d = dw.decode_rds(y, fs)
    return d, d.ok and d.fields.get("pi") == f"{pi:04X}", f"PI {d.fields.get('pi')} vs {pi:04X}, groups {d.fields.get('groups')}, PS {d.fields.get('ps')}"


def case_ble(rng, snr):
    fs = 8e6
    buf = np.zeros(int(0.2 * fs), dtype=np.complex64)
    mac = bytes(int(v) for v in rng.integers(0, 256, 6))
    mac = bytes([mac[0] | 0xC0]) + mac[1:]
    ch = int(rng.choice([37, 38, 39]))
    t = 0.001
    n = 0
    while t < 0.19:
        pdu = dw.ble_adv_pdu(mac, 0 if rng.random() < 0.7 else 2, "ARGUS", rng, int(rng.integers(0, 10)))
        bits = dw.ble_adv_bits(pdu, ch)
        sym = bits.astype(np.float64) * 2 - 1
        place(buf, sm.cpm(sym, 1e6, fs, 0.5, 0.5, 2), fs, t)
        t += float(rng.uniform(0.004, 0.02))
        n += 1
    y = add_noise(buf, fs, 1.2e6, snr, float(rng.uniform(-60e3, 60e3)), rng)
    d = dw.decode_ble_adv(y, fs, ch)
    want = ":".join(f"{x:02X}" for x in mac)
    return d, d.ok and want in d.fields.get("adv_addresses", []), f"{d.n_ok}/{n} PDUs, {d.fields.get('adv_addresses')}, names {d.fields.get('names')}"


def case_lte(rng, snr):
    nid1, nid2 = int(rng.integers(0, 168)), int(rng.integers(0, 3))
    scs, cp = 15e3, 4.69e-6
    n_used = 300                                          # 5 MHz carrier (25 RBs)
    N = 512
    fs = N * scs
    fn = dw.lte_sync_overwrite(nid1, nid2, N, scs, cp)
    x = sm.ofdm(fs, scs, cp, n_used, 0.03, rng=rng, sym_fn=fn, cp_pattern_s=[5.21e-6] + [4.69e-6] * 6)[: int(0.03 * fs)]
    y = add_noise(x, fs, n_used * scs, snr, float(rng.uniform(-1.5e3, 1.5e3)), rng)
    d = dw.decode_lte(y, fs)
    return d, d.ok and d.fields.get("pci") == 3 * nid1 + nid2, f"PCI {d.fields.get('pci')} vs {3 * nid1 + nid2} ({d.fields.get('duplex')}, peak {d.fields.get('pss_peak')})"


def case_pocsag(rng, snr):
    rate = float(rng.choice([512.0, 1200.0, 2400.0]))
    fs = 76800.0
    bits, rics = dw.pocsag_stream(rng, 3, 200)
    sym = bits.astype(np.float64) * 2 - 1
    x = sm.cpm(sym, rate, fs, 2 * 4.5e3 / rate, None, 2)
    y = add_noise(x, fs, 20e3, snr, float(rng.uniform(-2e3, 2e3)), rng)
    d = dw.decode_pocsag(y, fs)
    got = {a["ric"] for a in d.fields.get("addresses", [])}
    return d, d.ok and d.fields.get("baud") == rate and got <= set(rics) and len(got) >= 1, f"{rate:g} bps -> {d.fields.get('baud')}, {len(got)}/{len(set(rics))} RICs, {d.fields.get('codewords_ok')} cw"


def case_flex(rng, snr):
    fs = 76800.0
    cyc, frm = int(rng.integers(0, 15)), int(rng.integers(0, 128))
    bits = np.concatenate([rng.integers(0, 2, 500).astype(np.uint8), dw.flex_frame_bits(cyc, frm, rng)])
    sym = bits.astype(np.float64) * 2 - 1
    x = sm.cpm(sym, 1600.0, fs, 2 * 4.8e3 / 1600.0, None, 2)
    y = add_noise(x, fs, 20e3, snr, float(rng.uniform(-2e3, 2e3)), rng)
    d = dw.decode_flex(y, fs)
    fr = d.fields.get("frames", [{}])
    fr = fr[0] if fr else {}
    return d, d.ok and fr.get("cycle") == cyc and fr.get("frame") == frm, f"cycle/frame {fr.get('cycle')}/{fr.get('frame')} vs {cyc}/{frm}"


def case_ert(rng, snr):
    fs = 524288.0
    buf = np.zeros(int(0.25 * fs), dtype=np.complex64)
    ids = []
    t = 0.01
    while t < 0.22:
        eid = int(rng.integers(1, 2 ** 26))
        if rng.random() < 0.6:
            bits = dw.scm_packet(eid, int(rng.integers(0, 16)), int(rng.integers(0, 2 ** 24)))
        else:
            bits = dw.scmplus_packet(eid, 0x9C, int(rng.integers(0, 2 ** 24)))
        ids.append(eid)
        chips = dw.ert_manchester_chips(bits)
        x = np.repeat(chips, 16).astype(np.complex64)
        place(buf, x, fs, t)
        t += len(x) / fs + float(rng.uniform(0.01, 0.04))
    y = add_noise(buf, fs, 100e3, snr, float(rng.uniform(-20e3, 20e3)), rng)
    d = dw.decode_ert(y, fs)
    got = {p["id"] for p in d.fields.get("packets", [])}
    return d, d.ok and got <= set(ids) and len(got) >= 1, f"{len(got)}/{len(ids)} meters ({', '.join(sorted({p['kind'] for p in d.fields.get('packets', [])}))})"


def case_p25(rng, snr):
    fs = 48e3
    nac = int(rng.integers(0, 0x1000))
    sym = np.concatenate([dw.p25_frame_symbols(nac, int(rng.choice([0, 5, 10])), 800, rng) for _ in range(3)])
    x = sm.cpm(sym, 4800.0, fs, 0.75, None, 4)
    y = add_noise(x, fs, 12.5e3, snr, float(rng.uniform(-1e3, 1e3)), rng)
    d = dw.decode_p25(y, fs)
    nacs = {n["nac"] for n in d.fields.get("nids", [])}
    return d, d.ok and not d.weak and nacs == {f"{nac:03X}"}, f"NAC {nacs} vs {nac:03X}, syncs {d.fields.get('frame_syncs')}"


def case_dmr(rng, snr):
    fs = 48e3
    buf = np.zeros(int(0.3 * fs), dtype=np.complex64)
    name = str(rng.choice(["BS_voice", "BS_data"]))
    t = 0.005
    n = 0
    while t + 0.0275 < 0.3:
        x = sm.cpm(dw.dmr_burst_symbols(name, rng), 4800.0, fs, 0.81, None, 4)
        place(buf, x[: int(0.0275 * fs)], fs, t)
        t += 0.030
        n += 1
    y = add_noise(buf, fs, 12.5e3, snr, float(rng.uniform(-1e3, 1e3)), rng)
    d = dw.decode_dmr(y, fs)
    return d, d.ok and name in d.fields.get("syncs", {}) and d.n_ok >= n // 2, f"{d.fields.get('syncs')} of {n} bursts, spacing {d.fields.get('burst_spacing_s')}"


def case_dect(rng, snr):
    fs = 4.608e6
    buf = np.zeros(int(0.05 * fs), dtype=np.complex64)
    rfpi = bytes(int(v) for v in rng.integers(0, 256, 5))
    t = 0.0005
    n = 0
    while t + 0.0004 < 0.05:
        bits = dw.dect_packet_bits(True, rfpi if rng.random() < 0.5 else None, rng)
        sym = bits.astype(np.float64) * 2 - 1
        place(buf, sm.cpm(sym, 1.152e6, fs, 0.5, 0.5, 2), fs, t)
        t += 0.01 / 24 * int(rng.integers(1, 6))
        n += 1
    y = add_noise(buf, fs, 1.4e6, snr, float(rng.uniform(-50e3, 50e3)), rng)
    d = dw.decode_dect(y, fs)
    return d, d.ok and rfpi.hex().upper() in d.fields.get("rfpi", []), f"{d.n_ok}/{n} A-fields, RFPI {d.fields.get('rfpi')} vs {rfpi.hex().upper()}"


def case_zigbee(rng, snr):
    fs = 8e6
    buf = np.zeros(int(0.05 * fs), dtype=np.complex64)
    pan = int(rng.integers(0, 0xFFFF))
    t = 0.001
    n = 0
    while t < 0.045:
        psdu = dw.zigbee_frame_bytes(bytes(int(v) for v in rng.integers(0, 256, int(rng.integers(4, 40)))), n, pan, 0x0000, int(rng.integers(1, 0xFFF0)))
        x = dw.oqpsk_chips_waveform(fs, dw.zigbee_chips(dw.zigbee_ppdu_symbols(psdu)))
        place(buf, x, fs, t)
        t += len(x) / fs + float(rng.uniform(0.001, 0.004))
        n += 1
    y = add_noise(buf, fs, 2e6, snr, float(rng.uniform(-40e3, 40e3)), rng)
    d = dw.decode_zigbee(y, fs)
    pans = {f["pan"] for f in d.fields.get("frames", []) if "pan" in f}
    return d, d.ok and pans == {f"{pan:04X}"} and d.n_ok >= n // 2, f"{d.n_ok}/{n} frames, PAN {pans} vs {pan:04X}"


def case_uat(rng, snr):
    fs = 4 * 1041666.67
    buf = np.zeros(int(0.05 * fs), dtype=np.complex64)
    t = 0.001
    n = 0
    while t < 0.045:
        bits = dw.uat_frame_bits(False, 384 if rng.random() < 0.5 else 240, rng)
        sym = bits.astype(np.float64) * 2 - 1
        place(buf, sm.cpm(sym, 1041666.67, fs, 0.6, None, 2), fs, t)
        t += len(bits) / 1041666.67 + float(rng.uniform(0.001, 0.005))
        n += 1
    y = add_noise(buf, fs, 1.3e6, snr, float(rng.uniform(-30e3, 30e3)), rng)
    d = dw.decode_uat(y, fs)
    return d, d.ok and d.fields.get("syncs", {}).get("ADS-B", 0) >= n // 2, f"{d.fields.get('syncs')} of {n}"


CASES = {"adsb": case_adsb, "rds": case_rds, "ble_adv": case_ble, "lte": case_lte, "pocsag": case_pocsag, "flex": case_flex, "ert": case_ert,
         "p25": case_p25, "dmr": case_dmr, "dect": case_dect, "zigbee": case_zigbee, "uat": case_uat}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", nargs="*", type=float, default=[10.0, 15.0, 20.0, 30.0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    rows = []
    fails = 0
    for name, fn in CASES.items():
        if a.only and name not in a.only:
            continue
        for snr in a.snr:
            for r in range(a.reps):
                rng = np.random.default_rng(a.seed + 1000 * r + int(snr))
                sm.set_seed(a.seed + r)
                t0 = time.time()
                try:
                    d, ok, note = fn(rng, snr)
                except Exception as ex:  # noqa: BLE001
                    d, ok, note = dw.Decode(name), False, f"ERROR {type(ex).__name__}: {ex}"
                dt = time.time() - t0
                fails += 0 if ok else 1
                rows.append((name, snr, ok, note, d.evidence, dt))
                print(f"{'PASS' if ok else 'FAIL'} {name:8s} {snr:4.0f} dB  {note}  | {d.evidence[:90]}  ({dt:.1f} s)", flush=True)
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    L = ["# decode_waveforms.py on synthetic frames", "", f"{len(rows) - fails}/{len(rows)} cases pass (seed {a.seed}).", "",
         "| protocol | SNR dB | result | recovered | evidence | s |", "|---|---|---|---|---|---|"]
    for name, snr, ok, note, ev, dt in rows:
        L.append(f"| {name} | {snr:g} | {'pass' if ok else 'FAIL'} | {note} | {ev[:120]} | {dt:.1f} |")
    (res / "decode_synthetic_eval.md").write_text("\n".join(L) + "\n")
    print(f"{len(rows) - fails}/{len(rows)} pass -> {res / 'decode_synthetic_eval.md'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
