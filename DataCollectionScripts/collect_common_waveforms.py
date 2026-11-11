#!/usr/bin/env python3
"""Sweep a USRP over the common US waveforms and store short SigMF snippets of each look.

The sweep is planned from common_ota_waveforms_manifest_usa.yaml: every entry with `sweep.default: true`
contributes its centre frequencies x sample rates (plus explicit `looks`). The captures reproduce the
conventions of the N320 stress set used to validate the detector (Holohub-Signal-Detection,
applications/usrp_wideband_signal_detection): ci16 SigMF pairs, one folder per capture,
    <radio>_<entry>[_<tag>]_<lo>-<hi>MHz_c<fc>M_r<rate>M_g<gain>_ch<ch>/<same>.sigmf-{data,meta}
with the band label, expected waveforms, probe statistics and overflow count in the metadata.

Examples
  # whole default plan (46 looks, ~1 s each), auto gain, device auto-discovered
  python3 collect_common_waveforms.py --out ~/captures/site_a --site "roof, Austin TX" --antenna "discone"
  # only FM + TV, all default rates
  python3 collect_common_waveforms.py --out ~/captures/fm_tv --entries fm_broadcast,atsc_tv_uhf
  # fixed frequencies at fixed rates (the manifest only supplies labels / expected waveforms)
  python3 collect_common_waveforms.py --out ~/captures/custom --freqs 98e6,881.5e6,2437e6 --rates 25e6,100e6
  # one rate for the whole default plan, 0.5 s each, two antennas (ch1 below 1.5 GHz, ch0 above)
  python3 collect_common_waveforms.py --out ~/captures/s --rates 50e6 --dur 0.5 --channel-below 1 --channel-above 0
  # just print the plan / survey the power per look without saving IQ
  python3 collect_common_waveforms.py --list ;  python3 collect_common_waveforms.py --survey --out ~/captures/survey

Requirements: UHD 4.x with the `rx_samples_to_file` example (ships in <prefix>/lib/uhd/examples) or the
UHD Python module (`import uhd`), numpy, PyYAML. Run as a user that may open the radio (sudo for X4xx
DPDK/raw sockets). /dev/shm is used as staging when it is large enough, so a slow disk cannot overflow
the stream; set --no-shm to write directly to --out.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest_io as mio  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "common_ota_waveforms_manifest_usa.yaml"
VERSION = "collect_common_waveforms.py 1.0 (ARGUS-RFAI-Dataset)"


# --------------------------------------------------------------------------------------------- plan
@dataclass
class Look:
    entry_id: str
    label: str
    name: str
    fc_hz: float
    rate_req_hz: float
    tag: str
    dur_s: float
    gain0_db: float
    antenna_group: str
    expected: str
    tier: str
    # filled when the radio is known
    rate_hz: float = 0.0
    mcr_hz: float | None = None
    decim: int | None = None
    channel: int = 0

    @property
    def span(self) -> tuple[float, float]:
        return self.fc_hz - self.rate_hz / 2, self.fc_hz + self.rate_hz / 2

    def band_label(self) -> str:
        lo, hi = self.span
        t = f"_{self.tag}" if self.tag else ""
        return f"{self.entry_id}{t}_{lo / 1e6:g}-{hi / 1e6:g}MHz"


def _num_list(s: str | None) -> list[float]:
    if not s:
        return []
    return [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]


def plan_from_manifest(m: dict, a: argparse.Namespace) -> list[Look]:
    d = m.get("defaults", {})
    dur_default = a.dur if a.dur is not None else float(d.get("dur_s", 1.0))
    gain_default = float(d.get("gain_start_db", 35))
    want = set(_split(a.entries)) if a.entries else None
    tiers = set(a.tier.upper().split(",")) if a.tier else None
    looks: list[Look] = []
    for e in m["entries"]:
        s = e.get("sweep") or {}
        if e.get("relative_to_capture_center"):
            continue
        selected = (want and e["id"] in want) or (tiers and e["tier"] in tiers) or a.all or (not want and not tiers and s.get("default"))
        if not selected:
            continue
        if e.get("radio_min") and a.radio not in ("auto", e["radio_min"]):
            print(f"   skip {e['id']}: needs {e['radio_min']}", file=sys.stderr)
            continue
        centers = _num_list(a.freqs) if a.freqs else list(s.get("centers_hz", []))
        rates = _num_list(a.rates) if a.rates else list(s.get("rates_hz", []))
        explicit = [] if (a.freqs or a.rates) else list(s.get("looks", []))
        if not centers and not explicit:
            print(f"   skip {e['id']}: no sweep centres in the manifest (use --freqs)", file=sys.stderr)
            continue
        g0 = float(s.get("gain_start_db", gain_default))
        dur = a.dur if a.dur is not None else float(s.get("dur_s", dur_default))
        grp = s.get("antenna_group", "low" if centers and centers[0] < 1.5e9 else "high")
        for fc in centers:
            for r in rates:
                looks.append(Look(e["id"], e["label"], e["name"], fc, r, "", dur, g0, grp, e.get("name", ""), e["tier"]))
        for lk in explicit:
            looks.append(Look(e["id"], e["label"], e["name"], float(lk["fc_hz"]), float(lk["rate_hz"]), str(lk.get("tag", "")),
                              dur, g0, grp, e.get("name", ""), e["tier"]))
    return looks


def plan_from_freqs(m: dict, a: argparse.Namespace) -> list[Look]:
    """--freqs given without --entries: one look per (freq, rate); label from the manifest entry that covers it."""
    d = m.get("defaults", {})
    dur = a.dur if a.dur is not None else float(d.get("dur_s", 1.0))
    g0 = float(d.get("gain_start_db", 35))
    rates = _num_list(a.rates) or [25.0e6]
    looks = []
    for fc in _num_list(a.freqs):
        cov = mio.entries_covering(m, fc)
        if cov:
            e = sorted(cov, key=lambda x: (x["tier"], -x.get("band_exclusive", False)))[0]
            eid, lab, name, tier = e["id"], e["label"], e["name"], e["tier"]
            exp = "; ".join(x["name"] for x in cov)
            grp = (e.get("sweep") or {}).get("antenna_group", "low" if fc < 1.5e9 else "high")
            g = float((e.get("sweep") or {}).get("gain_start_db", g0))
        else:
            eid, lab, name, tier, exp, grp, g = "custom", "signal", "user-selected frequency", "-", "no manifest entry covers this frequency", ("low" if fc < 1.5e9 else "high"), g0
        for r in rates:
            looks.append(Look(eid, lab, name, fc, r, "", dur, g, grp, exp, tier))
    return looks


def _split(s: str) -> list[str]:
    return [x for x in re.split(r"[,\s]+", s.strip()) if x]


def finalize_plan(looks: list[Look], radio: str, a: argparse.Namespace) -> list[Look]:
    spec = mio.RADIOS.get(radio, mio.RADIOS["generic"])
    out = []
    for lk in looks:
        if not (spec["tune_min"] <= lk.fc_hz <= spec["tune_max"]):
            print(f"   skip {lk.entry_id} @ {lk.fc_hz / 1e6:g} MHz: outside the {radio} tuning range", file=sys.stderr)
            continue
        r, mcr, n = mio.snap_rate(lk.rate_req_hz, radio)
        if abs(r - lk.rate_req_hz) / lk.rate_req_hz > 0.02:
            print(f"   note {lk.entry_id}: rate {lk.rate_req_hz / 1e6:g} -> {r / 1e6:g} MSps on the {radio}", file=sys.stderr)
        lk.rate_hz, lk.mcr_hz, lk.decim = r, mcr, n
        if a.channel_below is not None or a.channel_above is not None:
            split = a.split_hz
            lk.channel = (a.channel_below if lk.fc_hz < split else a.channel_above)
            if lk.channel is None:
                lk.channel = a.channel
        else:
            lk.channel = a.channel
        # cap duration by size
        max_bytes = a.max_gb * 1e9
        lk.dur_s = float(min(lk.dur_s, max_bytes / (lk.rate_hz * 4)))
        out.append(lk)
    return out


# --------------------------------------------------------------------------------------------- radio
def find_rx_binary(explicit: str | None) -> str | None:
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("RX_BIN"):
        cands.append(os.environ["RX_BIN"])
    w = shutil.which("rx_samples_to_file")
    if w:
        cands.append(w)
    try:
        prefix = subprocess.run(["uhd_config_info", "--pkg-path"], capture_output=True, text=True, timeout=10).stdout.strip()
        if prefix:
            cands.append(str(Path(prefix) / "lib" / "uhd" / "examples" / "rx_samples_to_file"))
    except Exception:
        pass
    for p in ("/usr/local/lib/uhd/examples/rx_samples_to_file", "/usr/lib/uhd/examples/rx_samples_to_file",
              "/usr/lib/x86_64-linux-gnu/uhd/examples/rx_samples_to_file", "/usr/lib/aarch64-linux-gnu/uhd/examples/rx_samples_to_file",
              "/opt/uhd/lib/uhd/examples/rx_samples_to_file"):
        cands.append(p)
    for c in cands:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def discover_radio(args: str) -> tuple[str, dict]:
    """Return (radio_key, info dict) via the UHD Python API or uhd_find_devices."""
    info: dict = {}
    try:
        import uhd  # type: ignore
        devs = uhd.find(args)
        if devs:
            d = devs[0]
            info = {k: d.get(k) for k in ("type", "product", "serial", "addr", "mgmt_addr", "name") if d.get(k)}
    except Exception:
        try:
            out = subprocess.run(["uhd_find_devices", "--args", args], capture_output=True, text=True, timeout=30).stdout
            for key in ("type", "product", "serial", "addr", "mgmt_addr", "name"):
                mt = re.search(rf"^\s*{key}:\s*(\S+)", out, re.M)
                if mt:
                    info[key] = mt.group(1)
        except Exception:
            pass
    if not info:
        return "generic", {}
    return mio.radio_key(info.get("product") or info.get("type") or ""), info


def iq_stats(path: Path, max_samps: int = 4_000_000) -> dict:
    raw = np.fromfile(path, dtype="<i2", count=2 * max_samps)
    if raw.size < 2:
        return {"peak_fs": 0.0, "rms_dbfs": -200.0, "clipped_frac": 0.0}
    iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    peak = float(np.max(np.abs(raw))) / 32767.0
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) / 32767.0
    # spectral span (max - median of an averaged 4096-bin PSD): a flat spectrum means nothing is being received
    span = 0.0
    nfft = 4096
    n = (len(iq) // nfft) * nfft
    if n >= 4 * nfft:
        P = np.mean(np.abs(np.fft.fft(iq[:n].reshape(-1, nfft), axis=1)) ** 2, axis=0)
        Pdb = 10 * np.log10(P + 1e-12)
        span = float(Pdb.max() - np.median(Pdb))
    return {"peak_fs": round(peak, 4), "rms_dbfs": round(20 * np.log10(max(rms, 1e-9)), 1),
            "clipped_frac": float(np.mean(np.abs(raw) >= 32700)), "psd_span_db": round(span, 1)}


class Radio:
    def __init__(self, a: argparse.Namespace):
        self.args = a.args
        self.ant = a.ant
        self.rx_bin = None if a.backend == "python" else find_rx_binary(a.rx_bin)
        self.backend = "python" if (a.backend == "python" or (a.backend == "auto" and not self.rx_bin)) else "rx_samples_to_file"
        self.key, self.info = (a.radio, {}) if a.radio != "auto" else discover_radio(a.args)
        if a.radio == "auto" and self.key == "generic" and self.info:
            print(f"   unknown product {self.info}: treating as generic (rates passed through)", file=sys.stderr)
        self._usrp = None
        self._mcr = None

    def hw_string(self) -> str:
        bits = [self.info.get("product", self.key).upper()]
        if self.info.get("serial"):
            bits.append(f"serial {self.info['serial']}")
        return "USRP " + " ".join(bits)

    # ---- capture one look -> (ok, stats)
    def capture(self, lk: Look, gain: float, dur_s: float, out_file: Path, quiet: bool = False) -> dict:
        if self.backend == "rx_samples_to_file":
            return self._capture_bin(lk, gain, dur_s, out_file, quiet)
        return self._capture_py(lk, gain, dur_s, out_file, quiet)

    def _dev_args(self, lk: Look) -> str:
        args = self.args or ""
        if lk.mcr_hz and "master_clock_rate" not in args:
            args = (args + "," if args else "") + f"master_clock_rate={lk.mcr_hz:.0f}"
        return args

    def _capture_bin(self, lk: Look, gain: float, dur_s: float, out_file: Path, quiet: bool) -> dict:
        cmd = [self.rx_bin, "--args", self._dev_args(lk), "--freq", f"{lk.fc_hz:.0f}", "--rate", f"{lk.rate_hz:.6f}",
               "--gain", f"{gain:g}", "--duration", f"{dur_s:.4f}", "--channel", str(lk.channel), "--type", "short",
               "--file", str(out_file)]
        if self.ant:
            cmd += ["--ant", self.ant]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        log = (p.stdout or "") + (p.stderr or "")
        ovf = len(re.findall(r"(?<![A-Za-z])[OD](?![A-Za-z])", log))  # UHD prints a bare 'O' (overflow) / 'D' (dropped)
        act_rate = re.search(r"Actual RX Rate:\s*([0-9.]+)\s*Msps", log)
        act_freq = re.search(r"Actual RX Freq:\s*([0-9.]+)\s*MHz", log)
        act_gain = re.search(r"Actual RX Gain:\s*([0-9.]+)\s*dB", log)
        ok = p.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0
        st = {"ok": ok, "overflow_events": ovf, "elapsed_s": round(time.time() - t0, 1), "backend": "rx_samples_to_file",
              "actual_rate_hz": float(act_rate.group(1)) * 1e6 if act_rate else None,
              "actual_freq_hz": float(act_freq.group(1)) * 1e6 if act_freq else None,
              "actual_gain_db": float(act_gain.group(1)) if act_gain else None}
        if not ok:
            err = [ln for ln in log.splitlines() if re.search(r"error|Error|RuntimeError|LookupError|refused|timeout", ln)]
            st["error"] = " | ".join(err[-3:]) or log.strip()[-400:]
            if not quiet:
                print(f"   !! capture failed: {st['error']}", file=sys.stderr)
        return st

    def _capture_py(self, lk: Look, gain: float, dur_s: float, out_file: Path, quiet: bool) -> dict:
        import uhd  # type: ignore
        t0 = time.time()
        args = self._dev_args(lk)
        if self._usrp is None or self._mcr != lk.mcr_hz:
            self._usrp = uhd.usrp.MultiUSRP(args)
            self._mcr = lk.mcr_hz
        u = self._usrp
        ch = lk.channel
        u.set_rx_rate(lk.rate_hz, ch)
        u.set_rx_freq(uhd.types.TuneRequest(lk.fc_hz), ch)
        u.set_rx_gain(gain, ch)
        if self.ant:
            u.set_rx_antenna(self.ant, ch)
        time.sleep(0.05)
        n = int(dur_s * lk.rate_hz)
        sa = uhd.usrp.StreamArgs("fc32", "sc16")
        sa.channels = [ch]
        rx = u.get_rx_stream(sa)
        spp = rx.get_max_num_samps()
        buf = np.empty((1, spp * 64), dtype=np.complex64)
        md = uhd.types.RXMetadata()
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        cmd.num_samps = n
        cmd.stream_now = True
        rx.issue_stream_cmd(cmd)
        got, ovf = 0, 0
        with open(out_file, "wb") as f:
            while got < n:
                k = rx.recv(buf, md, 1.0)
                if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    ovf += 1
                elif md.error_code != uhd.types.RXMetadataErrorCode.none:
                    if not quiet:
                        print(f"   !! stream error: {md.strerror()}", file=sys.stderr)
                    break
                if k:
                    k = min(k, n - got)
                    x = buf[0, :k]
                    i16 = np.empty(2 * k, dtype="<i2")
                    i16[0::2] = np.clip(np.round(x.real * 32767.0), -32768, 32767)
                    i16[1::2] = np.clip(np.round(x.imag * 32767.0), -32768, 32767)
                    i16.tofile(f)
                    got += k
        return {"ok": got > 0, "overflow_events": ovf, "elapsed_s": round(time.time() - t0, 1), "backend": "uhd-python",
                "actual_rate_hz": float(u.get_rx_rate(ch)), "actual_freq_hz": float(u.get_rx_freq(ch)),
                "actual_gain_db": float(u.get_rx_gain(ch))}


# --------------------------------------------------------------------------------------------- gain probe
def probe_gain(radio: Radio, lk: Look, gain: float, stage: Path, a: argparse.Namespace) -> tuple[float, dict]:
    """0.2 s probes: lower the gain when peak > 0.5 FS, raise it when rms < -40 dBFS (max 3 steps)."""
    spec = mio.RADIOS.get(radio.key, mio.RADIOS["generic"])
    gmax = a.gain_max if a.gain_max is not None else spec["max_gain"]
    hist = []
    for _ in range(3):
        f = stage / "probe.bin"
        st = radio.capture(lk, gain, 0.2, f, quiet=True)
        if not st["ok"]:
            return gain, {"probe_error": st.get("error", "probe failed"), "history": hist}
        s = iq_stats(f)
        f.unlink(missing_ok=True)
        hist.append({"gain_db": gain, **s})
        if s["peak_fs"] > 0.5:
            new = max(0.0, gain - 15.0)
        elif s["rms_dbfs"] < -40.0:
            new = min(gmax, gain + 15.0)
        else:
            break
        if new == gain:
            break
        gain = new
    return gain, {"history": hist}


# --------------------------------------------------------------------------------------------- output
def write_meta(path: Path, lk: Look, gain: float, dur_s: float, st: dict, stats: dict, radio: Radio, a: argparse.Namespace, m: dict, n_samps: int):
    now = dt.datetime.now(dt.timezone.utc)
    hw = f"{radio.hw_string()}, RX ch{lk.channel}" + (f" antenna port {a.ant}" if a.ant else "") + (f" ({a.antenna})" if a.antenna else "") + \
         f", gain {gain:g} dB" + (f", master_clock_rate {lk.mcr_hz:.0f}" if lk.mcr_hz else "")
    desc = (f"{dur_s:g} s OTA snippet, {lk.band_label()}: {lk.expected}. "
            + (f"{a.site}. " if a.site else "") + f"{now.astimezone().isoformat(timespec='seconds')}. "
            f"Overflow events {st['overflow_events']}; peak {stats['peak_fs']:.3f} FS, rms {stats['rms_dbfs']:.1f} dBFS, "
            f"clipped {100 * stats['clipped_frac']:.4f}%"
            + (f"; UHD actual rate {st['actual_rate_hz'] / 1e6:.6f} Msps" if st.get("actual_rate_hz") else "")
            + (f", freq {st['actual_freq_hz'] / 1e6:.6f} MHz" if st.get("actual_freq_hz") else ""))
    g = {
        "core:datatype": "ci16_le", "core:sample_rate": float(lk.rate_hz), "core:version": "1.0.0", "core:num_channels": 1,
        "core:hw": hw, "core:recorder": f"UHD {st['backend']} ({VERSION})", "core:description": desc,
        "argus:band_label": lk.band_label(), "argus:expected_waveforms": lk.expected,
        "argus:manifest_entry": lk.entry_id, "argus:manifest_label": lk.label, "argus:manifest_tier": lk.tier,
        "argus:manifest_file": Path(m["_path"]).name, "argus:look_tag": lk.tag,
        "argus:radio": radio.key, "argus:rx_channel": int(lk.channel), "argus:gain_db": float(gain),
        "argus:overflow_events": int(st["overflow_events"]), "argus:peak_fs": stats["peak_fs"], "argus:rms_dbfs": stats["rms_dbfs"],
        "argus:clipped_frac": stats["clipped_frac"], "argus:requested_rate_hz": float(lk.rate_req_hz),
        "argus:collector": VERSION, "argus:num_samples": int(n_samps),
    }
    if a.author:
        g["core:author"] = a.author
    if a.site:
        g["argus:site"] = a.site
    if a.antenna:
        g["argus:antenna"] = a.antenna
    if lk.mcr_hz:
        g["argus:master_clock_rate_hz"] = float(lk.mcr_hz)
        g["argus:decimation"] = int(lk.decim or 0)
    if radio.info.get("serial"):
        g["argus:radio_serial"] = str(radio.info["serial"])
    meta = {"global": g,
            "captures": [{"core:sample_start": 0, "core:frequency": float(lk.fc_hz), "core:datetime": now.isoformat().replace("+00:00", "Z")}],
            "annotations": []}
    path.write_text(json.dumps(meta, indent=2) + "\n")


def stem_for(lk: Look, radio: Radio, gain: float) -> str:
    return f"{radio.key}_{lk.band_label()}_c{mio.hz_tag(lk.fc_hz)}_r{mio.hz_tag(lk.rate_hz)}_g{gain:g}_ch{lk.channel}"


def print_plan(looks: list[Look], radio_key: str):
    tot_b = 0.0
    print(f"{len(looks)} looks on a {radio_key}:")
    print(f"{'#':>3} {'entry':<22} {'tag':<16} {'fc MHz':>9} {'rate MSps':>10} {'dur s':>6} {'ch':>2}  {'size MB':>8}  label")
    for i, lk in enumerate(looks):
        b = lk.dur_s * lk.rate_hz * 4
        tot_b += b
        print(f"{i:>3} {lk.entry_id:<22} {lk.tag:<16} {lk.fc_hz / 1e6:>9.3f} {lk.rate_hz / 1e6:>10.3f} {lk.dur_s:>6.2f} {lk.channel:>2}  {b / 1e6:>8.0f}  {lk.label}")
    print(f"total ~{tot_b / 1e9:.1f} GB of ci16 IQ")


# --------------------------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=None, help="descriptor library (lut_library_usa.yaml): plan from the merged library, which adds the entries defined only there")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=None, help="output root (one folder per capture)")
    sel = ap.add_argument_group("what to capture")
    sel.add_argument("--entries", help="comma-separated manifest entry ids (default: entries with sweep.default true)")
    sel.add_argument("--tier", help="select all entries of these tiers, e.g. A or A,B")
    sel.add_argument("--all", action="store_true", help="every entry with sweep centres")
    sel.add_argument("--freqs", help="fixed centre frequencies in Hz (comma list, e.g. 98e6,2437e6); with --entries they replace the entries' centres, alone they are captured with manifest labels looked up by frequency")
    sel.add_argument("--rates", help="fixed sample rate(s) in Hz (comma list); replaces the manifest rates for every look")
    sel.add_argument("--dur", type=float, default=None, help="seconds per capture (default: manifest, 1 s)")
    sel.add_argument("--max-gb", type=float, default=1.0, help="cap each capture to this many GB (shortens --dur at high rates)")
    rad = ap.add_argument_group("radio")
    rad.add_argument("--args", default="", help='UHD device args, e.g. "addr=192.168.10.2" (default: auto-discover)')
    rad.add_argument("--radio", default="auto", choices=["auto"] + sorted(mio.RADIOS), help="radio family for rate snapping (default: discover)")
    rad.add_argument("--channel", type=int, default=0)
    rad.add_argument("--channel-below", type=int, default=None, help="RX channel for looks below --split-hz (e.g. a VHF/UHF antenna)")
    rad.add_argument("--channel-above", type=int, default=None, help="RX channel for looks at/above --split-hz (e.g. a 2.4/5 GHz antenna)")
    rad.add_argument("--split-hz", type=float, default=1.5e9)
    rad.add_argument("--ant", default=None, help="antenna port name, e.g. TX/RX or RX2 (default: UHD default)")
    rad.add_argument("--gain", default="auto", help="'auto' (probe per look, default) or a fixed dB value")
    rad.add_argument("--gain-max", type=float, default=None, help="cap for the auto gain (default: radio max)")
    rad.add_argument("--backend", default="auto", choices=["auto", "rx_samples_to_file", "python"], help="capture backend (default: rx_samples_to_file when found, else UHD Python)")
    rad.add_argument("--rx-bin", default=None, help="path to rx_samples_to_file")
    misc = ap.add_argument_group("metadata / behaviour")
    misc.add_argument("--site", default="", help="free text: where / which antenna / height (goes into the metadata)")
    misc.add_argument("--antenna", default="", help="antenna model (metadata)")
    misc.add_argument("--author", default="", help="core:author")
    misc.add_argument("--list", action="store_true", help="print the plan and exit")
    misc.add_argument("--dry-run", action="store_true", help="plan + radio discovery, no capture")
    misc.add_argument("--survey", action="store_true", help="probe the power of every look (0.2 s) but keep no IQ; writes survey.csv")
    misc.add_argument("--no-shm", action="store_true", help="stream straight into --out instead of /dev/shm staging")
    misc.add_argument("--skip-existing", action="store_true", help="skip looks whose capture folder already exists")
    misc.add_argument("--pause", type=float, default=0.3, help="seconds between captures")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    m = mio.load_library(a.manifest, a.library) if a.library else mio.load_manifest(a.manifest)
    if a.freqs and not a.entries and not a.tier and not a.all:
        looks = plan_from_freqs(m, a)
    else:
        looks = plan_from_manifest(m, a)
    if not looks:
        print("nothing to capture (check --entries / --freqs)", file=sys.stderr)
        return 2
    if a.list:
        radio_key = a.radio if a.radio != "auto" else "x410"
        looks = finalize_plan(looks, radio_key, a)
        print_plan(looks, radio_key + (" (assumed; use --radio)" if a.radio == "auto" else ""))
        return 0
    if not a.out:
        print("--out is required (or --list)", file=sys.stderr)
        return 2
    radio = Radio(a)
    print(f"radio: {radio.key} {radio.info or ''}  backend: {radio.backend}" + (f" ({radio.rx_bin})" if radio.rx_bin else ""))
    looks = finalize_plan(looks, radio.key, a)
    print_plan(looks, radio.key)
    if a.dry_run:
        return 0
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stage = None if a.no_shm else Path(tempfile.mkdtemp(prefix="argus_sweep_", dir="/dev/shm")) if Path("/dev/shm").is_dir() else None
    if stage is None:
        stage = Path(tempfile.mkdtemp(prefix="argus_sweep_", dir=str(out)))
    rows = []
    log = open(out / f"sweep_{ts}.log", "a")

    def say(s: str):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say(f"# {VERSION}  {dt.datetime.now().isoformat(timespec='seconds')}  manifest {Path(m['_path']).name}  radio {radio.key} {radio.info}")
    try:
        for i, lk in enumerate(looks):
            gain = float(lk.gain0_db) if a.gain == "auto" else float(a.gain)
            say(f"==> [{i + 1}/{len(looks)}] {lk.entry_id} {lk.tag} @ {lk.fc_hz / 1e6:g} MHz, {lk.rate_hz / 1e6:g} MSps, ch{lk.channel}")
            probe = {}
            if a.gain == "auto":
                gain, probe = probe_gain(radio, lk, gain, stage, a)
                if probe.get("history"):
                    h = probe["history"][-1]
                    say(f"    probe: gain {gain:g} dB -> peak {h['peak_fs']:.3f} FS, rms {h['rms_dbfs']:.1f} dBFS, spectral span {h.get('psd_span_db', 0):.1f} dB")
                    if h.get("psd_span_db", 99) < 4.0:
                        say("    WARNING: flat spectrum: is an antenna on this port? (UHD defaults to RX2 on N3xx/X3xx; pass --ant TX/RX)")
                if probe.get("probe_error"):
                    say(f"    !! probe failed: {probe['probe_error']}")
                    rows.append({"index": i, "entry": lk.entry_id, "tag": lk.tag, "fc_hz": lk.fc_hz, "rate_hz": lk.rate_hz, "status": "probe_failed"})
                    continue
            if a.survey:
                h = (probe.get("history") or [{}])[-1]
                rows.append({"index": i, "entry": lk.entry_id, "tag": lk.tag, "fc_hz": lk.fc_hz, "rate_hz": lk.rate_hz, "channel": lk.channel,
                             "gain_db": gain, "peak_fs": h.get("peak_fs"), "rms_dbfs": h.get("rms_dbfs"), "status": "surveyed"})
                continue
            stem = stem_for(lk, radio, gain)
            dest_dir = out / stem
            if a.skip_existing and dest_dir.exists():
                say(f"    skip (exists) {dest_dir}")
                continue
            need = int(lk.dur_s * lk.rate_hz * 4 * 1.05)
            use_stage = stage if (shutil.disk_usage(stage).free > need) else None
            if use_stage is None and not a.no_shm:
                say("    note: /dev/shm too small for this capture, streaming straight to the output disk")
            tmp_dir = use_stage or (out / f"tmp_{os.getpid()}")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            data_tmp = tmp_dir / f"{stem}.sigmf-data"
            st = radio.capture(lk, gain, lk.dur_s, data_tmp)
            if not st["ok"]:
                rows.append({"index": i, "entry": lk.entry_id, "tag": lk.tag, "fc_hz": lk.fc_hz, "rate_hz": lk.rate_hz, "status": "failed", "error": st.get("error", "")})
                data_tmp.unlink(missing_ok=True)
                continue
            stats = iq_stats(data_tmp)
            n_samps = data_tmp.stat().st_size // 4
            dest_dir.mkdir(parents=True, exist_ok=True)
            write_meta(dest_dir / f"{stem}.sigmf-meta", lk, gain, lk.dur_s, st, stats, radio, a, m, n_samps)
            shutil.move(str(data_tmp), str(dest_dir / f"{stem}.sigmf-data"))
            say(f"    {n_samps} samples, overflow events {st['overflow_events']}, peak {stats['peak_fs']:.3f} FS, rms {stats['rms_dbfs']:.1f} dBFS -> {dest_dir}")
            rows.append({"index": i, "entry": lk.entry_id, "label": lk.label, "tag": lk.tag, "fc_hz": lk.fc_hz, "rate_hz": lk.rate_hz,
                         "channel": lk.channel, "gain_db": gain, "dur_s": round(lk.dur_s, 4), "samples": n_samps,
                         "overflow_events": st["overflow_events"], "peak_fs": stats["peak_fs"], "rms_dbfs": stats["rms_dbfs"],
                         "clipped_frac": stats["clipped_frac"], "folder": stem, "status": "ok"})
            if use_stage is None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            time.sleep(a.pause)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        name = "survey" if a.survey else "sweep"
        fields = sorted({k for r in rows for k in r})
        if rows:
            with open(out / f"{name}_{ts}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            (out / f"{name}_{ts}.json").write_text(json.dumps({"collector": VERSION, "manifest": Path(m["_path"]).name, "radio": radio.key,
                                                             "radio_info": radio.info, "site": a.site, "antenna": a.antenna,
                                                             "plan": [asdict(lk) for lk in looks], "captures": rows}, indent=1))
        ok = sum(1 for r in rows if r.get("status") in ("ok", "surveyed"))
        say(f"# done: {ok}/{len(looks)} looks -> {out}/{name}_{ts}.csv")
        log.close()
    return 0 if rows and all(r.get("status") in ("ok", "surveyed") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
