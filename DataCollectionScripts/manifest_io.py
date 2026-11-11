"""Shared helpers for the ARGUS data-collection scripts.

* load_manifest(): read common_ota_waveforms_manifest_usa.yaml. PyYAML implements YAML 1.1, where
  "98.0e6" is a *string* (the exponent needs a sign), so every numeric-looking string is coerced to a
  number after loading. Users can therefore write 98e6, 98.0e6 or 98.0e+6 interchangeably.
* app_fft_size() / frame geometry: the Holohub detector frames a recording as 512 rows x F samples with
  F derived from the sample rate (mirrors applications/usrp_wideband_signal_detection/fft_runtime_config.hpp).
  The fine-tuned DINO mask is 512 x 1024 per frame at every rate; the coherent-power mask is 512 x F.
* snap_rate(): the sample rates a USRP can deliver (master clock / integer decimation) so a requested
  rate is replaced by the closest achievable one before tuning.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from pathlib import Path

_NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

# --------------------------------------------------------------------------------------------- manifest
def _coerce(x):
    if isinstance(x, dict):
        return {k: _coerce(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_coerce(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if _NUM_RE.match(s):
            try:
                return float(s) if any(c in s for c in ".eE") else int(s)
            except ValueError:
                return x
    if isinstance(x, (_dt.date, _dt.datetime)):
        return x.isoformat()
    return x


REQUIRED_ENTRY_KEYS = ("id", "label", "tier", "freq_ranges_hz", "temporal")


def load_manifest(path: str | Path) -> dict:
    import yaml  # PyYAML
    with open(path) as f:
        raw = yaml.safe_load(f)
    m = _coerce(raw)
    if "entries" not in m or not isinstance(m["entries"], list):
        raise ValueError(f"{path}: no 'entries' list")
    seen = set()
    for e in m["entries"]:
        for k in REQUIRED_ENTRY_KEYS:
            if k not in e:
                raise ValueError(f"{path}: entry {e.get('id', '?')} lacks '{k}'")
        if e["id"] in seen:
            raise ValueError(f"{path}: duplicate entry id {e['id']}")
        seen.add(e["id"])
        e.setdefault("sweep", {"default": False})
        e.setdefault("requires", [])
        e.setdefault("band_exclusive", False)
        e.setdefault("iq_signature", "any")
        if "bandwidth_hz" in e:
            bw = e["bandwidth_hz"]
            bw.setdefault("nominal", 0.5 * (bw["min"] + bw["max"]))
    m["_path"] = str(Path(path).resolve())
    return m


def entry_by_id(manifest: dict) -> dict:
    return {e["id"]: e for e in manifest["entries"]}


def entries_covering(manifest: dict, fc_hz: float) -> list[dict]:
    """Entries whose freq_ranges contain fc (absolute entries only)."""
    out = []
    for e in manifest["entries"]:
        if e.get("relative_to_capture_center"):
            continue
        if any(lo <= fc_hz <= hi for lo, hi in e["freq_ranges_hz"]):
            out.append(e)
    return out


# --------------------------------------------------------------------------------------------- geometry
ROWS_WIDE = 512
NFFT_MODEL = 1024
_REF_SPAN_HZ = 500.0e6
_REF_FFT = 20480
_PACKET = 1024


def _cpp_round(x: float) -> float:
    return math.copysign(math.floor(abs(x) + 0.5), x)


def app_fft_size(rate_hz: float) -> int:
    """Wide-FFT size the Holohub app uses at this rate (C++ half-away-from-zero rounding)."""
    span_ratio = rate_hz / _REF_SPAN_HZ
    snapped = 2.0 ** _cpp_round(math.log2(span_ratio)) if span_ratio > 0 else 1.0
    packets = max(1, int(_cpp_round(_REF_FFT * snapped / _PACKET)))
    return max(_PACKET, packets * _PACKET)


def frame_geometry(rate_hz: float, detector_cols: int | None = None) -> dict:
    """Frame / mask geometry at a rate. detector_cols: 1024 for the fine-tuned DINO mask (default),
    app_fft_size(rate) for the coherent-power / zero-shot masks."""
    F = app_fft_size(rate_hz)
    cols = detector_cols or NFFT_MODEL
    return {"rate_hz": float(rate_hz), "fft_size": F, "rows": ROWS_WIDE, "cols": cols,
            "frame_samples": ROWS_WIDE * F, "frame_s": ROWS_WIDE * F / rate_hz,
            "row_s": F / rate_hz, "bin_hz": rate_hz / cols}


# --------------------------------------------------------------------------------------------- radios
# master clock rates (Hz) per radio family; None = variable clock (any rate within limits)
RADIOS = {
    "x410": {"mcrs": [491.52e6, 500.0e6], "max_rate": 500.0e6, "tune_min": 1.0e6, "tune_max": 7.2e9, "max_gain": 60.0, "max_decim": 2048, "even_only": True},
    "x440": {"mcrs": [500.0e6, 1000.0e6, 1600.0e6], "max_rate": 1600.0e6, "tune_min": 30.0e6, "tune_max": 4.0e9, "max_gain": 0.0, "max_decim": 2048, "even_only": True},
    "n320": {"mcrs": [200.0e6, 245.76e6, 250.0e6], "max_rate": 250.0e6, "tune_min": 3.0e6, "tune_max": 6.0e9, "max_gain": 60.0, "max_decim": 1024},
    "n321": {"mcrs": [200.0e6, 245.76e6, 250.0e6], "max_rate": 250.0e6, "tune_min": 3.0e6, "tune_max": 6.0e9, "max_gain": 60.0, "max_decim": 1024},
    "n310": {"mcrs": [122.88e6, 125.0e6, 153.6e6], "max_rate": 153.6e6, "tune_min": 10.0e6, "tune_max": 6.0e9, "max_gain": 75.0, "max_decim": 1024},
    "n300": {"mcrs": [122.88e6, 125.0e6, 153.6e6], "max_rate": 153.6e6, "tune_min": 10.0e6, "tune_max": 6.0e9, "max_gain": 75.0, "max_decim": 1024},
    "x310": {"mcrs": [200.0e6, 184.32e6], "max_rate": 200.0e6, "tune_min": 10.0e6, "tune_max": 6.0e9, "max_gain": 37.5, "max_decim": 1024},
    "b200": {"mcrs": None, "max_rate": 56.0e6, "tune_min": 70.0e6, "tune_max": 6.0e9, "max_gain": 76.0, "max_decim": 1},
    "generic": {"mcrs": None, "max_rate": 1.0e12, "tune_min": 0.0, "tune_max": 1.0e12, "max_gain": 60.0, "max_decim": 1},
}


def radio_key(product_or_type: str) -> str:
    s = (product_or_type or "").lower()
    for k in ("x410", "x440", "n320", "n321", "n310", "n300", "x310", "b200", "b210", "b205"):
        if k in s:
            return "b200" if k.startswith("b2") else k
    if "x4xx" in s:
        return "x410"
    if "n3xx" in s:
        return "n320"
    if "x300" in s:
        return "x310"
    return "generic"


def snap_rate(rate_hz: float, radio: str, prefer_even: bool = True) -> tuple[float, float | None, int | None]:
    """Closest achievable (rate, master_clock_rate, decimation) for the radio. Variable-clock radios
    return (rate, None, None). Ties prefer an even decimation (UHD deprecates odd CIC factors on X4xx)."""
    spec = RADIOS.get(radio, RADIOS["generic"])
    if spec["mcrs"] is None:
        return float(min(rate_hz, spec["max_rate"])), None, None
    even_only = spec.get("even_only", False)   # X4xx: odd CIC decimations are deprecated by UHD (passband ripple)
    best = None
    for mcr in spec["mcrs"]:
        for n in range(1, spec["max_decim"] + 1):
            r = mcr / n
            if r > spec["max_rate"] + 1e-3:
                continue
            if even_only and n != 1 and n % 2:
                continue
            err = abs(r - rate_hz) / rate_hz
            penalty = 0.0 if (n == 1 or n % 2 == 0 or not prefer_even) else 0.002
            key = (err + penalty, n)
            if best is None or key < best[0]:
                best = (key, r, mcr, n)
            if r < rate_hz * 0.5:
                break
    _, r, mcr, n = best
    return float(r), float(mcr), int(n)


def achievable_rates(radio: str, lo: float = 5.0e6, hi: float | None = None) -> list[float]:
    spec = RADIOS.get(radio, RADIOS["generic"])
    if spec["mcrs"] is None:
        return []
    hi = hi or spec["max_rate"]
    out = set()
    for mcr in spec["mcrs"]:
        for n in range(1, spec["max_decim"] + 1):
            r = mcr / n
            if lo <= r <= hi:
                out.add(round(r, 3))
    return sorted(out)


def hz_tag(hz: float) -> str:
    """98e6 -> '98M', 24.576e6 -> '24p576M' (same convention as the N320 stress captures)."""
    return f"{hz / 1e6:g}".replace(".", "p") + "M"


# --------------------------------------------------------------------------------------------- LUT library
DEFAULT_LIBRARY = Path(__file__).resolve().parent / "lut_library_usa.yaml"
LIBRARY_ONLY_KEYS = ("new", "descriptor", "variants", "synth", "source", "requires_descriptors")
DESCRIPTOR_WEIGHTS = {"symbol_rate": 3.0, "ofdm": 3.0, "chirp": 3.0, "pri": 2.0, "period": 2.0, "gap": 2.0, "pulse": 2.0,
                      "pulse_pair": 3.0, "modulation": 2.0, "deviation": 1.0, "fm_tones": 3.0, "am_tones": 2.0,
                      "carrier_line": 1.0, "hop": 2.0, "hd_sidebands": 3.0, "duty": 1.0, "burst": 1.5}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_library(manifest_path: str | Path | None = None, library_path: str | Path | None = None) -> dict:
    """Manifest + descriptor layer -> one library. Library entries with an existing id add `descriptor`,
    `variants`, `synth`, `source` (and may override manifest fields); `new: true` entries are appended whole."""
    import yaml
    lib_path = Path(library_path) if library_path else DEFAULT_LIBRARY
    with open(lib_path) as f:
        lib = _coerce(yaml.safe_load(f))
    mpath = Path(manifest_path) if manifest_path else lib_path.parent / lib.get("extends", "common_ota_waveforms_manifest_usa.yaml")
    m = load_manifest(mpath)
    by_id = {e["id"]: e for e in m["entries"]}
    n_ext = n_new = 0
    for le in lib.get("entries", []):
        if le.get("new"):
            if le["id"] in by_id:
                raise ValueError(f"{lib_path}: entry {le['id']} is marked new but exists in the manifest")
            e = dict(le)
            for k in REQUIRED_ENTRY_KEYS:
                if k not in e:
                    raise ValueError(f"{lib_path}: new entry {e['id']} lacks '{k}'")
            e.setdefault("sweep", {"default": False})
            e.setdefault("requires", [])
            e.setdefault("band_exclusive", False)
            e.setdefault("iq_signature", "any")
            if "bandwidth_hz" in e:
                e["bandwidth_hz"].setdefault("nominal", 0.5 * (e["bandwidth_hz"]["min"] + e["bandwidth_hz"]["max"]))
            m["entries"].append(e)
            by_id[e["id"]] = e
            n_new += 1
        else:
            if le["id"] not in by_id:
                raise ValueError(f"{lib_path}: entry {le['id']} is not in the manifest (add new: true to define it here)")
            e = by_id[le["id"]]
            for k, v in le.items():
                if k == "id":
                    continue
                if k in ("descriptor",) and isinstance(e.get(k), dict):
                    e[k] = _deep_merge(e[k], v)
                else:
                    e[k] = v
            n_ext += 1
    for e in m["entries"]:
        e.setdefault("descriptor", {})
        e.setdefault("variants", [])
        e.setdefault("source", "")
        # a descriptor-level burst range defaults to the manifest burst_s
        d = e["descriptor"]
        if "burst_s" in e and e.get("temporal") in ("bursty", "tdd") and "burst_s" not in d.get("timing", {}):
            d.setdefault("timing", {})["burst_s"] = e["burst_s"]
        if e.get("requires_descriptors") is None:
            e["requires_descriptors"] = 0 if e.get("band_exclusive") else 1
    m["_library_path"] = str(lib_path.resolve())
    m["_library_counts"] = {"extended": n_ext, "new": n_new}
    m["name"] = lib.get("name", m.get("name"))
    return m


def descriptor_fields(e: dict) -> list[str]:
    """Which descriptor tests an entry (or variant descriptor) defines."""
    d = e.get("descriptor", {}) if "descriptor" in e else e
    out = []
    if d.get("modulation"):
        out.append("modulation")
    if d.get("symbol_rate_hz"):
        out.append("symbol_rate")
    if d.get("ofdm", {}).get("scs_hz"):
        out.append("ofdm")
    if d.get("chirp"):
        out.append("chirp")
    if d.get("deviation_hz"):
        out.append("deviation")
    t = d.get("timing", {})
    for k, name in (("pulse_s", "pulse"), ("burst_s", "burst"), ("pri_s", "pri"), ("gap_s", "gap"), ("period_s", "period"),
                    ("pulse_pair_s", "pulse_pair"), ("duty", "duty")):
        if t.get(k):
            out.append(name)
    s = d.get("spectral", {})
    if s.get("fm_tones_hz"):
        out.append("fm_tones")
    if s.get("am_tones_hz"):
        out.append("am_tones")
    if s.get("hd_sidebands"):
        out.append("hd_sidebands")
    if "carrier_line" in s:
        out.append("carrier_line")
    if d.get("hop"):
        out.append("hop")
    return out
