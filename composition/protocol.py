"""Shared TX/RX protocol: time layout, metadata serialization, shared modem.

Both the composer and the receiver import this so they agree on:
- where the metadata burst and data window sit relative to the ZC start, and
- how the per-block metadata descriptor is serialized into modem bytes.
"""

from __future__ import annotations
import json
import zlib
import numpy as np
from scipy.signal import resample_poly

from metadata_modem import MetaModem
import geometry as geo

# Time-layout constants (samples @ 245.76 MSps), known to TX and RX.
GAP_ZC_META = 2048    # small fixed gap ZC -> metadata (RX-known; not a guard time)
UPSAMPLE = 8          # metadata native 30.72 MHz -> 245.76 MHz

# Metadata modem profiles. "fast" = short burst, light FEC (high-SNR / quick).
# "robust" = long burst with heavy repetition for very-low-SNR links. The RX
# tries the profiles in PROFILE_TRY_ORDER and the CRC selects the right one, so a
# capture is self-describing (no annotation needed to know which was used).
PROFILES = {
    "fast":   dict(rep=3,  n_data_syms=20),                       # no channel smoothing
    "robust": dict(rep=24, n_data_syms=160, smooth_width=21),     # heavy smoothing for low SNR
}
DEFAULT_PROFILE = "fast"
PROFILE_TRY_ORDER = ("fast", "robust")
META_MODEMS = {name: MetaModem(**kw) for name, kw in PROFILES.items()}
META_MODEM = META_MODEMS[DEFAULT_PROFILE]   # back-compat alias


def meta_len_245(profile=DEFAULT_PROFILE):
    return META_MODEMS[profile].burst_len_samples() * UPSAMPLE


def meta_start(zc_len):
    """Metadata-burst start sample relative to ZC start (RX-known, profile-free)."""
    return zc_len + GAP_ZC_META


def data_start(zc_len, guard1_samples, profile=DEFAULT_PROFILE):
    """Waveform start sample relative to ZC start. guard1 is the (configurable)
    guard time between the metadata burst and the waveform."""
    return meta_start(zc_len) + meta_len_245(profile) + int(guard1_samples)


def layout(zc_len, guard1_samples=2048, profile=DEFAULT_PROFILE):
    """Back-compat: (meta_start, data_start)."""
    return meta_start(zc_len), data_start(zc_len, guard1_samples, profile)


def meta_native_to_245(burst_native):
    return resample_poly(burst_native, UPSAMPLE, 1).astype(np.complex64)


def meta_245_to_native(burst_245):
    return resample_poly(burst_245, 1, UPSAMPLE).astype(np.complex64)


def serialize(descriptor: dict) -> bytes:
    raw = json.dumps(descriptor, separators=(",", ":")).encode("utf-8")
    return zlib.compress(raw, level=9)


def deserialize(payload: bytes):
    return json.loads(zlib.decompress(payload).decode("utf-8"))


def encode_metadata(descriptor: dict, profile=DEFAULT_PROFILE) -> np.ndarray:
    """Descriptor dict -> 245.76 MSps baseband metadata burst (raises if too big)."""
    payload = serialize(descriptor)
    burst_native = META_MODEMS[profile].encode(payload)
    return meta_native_to_245(burst_native)


def decode_metadata(burst_245: np.ndarray, profile=DEFAULT_PROFILE):
    """245.76 MSps baseband metadata burst -> descriptor dict, or None on failure."""
    native = meta_245_to_native(burst_245)
    payload = META_MODEMS[profile].decode(native)
    if payload is None:
        return None
    try:
        return deserialize(payload)
    except Exception:
        return None
