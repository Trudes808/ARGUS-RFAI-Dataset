#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_QUANTITY_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*$"
)

_FREQ_MULTIPLIERS = {
    "": 1.0,
    "hz": 1.0,
    "k": 1e3,
    "khz": 1e3,
    "m": 1e6,
    "mhz": 1e6,
    "g": 1e9,
    "ghz": 1e9,
}

_SIGMF_SAMPLE_DTYPES = {
    "cf32_le": 8,
    "cf32_be": 8,
    "cf64_le": 16,
    "cf64_be": 16,
    "ci16_le": 4,
    "ci16_be": 4,
    "ci8": 2,
    "cu8": 2,
}

_CLASS_TEXT_FIELDS = (
    "compose:class",
    "waveform:class",
    "sigmf:class",
    "rfml:class",
    "core:class",
    "core:label",
    "compose:waveform_name",
    "eye_chart:waveform_name",
    "compose:source_path",
    "rx_sync:tx_plan",
)

_WAVEFORM_CLASS_PATTERNS = (
    ("5G_Downlink", ("5G_Downlink", "5G_downlink", "5G_DL", "NR_Downlink", "NR_DL", "fiveg")),
    ("5G_Uplink", ("5G_Uplink", "5G_uplink", "5G_UL", "NR_Uplink", "NR_UL")),
    ("LTE_Downlink", ("LTE_Downlink", "LTE_downlink", "LTE_DL")),
    ("LTE_Uplink", ("LTE_Uplink", "LTE_uplink", "LTE_UL")),
    ("802_11ax", ("802_11ax", "802.11ax", "WiFi_6", "Wi-Fi_6", "WLAN_11ax")),
    ("802_11ac", ("802_11ac", "802.11ac", "WiFi_5", "Wi-Fi_5", "WLAN_11ac")),
    ("802_11n", ("802_11n", "802.11n", "WiFi_4", "Wi-Fi_4", "WLAN_11n")),
    ("Bluetooth", ("Bluetooth", "BLE")),
    ("Broadband_FM", ("Broadband_FM", "wideband_fm", "WBFM")),
    ("Broadcast_FM", ("Broadcast_FM", "broadcastfm")),
    ("Narrowband_FM", ("Narrowband_FM", "NBFM")),
    ("16QAM", ("16QAM", "QAM16")),
    ("64QAM", ("64QAM", "QAM64")),
    ("256QAM", ("256QAM", "QAM256")),
    ("QPSK", ("QPSK",)),
    ("BPSK", ("BPSK",)),
    ("OFDM", ("OFDM",)),
)

_GENERATED_CLASS_RE = re.compile(
    r"(?:^|_)(?:payload|sync_pn)_evt\d+_slot\d+_(.+?)(?:_snr_|_payloadSNR_)",
    re.IGNORECASE,
)

_VIRIDIS_POSITIONS = np.array(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    dtype=np.float32,
)
_VIRIDIS_COLORS = np.array(
    [
        (68, 1, 84),
        (72, 35, 116),
        (64, 67, 135),
        (52, 94, 141),
        (41, 120, 142),
        (32, 144, 140),
        (34, 167, 132),
        (68, 190, 112),
        (121, 209, 81),
        (189, 223, 38),
        (253, 231, 37),
    ],
    dtype=np.float32,
)
_COLORMAP_LUTS: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class SpectrogramLayout:
    sample_rate_hz: float
    total_samples: int
    fft_size: int
    hop_size: int
    total_frames: int
    freq_min_hz: float
    freq_max_hz: float
    freq_bins: int
    time_box_mode: str


@dataclass(frozen=True)
class FullBox:
    annotation_index: int
    class_name: str
    class_id: int
    x: float
    y: float
    width: float
    height: float
    sample_start: int
    sample_stop: int
    freq_lower_hz: float
    freq_upper_hz: float
    label: str


@dataclass(frozen=True)
class Tile:
    id: int
    file_name: str
    x: int
    y: int
    width: int
    height: int


class SigMFReader:
    _COMPLEX_DTYPES = {
        "cf32_le": np.dtype("<c8"),
        "cf32_be": np.dtype(">c8"),
        "cf64_le": np.dtype("<c16"),
        "cf64_be": np.dtype(">c16"),
    }
    _INTERLEAVED_DTYPES = {
        "ci16_le": np.dtype("<i2"),
        "ci16_be": np.dtype(">i2"),
        "ci8": np.dtype("i1"),
        "cu8": np.dtype("u1"),
    }

    def __init__(self, data_path: Path, meta: dict[str, Any]):
        self.data_path = Path(data_path)
        global_meta = meta.get("global", {})
        self.datatype = str(global_meta.get("core:datatype", "")).strip().lower()
        self.header_bytes = int(global_meta.get("core:header_bytes", 0))
        self.trailing_bytes = int(global_meta.get("core:trailing_bytes", 0))
        payload_bytes = self.data_path.stat().st_size - self.header_bytes - self.trailing_bytes
        if payload_bytes < 0:
            raise ValueError("SigMF header/trailing bytes exceed data file size")

        if self.datatype in self._COMPLEX_DTYPES:
            self.mode = "complex"
            self.dtype = self._COMPLEX_DTYPES[self.datatype]
            if payload_bytes % self.dtype.itemsize:
                raise ValueError(f"{self.datatype} payload is not aligned to complex samples")
            self.num_samples = payload_bytes // self.dtype.itemsize
            self._mm = np.memmap(
                self.data_path,
                dtype=self.dtype,
                mode="r",
                offset=self.header_bytes,
                shape=(self.num_samples,),
            )
        elif self.datatype in self._INTERLEAVED_DTYPES:
            self.mode = "interleaved"
            self.dtype = self._INTERLEAVED_DTYPES[self.datatype]
            bytes_per_sample = 2 * self.dtype.itemsize
            if payload_bytes % bytes_per_sample:
                raise ValueError(f"{self.datatype} payload is not aligned to I/Q sample pairs")
            self.num_samples = payload_bytes // bytes_per_sample
            self._mm = np.memmap(
                self.data_path,
                dtype=self.dtype,
                mode="r",
                offset=self.header_bytes,
                shape=(self.num_samples, 2),
            )
        else:
            supported = sorted([*self._COMPLEX_DTYPES, *self._INTERLEAVED_DTYPES])
            raise ValueError(f"Unsupported SigMF datatype {self.datatype!r}; supported: {supported}")

    def read(self, start: int, stop: int) -> np.ndarray:
        start = max(0, int(start))
        stop = min(int(stop), int(self.num_samples))
        if stop <= start:
            return np.empty((0,), dtype=np.complex64)
        if self.mode == "complex":
            return np.asarray(self._mm[start:stop], dtype=np.complex64)
        raw = np.asarray(self._mm[start:stop])
        if self.datatype == "cu8":
            i = raw[:, 0].astype(np.float32) - 127.5
            q = raw[:, 1].astype(np.float32) - 127.5
        else:
            i = raw[:, 0].astype(np.float32, copy=False)
            q = raw[:, 1].astype(np.float32, copy=False)
        return i + 1j * q


def parse_frequency(value: Any, field_name: str) -> float:
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        match = _QUANTITY_RE.match(value.replace("_", "").strip())
        if not match:
            raise ValueError(f"Could not parse {field_name}={value!r}")
        number, suffix = match.groups()
        suffix = suffix.lower()
        if suffix not in _FREQ_MULTIPLIERS:
            raise ValueError(f"Unsupported unit suffix for {field_name}: {suffix!r}")
        out = float(number) * _FREQ_MULTIPLIERS[suffix]
    else:
        raise TypeError(f"{field_name} must be numeric or a frequency string")
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    return out


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def infer_sigmf_data_path(meta_path: Path) -> Path:
    if meta_path.name.endswith(".sigmf-meta"):
        return meta_path.with_name(meta_path.name[: -len(".sigmf-meta")] + ".sigmf-data")
    return meta_path.with_suffix(".sigmf-data")


def sigmf_sample_count(data_path: Path, meta: dict[str, Any]) -> int:
    global_meta = meta.get("global", {})
    datatype = str(global_meta.get("core:datatype", "")).strip().lower()
    if datatype not in _SIGMF_SAMPLE_DTYPES:
        raise ValueError(f"Unsupported or missing SigMF datatype {datatype!r}")
    header_bytes = int(global_meta.get("core:header_bytes", 0))
    trailing_bytes = int(global_meta.get("core:trailing_bytes", 0))
    payload_bytes = data_path.stat().st_size - header_bytes - trailing_bytes
    if payload_bytes < 0:
        raise ValueError("SigMF header/trailing bytes exceed data file size")
    bytes_per_sample = _SIGMF_SAMPLE_DTYPES[datatype]
    if payload_bytes % bytes_per_sample:
        raise ValueError("SigMF data size is not aligned to the declared datatype")
    return payload_bytes // bytes_per_sample


def annotation_stop(annotation: dict[str, Any]) -> int:
    return int(annotation.get("core:sample_start", 0)) + int(annotation.get("core:sample_count", 0))


def infer_total_samples(meta: dict[str, Any], data_path: Path | None) -> int:
    if data_path is not None and data_path.exists():
        return sigmf_sample_count(data_path, meta)
    global_meta = meta.get("global", {})
    for key in ("compose:total_samples", "core:sample_count"):
        if key in global_meta:
            return int(global_meta[key])
    annotations = meta.get("annotations", [])
    if annotations:
        return max(annotation_stop(annotation) for annotation in annotations)
    raise ValueError("Could not infer total samples; pass the matching .sigmf-data file")


def fft_freq_extent(sample_rate_hz: float, fft_size: int, fmin_hz: float, fmax_hz: float) -> tuple[float, float, int]:
    if fmax_hz <= fmin_hz:
        raise ValueError("Frequency max must be greater than frequency min")
    fmin_hz = max(-0.5 * sample_rate_hz, float(fmin_hz))
    fmax_hz = min(0.5 * sample_rate_hz, float(fmax_hz))
    if fmax_hz <= fmin_hz:
        raise ValueError("Frequency range does not overlap the sampled Nyquist span")
    df = sample_rate_hz / fft_size
    first = math.ceil((fmin_hz + 0.5 * sample_rate_hz) / df)
    last = math.floor((fmax_hz + 0.5 * sample_rate_hz) / df)
    first = max(0, min(fft_size - 1, first))
    last = max(0, min(fft_size - 1, last))
    if last < first:
        raise ValueError("Requested frequency range does not include any FFT bin centers")
    return float(fmin_hz), float(fmax_hz), int(last - first + 1)


def build_layout(
    *,
    meta: dict[str, Any],
    data_path: Path | None,
    fft_size: int,
    hop_size: int,
    freq_min_hz: float | None,
    freq_max_hz: float | None,
    time_box_mode: str,
) -> SpectrogramLayout:
    sample_rate_hz = float(meta.get("global", {}).get("core:sample_rate"))
    total_samples = infer_total_samples(meta, data_path)
    total_frames = 1 if total_samples < fft_size else 1 + (total_samples - fft_size) // hop_size
    requested_min = -0.5 * sample_rate_hz if freq_min_hz is None else float(freq_min_hz)
    requested_max = 0.5 * sample_rate_hz if freq_max_hz is None else float(freq_max_hz)
    axis_min, axis_max, freq_bins = fft_freq_extent(sample_rate_hz, fft_size, requested_min, requested_max)
    return SpectrogramLayout(
        sample_rate_hz=sample_rate_hz,
        total_samples=int(total_samples),
        fft_size=int(fft_size),
        hop_size=int(hop_size),
        total_frames=int(total_frames),
        freq_min_hz=float(axis_min),
        freq_max_hz=float(axis_max),
        freq_bins=int(freq_bins),
        time_box_mode=time_box_mode,
    )


def is_pn_annotation(annotation: dict[str, Any]) -> bool:
    text = " ".join(
        str(annotation.get(key, ""))
        for key in ("core:label", "compose:waveform_name", "compose:source_path", "rx_sync:tx_plan")
    ).lower()
    return "sync_pn" in text or "pn9_bpsk_sync" in text or "pn_preamble" in text


def sanitize_class_label(value: Any, fallback: str = "signal") -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_.+-]+", "_", text).strip("_")
    return text or fallback


def normalized_class_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def contains_class_pattern(text: str, pattern: str) -> bool:
    normalized_pattern = normalized_class_text(pattern)
    if not normalized_pattern:
        return False
    return re.search(rf"(?:^|_){re.escape(normalized_pattern)}(?:_|$)", text) is not None


def infer_waveform_class(annotation: dict[str, Any], fallback: str) -> str:
    text_parts = [str(annotation.get(key, "")) for key in _CLASS_TEXT_FIELDS if annotation.get(key) not in (None, "")]
    normalized_text = normalized_class_text(" ".join(text_parts))

    for class_name, patterns in _WAVEFORM_CLASS_PATTERNS:
        if any(contains_class_pattern(normalized_text, pattern) for pattern in patterns):
            return class_name

    for value in text_parts:
        match = _GENERATED_CLASS_RE.search(normalized_class_text(value))
        if match:
            return sanitize_class_label(match.group(1), fallback=fallback)

    for key in ("compose:class", "waveform:class", "sigmf:class", "rfml:class", "core:class"):
        value = annotation.get(key)
        if value not in (None, ""):
            return sanitize_class_label(value, fallback=fallback)

    return sanitize_class_label(fallback, fallback="signal")


def class_name_for_annotation(
    annotation: dict[str, Any],
    *,
    class_mode: str,
    class_name: str,
    class_field: str,
) -> str:
    if class_mode == "single":
        return class_name
    if class_mode == "waveform-class":
        return infer_waveform_class(annotation, fallback=class_name)
    value = annotation.get(class_field)
    if value is None or value == "":
        return class_name
    return str(value)


def annotation_label(annotation: dict[str, Any], index: int) -> str:
    return str(
        annotation.get("core:label")
        or annotation.get("compose:waveform_name")
        or annotation.get("eye_chart:waveform_name")
        or f"annotation_{index}"
    )


def annotation_frequency_bounds(annotation: dict[str, Any]) -> tuple[float, float]:
    lo = float(annotation.get("core:freq_lower_edge", 0.0))
    hi = float(annotation.get("core:freq_upper_edge", 0.0))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def clipped_interval(lo: float, hi: float, lower: float, upper: float) -> tuple[float, float] | None:
    out_lo = max(lower, lo)
    out_hi = min(upper, hi)
    if out_hi <= out_lo:
        return None
    return out_lo, out_hi


def expand_interval_to_min(lo: float, hi: float, minimum: float, lower: float, upper: float) -> tuple[float, float]:
    if hi - lo >= minimum:
        return lo, hi
    center = 0.5 * (lo + hi)
    lo = center - 0.5 * minimum
    hi = center + 0.5 * minimum
    if lo < lower:
        hi += lower - lo
        lo = lower
    if hi > upper:
        lo -= hi - upper
        hi = upper
    lo = max(lower, lo)
    hi = min(upper, hi)
    return lo, hi


def sample_interval_to_x(start: int, stop: int, layout: SpectrogramLayout) -> tuple[float, float] | None:
    if stop <= start:
        return None
    hop = layout.hop_size
    nfft = layout.fft_size
    if layout.time_box_mode == "sample":
        x0 = start / hop
        x1 = stop / hop
    elif layout.time_box_mode == "frame-center":
        first = math.ceil((start - 0.5 * nfft) / hop)
        last = math.ceil((stop - 0.5 * nfft) / hop) - 1
        x0 = first
        x1 = last + 1
    else:
        first = math.floor((start - nfft + 1) / hop)
        last = math.ceil(stop / hop) - 1
        x0 = first
        x1 = last + 1
    return clipped_interval(x0, x1, 0.0, float(layout.total_frames))


def freq_interval_to_y(freq_lo: float, freq_hi: float, layout: SpectrogramLayout) -> tuple[float, float] | None:
    clipped = clipped_interval(freq_lo, freq_hi, layout.freq_min_hz, layout.freq_max_hz)
    if clipped is None:
        return None
    lo, hi = clipped
    span = layout.freq_max_hz - layout.freq_min_hz
    y_from_bottom_lo = (lo - layout.freq_min_hz) / span * layout.freq_bins
    y_from_bottom_hi = (hi - layout.freq_min_hz) / span * layout.freq_bins
    y_top = layout.freq_bins - y_from_bottom_hi
    y_bottom = layout.freq_bins - y_from_bottom_lo
    return clipped_interval(y_top, y_bottom, 0.0, float(layout.freq_bins))


def annotations_to_full_boxes(
    annotations: list[dict[str, Any]],
    layout: SpectrogramLayout,
    *,
    class_mode: str,
    class_name: str,
    class_field: str,
    exclude_pn: bool,
    min_box_width_px: float,
    min_box_height_px: float,
) -> tuple[list[FullBox], list[str]]:
    categories: dict[str, int] = {}
    boxes: list[FullBox] = []

    for index, annotation in enumerate(annotations):
        if exclude_pn and is_pn_annotation(annotation):
            continue
        sample_start = int(annotation.get("core:sample_start", 0))
        sample_stop = sample_start + int(annotation.get("core:sample_count", 0))
        x_interval = sample_interval_to_x(sample_start, sample_stop, layout)
        freq_lo, freq_hi = annotation_frequency_bounds(annotation)
        y_interval = freq_interval_to_y(freq_lo, freq_hi, layout)
        if x_interval is None or y_interval is None:
            continue

        x0, x1 = expand_interval_to_min(
            x_interval[0], x_interval[1], min_box_width_px, 0.0, float(layout.total_frames)
        )
        y0, y1 = expand_interval_to_min(
            y_interval[0], y_interval[1], min_box_height_px, 0.0, float(layout.freq_bins)
        )
        if x1 <= x0 or y1 <= y0:
            continue

        cls = class_name_for_annotation(
            annotation,
            class_mode=class_mode,
            class_name=class_name,
            class_field=class_field,
        )
        if cls not in categories:
            categories[cls] = len(categories)

        boxes.append(
            FullBox(
                annotation_index=index,
                class_name=cls,
                class_id=categories[cls],
                x=x0,
                y=y0,
                width=x1 - x0,
                height=y1 - y0,
                sample_start=sample_start,
                sample_stop=sample_stop,
                freq_lower_hz=freq_lo,
                freq_upper_hz=freq_hi,
                label=annotation_label(annotation, index),
            )
        )

    category_names = [name for name, _ in sorted(categories.items(), key=lambda item: item[1])]
    return boxes, category_names


def tile_starts(total: int, tile_size: int | None, overlap: int) -> list[int]:
    if tile_size is None or tile_size <= 0 or tile_size >= total:
        return [0]
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("Tile overlap must be smaller than tile size")
    starts = list(range(0, total - tile_size + 1, step))
    if starts[-1] != total - tile_size:
        starts.append(total - tile_size)
    return starts


def build_tiles(
    *,
    stem: str,
    total_width: int,
    total_height: int,
    tile_frames: int | None,
    tile_freq_bins: int | None,
    overlap_frames: int,
    overlap_freq_bins: int,
) -> list[Tile]:
    width = total_width if tile_frames is None else min(tile_frames, total_width)
    height = total_height if tile_freq_bins is None else min(tile_freq_bins, total_height)
    tiles: list[Tile] = []
    for y in tile_starts(total_height, tile_freq_bins, overlap_freq_bins):
        for x in tile_starts(total_width, tile_frames, overlap_frames):
            tile_id = len(tiles) + 1
            file_name = f"{stem}_x{x:07d}_y{y:05d}_w{width}_h{height}.png"
            tiles.append(Tile(id=tile_id, file_name=file_name, x=x, y=y, width=width, height=height))
    return tiles


def intersect_box_with_tile(box: FullBox, tile: Tile) -> tuple[float, float, float, float] | None:
    x0 = max(box.x, float(tile.x))
    y0 = max(box.y, float(tile.y))
    x1 = min(box.x + box.width, float(tile.x + tile.width))
    y1 = min(box.y + box.height, float(tile.y + tile.height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0 - tile.x, y0 - tile.y, x1 - x0, y1 - y0


def write_outputs(
    *,
    outdir: Path,
    stem: str,
    layout: SpectrogramLayout,
    tiles: list[Tile],
    boxes: list[FullBox],
    category_names: list[str],
    include_empty_tiles: bool,
    min_visible_fraction: float,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    labels_dir = outdir / "yolo_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for old_label in labels_dir.glob("*.txt"):
        old_label.unlink()

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    images_csv_rows: list[dict[str, Any]] = []
    annotation_id = 1
    kept_tiles = 0
    kept_boxes = 0

    for tile in tiles:
        tile_records: list[tuple[FullBox, tuple[float, float, float, float], float]] = []
        for box in boxes:
            clipped = intersect_box_with_tile(box, tile)
            if clipped is None:
                continue
            visible_fraction = (clipped[2] * clipped[3]) / max(box.width * box.height, 1e-12)
            if visible_fraction < min_visible_fraction:
                continue
            tile_records.append((box, clipped, visible_fraction))

        if not tile_records and not include_empty_tiles:
            continue

        kept_tiles += 1
        coco_images.append(
            {
                "id": tile.id,
                "file_name": tile.file_name,
                "width": tile.width,
                "height": tile.height,
                "spectrogram:x_frame_start": tile.x,
                "spectrogram:y_bin_start_top": tile.y,
            }
        )
        images_csv_rows.append(
            {
                "image_id": tile.id,
                "file_name": tile.file_name,
                "width": tile.width,
                "height": tile.height,
                "x_frame_start": tile.x,
                "y_bin_start_top": tile.y,
            }
        )

        label_path = labels_dir / tile.file_name.replace(".png", ".txt")
        with label_path.open("w", encoding="utf-8") as label_file:
            for box, (x, y, width, height), visible_fraction in tile_records:
                x_center = (x + 0.5 * width) / tile.width
                y_center = (y + 0.5 * height) / tile.height
                width_norm = width / tile.width
                height_norm = height / tile.height
                label_file.write(
                    f"{box.class_id} {x_center:.9f} {y_center:.9f} "
                    f"{width_norm:.9f} {height_norm:.9f}\n"
                )

                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": tile.id,
                        "category_id": box.class_id + 1,
                        "bbox": [x, y, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "sigmf:annotation_index": box.annotation_index,
                        "sigmf:label": box.label,
                        "sigmf:sample_start": box.sample_start,
                        "sigmf:sample_stop": box.sample_stop,
                        "sigmf:freq_lower_hz": box.freq_lower_hz,
                        "sigmf:freq_upper_hz": box.freq_upper_hz,
                        "spectrogram:visible_fraction": visible_fraction,
                    }
                )
                rows.append(
                    {
                        "image_id": tile.id,
                        "file_name": tile.file_name,
                        "class_id": box.class_id,
                        "class_name": box.class_name,
                        "x_px": x,
                        "y_px": y,
                        "width_px": width,
                        "height_px": height,
                        "yolo_x_center": x_center,
                        "yolo_y_center": y_center,
                        "yolo_width": width_norm,
                        "yolo_height": height_norm,
                        "sigmf_annotation_index": box.annotation_index,
                        "sigmf_label": box.label,
                        "sample_start": box.sample_start,
                        "sample_stop": box.sample_stop,
                        "freq_lower_hz": box.freq_lower_hz,
                        "freq_upper_hz": box.freq_upper_hz,
                        "visible_fraction": visible_fraction,
                    }
                )
                kept_boxes += 1
                annotation_id += 1

    categories = [
        {"id": index + 1, "name": name, "supercategory": "signal"}
        for index, name in enumerate(category_names)
    ]
    coco = {
        "info": {
            "description": "SigMF annotation boxes mapped onto STFT spectrogram image coordinates",
            "spectrogram": {
                "fft_size": layout.fft_size,
                "hop_size": layout.hop_size,
                "sample_rate_hz": layout.sample_rate_hz,
                "total_samples": layout.total_samples,
                "total_frames": layout.total_frames,
                "freq_min_hz": layout.freq_min_hz,
                "freq_max_hz": layout.freq_max_hz,
                "freq_bins": layout.freq_bins,
                "image_coordinate_origin": "top-left",
                "frequency_orientation": "high frequency at top, low frequency at bottom",
                "time_box_mode": layout.time_box_mode,
            },
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    (outdir / "dino_coco.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (outdir / "classes.txt").write_text("\n".join(category_names) + ("\n" if category_names else ""), encoding="utf-8")
    (outdir / "dataset.yaml").write_text(
        "path: .\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        + "".join(f"  {idx}: {name}\n" for idx, name in enumerate(category_names)),
        encoding="utf-8",
    )

    with (outdir / "boxes.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "image_id",
            "file_name",
            "class_id",
            "class_name",
            "x_px",
            "y_px",
            "width_px",
            "height_px",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (outdir / "spectrogram_tiles.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(images_csv_rows[0].keys()) if images_csv_rows else [
            "image_id",
            "file_name",
            "width",
            "height",
            "x_frame_start",
            "y_bin_start_top",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(images_csv_rows)

    summary = {
        "output_dir": str(outdir),
        "tile_count": kept_tiles,
        "box_count": kept_boxes,
        "source_box_count": len(boxes),
        "category_count": len(category_names),
        "layout": {
            "sample_rate_hz": layout.sample_rate_hz,
            "total_samples": layout.total_samples,
            "fft_size": layout.fft_size,
            "hop_size": layout.hop_size,
            "total_frames": layout.total_frames,
            "freq_min_hz": layout.freq_min_hz,
            "freq_max_hz": layout.freq_max_hz,
            "freq_bins": layout.freq_bins,
            "time_box_mode": layout.time_box_mode,
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_written_tiles(outdir: Path) -> list[Tile]:
    tiles_path = outdir / "spectrogram_tiles.csv"
    with tiles_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        Tile(
            id=int(row["image_id"]),
            file_name=row["file_name"],
            x=int(row["x_frame_start"]),
            y=int(row["y_bin_start_top"]),
            width=int(row["width"]),
            height=int(row["height"]),
        )
        for row in rows
    ]


def frequency_mask_for_layout(layout: SpectrogramLayout) -> np.ndarray:
    freqs = np.fft.fftshift(np.fft.fftfreq(layout.fft_size, d=1.0 / layout.sample_rate_hz))
    mask = (freqs >= layout.freq_min_hz) & (freqs <= layout.freq_max_hz)
    if int(np.count_nonzero(mask)) != layout.freq_bins:
        raise RuntimeError(
            "Internal frequency-bin mismatch between detection layout and spectrogram renderer: "
            f"layout={layout.freq_bins}, mask={int(np.count_nonzero(mask))}"
        )
    return mask


def stft_power_db_for_frame_span(
    reader: SigMFReader,
    layout: SpectrogramLayout,
    *,
    frame_start: int,
    frame_count: int,
    window: np.ndarray,
    freq_mask: np.ndarray,
) -> np.ndarray:
    sample_start = int(frame_start) * layout.hop_size
    sample_stop = sample_start + (int(frame_count) - 1) * layout.hop_size + layout.fft_size
    samples = reader.read(sample_start, sample_stop)
    needed = sample_stop - sample_start
    if samples.size < needed:
        padded = np.zeros(needed, dtype=np.complex64)
        padded[: samples.size] = samples
        samples = padded

    frames = np.lib.stride_tricks.sliding_window_view(samples, layout.fft_size)[:: layout.hop_size]
    frames = frames[:frame_count]
    fft = np.fft.fftshift(
        np.fft.fft(frames * window[None, :], n=layout.fft_size, axis=1),
        axes=1,
    )
    eps = np.float32(1e-12)
    power_db = (20.0 * np.log10(np.maximum(np.abs(fft), eps))).astype(np.float32).T
    # FFT bins are low-to-high after fftshift. Images use top-left origin, so
    # row 0 should be high frequency.
    return power_db[freq_mask, :][::-1, :]


def scale_spectrogram_tile_to_uint8(
    tile_db: np.ndarray,
    *,
    dynamic_range_db: float,
    percentile: float,
    vmin_db: float | None,
    vmax_db: float | None,
) -> np.ndarray:
    if tile_db.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    finite = tile_db[np.isfinite(tile_db)]
    if finite.size == 0:
        return np.zeros(tile_db.shape, dtype=np.uint8)

    if vmax_db is None:
        vmax = float(np.percentile(finite, percentile))
    else:
        vmax = float(vmax_db)
    if vmin_db is None:
        vmin = vmax - float(dynamic_range_db)
    else:
        vmin = float(vmin_db)
    if vmax <= vmin:
        vmax = vmin + 1.0

    scaled = (tile_db - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def spectrogram_colormap_lut(colormap: str) -> np.ndarray:
    if colormap not in _COLORMAP_LUTS:
        if colormap != "viridis":
            raise ValueError(f"Unsupported spectrogram colormap {colormap!r}")
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        channels = [
            np.interp(x, _VIRIDIS_POSITIONS, _VIRIDIS_COLORS[:, channel])
            for channel in range(3)
        ]
        _COLORMAP_LUTS[colormap] = np.round(np.stack(channels, axis=1)).astype(np.uint8)
    return _COLORMAP_LUTS[colormap]


def apply_spectrogram_colormap(image: np.ndarray, colormap: str) -> np.ndarray:
    if colormap == "gray":
        return image
    return spectrogram_colormap_lut(colormap)[image]


def render_spectrogram_tiles(
    *,
    meta: dict[str, Any],
    data_path: Path,
    outdir: Path,
    image_dirname: str,
    layout: SpectrogramLayout,
    tiles: list[Tile],
    dynamic_range_db: float,
    percentile: float,
    vmin_db: float | None,
    vmax_db: float | None,
    colormap: str,
    max_tiles: int,
    verbose: bool,
) -> int:
    from PIL import Image

    image_dir = outdir / image_dirname
    image_dir.mkdir(parents=True, exist_ok=True)
    for old_image in image_dir.glob("*.png"):
        old_image.unlink()

    render_tiles = tiles[: max_tiles if max_tiles > 0 else None]
    if not render_tiles:
        return 0

    by_x: dict[int, list[Tile]] = {}
    for tile in render_tiles:
        by_x.setdefault(tile.x, []).append(tile)

    reader = SigMFReader(data_path, meta)
    window = np.hanning(layout.fft_size).astype(np.float32)
    freq_mask = frequency_mask_for_layout(layout)
    rendered = 0
    total_groups = len(by_x)

    for group_index, (frame_start, group_tiles) in enumerate(sorted(by_x.items()), start=1):
        frame_count = max(tile.width for tile in group_tiles)
        spec_db = stft_power_db_for_frame_span(
            reader,
            layout,
            frame_start=frame_start,
            frame_count=frame_count,
            window=window,
            freq_mask=freq_mask,
        )

        for tile in group_tiles:
            tile_db = spec_db[tile.y : tile.y + tile.height, : tile.width]
            image = scale_spectrogram_tile_to_uint8(
                tile_db,
                dynamic_range_db=dynamic_range_db,
                percentile=percentile,
                vmin_db=vmin_db,
                vmax_db=vmax_db,
            )
            Image.fromarray(apply_spectrogram_colormap(image, colormap)).save(image_dir / tile.file_name)
            rendered += 1

        if verbose and (group_index == 1 or group_index % 25 == 0 or group_index == total_groups):
            print(
                f"Rendered spectrogram frame group {group_index}/{total_groups} "
                f"({rendered}/{len(render_tiles)} tiles)",
                flush=True,
            )

    return rendered


def update_summary_with_images(outdir: Path, *, image_dirname: str, image_count: int, colormap: str) -> None:
    summary_path = outdir / "summary.json"
    summary = load_json(summary_path)
    summary["spectrogram_images_dir"] = image_dirname
    summary["spectrogram_images_written"] = int(image_count)
    summary["spectrogram_colormap"] = colormap
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SigMF annotations into YOLO labels and COCO/DINO detection "
            "boxes for an STFT spectrogram grid."
        )
    )
    parser.add_argument("meta", type=Path, help="Input .sigmf-meta with annotations.")
    parser.add_argument("data", type=Path, nargs="?", help="Optional matching .sigmf-data used to infer duration.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for labels and manifests.")
    parser.add_argument("--fft-size", "--nfft", type=int, default=4096, help="Spectrogram FFT size / frequency bins.")
    parser.add_argument("--hop-size", "--hop", type=int, default=None, help="STFT hop size. Default: fft_size / 4.")
    parser.add_argument("--freq-min-hz", type=str, default=None, help="Minimum displayed baseband frequency.")
    parser.add_argument("--freq-max-hz", type=str, default=None, help="Maximum displayed baseband frequency.")
    parser.add_argument("--tile-frames", type=int, default=None, help="Tile width in STFT frames. Omit for full width.")
    parser.add_argument("--tile-freq-bins", type=int, default=None, help="Tile height in frequency bins. Omit for full height.")
    parser.add_argument("--tile-overlap-frames", type=int, default=0, help="Horizontal tile overlap in frames.")
    parser.add_argument("--tile-overlap-freq-bins", type=int, default=0, help="Vertical tile overlap in bins.")
    parser.add_argument(
        "--time-box-mode",
        choices=("stft-overlap", "sample", "frame-center"),
        default="stft-overlap",
        help=(
            "How annotation sample spans map to STFT columns. stft-overlap covers "
            "all columns whose FFT windows intersect the annotation."
        ),
    )
    parser.add_argument(
        "--class-mode",
        choices=("single", "field", "waveform-class"),
        default="single",
        help=(
            "single uses one detector class, field uses --class-field directly, and "
            "waveform-class infers families such as 5G_Downlink, LTE_Uplink, 802_11ax, or QPSK."
        ),
    )
    parser.add_argument("--class-name", default="signal", help="Class name for --class-mode single or fallback.")
    parser.add_argument("--class-field", default="compose:waveform_name", help="Annotation key for --class-mode field.")
    parser.add_argument("--exclude-pn", action="store_true", help="Skip PN sync/preamble annotations.")
    parser.add_argument("--include-empty-tiles", action="store_true", help="Write image records and empty label files.")
    parser.add_argument("--min-visible-fraction", type=float, default=0.0, help="Drop tile-clipped boxes below this fraction.")
    parser.add_argument("--min-box-width-px", type=float, default=1.0, help="Minimum box width after full-image mapping.")
    parser.add_argument("--min-box-height-px", type=float, default=1.0, help="Minimum box height after full-image mapping.")
    parser.add_argument(
        "--write-spectrograms",
        action="store_true",
        help="Also render spectrogram PNG tiles matching the COCO image records.",
    )
    parser.add_argument("--image-dirname", default="images", help="Subdirectory under outdir for spectrogram PNGs.")
    parser.add_argument(
        "--spectrogram-dynamic-range-db",
        type=float,
        default=80.0,
        help="Per-tile displayed dynamic range when rendering spectrogram images.",
    )
    parser.add_argument(
        "--spectrogram-percentile",
        type=float,
        default=99.5,
        help="Per-tile percentile used as vmax when rendering spectrogram images.",
    )
    parser.add_argument("--spectrogram-vmin-db", type=float, default=None, help="Fixed spectrogram vmin in dB.")
    parser.add_argument("--spectrogram-vmax-db", type=float, default=None, help="Fixed spectrogram vmax in dB.")
    parser.add_argument(
        "--spectrogram-colormap",
        choices=("viridis", "gray"),
        default="viridis",
        help="Colormap for rendered spectrogram PNG tiles.",
    )
    parser.add_argument(
        "--max-spectrogram-tiles",
        type=int,
        default=0,
        help="Render only the first N spectrogram tiles. Default 0 renders all written image records.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress while rendering spectrogram PNGs.")
    args = parser.parse_args()
    if args.fft_size <= 0:
        parser.error("--fft-size must be positive")
    if args.hop_size is not None and args.hop_size <= 0:
        parser.error("--hop-size must be positive")
    if args.tile_frames is not None and args.tile_frames <= 0:
        parser.error("--tile-frames must be positive")
    if args.tile_freq_bins is not None and args.tile_freq_bins <= 0:
        parser.error("--tile-freq-bins must be positive")
    if not 0.0 <= args.min_visible_fraction <= 1.0:
        parser.error("--min-visible-fraction must be between 0 and 1")
    if not 0.0 < args.spectrogram_percentile <= 100.0:
        parser.error("--spectrogram-percentile must be in (0, 100]")
    if args.spectrogram_dynamic_range_db <= 0:
        parser.error("--spectrogram-dynamic-range-db must be positive")
    if args.max_spectrogram_tiles < 0:
        parser.error("--max-spectrogram-tiles must be >= 0")
    return args


def main() -> int:
    args = parse_args()
    meta_path = args.meta.resolve()
    data_path = args.data.resolve() if args.data else infer_sigmf_data_path(meta_path).resolve()
    if not data_path.exists():
        data_path = None
    if args.write_spectrograms and data_path is None:
        raise FileNotFoundError("--write-spectrograms requires an existing .sigmf-data file")

    meta = load_json(meta_path)
    hop_size = args.hop_size or max(1, args.fft_size // 4)
    layout = build_layout(
        meta=meta,
        data_path=data_path,
        fft_size=args.fft_size,
        hop_size=hop_size,
        freq_min_hz=parse_frequency(args.freq_min_hz, "freq_min_hz") if args.freq_min_hz is not None else None,
        freq_max_hz=parse_frequency(args.freq_max_hz, "freq_max_hz") if args.freq_max_hz is not None else None,
        time_box_mode=args.time_box_mode,
    )
    boxes, category_names = annotations_to_full_boxes(
        list(meta.get("annotations", [])),
        layout,
        class_mode=args.class_mode,
        class_name=args.class_name,
        class_field=args.class_field,
        exclude_pn=bool(args.exclude_pn),
        min_box_width_px=float(args.min_box_width_px),
        min_box_height_px=float(args.min_box_height_px),
    )
    tiles = build_tiles(
        stem=meta_path.name.replace(".sigmf-meta", ""),
        total_width=layout.total_frames,
        total_height=layout.freq_bins,
        tile_frames=args.tile_frames,
        tile_freq_bins=args.tile_freq_bins,
        overlap_frames=args.tile_overlap_frames,
        overlap_freq_bins=args.tile_overlap_freq_bins,
    )
    summary = write_outputs(
        outdir=args.outdir.resolve(),
        stem=meta_path.name.replace(".sigmf-meta", ""),
        layout=layout,
        tiles=tiles,
        boxes=boxes,
        category_names=category_names,
        include_empty_tiles=bool(args.include_empty_tiles),
        min_visible_fraction=float(args.min_visible_fraction),
    )
    images_written = 0
    if args.write_spectrograms:
        written_tiles = load_written_tiles(args.outdir.resolve())
        images_written = render_spectrogram_tiles(
            meta=meta,
            data_path=data_path,
            outdir=args.outdir.resolve(),
            image_dirname=args.image_dirname,
            layout=layout,
            tiles=written_tiles,
            dynamic_range_db=float(args.spectrogram_dynamic_range_db),
            percentile=float(args.spectrogram_percentile),
            vmin_db=args.spectrogram_vmin_db,
            vmax_db=args.spectrogram_vmax_db,
            colormap=args.spectrogram_colormap,
            max_tiles=int(args.max_spectrogram_tiles),
            verbose=bool(args.verbose),
        )
        update_summary_with_images(
            args.outdir.resolve(),
            image_dirname=args.image_dirname,
            image_count=images_written,
            colormap=args.spectrogram_colormap,
        )
    print(f"Meta: {meta_path}")
    print(f"Data: {data_path if data_path is not None else 'not used'}")
    print(f"Output: {args.outdir.resolve()}")
    print(f"Spectrogram frames: {layout.total_frames}")
    print(f"Frequency bins: {layout.freq_bins}")
    print(f"Source boxes: {summary['source_box_count']}")
    print(f"Dataset tiles: {summary['tile_count']}")
    print(f"Tile-clipped boxes: {summary['box_count']}")
    print(f"Categories: {summary['category_count']}")
    print(f"YOLO labels: {args.outdir.resolve() / 'yolo_labels'}")
    print(f"DINO/COCO JSON: {args.outdir.resolve() / 'dino_coco.json'}")
    if args.write_spectrograms:
        print(f"Spectrogram images: {args.outdir.resolve() / args.image_dirname} ({images_written})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
