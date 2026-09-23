"""Comprehensive proof-of-concept: pack the ENTIRE library into one composition,
then decode + re-annotate clean and through a wireless channel, validating every
waveform's recovered annotation against ground truth.
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os
import gc
import shutil
import numpy as np

from build_comprehensive_config import build
from compose import compose
from decode import decode_composite, validate
from channel import apply_channel, WIRELESS
import geometry as geo

HERE = os.path.dirname(os.path.abspath(__file__))
ZC = os.path.join(HERE, "..", "generated_sync_sequences", "ZadoffChu_bw50MHz_R25_N601.mat")
MANIFEST = os.path.join(HERE, "..", "generated_waveforms_24576", "waveform_manifest.csv")


def _summary(tag, v):
    starts = [d["start_err"] for d in v["details"] if d.get("matched")]
    print(f"  {tag:8s}: {v['n_full']}/{v['n_truth']} fully recovered, "
          f"{v['n_matched']} matched"
          + (f", max start err {max(starts)} samp" if starts else ""))
    miss = [d["variation"] for d in v["details"] if not d["matched"]]
    if miss:
        print(f"            {len(miss)} not matched, e.g. {miss[:3]}")


def main():
    cfg = build(MANIFEST, os.path.join(HERE, "comprehensive.yaml"),
                "../generated_waveforms_24576",
                "../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat")
    print("composing (this writes a multi-GB capture) ...")
    s = compose(cfg)
    base = s["data"].replace(".sigmf-data", "")
    tx_data, tx_meta, rx_meta = base + ".sigmf-data", s["tx_meta"], s["rx_meta"]
    gc.collect()

    # ---- CLEAN (received data == TX data) ----
    print("\ndecoding CLEAN ...")
    rec, _ = decode_composite(tx_data, rx_meta, ZC,
                              out_meta_path=base + ".rx.decoded.sigmf-meta")
    vc = validate(tx_meta, rec, sample_tol=4)
    _summary("CLEAN", vc)
    del rec; gc.collect()

    # ---- CHANNEL ----
    print("\napplying wireless channel + decoding ...")
    iq = np.fromfile(tx_data, dtype="<c8")
    ych = apply_channel(iq, geo.FS, **WIRELESS)
    del iq; gc.collect()
    ych.astype("<c8").tofile(base + ".rxch.sigmf-data")
    del ych; gc.collect()
    shutil.copy(rx_meta, base + ".rxch.sigmf-meta")
    recch, _ = decode_composite(base + ".rxch.sigmf-data", base + ".rxch.sigmf-meta", ZC,
                                out_meta_path=base + ".rxch.decoded.sigmf-meta")
    vch = validate(tx_meta, recch, sample_tol=256)
    _summary("CHANNEL", vch)

    print(f"\nWireless channel: {WIRELESS}")
    print("=" * 70)
    print(f"COMPREHENSIVE: {s['n_waveforms']} waveforms in {s['n_groups']} time groups, "
          f"{s['duration_s']:.2f} s")
    print(f"  CLEAN   {vc['n_full']}/{vc['n_truth']} fully recovered")
    print(f"  CHANNEL {vch['n_full']}/{vch['n_truth']} fully recovered")


if __name__ == "__main__":
    main()
