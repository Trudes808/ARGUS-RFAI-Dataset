"""End-to-end composition example + validation (clean and wireless channel).

For each example config:
  1. compose() -> 200 MHz SigMF recording (+ TX truth meta + bare RX meta)
  2. CLEAN  : decode the RX capture, rebuild annotations, validate vs truth
  3. CHANNEL: pass the composite through a simple wireless channel, decode the
              annotation-less capture, rebuild annotations, validate vs truth
Also demonstrates the helpful error raised by an over-stuffed block.
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os
import shutil
import numpy as np

from compose import compose
from decode import decode_composite, validate
from channel import apply_channel, WIRELESS
import geometry as geo

HERE = os.path.dirname(os.path.abspath(__file__))
ZC = os.path.join(HERE, "..", "generated_sync_sequences", "ZadoffChu_bw50MHz_R25_N601.mat")
CONFIGS = ["example_mixed.yaml", "example_wide80.yaml",
           "example_full160.yaml", "example_5g100.yaml",
           "example_timegroups.yaml"]


def _fmt(v):
    starts = [d["start_err"] for d in v["details"] if d.get("matched")]
    return (f"{v['n_full']}/{v['n_truth']} fully recovered, "
            f"{v['n_matched']} matched"
            + (f", max start err {max(starts)} samp" if starts else ""))


def run_one(cfg):
    print("=" * 78)
    s = compose(os.path.join(HERE, cfg))
    base = s["data"].replace(".sigmf-data", "")
    tx_meta = s["tx_meta"]

    # ---- CLEAN (the clean reception data == the TX data) ----
    rec, _ = decode_composite(base + ".sigmf-data", s["rx_meta"], ZC,
                              out_meta_path=base + ".rx.decoded.sigmf-meta")
    vc = validate(tx_meta, rec, sample_tol=4)
    print(f"  CLEAN   : {_fmt(vc)}")

    # ---- CHANNEL ----
    iq = np.fromfile(base + ".sigmf-data", dtype="<c8")
    ych = apply_channel(iq, geo.FS, **WIRELESS)
    ych.astype("<c8").tofile(base + ".rxch.sigmf-data")
    shutil.copy(s["rx_meta"], base + ".rxch.sigmf-meta")
    recch, _ = decode_composite(base + ".rxch.sigmf-data", base + ".rxch.sigmf-meta", ZC,
                                out_meta_path=base + ".rxch.decoded.sigmf-meta")
    vch = validate(tx_meta, recch, sample_tol=64)
    print(f"  CHANNEL : {_fmt(vch)}")
    return vc, vch


def main():
    print(f"Wireless channel profile: {WIRELESS}\n")
    totals = {"clean_full": 0, "clean_truth": 0, "chan_full": 0, "chan_truth": 0}
    for cfg in CONFIGS:
        vc, vch = run_one(cfg)
        totals["clean_full"] += vc["n_full"]; totals["clean_truth"] += vc["n_truth"]
        totals["chan_full"] += vch["n_full"]; totals["chan_truth"] += vch["n_truth"]

    # error-handling demonstration
    print("=" * 78)
    print("Error demo (over-stuffed block):")
    try:
        compose(os.path.join(HERE, "example_bad.yaml"))
        print("  ERROR: expected a validation failure but none was raised")
    except ValueError as e:
        print(f"  raised ValueError as expected -> {e}")

    print("=" * 78)
    print(f"OVERALL  CLEAN   {totals['clean_full']}/{totals['clean_truth']} waveforms fully recovered")
    print(f"OVERALL  CHANNEL {totals['chan_full']}/{totals['chan_truth']} waveforms fully recovered")


if __name__ == "__main__":
    main()
