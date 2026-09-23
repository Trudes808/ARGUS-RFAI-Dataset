"""Build + compose the ordered comprehensive composition, then clean-decode it."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, gc, time, collections
from build_comprehensive_ordered import build, DURATIONS_MS
from compose import compose
from decode import decode_composite, validate

HERE = os.path.dirname(os.path.abspath(__file__))
ZC = os.path.join(HERE, "..", "generated_sync_sequences", "ZadoffChu_bw50MHz_R25_N601.mat")
MANIFEST = os.path.join(HERE, "..", "generated_waveforms_24576", "waveform_manifest.csv")

cfg = build(MANIFEST, os.path.join(HERE, "comprehensive_ordered.yaml"),
            "../generated_waveforms_24576",
            "../generated_sync_sequences/ZadoffChu_bw50MHz_R25_N601.mat")
print("composing (~12 GB) ...", flush=True)
t0 = time.time(); s = compose(cfg); print("compose: %.0fs" % (time.time() - t0), flush=True)
base = s["data"].replace(".sigmf-data", "")
gc.collect()

print("decoding CLEAN ...", flush=True)
t0 = time.time()
rec, _ = decode_composite(base + ".sigmf-data", s["rx_meta"], ZC,
                          out_meta_path=base + ".rx.decoded.sigmf-meta")
print("decode: %.0fs" % (time.time() - t0), flush=True)

v = validate(s["tx_meta"], rec, sample_tol=4)
print(f"CLEAN: {v['n_full']}/{v['n_truth']} fully recovered, {v['n_matched']} matched")

# breakdown: count fully-recovered per (class) and confirm all 6 durations present
byc = collections.Counter()
for d in v["details"]:
    if d["matched"] and d["class_ok"] and d["start_ok"] and d["freq_ok"] and d["count_ok"]:
        byc[d["variation"].split("_")[0]] += 1
print("fully recovered by name-prefix:", dict(byc))
durs = collections.Counter(round(a["core:sample_count"] / 245760.0, 3)
                           for a in rec if a.get("wfgt:kind") == "waveform")
print("recovered waveform durations (ms -> count):", dict(durs))
