"""Receiver: sync all ZCs, decode per-block metadata, rebuild SigMF annotations.

On reception the .sigmf-meta carries only receiver parameters (no annotations).
This module scans the known candidate block centers for the ZC, decodes the
OFDM metadata burst that follows each ZC, reconstructs each waveform's absolute
position / bandwidth / class / variation (positions are relative to the ZC), and
writes those annotations into the received .sigmf-meta.
"""

from __future__ import annotations
import os
import json
import numpy as np

import geometry as geo
import protocol as proto
from zc_sync import load_zc, freq_shift, detect_multi

FS = geo.FS


def _read_meta(path):
    with open(path) as f:
        return json.load(f)


def decode_composite(data_path, rx_meta_path, zc_path, out_meta_path=None,
                     candidate_centers=geo.RX_CANDIDATE_CENTERS, detector=None,
                     profile="fast"):
    """Recover annotations from a composite capture and write them into the meta.

    `detector(iq, zc, fs, centers) -> [{center_hz, start_sample, metric, [cfo_hz]}]`
    defaults to the fast energy-gated detector. Pass fine_comb_decode.detect_fine
    for a low-SNR full-sweep CFAR detector. Returns recovered annotation dicts.

    `profile` selects the compose-time metadata modem:
      - "fast"/"robust": decode with exactly that profile (one cheap attempt per
        detection; detections that aren't a real ZC just fail the CRC).
      - "auto": self-describing -- try PROFILE_TRY_ORDER and let the CRC pick.
        Detections are processed strongest-first and the profile is locked after
        the first success, so false alarms don't each pay for the slow profile.
    """
    if detector is None:
        detector = lambda iq, zc, fs, c: detect_multi(iq, zc, fs, c, thresh=0.35)
    auto = (profile == "auto")
    try_profiles = list(proto.PROFILE_TRY_ORDER) if auto else [profile]
    meta = _read_meta(rx_meta_path)
    fs = float(meta["global"].get("core:sample_rate", FS))
    iq = np.fromfile(data_path, dtype="<c8").astype(np.complex64)

    zc = load_zc(zc_path)
    zc_len = zc.size
    meta_start = proto.meta_start(zc_len)   # RX-known; independent of guard times

    recovered = []
    blocks = []
    failed = []   # ZC bursts detected but whose metadata failed CRC on every center
    # Find every ZC burst (all time groups, all candidate centers). In "auto" mode
    # process the strongest detections first so the metadata profile locks after
    # the first success and false alarms don't each pay for the slow profile.
    dets = detector(iq, zc, fs, candidate_centers)
    if auto:
        dets = sorted(dets, key=lambda d: -d.get("metric", 0))
    locked = None
    for d in dets:
        fc = d["center_hz"]; t_zc = d["start_sample"]; metric = d["metric"]
        cfo = float(d.get("cfo_hz", 0.0))
        s0 = t_zc + meta_start
        # CRC validates each attempt; in auto mode lock the profile after success.
        order = [locked] if (auto and locked) else try_profiles
        desc, meta_len = None, 0
        for prof in order:
            ml = proto.meta_len_245(prof)
            if s0 < 0 or s0 + ml > iq.size:
                continue
            cand = proto.decode_metadata(freq_shift(iq[s0:s0 + ml], -(fc + cfo), fs), prof)
            if cand is not None:
                desc, meta_len, locked = cand, ml, prof
                break
        if desc is None:
            failed.append({"t_zc": t_zc, "center_hz": fc, "metric": metric})
            continue
        gi = int(desc.get("g", 0))
        blocks.append({"center_hz": fc, "t_zc": t_zc, "metric": metric,
                       "n_waveforms": len(desc["wfs"]), "time_group": gi})
        for w in desc["wfs"]:
            fcw = float(w["fc"]); bw = float(w["bw"])
            recovered.append({
                "core:sample_start": int(t_zc + w["rs"]),
                "core:sample_count": int(w["n"]),
                "core:freq_lower_edge": fcw - bw / 2.0,
                "core:freq_upper_edge": fcw + bw / 2.0,
                "core:label": w["cls"],
                "wfgt:kind": "waveform",
                "wfgt:class": w["cls"],
                "wfgt:variation": w["var"],
                "wfgt:occupied_bw_hz": bw,
                "wfgt:power_db": w.get("pdb", 0.0),
                "wfgt:length_samples": int(w["n"]),
                "wfgt:original_length_samples": int(w.get("ol", w["n"])),
                "wfgt:block_center_hz": float(desc["fc"]),
                "wfgt:time_group": gi,
                "wfgt:zc_sample": t_zc,
                "wfgt:zc_metric": round(metric, 4),
            })
        recovered.append({
            "core:sample_start": t_zc, "core:sample_count": zc_len,
            "core:freq_lower_edge": fc - 25e6, "core:freq_upper_edge": fc + 25e6,
            "core:label": "ZC", "wfgt:kind": "zadoff_chu",
            "wfgt:block_center_hz": fc, "wfgt:time_group": gi,
            "wfgt:zc_metric": round(metric, 4)})
        recovered.append({
            "core:sample_start": t_zc + meta_start, "core:sample_count": meta_len,
            "core:freq_lower_edge": fc - 13e6, "core:freq_upper_edge": fc + 13e6,
            "core:label": "METADATA", "wfgt:kind": "metadata",
            "wfgt:block_center_hz": fc, "wfgt:time_group": gi})

    # de-duplicate (a non-block candidate center can occasionally also decode)
    seen = set()
    uniq = []
    for a in sorted(recovered, key=lambda r: (r["core:sample_start"],
                                              r["core:freq_lower_edge"])):
        key = (a["wfgt:kind"], a["core:sample_start"],
               round(a["core:freq_lower_edge"]), round(a["core:freq_upper_edge"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)

    meta["annotations"] = uniq
    if out_meta_path is None:
        out_meta_path = rx_meta_path
    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    nwf = sum(b["n_waveforms"] for b in blocks)
    gids = {b["time_group"] for b in blocks}
    ngroups = len(gids)
    print(f"decoded {len(blocks)} block(s) across {ngroups} time group(s), {nwf} waveforms")
    print(f"  wrote {len(uniq)} annotations -> {out_meta_path}")

    # Under-report guard: the composer numbers time groups 0..N-1 contiguously,
    # so a gap in the decoded group indices (or a missing group 0) means whole
    # groups were dropped. Classify each drop by *cause* so the remedy is clear:
    #   - "decode issue": a ZC burst WAS detected there but its metadata failed
    #     CRC on every center. Detection tuning won't help (bad SNR/corrupt
    #     burst / wrong profile) -- the ZC is found, the payload isn't readable.
    #   - "unfound ZC": no ZC cleared the detector at all. Detection tuning CAN
    #     help (lower active_factor, or fine_comb_decode.py's full sweep).
    # A detected-but-undecoded burst that sits on top of a group that DID decode
    # at another center is just a false alarm, not a missing group, so drop it.
    if gids:
        missing = sorted(set(range(0, max(gids) + 1)) - gids)
        if missing:
            decoded_t = sorted({b["t_zc"] for b in blocks})
            near_decoded = lambda t: any(abs(t - dt) < zc_len for dt in decoded_t)
            # collapse the per-center failed detections into one per ZC burst
            fpos = []
            for d in sorted(failed, key=lambda x: x["t_zc"]):
                if near_decoded(d["t_zc"]):
                    continue
                if fpos and d["t_zc"] - fpos[-1]["t_zc"] < zc_len:
                    if d["metric"] > fpos[-1]["metric"]:
                        fpos[-1] = d
                else:
                    fpos.append(d)
            n_decode = min(len(fpos), len(missing))   # decode-issue bursts <-> missing groups
            n_unfound = len(missing) - n_decode
            print(f"  WARNING: {len(missing)} time group(s) missing (indices "
                  f"{missing}); decoded {sorted(gids)}.")
            if fpos:
                locs = ", ".join(f"t={d['t_zc']/fs*1000:.2f}ms "
                                 f"center={d['center_hz']/1e6:+.0f}MHz "
                                 f"metric={d['metric']:.3f}" for d in fpos)
                print(f"    decode issue ({n_decode}): ZC found but metadata failed "
                      f"CRC at [{locs}] -- not recoverable by detection tuning "
                      f"(SNR/corrupt burst/profile).")
            if n_unfound > 0:
                print(f"    unfound ZC ({n_unfound}): no ZC cleared the detector -- "
                      f"retry with a lower active_factor or fine_comb_decode.py "
                      f"(full sweep, no energy gate).")
            print(f"    (trailing groups with index > {max(gids)} cannot be "
                  f"checked this way.)")
    return uniq, blocks


# --------------------------------------------------------------------------- #
def validate(tx_meta_path, recovered, sample_tol=64):
    """Compare recovered waveform annotations against TX ground truth."""
    tx = _read_meta(tx_meta_path)
    truth = [a for a in tx["annotations"] if a.get("wfgt:kind") == "waveform"]
    rec = [a for a in recovered if a.get("wfgt:kind") == "waveform"]

    results = []
    used = set()
    for t in truth:
        tc = (t["core:freq_lower_edge"] + t["core:freq_upper_edge"]) / 2
        # candidates: same variation, freq center within 1 MHz, unused; pick the
        # one closest in sample_start (handles a variation reused across groups).
        cands = [(i, r) for i, r in enumerate(rec) if i not in used
                 and r["wfgt:variation"] == t["wfgt:variation"]
                 and abs((r["core:freq_lower_edge"] + r["core:freq_upper_edge"]) / 2 - tc) < 1e6]
        if not cands:
            results.append({"variation": t["wfgt:variation"], "matched": False})
            continue
        i, r = min(cands, key=lambda ir: abs(ir[1]["core:sample_start"] - t["core:sample_start"]))
        used.add(i)
        ds = abs(r["core:sample_start"] - t["core:sample_start"])
        df_lo = abs(r["core:freq_lower_edge"] - t["core:freq_lower_edge"])
        df_hi = abs(r["core:freq_upper_edge"] - t["core:freq_upper_edge"])
        results.append({
            "variation": t["wfgt:variation"], "matched": True,
            "class_ok": r["wfgt:class"] == t["wfgt:class"],
            "start_err": ds, "start_ok": ds <= sample_tol,
            "freq_err_hz": max(df_lo, df_hi), "freq_ok": max(df_lo, df_hi) < 1.0,
            "count_ok": r["core:sample_count"] == t["core:sample_count"]})

    n = len(truth)
    matched = sum(1 for r in results if r["matched"])
    full = sum(1 for r in results if r["matched"] and r["class_ok"]
               and r["start_ok"] and r["freq_ok"] and r["count_ok"])
    return {"n_truth": n, "n_matched": matched, "n_full": full, "details": results}


if __name__ == "__main__":
    import argparse
    from zc_sync import detect_multi
    ap = argparse.ArgumentParser(
        description="Recover SigMF annotations from a composite capture.")
    ap.add_argument("data"); ap.add_argument("rxmeta"); ap.add_argument("zc")
    ap.add_argument("profile", nargs="?", default="fast",
                    choices=["fast", "robust", "auto"],
                    help="metadata modem profile (default: fast, the cheapest).")
    ap.add_argument("--centers", metavar="MHz",
                    help="comma-separated block centers to scan, e.g. '-60,0,60'. "
                         "The default scans all 5 RX candidates (-60,-30,0,30,60); "
                         "if your composition never uses the +/-30 merged 2-block "
                         "slots, passing '-60,0,60' skips 2 of 5 matched filters "
                         "(~40%% faster) with no loss.")
    ap.add_argument("--active-factor", type=float, default=None, metavar="X",
                    help="energy-gate sensitivity (default 1.5). Higher = fewer "
                         "search windows = faster, but drops weak/isolated ZCs.")
    args = ap.parse_args()

    centers = geo.RX_CANDIDATE_CENTERS
    if args.centers:
        centers = tuple(float(x) * 1e6 for x in args.centers.split(","))
    detector = None
    if args.active_factor is not None:
        af = args.active_factor
        detector = lambda iq, zc, fs, c: detect_multi(iq, zc, fs, c,
                                                       thresh=0.35, active_factor=af)
    decode_composite(args.data, args.rxmeta, args.zc, candidate_centers=centers,
                     detector=detector, profile=args.profile)
