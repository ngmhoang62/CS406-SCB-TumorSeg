"""Check every {CASE}.npz against the Data Contract (README section 2) and splits_metadata.csv."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def check(args):
    f, sid = args
    try:
        z = np.load(f)
        img, m, am = z["image"], z["mask"], z["auto_mask"]
        problems = []
        if img.ndim != 4 or img.shape[0] != 3:
            problems.append(f"image shape {img.shape}")
        if img.dtype != np.float16:
            problems.append(f"image dtype {img.dtype}")
        if min(img.shape[1:]) < 96:
            problems.append(f"VOI smaller than 96: {img.shape}")
        if m.shape != (1,) + img.shape[1:] or am.shape != m.shape:
            problems.append("mask shape mismatch")
        if int(z["subtype_id"]) != sid:
            problems.append(f"subtype_id {int(z['subtype_id'])} != {sid}")
        if not np.isfinite(img.astype(np.float32)).all():
            problems.append("non-finite values")
        if set(np.unique(m)) - {0, 1}:
            problems.append("mask not binary")
        if set(np.unique(am)) - {0, 1}:
            problems.append("auto_mask not binary")
        return f.stem, problems, int(m.sum()), img.shape[1:], int(am.sum())
    except Exception as exc:
        return f.stem, [f"unreadable: {type(exc).__name__}: {exc}"], -1, None, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    root = Path(a.data_root)
    meta = pd.read_csv(root / "splits" / "splits_metadata.csv")
    npz = root / "processed" / "npz_96x96x96"
    jobs = [(npz / f"{r.patient_id}.npz", int(r.subtype_id)) for r in meta.itertuples() if (npz / f"{r.patient_id}.npz").exists()]
    missing = [r.patient_id for r in meta.itertuples() if not (npz / f"{r.patient_id}.npz").exists()]
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(check, jobs))
    bad = [(c, p) for c, p, _, _, _ in res if p]
    empty = [c for c, p, v, _, _ in res if not p and v == 0]
    big = [c for c, p, _, s, _ in res if s is not None and max(s) > 96]
    no_auto = [c for c, p, _, _, av in res if not p and av == 0]
    print(f"auto_mask all-zero: {len(no_auto)} {no_auto[:8]}")
    print(f"expected {len(meta)} | found {len(jobs)} | missing {missing}")
    print(f"contract violations: {len(bad)} {bad[:5]}")
    print(f"empty masks: {len(empty)} {empty[:5]}")
    print(f"cases with an axis > 96 (kept whole): {len(big)}")
    print(meta.merge(pd.DataFrame({'patient_id': [c for c, _, _, _, _ in res]}), on="patient_id")
          .groupby(["split", "tumor_subtype"]).size().unstack(fill_value=0))
    print("npz total GB: %.2f" % (sum(f.stat().st_size for f, _ in jobs) / 1e9))


if __name__ == "__main__":
    main()
