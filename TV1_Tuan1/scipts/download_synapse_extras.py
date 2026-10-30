"""
Download from Synapse (syn60868042) only what the XNAT mirror lacks:
  - segmentations/automatic/{CASE}.nii.gz   (needed for baseline B0.2 and for `auto_mask` in the .npz)
  - train_test_splits.csv                    (official paper split, for verification)
  - a few segmentations/expert masks         (to verify the XNAT `_segmentation` files are the expert masks)

Token: environment variable SYNAPSE_AUTH_TOKEN (never printed or written to disk).
Usage: python scripts/download_synapse_extras.py --out D:/scb_syn [--expert-sample 12] [--workers 8]
"""

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
import synapseclient

AUTO_ID, EXPERT_ID, SPLITS_ID = "syn64874864", "syn64871175", "syn60880777"


def get_one(syn, sid, name, dest_dir: Path):
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return name, "skipped"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            ent = syn.get(sid, downloadLocation=str(dest_dir), ifcollision="overwrite.local")
            if Path(ent.path).name != name:
                Path(ent.path).replace(dest)
            return name, "downloaded"
        except Exception as exc:
            err = exc
            time.sleep(2 ** attempt)
    return name, f"failed: {type(err).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/scb_syn")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--expert-sample", type=int, default=12)
    a = ap.parse_args()
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        sys.exit("SYNAPSE_AUTH_TOKEN is not set")
    syn = synapseclient.Synapse(skip_checks=True)
    syn.login(authToken=token, silent=True)
    out = Path(a.out)

    print("splits:", get_one(syn, SPLITS_ID, "train_test_splits.csv", out), flush=True)
    auto = [(c["id"], c["name"]) for c in syn.getChildren(AUTO_ID)]
    expert = [(c["id"], c["name"]) for c in syn.getChildren(EXPERT_ID)][: a.expert_sample]
    print(f"automatic masks: {len(auto)} | expert sample: {len(expert)}", flush=True)

    jobs = [(i, n, out / "automatic") for i, n in auto] + [(i, n, out / "expert") for i, n in expert]
    counts, failed, t0 = {"downloaded": 0, "skipped": 0, "failed": 0}, [], time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(get_one, syn, i, n, d) for i, n, d in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            name, st = f.result()
            counts["failed" if st.startswith("failed") else st] += 1
            if st.startswith("failed"):
                failed.append(f"{name}\t{st}")
            if k % 100 == 0:
                print(f"[{k}/{len(jobs)}] {counts} {(time.time() - t0) / 60:.1f} min", flush=True)
    if failed:
        (out / "failed.tsv").write_text("\n".join(failed), encoding="utf-8")
    print("FINISHED", counts, flush=True)


if __name__ == "__main__":
    main()
