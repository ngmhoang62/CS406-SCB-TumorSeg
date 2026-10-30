"""
Copy raw MAMA-MIA cases from a mounted Google Drive (G:) to a local scratch dir in a streaming fashion, preprocess
them locally with many processes, write {CASE}.npz back to the Drive and delete the local raw copy.

Reading NIfTI directly from the Drive stream is slow/unstable under concurrency; plain sequential file copies with
a few threads are robust, and the CPU-heavy part then runs on local disk. Resumable (skips cases with an .npz).

Usage:
  python scripts/preprocess_via_local_copy.py --data-root "G:\\My Drive\\CS406-SCB-TumorSeg\\data_real" \
      --scratch D:\\scb_raw --copy-workers 8 --workers 10
"""

import argparse
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, seed_everything
from src.preprocessing import run_step1_pipeline
from scripts.preprocess_from_drive import _init, process_case

BUF = 8 * 1024 * 1024


def copy_file(src: Path, dst: Path, retries: int = 6) -> bool:
    """Copy with retries; returns False if the source does not exist."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    for attempt in range(retries):
        try:
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo, BUF)
            if dst.stat().st_size == size:
                return True
        except OSError:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"copy failed: {src}")


def copy_case(case_id: str, raw: Path, scratch: Path) -> str:
    for rel in (f"images/{case_id}/{case_id}_0000.nii.gz", f"images/{case_id}/{case_id}_0001.nii.gz",
                f"segmentations/expert/{case_id}.nii.gz"):
        if not copy_file(raw / rel, scratch / rel):
            raise FileNotFoundError(str(raw / rel))
    copy_file(raw / f"segmentations/automatic/{case_id}.nii.gz", scratch / f"segmentations/automatic/{case_id}.nii.gz")
    return case_id


def cleanup_case(case_id: str, scratch: Path) -> None:
    shutil.rmtree(scratch / "images" / case_id, ignore_errors=True)
    for kind in ("expert", "automatic"):
        (scratch / "segmentations" / kind / f"{case_id}.nii.gz").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--scratch", default="D:/scb_raw")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default_config.yaml"))
    ap.add_argument("--copy-workers", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-inflight", type=int, default=30, help="max cases copied but not yet processed")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N cases that still need work")
    a = ap.parse_args()

    cfg = load_config(a.config)
    seed_everything(cfg["project"]["seed"])
    root, scratch = Path(a.data_root), Path(a.scratch)
    raw, npz_dir = root / "raw", root / "processed" / "npz_96x96x96"
    npz_dir.mkdir(parents=True, exist_ok=True)

    meta = run_step1_pipeline(str(raw / "clinical_and_imaging_info.xlsx"), str(raw / "train_test_splits.csv"),
                              str(root / "splits" / "splits_metadata.csv"),
                              val_ratio=float(cfg["step1_ingestion"]["val_ratio"]), seed=int(cfg["project"]["seed"]))
    done = {p.stem for p in npz_dir.glob("*.npz") if not p.stem.endswith(".tmp")}
    todo = [(str(r.patient_id), int(r.subtype_id)) for r in meta.itertuples() if str(r.patient_id) not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"Valid {len(meta)} | already done {len(done)} | to do {len(todo)}", flush=True)

    # clean leftovers of a previous stop/kill: half-written npz and half-copied raw cases
    for p in npz_dir.glob("*.tmp.npz"):
        p.unlink(missing_ok=True)
    shutil.rmtree(scratch / "images", ignore_errors=True)
    shutil.rmtree(scratch / "segmentations", ignore_errors=True)

    stop_file = scratch / "STOP"          # create this file to stop gracefully; delete it to allow a new run
    scratch.mkdir(parents=True, exist_ok=True)
    stop_evt = threading.Event()
    sem = threading.Semaphore(a.max_inflight)
    counts, failed, t0 = {"done": 0, "skipped": 0, "failed": 0}, [], time.time()
    subtype = dict(todo)
    with ThreadPoolExecutor(max_workers=a.copy_workers) as cp, \
            ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(cfg,)) as pp:

        def copier(cid):
            sem.acquire()
            if stop_evt.is_set() or stop_file.exists():
                stop_evt.set()
                sem.release()
                return None
            return copy_case(cid, raw, scratch)

        copy_futs = {cp.submit(copier, cid): cid for cid, _ in todo}
        proc_futs = {}
        n = 0
        pending = set(copy_futs)
        announced = False
        while pending or proc_futs:
            if not announced and stop_file.exists():
                announced = True
                stop_evt.set()
                for f in list(pending):
                    if f.cancel():
                        pending.discard(f)
                print("STOP requested: finishing cases already copied, then exiting. Progress is saved.", flush=True)
            for f in [f for f in list(pending) if f.done()]:
                pending.discard(f)
                cid = copy_futs[f]
                try:
                    if f.result() is None:      # copier skipped it because of STOP
                        continue
                    proc_futs[pp.submit(process_case, cid, subtype[cid], str(scratch), str(npz_dir))] = cid
                except Exception as exc:
                    sem.release(); counts["failed"] += 1; failed.append(f"{cid}\t{exc}"); cleanup_case(cid, scratch)
            for f in [f for f in list(proc_futs) if f.done()]:
                cid = proc_futs.pop(f)
                _, st, err = f.result()
                counts[st] += 1
                if st == "failed":
                    failed.append(f"{cid}\t{err}")
                cleanup_case(cid, scratch)
                sem.release()
                n += 1
                if n % 5 == 0 or st == "failed":
                    el = (time.time() - t0) / 60
                    print(f"[{n}/{len(todo)}] {counts} elapsed={el:.1f} min rate={n / max(el, 1e-6):.2f}/min", flush=True)
            time.sleep(0.5)

    if failed:
        (root / "preprocess_failed_cases.tsv").write_text("\n".join(failed), encoding="utf-8")
        print(f"{len(failed)} failed -> {root / 'preprocess_failed_cases.tsv'} (re-run to retry)")
    print("FINISHED", counts, flush=True)


if __name__ == "__main__":
    main()
