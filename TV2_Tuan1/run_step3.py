"""Run TV2 Step 3 on Colab/Drive without repeating TV1 Steps 1–2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from src.step3 import synthesize_three_channels

HERE = Path(__file__).resolve().parent
REQUIRED_KEYS = {"image", "mask", "auto_mask", "subtype_id"}
VOI_KEYS = {"pre_voi", "post1_voi"}


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"TV1 metadata missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    needed = {"patient_id", "dataset", "tumor_subtype", "subtype_id", "split"}
    if not rows or not needed.issubset(rows[0]):
        raise ValueError(f"Metadata must have columns {sorted(needed)}")
    ids = [r["patient_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient_id in metadata")
    return rows


def load_voi(path: Path, expected_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if not VOI_KEYS.issubset(data.files):
            raise ValueError(f"{path.name}: need pre_voi and post1_voi")
        pre = data["pre_voi"]
        post = data["post1_voi"]
    if pre.shape != expected_shape or post.shape != expected_shape or pre.ndim != 3:
        raise ValueError(f"{path.name}: VOI shape does not match TV1 image {expected_shape}")
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError(f"{path.name}: non-finite VOI values")
    return pre, post


def load_tv1(path: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != REQUIRED_KEYS:
            raise ValueError(f"{path.name}: expected keys {sorted(REQUIRED_KEYS)}")
        image, mask, auto_mask = data["image"], data["mask"], data["auto_mask"]
        subtype_id = int(data["subtype_id"])
    errors = check_arrays(image, mask, auto_mask, subtype_id, row)
    if errors:
        raise ValueError(f"{path.name}: {'; '.join(errors)}")
    return image, mask, auto_mask, subtype_id


def check_arrays(image, mask, auto_mask, subtype_id, row) -> list[str]:
    errors = []
    if image.ndim != 4 or image.shape[0] != 3 or min(image.shape[1:]) < 96:
        errors.append(f"image shape {image.shape}")
    if image.dtype != np.float16:
        errors.append(f"image dtype {image.dtype}")
    if mask.shape != (1,) + image.shape[1:] or auto_mask.shape != mask.shape:
        errors.append("mask/auto_mask shape mismatch")
    if mask.dtype != np.uint8 or auto_mask.dtype != np.uint8:
        errors.append("mask/auto_mask dtype must be uint8")
    if not np.isfinite(image).all():
        errors.append("non-finite image")
    if not np.isin(mask, [0, 1]).all() or not np.isin(auto_mask, [0, 1]).all():
        errors.append("mask/auto_mask not binary")
    if subtype_id != int(row["subtype_id"]):
        errors.append("subtype_id mismatch")
    if mask.sum() == 0:
        errors.append("empty expert mask")
    return errors


def inspect_case(path: Path, row: dict[str, str]) -> dict:
    result = {k: row[k] for k in ("patient_id", "dataset", "tumor_subtype", "split")}
    result.update(status="invalid", error="", auto_mask_empty="", oversized="")
    if not path.is_file():
        result.update(status="missing", error="file missing")
        return result
    try:
        image, mask, auto_mask, _ = load_tv1(path, row)
        result["status"] = "valid"
        result["auto_mask_empty"] = int(auto_mask.sum() == 0)
        result["oversized"] = int(max(image.shape[1:]) > 96)
        result["shape"] = "x".join(map(str, image.shape))
        for i, name in enumerate(("post1", "sub", "pe_clahe")):
            channel = image[i].astype(np.float32)
            result[f"{name}_mean"] = round(float(channel.mean()), 5)
            result[f"{name}_std"] = round(float(channel.std()), 5)
            tumor = mask[0] > 0
            result[f"{name}_bg_std"] = round(float(channel[~tumor].std()), 5) if (~tumor).any() else ""
            result[f"{name}_tumor_minus_bg"] = round(
                float(channel[tumor].mean() - channel[~tumor].mean()), 5
            ) if (~tumor).any() else ""
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qc_dataset(rows: list[dict[str, str]], source_dir: Path, report_dir: Path) -> dict:
    records = [inspect_case(source_dir / f"{r['patient_id']}.npz", r) for r in rows]
    write_csv(report_dir / "qc_cases.csv", records)
    grouped = []
    for field in ("dataset", "tumor_subtype"):
        for value in sorted({r[field] for r in records}):
            members = [r for r in records if r["status"] == "valid" and r[field] == value]
            if not members:
                continue
            item = {"group_type": field, "group_value": value, "cases": len(members)}
            for name in ("post1", "sub", "pe_clahe"):
                for metric in ("bg_std", "tumor_minus_bg"):
                    key = f"{name}_{metric}"
                    values = [float(r[key]) for r in members if r.get(key) != ""]
                    item[f"mean_{key}"] = round(float(np.mean(values)), 5) if values else ""
            grouped.append(item)
    write_csv(report_dir / "channel_summary_by_group.csv", grouped)
    counts = Counter(r["status"] for r in records)
    summary = {
        "source_dir": str(source_dir), "expected": len(rows),
        "valid": counts["valid"], "missing": counts["missing"],
        "invalid": counts["invalid"],
        "auto_mask_empty": sum(r["auto_mask_empty"] == 1 for r in records),
        "oversized_voi": sum(r["oversized"] == 1 for r in records),
        "by_dataset": dict(Counter(r["dataset"] for r in records if r["status"] == "valid")),
        "by_subtype": dict(Counter(r["tumor_subtype"] for r in records if r["status"] == "valid")),
        "by_split": dict(Counter(r["split"] for r in records if r["status"] == "valid")),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def select_pilot(rows: list[dict[str, str]], size: int) -> list[dict[str, str]]:
    if size < len({r["dataset"] for r in rows}):
        raise ValueError("pilot-size must cover every dataset")
    rng = random.Random(42)
    chosen, seen = [], set()

    def take(group):
        options = [r for r in group if r["patient_id"] not in seen]
        if options and len(chosen) < size:
            row = rng.choice(options)
            chosen.append(row)
            seen.add(row["patient_id"])

    for dataset in sorted({r["dataset"] for r in rows}):
        take([r for r in rows if r["dataset"] == dataset])
    for subtype in sorted({r["tumor_subtype"] for r in rows}):
        if subtype not in {r["tumor_subtype"] for r in chosen}:
            take([r for r in rows if r["tumor_subtype"] == subtype])
    for dataset, subtype in sorted({(r["dataset"], r["tumor_subtype"]) for r in rows}):
        take([r for r in rows if r["dataset"] == dataset and r["tumor_subtype"] == subtype])
    rest = [r for r in rows if r["patient_id"] not in seen]
    rng.shuffle(rest)
    return chosen + rest[:max(0, size - len(chosen))]


def plot_case(image: np.ndarray, mask: np.ndarray, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    area = mask[0].sum(axis=(1, 2))
    z = int(np.argmax(area)) if area.any() else image.shape[1] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (ax, name) in enumerate(zip(axes, ("Post1", "Subtraction", "PE + CLAHE"))):
        plane = image[i, z].astype(np.float32)
        lo, hi = np.percentile(plane, [1, 99])
        ax.imshow(plane, cmap="gray", vmin=lo, vmax=hi)
        if area[z]:
            ax.contour(mask[0, z], levels=[0.5], colors="red", linewidths=0.7)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pilot(rows, tv1_dir: Path, voi_dir: Path, report_dir: Path, config: dict, recompute: bool, size: int) -> None:
    selected = select_pilot(rows, size)
    if not any(max(load_tv1(tv1_dir / f"{r['patient_id']}.npz", r)[0].shape[1:]) > 96 for r in selected):
        selected_ids = {r["patient_id"] for r in selected}
        for row in rows:
            if row["patient_id"] in selected_ids:
                continue
            with np.load(tv1_dir / f"{row['patient_id']}.npz", allow_pickle=False) as data:
                if max(data["image"].shape[1:]) > 96:
                    selected.append(row)
                    break
    comparisons = []
    for row in selected:
        case_id = row["patient_id"]
        old, mask, _, _ = load_tv1(tv1_dir / f"{case_id}.npz", row)
        image = old
        record = {"patient_id": case_id, "dataset": row["dataset"], "tumor_subtype": row["tumor_subtype"]}
        if recompute:
            pre, post = load_voi(voi_dir / f"{case_id}.npz", old.shape[1:])
            image = synthesize_three_channels(pre, post, **config)
            for i, name in enumerate(("post1", "sub", "pe_clahe")):
                record[f"{name}_mean_abs_diff_vs_tv1"] = float(
                    np.abs(image[i].astype(np.float32) - old[i].astype(np.float32)).mean()
                )
        plot_case(image, mask, f"{case_id} | {row['dataset']} | {row['tumor_subtype']}",
                  report_dir / "figures" / f"{case_id}.png")
        comparisons.append(record)
    write_csv(report_dir / "pilot_cases.csv", comparisons)


def build_dataset(rows, tv1_dir: Path, voi_dir: Path, out_dir: Path, report_dir: Path, config: dict) -> dict:
    source = HERE / "src" / "step3.py"
    signature = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode() + source.read_bytes()
    ).hexdigest()
    manifest = out_dir / "tv2_build_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError("Output directory has a different Step 3 code/config version")
    elif list(out_dir.glob("*.npz")):
        raise ValueError("Output directory has NPZ files without TV2 build manifest")
    else:
        manifest.write_text(json.dumps({"signature": signature, "config": config}, indent=2), encoding="utf-8")

    results = []
    for index, row in enumerate(rows, 1):
        case_id = row["patient_id"]
        out = out_dir / f"{case_id}.npz"
        if out.exists():
            checked = inspect_case(out, row)
            if checked["status"] != "valid":
                raise ValueError(f"Existing output {out} is invalid; inspect before resuming")
            results.append({"patient_id": case_id, "status": "skipped", "error": ""})
            continue
        tmp = out_dir / f"{case_id}.tmp.npz"
        try:
            old, mask, auto_mask, subtype_id = load_tv1(tv1_dir / f"{case_id}.npz", row)
            pre, post = load_voi(voi_dir / f"{case_id}.npz", old.shape[1:])
            image = synthesize_three_channels(pre, post, **config)
            problems = check_arrays(image, mask, auto_mask, subtype_id, row)
            if problems:
                raise ValueError("; ".join(problems))
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, image=image, mask=mask, auto_mask=auto_mask,
                                    subtype_id=np.int64(subtype_id))
            os.replace(tmp, out)
            results.append({"patient_id": case_id, "status": "done", "error": ""})
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            results.append({"patient_id": case_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)}", flush=True)
            write_csv(report_dir / "build_cases.csv", results)
    write_csv(report_dir / "build_cases.csv", results)
    return dict(Counter(r["status"] for r in results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Drive data_real directory")
    parser.add_argument("--mode", choices=("auto", "validate", "recompute"), default="auto")
    parser.add_argument("--voi-dir", type=Path, help="Step 2 VOI NPZ files with pre_voi/post1_voi")
    parser.add_argument("--output-dir", type=Path, help="Versioned TV2 output NPZ directory")
    parser.add_argument("--report-dir", type=Path, help="TV2 reports and figures directory")
    parser.add_argument("--pilot-size", type=int, default=12)
    parser.add_argument("--skip-pilot", action="store_true")
    args = parser.parse_args()

    root = args.data_root
    rows = read_metadata(root / "splits" / "splits_metadata.csv")
    tv1_dir = root / "processed" / "npz_96x96x96"
    voi_dir = args.voi_dir or root / "processed" / "voi_step2"
    out_dir = args.output_dir or root / "processed" / "npz_tv2_step3"
    report_dir = args.report_dir or root / "tv2_step3" / "reports"
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    missing_tv1 = [r["patient_id"] for r in rows if not (tv1_dir / f"{r['patient_id']}.npz").is_file()]
    missing_voi = [r["patient_id"] for r in rows if not (voi_dir / f"{r['patient_id']}.npz").is_file()]
    available_voi = len(rows) - len(missing_voi)
    inventory = {"expected": len(rows), "missing_tv1": missing_tv1,
                 "available_voi": available_voi, "missing_voi": missing_voi,
                 "voi_dir": str(voi_dir)}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"TV1 NPZ: {len(rows) - len(missing_tv1)}/{len(rows)} | "
          f"Step 2 VOI: {available_voi}/{len(rows)} | inventory: {report_dir / 'inventory.json'}", flush=True)
    if missing_tv1:
        raise SystemExit("TV1 NPZ dataset is incomplete; see inventory.json")

    recompute = args.mode == "recompute" or (args.mode == "auto" and available_voi == len(rows))
    if args.mode == "recompute" and available_voi != len(rows):
        raise SystemExit("Cannot recompute: not all Step 2 VOI files are available")
    provenance = {
        "requested_mode": args.mode,
        "resolved_mode": "recompute" if recompute else "validate",
        "step3_source_sha256": hashlib.sha256((HERE / "src" / "step3.py").read_bytes()).hexdigest(),
        "config": config,
        "tv1_input_dir": str(tv1_dir),
        "final_output_dir": str(out_dir if recompute else tv1_dir),
    }
    (report_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if args.mode == "auto" and not recompute:
        print("No complete Step 2 VOI handoff; validating existing TV1 3-channel NPZ files.")
    if not args.skip_pilot:
        pilot(rows, tv1_dir, voi_dir, report_dir, config, recompute, args.pilot_size)
    source_dir = tv1_dir
    if recompute:
        build_counts = build_dataset(rows, tv1_dir, voi_dir, out_dir, report_dir, config)
        print("Build:", build_counts)
        if build_counts.get("failed", 0):
            raise SystemExit("Some cases failed; see build_cases.csv, then rerun to resume")
        source_dir = out_dir
    summary = qc_dataset(rows, source_dir, report_dir)
    print("QC:", json.dumps(summary, indent=2))
    if summary["valid"] != len(rows):
        raise SystemExit("QC incomplete; see qc_cases.csv")


if __name__ == "__main__":
    main()
