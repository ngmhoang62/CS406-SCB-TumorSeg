"""Entry point for preprocessing, baseline training, selection and inference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.datasets import MAMA_MIA_Dataset, build_loaders
from src.datasets.mama_mia import LoadNPZd
from src.engine import LinearWarmupCosineAnnealingLR, Trainer
from src.engine.inference import predict_logits, write_prediction
from src.evaluation.metrics import segmentation_metrics
from src.losses import BoundaryAuxiliaryLoss, DiceBCELoss
from src.models import MODEL_VARIANTS, build_model, validate_patch_size
from src.preprocessing import CONTRACT_VERSION, preprocess_case, run_step1_pipeline, save_case
from src.utils.config import load_config, resolve_device, seed_everything


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def preprocessing_signature(cfg):
    paths = cfg["paths"]
    return {
        "contract": CONTRACT_VERSION,
        "preprocessing": cfg["preprocessing"],
        "metadata_sha256": file_digest(paths["splits_metadata_csv"]),
        "raw_images_dir": paths["raw_images_dir"],
        "raw_expert_seg_dir": paths["raw_expert_seg_dir"],
    }


def preprocess(args, cfg):
    paths = cfg["paths"]
    options = cfg["preprocessing"]
    metadata_path = Path(paths["splits_metadata_csv"])
    if args.rebuild_splits or not metadata_path.exists():
        run_step1_pipeline(
            paths["clinical_excel"], paths["official_splits_csv"], metadata_path,
            val_ratio=options["val_ratio"], seed=cfg["project"]["seed"],
        )
    metadata = pd.read_csv(metadata_path)
    if not {"patient_id", "split"}.issubset(metadata):
        raise ValueError("Metadata requires patient_id and split")
    if metadata.patient_id.duplicated().any() or not metadata.split.isin(["train", "val", "test"]).all():
        raise ValueError("Metadata contains duplicate patients or invalid splits")
    destination = Path(paths["processed_npz_dir"])
    signature = preprocessing_signature(cfg)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Preprocessing settings changed; choose a fresh processed_npz_dir")
    destination.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must not look like a completed training dataset.
    save_json(manifest_path, {"signature": signature, "processed": [], "errors": {},
                              "complete": False})
    # Case-insensitive discovery also supports the lowercase MAMA-MIA release.
    folders = {p.name.lower(): p for p in Path(paths["raw_images_dir"]).iterdir() if p.is_dir()}
    masks = {p.name.lower(): p for p in Path(paths["raw_expert_seg_dir"]).glob("*.nii.gz")}
    processed, errors, case_hashes = [], {}, {}
    for index, case_id in enumerate(metadata.patient_id.astype(str), 1):
        try:
            folder = folders.get(case_id.lower(), Path(paths["raw_images_dir"]))
            phases = [p for p in sorted(folder.glob("*_[0-9][0-9][0-9][0-9].nii.gz"))
                      if p.name.rsplit("_", 1)[0].lower() == case_id.lower()]
            expert = masks.get(f"{case_id.lower()}.nii.gz")
            if expert is None:
                raise FileNotFoundError(f"Expert mask missing for {case_id}")
            phase_signature = [
                [str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in phases
            ]
            source_signature = json.dumps({
                "phases": phase_signature,
                "mask": [str(expert.resolve()), expert.stat().st_size, expert.stat().st_mtime_ns],
            }, sort_keys=True)
            target = destination / f"{case_id}.npz"
            reuse = False
            if target.exists() and not args.overwrite:
                try:
                    with np.load(target, allow_pickle=False) as data:
                        reuse = ("source_signature" in data and
                                 str(data["source_signature"]) == source_signature and
                                 str(data["preprocessing_signature"]) == json.dumps(signature, sort_keys=True))
                    if reuse:
                        LoadNPZd("IP3")({"npz_path": str(target)})
                except Exception:
                    # Regenerate an incomplete/corrupt case from its raw images.
                    reuse = False
            if not reuse:
                data = preprocess_case(phases, expert, tuple(options["spacing"]),
                                       options["zscore_eps"], options["output_dtype"])
                data["source_signature"] = np.array(source_signature)
                data["preprocessing_signature"] = np.array(json.dumps(signature, sort_keys=True))
                save_case(target, data)
            case_hashes[case_id] = file_digest(target)
            processed.append(case_id)
            print(f"[{index}/{len(metadata)}] {case_id} {'reused' if reuse else 'processed'}", flush=True)
        except Exception as exc:
            errors[case_id] = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(metadata)}] {case_id} FAILED {exc}", flush=True)
    save_json(manifest_path, {"signature": signature, "processed": processed,
                              "errors": errors, "complete": not errors,
                              "case_sha256": case_hashes})
    if errors:
        raise RuntimeError(f"{len(errors)} cases failed; see {manifest_path}. Rerun to retry.")


def data_signature(cfg):
    manifest = Path(cfg["paths"]["processed_npz_dir"]) / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("Run preprocess first; processed dataset manifest is missing")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not data.get("complete") or data["errors"] or data["signature"] != preprocessing_signature(cfg):
        raise ValueError("Processed dataset is incomplete or its settings/metadata differ")
    expected = set(pd.read_csv(cfg["paths"]["splits_metadata_csv"]).patient_id.astype(str))
    if set(data["processed"]) != expected:
        raise ValueError("Processed manifest does not cover every metadata patient")
    if set(data.get("case_sha256", {})) != expected:
        raise ValueError("Run preprocess again to record all processed case checksums")
    return file_digest(manifest)


def resolve_ip(value, cfg):
    if value != "selected":
        return value
    path = Path(cfg["paths"]["outputs_dir"]) / "selected_input.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection["data_signature"] != data_signature(cfg):
        raise ValueError("Input selection belongs to a different processed dataset")
    chosen_run = Path(cfg["paths"]["outputs_dir"]) / f"M1_{selection['ip']}"
    if file_digest(chosen_run / "best.pt") != selection["checkpoint_sha256"]:
        raise ValueError("Selected M1 checkpoint changed; repeat input selection")
    return selection["ip"]


def run_directory(cfg, model, ip, variant="baseline"):
    name = f"{model}_{ip}"
    if variant != "baseline":
        name += f"_{variant}"
    return Path(cfg["paths"]["outputs_dir"]) / name


def train(args, cfg):
    variant = getattr(args, "variant", "baseline")
    ip = resolve_ip(args.ip, cfg)
    if args.model != "M1":
        selected_ip = resolve_ip("selected", cfg)
        if ip != selected_ip:
            raise ValueError(f"Backbone comparison must use selected input {selected_ip}")
    training = cfg["training"]
    patch_size = tuple(training["patch_size"])
    validate_patch_size(args.model, patch_size, cfg["model"])
    signature = data_signature(cfg)
    device = resolve_device(args.device)
    model = build_model(args.model, ip_mode=ip, config=cfg["model"], patch_size=patch_size,
                        variant=variant).to(device)
    boundary_aux = variant == "boundary"
    boundary_weight = getattr(args, "boundary_weight", 0.2)
    boundary_radius = getattr(args, "boundary_radius", 1)
    criterion = BoundaryAuxiliaryLoss(boundary_weight) if boundary_aux else DiceBCELoss()
    loaders = build_loaders(
        cfg["paths"]["splits_metadata_csv"], cfg["paths"]["processed_npz_dir"],
        ip_mode=ip, batch_size=training["batch_size"], patch_size=patch_size,
        num_workers=cfg["project"]["num_workers"], cache_rate=training["cache_rate"],
        seed=cfg["project"]["seed"],
        positive_patch_probability=training["positive_patch_probability"],
        patches_per_case=training["patches_per_case"],
        boundary_aux=boundary_aux, boundary_radius=boundary_radius,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"],
                                  weight_decay=training["weight_decay"])
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=training["warmup_epochs"], max_epochs=training["epochs"])
    trainer = Trainer(
        model, optimizer, criterion, scheduler, device=device,
        use_amp=training["use_amp"], val_roi_size=patch_size,
        sw_batch_size=training["sw_batch_size"], sw_overlap=training["sw_overlap"],
        prob_threshold=cfg["inference"]["threshold"],
    )
    run_config = {"model": args.model, "ip": ip, "contract": CONTRACT_VERSION,
                  "data_signature": signature, "config": cfg}
    if variant != "baseline":
        run_config["variant"] = variant
    if boundary_aux:
        run_config.update(boundary_weight=boundary_weight, boundary_radius=boundary_radius)
    trainer.fit(loaders["train"], loaders["val"], training["epochs"],
                run_directory(cfg, args.model, ip, variant), run_config=run_config)


def select_input(args, cfg):
    signature = data_signature(cfg)
    candidates = []
    for ip in ("IP1", "IP2", "IP3"):
        directory = run_directory(cfg, "M1", ip)
        summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        if not summary["completed"] or summary["last_epoch"] != cfg["training"]["epochs"]:
            raise ValueError(f"{ip} training is incomplete")
        expected = {"model": "M1", "ip": ip, "contract": CONTRACT_VERSION,
                    "data_signature": signature, "config": cfg}
        if summary["run_config"] != expected or checkpoint["run_config"] != expected:
            raise ValueError(f"{ip} configuration/data differ from this comparison")
        score = float(checkpoint["val_dice"])
        if not np.isfinite(score) or score != summary["best_val_dice"]:
            raise ValueError(f"{ip} summary/checkpoint score mismatch")
        candidates.append({"ip": ip, "val_dice": score,
                           "checkpoint_sha256": file_digest(directory / "best.pt")})
    # Stable tie breaking follows IP1, IP2, IP3.
    best = max(candidates, key=lambda item: item["val_dice"])
    selection = {**best, "data_signature": signature, "candidates": candidates}
    save_json(Path(cfg["paths"]["outputs_dir"]) / "selected_input.json", selection)
    print(json.dumps(selection, indent=2))


def load_run(args, cfg, verify_dataset=True, require_completed=False):
    variant = getattr(args, "variant", "baseline")
    ip = resolve_ip(args.ip, cfg)
    directory = run_directory(cfg, args.model, ip, variant)
    checkpoint_path = directory / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    run = checkpoint["run_config"]
    if run.get("variant", "baseline") != variant:
        raise ValueError("Checkpoint model variant differs from the requested experiment")
    if run["model"] != args.model or run["ip"] != ip or run["contract"] != CONTRACT_VERSION:
        raise ValueError("Checkpoint architecture/input contract differs")
    if verify_dataset and run["data_signature"] != data_signature(cfg):
        raise ValueError("Checkpoint belongs to a different processed dataset")
    if require_completed:
        summary_path = directory / "run_summary.json"
        if not summary_path.exists():
            raise ValueError("Test requires a completed training run")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_epochs = run["config"]["training"]["epochs"]
        if (not summary.get("completed") or summary.get("last_epoch") != expected_epochs
                or summary.get("run_config") != run
                or summary.get("best_val_dice") != checkpoint["val_dice"]):
            raise ValueError("Training summary does not match the completed run and best checkpoint")
    saved_cfg = run["config"]
    device = resolve_device(args.device)
    model = build_model(args.model, ip_mode=ip, config=saved_cfg["model"],
                        patch_size=saved_cfg["training"]["patch_size"], variant=variant)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, saved_cfg, ip, directory, device, file_digest(checkpoint_path)


def evaluate(args, cfg):
    model, saved_cfg, ip, directory, device, checkpoint_hash = load_run(
        args, cfg, require_completed=args.split == "test")
    variant = getattr(args, "variant", "baseline")
    if (args.split == "test" and variant == "baseline" and args.model != "M1"
            and ip != resolve_ip("selected", cfg)):
        raise ValueError("Test requires the input selected using validation")
    settings = saved_cfg["training"]
    threshold = saved_cfg["inference"]["threshold"]
    dataset = MAMA_MIA_Dataset(cfg["paths"]["splits_metadata_csv"],
                               cfg["paths"]["processed_npz_dir"], args.split, ip)
    destination = directory / args.split
    destination.mkdir(parents=True, exist_ok=True)
    lock = {"checkpoint_sha256": checkpoint_hash, "ip": ip, "threshold": threshold,
            "patch_size": settings["patch_size"], "overlap": settings["sw_overlap"],
            "data_signature": data_signature(cfg)}
    if variant != "baseline":
        lock["variant"] = variant
    lock_path = destination / "evaluation_config.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Evaluation output already belongs to a different locked checkpoint")
    save_json(lock_path, lock)
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        with torch.inference_mode():
            logits = predict_logits(model, item["image"].unsqueeze(0), device,
                                    settings["patch_size"], settings["sw_batch_size"],
                                    settings["sw_overlap"])
        probability = logits[0, 0].sigmoid().numpy()
        prediction = (probability >= threshold).astype(np.uint8)
        case_id = str(item["patient_id"])
        with np.load(Path(cfg["paths"]["processed_npz_dir"]) / f"{case_id}.npz") as geometry:
            scores = segmentation_metrics(prediction, item["mask"][0].numpy(),
                                           geometry["spacing"][::-1])
            write_prediction(probability, geometry, destination / f"{case_id}_prob.nii.gz")
            write_prediction(prediction, geometry, destination / f"{case_id}_mask.nii.gz")
        records.append({"patient_id": case_id, "dataset": item.get("dataset", "unknown"), **scores})
        print(f"[{index + 1}/{len(dataset)}] {case_id} Dice={scores['dice']:.4f}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(destination / "metrics.csv", index=False)
    grouped = frame.groupby("dataset")[["dice", "hd95_mm"]].agg(
        lambda values: values.to_numpy().mean())
    grouped["cases"] = frame.groupby("dataset").size()
    grouped.to_csv(destination / "metrics_by_cohort.csv")
    print(frame[["dice", "hd95_mm"]].mean().to_string())


def infer(args, cfg):
    model, saved_cfg, ip, _, device, _ = load_run(args, cfg, verify_dataset=False)
    item = LoadNPZd(ip, require_mask=False)({"npz_path": str(args.case)})
    settings = saved_cfg["training"]
    with torch.inference_mode():
        logits = predict_logits(model, item["image"].unsqueeze(0), device,
                                settings["patch_size"], settings["sw_batch_size"],
                                settings["sw_overlap"])
    probability = logits[0, 0].sigmoid().numpy()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(args.case, allow_pickle=False) as geometry:
        write_prediction(probability, geometry, destination / "probability.nii.gz")
        mask = (probability >= saved_cfg["inference"]["threshold"]).astype(np.uint8)
        write_prediction(mask, geometry, destination / "mask.nii.gz")


def smoke_test(args, cfg):
    from scripts.smoke_baseline import run_smoke_test

    print(json.dumps(run_smoke_test(args.device, improvements=getattr(args, "improvements", False)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke-test", help="Synthetic NIfTI to train/validation/export on CPU or CUDA")
    smoke.add_argument("--device", default=None)
    smoke.add_argument("--improvements", action="store_true", help="Also exercise dual encoders and the boundary head")
    smoke.set_defaults(handler=smoke_test)
    prep = subparsers.add_parser("preprocess", help="All phases to full-volume two-channel cases")
    prep.add_argument("--rebuild-splits", action="store_true")
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(handler=preprocess)
    training = subparsers.add_parser("train", help="Train or resume a baseline or M1 improvement")
    training.add_argument("--boundary-weight", type=float, default=0.2,
                          help="Auxiliary loss coefficient for --variant boundary")
    training.add_argument("--boundary-radius", type=int, default=1,
                          help="Boundary band radius in full-volume voxels")
    selection = subparsers.add_parser("select-input", help="Choose IP by M1 validation Dice")
    selection.set_defaults(handler=select_input)
    evaluation = subparsers.add_parser("evaluate", help="Full-volume Dice/HD95 and NIfTI predictions")
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    inference = subparsers.add_parser("infer", help="Predict a processed case without a mask")
    inference.add_argument("--case", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    for command, handler in ((training, train), (evaluation, evaluate), (inference, infer)):
        command.add_argument("--model", choices=["M1", "M2", "M3"], required=True)
        command.add_argument("--ip", choices=["IP1", "IP2", "IP3", "selected"], default="selected")
        command.add_argument("--device", default=None)
        command.add_argument("--variant", choices=MODEL_VARIANTS, default="baseline")
        command.set_defaults(handler=handler)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    args.handler(args, cfg)


if __name__ == "__main__":
    main()
