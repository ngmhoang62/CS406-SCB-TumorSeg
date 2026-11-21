"""
Download one or more cases (Pre, Post1, expert, automatic) from Synapse and run Step 2 + 3 on them, writing
{CASE}.npz to --out-dir. Used for cases that are broken on the XNAT mirror (e.g. ISPY2_601489) and to validate that
the Synapse and XNAT sources give the same VOI (compare a case with the npz that already exists).

Token: environment variable SYNAPSE_AUTH_TOKEN. Usage:
  python scripts/process_case_from_synapse.py --cases ISPY2_601489 --out-dir D:/scb_syn/out --subtypes-from <xlsx>
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import SimpleITK as sitk
import synapseclient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import standardize_and_extract_voi_fast, synthesize_three_channels
from src.preprocessing.step1_ingestion import SUBTYPE_TO_ID
from src.utils.config import load_config

IMAGES, AUTO, EXPERT = "syn64871114", "syn64874864", "syn64871175"


def child(syn, parent, name):
    return next(c["id"] for c in syn.getChildren(parent) if c["name"] == name)


def fetch(syn, sid, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    return Path(syn.get(sid, downloadLocation=str(dest), ifcollision="overwrite.local").path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clinical-xlsx", required=True)
    ap.add_argument("--work-dir", default="D:/scb_syn/cases")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default_config.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    s2, s3 = cfg["step2_spatial_voi"], cfg["step3_image_processing"]
    syn = synapseclient.Synapse(skip_checks=True)
    syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"], silent=True)

    info = pd.read_excel(a.clinical_xlsx, sheet_name="dataset_info").set_index("patient_id")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    img_children = {c["name"]: c["id"] for c in syn.getChildren(IMAGES)}
    for cid in a.cases:
        work = Path(a.work_dir) / cid
        files = {c["name"]: c["id"] for c in syn.getChildren(img_children[cid])}
        pre = fetch(syn, files[f"{cid}_0000.nii.gz"], work)
        post1 = fetch(syn, files[f"{cid}_0001.nii.gz"], work)
        expert = fetch(syn, child(syn, EXPERT, f"{cid}.nii.gz"), work / "expert")
        auto = fetch(syn, child(syn, AUTO, f"{cid}.nii.gz"), work / "automatic")
        subtype = info.loc[cid, "tumor_subtype"]
        sid = SUBTYPE_TO_ID[str(subtype)] if str(subtype) in SUBTYPE_TO_ID else cfg["step1_ingestion"]["raw_to_canonical_subtype"][str(subtype)]
        if isinstance(sid, str):
            sid = {v: k for k, v in cfg["step1_ingestion"]["subtype_classes"].items()}[sid]

        pre_voi, post1_voi, exp_voi, auto_voi, _ = standardize_and_extract_voi_fast(
            sitk.ReadImage(str(pre), sitk.sitkFloat32), sitk.ReadImage(str(post1), sitk.sitkFloat32),
            sitk.ReadImage(str(expert), sitk.sitkUInt8), sitk.ReadImage(str(auto), sitk.sitkUInt8),
            target_orientation=str(s2["target_orientation"]), target_spacing=tuple(float(x) for x in s2["target_spacing"]),
            voi_size=tuple(int(x) for x in s2["voi_size"]), margin_ratio=float(s2["voi_margin_ratio"]), pad_mode=str(s2["pad_mode"]))
        image = synthesize_three_channels(
            pre_voi=pre_voi, post1_voi=post1_voi,
            apply_n4=bool(s3["apply_n4_bias_correction"]), n4_shrink_factor=int(s3["n4_shrink_factor"]),
            apply_denoise=bool(s3["apply_edge_preserving_denoise"]), denoise_method=str(s3["denoise_method"]),
            nlmeans_patch_size=int(s3["nlmeans_patch_size"]), nlmeans_patch_distance=int(s3["nlmeans_patch_distance"]),
            nlmeans_h_factor=float(s3["nlmeans_h_factor"]), clip_percentiles=tuple(float(x) for x in s3["clip_percentiles"]),
            pe_tau_percentile=float(s3["pe_tau_percentile"]), clahe_clip_limit=float(s3["clahe_clip_limit"]),
            clahe_kernel_size=int(s3["clahe_kernel_size"]), zscore_eps=float(s3["zscore_eps"]), output_dtype=str(s3["output_dtype"]))
        np.savez_compressed(out / f"{cid}.npz", image=image, mask=exp_voi[None].astype(np.uint8),
                            auto_mask=auto_voi[None].astype(np.uint8), subtype_id=np.int64(sid))
        print(cid, "saved", image.shape, "subtype_id", sid, "tumour vox", int(exp_voi.sum()), "auto vox", int(auto_voi.sum()), flush=True)


if __name__ == "__main__":
    main()
