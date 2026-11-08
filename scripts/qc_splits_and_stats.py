"""Sanity checks on splits_metadata.csv + dataset statistics + a visual QC figure for the preprocessed .npz files."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {0: "Luminal", 1: "Triple Negative", 2: "HER2-enriched", 3: "HER2-pure"}


def check_splits(root: Path) -> None:
    meta = pd.read_csv(root / "splits" / "splits_metadata.csv")
    off = pd.read_csv(root / "raw" / "train_test_splits.csv")
    test_ids = set(off["test_split"].dropna().astype(str))
    train_ids = set(off["train_split"].dropna().astype(str))
    print(f"[splits] rows={len(meta)} unique ids={meta.patient_id.nunique()} duplicates={meta.patient_id.duplicated().sum()}")
    print(f"[splits] official csv: train={len(train_ids)} test={len(test_ids)} overlap={len(train_ids & test_ids)}")
    m_test = set(meta.loc[meta.split == "test", "patient_id"])
    m_trval = set(meta.loc[meta.split != "test", "patient_id"])
    print(f"[splits] our test={len(m_test)} not in official test={len(m_test - test_ids)} | "
          f"official test missing from ours={len(test_ids & set(meta.patient_id) - m_test)} | "
          f"our train/val that are official test={len(m_trval & test_ids)}")
    tv = meta[meta.split != "test"]
    print(f"[splits] val fraction of train+val = {(tv.split == 'val').mean():.3f} (target 0.15)")
    tab = pd.crosstab(meta.tumor_subtype, meta.split, normalize="index").round(3)
    print("[splits] split share per subtype:\n", tab)
    print("[splits] split share per dataset:\n", pd.crosstab(meta.dataset, meta.split))
    w = meta.groupby("subtype_id").sample_weight.agg(["min", "max", "mean"]).round(3)
    print("[splits] sample_weight per subtype_id:\n", w)
    print("[splits] sample_weight NaN:", int(meta.sample_weight.isna().sum()), "| non-positive:", int((meta.sample_weight <= 0).sum()))
    print("[splits] subtype share (%):", (meta.tumor_subtype.value_counts(normalize=True) * 100).round(1).to_dict())


def load_stats(args):
    f, sid = args
    z = np.load(f)
    img, mask = z["image"].astype(np.float32), z["mask"][0] > 0
    bg = ~mask
    return {
        "patient_id": f.stem, "subtype_id": sid, "shape": img.shape[1:], "tumor_vox": int(mask.sum()),
        "c1": float(img[0][mask].mean() - img[0][bg].mean()),
        "c2": float(img[1][mask].mean() - img[1][bg].mean()),
        "c3": float(img[2][mask].mean() - img[2][bg].mean()),
        "img_mean": [float(img[c].mean()) for c in range(3)], "img_std": [float(img[c].std()) for c in range(3)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fig", default="outputs/figures/qc_preprocessed_samples.png")
    a = ap.parse_args()
    root = Path(a.data_root)
    check_splits(root)

    meta = pd.read_csv(root / "splits" / "splits_metadata.csv")
    npz = root / "processed" / "npz_96x96x96"
    sample = pd.concat([
        g.sample(min(len(g), max(2, round(a.sample * len(g) / len(meta)))), random_state=0)
        for _, g in meta.groupby("subtype_id")
    ])
    with ThreadPoolExecutor(a.threads) as ex:
        rows = list(ex.map(load_stats, [(npz / f"{r.patient_id}.npz", int(r.subtype_id)) for r in sample.itertuples()]))
    st = pd.DataFrame(rows)
    print(f"\n[stats] sample of {len(st)} cases")
    print("[stats] tumour voxels (mm^3) by subtype:\n", st.groupby("subtype_id").tumor_vox.describe()[["min", "50%", "max"]].round(0))
    print("[stats] tumour-minus-background contrast (Z-score units), mean over cases:",
          st[["c1", "c2", "c3"]].mean().round(2).to_dict())
    print("[stats] cases where subtraction contrast <= post1 contrast:", int((st.c2 <= st.c1).sum()), "of", len(st))
    arr_m, arr_s = np.array(st.img_mean.tolist()), np.array(st.img_std.tolist())
    print("[stats] per-channel mean of image mean/std (expect ~0/~1 after Z-score on the VOI):",
          arr_m.mean(0).round(2), arr_s.mean(0).round(2))
    print("[stats] VOI shapes: all (96,96,96):", int((st["shape"].apply(lambda s: tuple(s) == (96, 96, 96))).sum()),
          "| larger:", int((st["shape"].apply(lambda s: max(s) > 96)).sum()))

    fig, ax = plt.subplots(4, 4, figsize=(13, 13))
    for r, sid in enumerate(range(4)):
        pid = st[st.subtype_id == sid].sort_values("tumor_vox").patient_id.iloc[len(st[st.subtype_id == sid]) // 2]
        z = np.load(npz / f"{pid}.npz")
        s = int(np.argmax(z["mask"][0].sum(axis=(1, 2))))
        for c, t in enumerate(["Post1", "Subtraction", "PE + CLAHE"]):
            ax[r, c].imshow(z["image"][c, s].astype(np.float32), cmap="gray")
            ax[r, c].contour(z["mask"][0, s], colors="r", linewidths=0.8)
            ax[r, c].set_title(f"{t}" if r == 0 else "", fontsize=11); ax[r, c].axis("off")
        ax[r, 3].imshow(z["mask"][0, s], cmap="gray"); ax[r, 3].axis("off")
        ax[r, 3].set_title("Expert mask" if r == 0 else "", fontsize=11)
        ax[r, 0].text(-0.05, 0.5, f"{NAMES[sid]}\n{pid}", transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=10)
    plt.tight_layout()
    out = Path(a.fig); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("figure ->", out.resolve())


if __name__ == "__main__":
    main()
