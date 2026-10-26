# SCB-TumorSeg: Subtype-Conditioned 3D Breast Tumor Segmentation Using DCE-MRI

> **Đề tài:** Phân đoạn khối u vú 3D có điều kiện hóa theo phân nhóm bệnh học sử dụng ảnh DCE-MRI  
> **Môn học:** Xử lý ảnh và Ứng dụng (CS406)  
> **Bộ dữ liệu gốc:** MAMA-MIA (*Nature Scientific Data*, 2025 — 1,506 ca DCE-MRI đa trung tâm)  
> **Kế hoạch kỹ thuật chi tiết:** [`docs/detailed_pipeline.md`](docs/detailed_pipeline.md)  
> **Phân công công việc nhóm:** [`docs/task_assignment.md`](docs/task_assignment.md)

---

## 1. Cấu trúc Cây thư mục (Modular ML Project Structure)

```text
Project/
├── configs/
│   └── default_config.yaml           # Cấu hình tổng thể cho toàn bộ 8 Steps
├── data/
│   ├── raw/                          # Dữ liệu gốc MAMA-MIA (hoặc Synthetic Raw .nii.gz, .xlsx, .csv)
│   │   ├── images/                   # {CASE_ID}/{CASE_ID}_0000.nii.gz (Pre), _0001.nii.gz (Post1)
│   │   ├── segmentations/
│   │   │   ├── expert/               # {CASE_ID}.nii.gz (Nhãn chuyên gia Ground Truth)
│   │   │   └── automatic/            # {CASE_ID}.nii.gz (Nhãn tự động sơ bộ cho Baseline B0.2)
│   │   ├── clinical_and_imaging_info.xlsx
│   │   └── train_test_splits.csv
│   ├── processed/
│   │   └── npz_96x96x96/             # Dữ liệu 3D VOI sau Step 2 & Step 3 ({CASE_ID}.npz)
│   └── splits/
│       └── splits_metadata.csv       # Metadata chuẩn hóa & phân chia Stratified Train/Val/Test (Step 1)
├── docs/                             # Tài liệu thiết kế pipeline, phân công & bài báo gốc
├── notebooks/                        # Jupyter Notebooks phục vụ trực quan hóa & demo báo cáo
├── outputs/
│   ├── checkpoints/                  # Trọng số tốt nhất của 18 runs huấn luyện
│   ├── logs/                         # Log huấn luyện (Loss, Validation Dice)
│   ├── predictions/                  # Mặt nạ dự đoán 3D (P0 & P1)
│   ├── figures/                      # Hình ảnh lát cắt 2D/3D & biểu đồ so sánh
│   └── tables/                       # Bảng tổng hợp 39 cấu hình & phân tích theo từng Subtype
├── scripts/
│   ├── generate_synthetic_raw.py     # Sinh dữ liệu thô giả lập chuẩn định dạng MAMA-MIA (32 ca + 2 ca N/A)
│   └── run_step1_2_3_pipeline.py     # Chạy thực thi toàn bộ Step 1 -> Step 2 -> Step 3
└── src/
    ├── preprocessing/                # Step 1, 2, 3 (Ingestion, 3D VOI 96^3, Denoising & 3-Channel Synthesis)
    ├── datasets/                     # Step 3.3 & Step 6 (Dataset, DataLoader, 3D On-the-fly Augmentation)
    ├── models/                       # Step 4 & Step 5 (Multi-Stage FiLM, 3D ResUNet, Attention U-Net, Swin UNETR)
    ├── losses/                       # Step 6 (Deep Supervision Dice + BCE/Focal Loss)
    ├── engine/                       # Step 6 (Trainer bf16 AMP, Sliding Window + 8-Way 3D TTA)
    ├── postprocessing/               # Step 7 (P0 Raw, P1: 3D CCA + Morphological Closing + Hole Filling)
    ├── evaluation/                   # Step 8 (DSC, HD95, Recall, Precision, 39-Row Benchmark)
    ├── visualization/                # Step 8 (2D/3D Contours & Preprocessing Progression)
    └── utils/                        # Quản lý cấu hình YAML, cố định random seed
```

---

## 2. Quy ước "Data Contract" giữa Nhóm A và Nhóm B

Khi Nhóm A hoàn tất tiền xử lý trên dữ liệu thực tế (1,480 ca), chỉ cần ghi đè vào `data/processed/npz_96x96x96/{CASE_ID}.npz` và `data/splits/splits_metadata.csv` theo đúng chuẩn dưới đây:

### 2.1. Định dạng file `data/processed/npz_96x96x96/{CASE_ID}.npz`
Mỗi file `.npz` nén chứa 4 keys:
* **`image`**: `np.ndarray`, shape `(3, 96, 96, 96)`, `dtype=float16`
  * Kênh `0` ($C_1$): $\hat{I}_{\text{post1}}$ — Ảnh pha sau tiêm thứ nhất chuẩn hóa Z-score.
  * Kênh `1` ($C_2$): $\hat{I}_{\text{sub}} = \text{Z-score}(\max(0, I_{\text{post1}} - I_{\text{pre}}))$ — Ảnh trừ số học triệt tiêu nền mô mỡ.
  * Kênh `2` ($C_3$): $\hat{I}_{\text{PE+CLAHE}}$ — Bản đồ tỷ lệ tăng cường tín hiệu kết hợp cân bằng histogram cục bộ 3D CLAHE chuẩn hóa Z-score.
* **`mask`**: `np.ndarray`, shape `(1, 96, 96, 96)`, `dtype=uint8` — Nhãn chuyên gia (`0` hoặc `1`).
* **`auto_mask`**: `np.ndarray`, shape `(1, 96, 96, 96)`, `dtype=uint8` — Nhãn tự động sơ bộ phục vụ đánh giá Baseline `B0.2`.
* **`subtype_id`**: `np.int64` scalar — Mã số phân nhóm bệnh học:
  * `0`: `Luminal` (`luminal`, `luminal_a`, `luminal_b`)
  * `1`: `Triple Negative` (`triple_negative`)
  * `2`: `HER2-enriched` (`her2_enriched`)
  * `3`: `HER2-pure` (`her2_pure`)

### 2.2. Định dạng bảng `data/splits/splits_metadata.csv`
Chứa các cột bắt buộc: `patient_id, dataset, tumor_subtype, subtype_id, split, sample_weight` (trong đó `split` $\in \{\text{train}, \text{val}, \text{test}\}$).

---

## 3. Hướng dẫn Chạy Tạo Synthetic Data & Pipeline Steps 1–3

```powershell
# 1. Sinh dữ liệu thô giả lập (32 ca hợp lệ + 2 ca N/A) vào data/raw/
.\venv\Scripts\python.exe scripts/generate_synthetic_raw.py

# 2. Chạy toàn bộ pipeline Step 1 -> Step 2 -> Step 3 để xuất .npz và splits_metadata.csv
.\venv\Scripts\python.exe scripts/run_step1_2_3_pipeline.py
```
