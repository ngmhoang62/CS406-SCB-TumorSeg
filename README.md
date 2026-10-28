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
│   ├── raw/                          # Dữ liệu gốc MAMA-MIA (1,506 ca DCE-MRI, .xlsx, .csv)
│   │   ├── images/                   # {CASE_ID}/{CASE_ID}_0000.nii.gz (Pre), _0001.nii.gz (Post1)
│   │   ├── segmentations/
│   │   │   ├── expert/               # {CASE_ID}.nii.gz (Nhãn chuyên gia Ground Truth)
│   │   │   └── automatic/            # {CASE_ID}.nii.gz (Nhãn tự động sơ bộ cho Baseline B0.2)
│   │   ├── patient_info_files/       # Metadata JSON từng ca (tuổi, dòng máy, từ trường 1.5T/3T, pCR)
│   │   ├── clinical_and_imaging_info.xlsx # Bảng thông tin lâm sàng chuẩn hóa
│   │   ├── preliminary_automatic_segmentations_quality_scores.csv # Bảng điểm chất lượng mặt nạ sơ bộ
│   │   └── train_test_splits.csv     # Bảng phân chia Train/Test gốc (1,200 train / 306 test)
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
│   ├── run_step1_2_3_pipeline.py     # Chạy thực thi toàn bộ Step 1 -> Step 2 -> Step 3
│   └── verify_models_step4_5.py      # Bộ kiểm thử xác minh Step 4, 5 & Smoke test 2-epoch trên GPU/CPU
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

> **Lưu ý quan trọng cho Nhóm A:** Toàn bộ chi tiết kỹ thuật và công thức toán học của **Step 1, Step 2, Step 3** (chuẩn hóa trục **RAS**, nội suy đẳng hướng $1\times 1\times 1\text{ mm}^3$ dùng `B-Spline` cho MRI và `Nearest-Neighbor` cho Mask, trích xuất VOI $+25\%$ lề, lọc nhiễu giữ biên 3D, cắt phân vị $[0.5\%, 99.5\%]$ trên vùng $>0$, và công thức tính 3 kênh) bắt buộc phải tuân thủ đúng mục **STEP 1, STEP 2, STEP 3** trong [`docs/detailed_pipeline.md`](docs/detailed_pipeline.md) (hoặc chạy trực tiếp script [`scripts/run_step1_2_3_pipeline.py`](scripts/run_step1_2_3_pipeline.py)).

Khi hoàn tất tiền xử lý trên 1,480 ca thực tế, dữ liệu đầu ra tại `data/processed/npz_96x96x96/{CASE_ID}.npz` và `data/splits/splits_metadata.csv` phải thỏa mãn đặc tả dưới đây:

### 2.1. Định dạng file `data/processed/npz_96x96x96/{CASE_ID}.npz`
Mỗi file `.npz` nén (`np.savez_compressed`) chứa đúng 4 keys theo thứ tự chiều không gian `(C, D, H, W)` = `(C, Z, Y, X)`:
* **`image`**: `np.ndarray`, shape `(3, D, H, W)`, `dtype=float16` (với $D, H, W \ge 96$: bằng `(3, 96, 96, 96)` nếu VOI $+25\%$ lề $\le 96^3$; giữ nguyên kích thước VOI thực $\ge 96$ ở trục vượt quá $96$ để không cắt mất voxel nào của u và phục vụ Sliding Window $96^3$ ở Step 6):
  * Kênh `0` ($C_1$): $\hat{I}_{\text{post1}}$ — Ảnh pha sau tiêm thứ nhất (`_0001`) chuẩn hóa Z-score.
  * Kênh `1` ($C_2$): $\hat{I}_{\text{sub}} = \text{Z-score}(\max(0, I_{\text{post1}} - I_{\text{pre}}))$ — Ảnh trừ số học triệt tiêu nền mô mỡ.
  * Kênh `2` ($C_3$): $\hat{I}_{\text{PE+CLAHE}}$ — Bản đồ tỷ lệ tăng cường tín hiệu kết hợp cân bằng histogram cục bộ 3D CLAHE chuẩn hóa Z-score.
* **`mask`**: `np.ndarray`, shape `(1, D, H, W)`, `dtype=uint8` — Nhãn chuyên gia nhị phân $\in \{0, 1\}$ (khớp kích thước không gian với `image`).
* **`auto_mask`**: `np.ndarray`, shape `(1, D, H, W)`, `dtype=uint8` — Nhãn tự động sơ bộ nhị phân $\in \{0, 1\}$ phục vụ đánh giá Baseline `B0.2`.
* **`subtype_id`**: `np.int64` scalar — Mã số phân nhóm bệnh học (sau khi đã loại bỏ 26 ca `N/A`):
  * `0`: `Luminal` (gộp từ `luminal`, `luminal_a`, `luminal_b`)
  * `1`: `Triple Negative` (`triple_negative`)
  * `2`: `HER2-enriched` (`her2_enriched`)
  * `3`: `HER2-pure` (`her2_pure`)

### 2.2. Định dạng bảng `data/splits/splits_metadata.csv`
Chứa danh sách các ca hợp lệ kèm các cột bắt buộc: `patient_id, dataset, tumor_subtype, subtype_id, split, sample_weight` (trong đó `split` $\in \{\text{train}, \text{val}, \text{test}\}$ giữ nguyên tập `test` gốc từ `train_test_splits.csv` của bài báo và chia Stratified 85% `train` / 15% `val` theo `(tumor_subtype, dataset)`).

---

## 3. Hướng dẫn Tái lập Dự án & Chạy Kiểm thử (Quick Start cho Thành viên)

Dưới đây là chu trình đầy đủ để một thành viên mới (ví dụ TV4) clone repository về máy, dựng môi trường, tạo dữ liệu giả lập chuẩn hóa và kiểm thử toàn bộ mô hình:

### 3.1. Clone Repository & Cài đặt Môi trường
```bash
# 1. Clone repository về máy
git clone https://github.com/ngmhoang62/CS406-SCB-TumorSeg.git
cd CS406-SCB-TumorSeg

# 2. Khởi tạo môi trường ảo Python (khuyến nghị Python 3.10 - 3.12)
python -m venv venv

# Kích hoạt môi trường:
# - Trên Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# - Trên Linux / macOS:
source venv/bin/activate

# 3. Cài đặt PyTorch phù hợp với phần cứng máy:
# - Nếu chạy trên CPU:
pip install torch
# - Nếu chạy trên GPU NVIDIA: Cài đặt bản PyTorch CUDA tương ứng với phiên bản CUDA của máy
#   (Tham khảo lệnh chuẩn xác tại https://pytorch.org/get-started/locally/)

# 4. Cài đặt các gói phụ thuộc còn lại
pip install -r requirements.txt
```

### 3.2. Sinh Dữ liệu Giả lập (Synthetic Raw Data) & Chạy Pipeline Steps 1–3
```bash
# 1. Sinh 32 ca giả lập chuẩn MAMA-MIA + 2 ca N/A vào data/raw/
python scripts/generate_synthetic_raw.py

# 2. Thực thi toàn bộ pipeline Step 1 -> Step 2 -> Step 3
# Xuất 32 file .npz vào data/processed/npz_96x96x96/ và tạo data/splits/splits_metadata.csv
python scripts/run_step1_2_3_pipeline.py
```

### 3.3. Kiểm thử Kiến trúc Mô hình & Chạy Smoke Test (Steps 4 & 5)
```bash
# Chạy bộ kiểm thử xác minh Zero-Init của FiLM, Gradient Flow, Deep Supervision,
# và thực hiện Smoke Test 2 epochs huấn luyện thực tế trên GPU/CPU cho cả 3 mô hình (M1, M2, M3)
python scripts/verify_models_step4_5.py
```
