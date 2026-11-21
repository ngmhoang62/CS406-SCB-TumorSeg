# SCB-TumorSeg: Subtype-Conditioned 3D Breast Tumor Segmentation Using DCE-MRI

> **Đề tài:** Phân đoạn khối u vú 3D có điều kiện hóa theo phân nhóm bệnh học sử dụng ảnh DCE-MRI  
> **Môn học:** Xử lý ảnh và Ứng dụng (CS406)  
> **Bộ dữ liệu gốc:** MAMA-MIA (*Nature Scientific Data*, 2025 — 1,506 ca DCE-MRI đa trung tâm; 1,480 ca hợp lệ có phân nhóm sinh học)  
> **Đặc tả kỹ thuật 8 Steps:** [`docs/detailed_pipeline.md`](docs/detailed_pipeline.md)  
> **Phân công công việc nhóm (4 tuần):** [`docs/task_assignment.md`](docs/task_assignment.md)

---

## 1. Cấu trúc Cây thư mục Dự án (Canonical Project Structure)

```text
SCB-TumorSeg/
├── configs/
│   ├── default_config.yaml           # Cấu hình tổng thể chuẩn cho toàn bộ 8 Steps (paths: data/*)
│   └── step3_config.json             # Cấu hình tham số Step 3 (Denoising, Clipping, 3-Channel Synthesis)
├── data/
│   ├── raw/                          # Dữ liệu gốc MAMA-MIA (.nii.gz bị ignore trên Git; giữ bảng .xlsx/.csv)
│   │   ├── clinical_and_imaging_info.xlsx
│   │   ├── preliminary_automatic_segmentations_quality_scores.csv
│   │   └── train_test_splits.csv
│   ├── processed/
│   │   └── npz_96x96x96/             # 1,480 file {CASE_ID}.npz 3 kênh sau Step 2 & Step 3 (5.60 GB)
│   └── splits/
│       ├── splits_metadata.csv       # Metadata chuẩn hóa 1,480 ca (Train: 1,003 | Val: 177 | Test: 300)
│       ├── auto_vs_expert_dice.csv   # Điểm Dice đối chiếu automatic mask vs expert mask
│       └── missing_auto_mask_cases.txt
├── docs/
│   ├── detailed_pipeline.md          # Đặc tả kỹ thuật chi tiết Steps 1 -> 8
│   ├── task_assignment.md            # Bảng phân công nhiệm vụ 4 thành viên theo 4 tuần
│   ├── tv1_week1.md                  # Báo cáo Tuần 1 của TV1 (Step 1 & Step 2)
│   ├── tv2_week1.md                  # Báo cáo Tuần 1 của TV2 (Step 3 & Thống kê QC 1,480 ca)
│   ├── tv4_week1.md                  # Báo cáo Tuần 1 của TV4 (Step 6 Dataset, Loss & Trainer)
│   ├── figures/                      # Hình ảnh trực quan hóa QC tiền xử lý & mẫu 3 kênh
│   └── reports/tv2_step3/            # Báo cáo thống kê định lượng 3 kênh trên 1,480 ca (.json, .csv)
├── notebooks/
│   ├── colab_download_preprocess.ipynb      # [TV1] Notebook tải dữ liệu & chạy Step 1-2-3 trên Colab
│   ├── colab_preprocess_from_drive.ipynb    # [TV1] Notebook tiền xử lý từ Google Drive & vẽ hình QC
│   ├── TV2_colab_step3.ipynb                # [TV2] Notebook kiểm định & thống kê 3 kênh trên 1,480 ca
│   ├── TV2_tao_VOI_va_3_kenh.ipynb          # [TV2] Notebook trích xuất VOI trung gian & tổng hợp 3 kênh
│   └── TV4_Week1_Training_SmokeTest.ipynb   # [TV4] Notebook Smoke Test huấn luyện M1, M2, M3
├── outputs/
│   ├── checkpoints/                  # Trọng số mô hình (.pt)
│   ├── logs/                         # Log quá trình huấn luyện
│   ├── predictions/                  # Mặt nạ dự đoán 3D (P0 & P1)
│   ├── figures/                      # Hình ảnh lát cắt 2D/3D & biểu đồ đánh giá
│   └── tables/                       # Bảng tổng hợp thực nghiệm 39 cấu hình
├── scripts/
│   ├── run_step1_2_3_pipeline.py     # Pipeline chuẩn chạy liên hoàn Step 1 -> Step 2 -> Step 3
│   ├── qc_npz_contract.py            # Kiểm tra nhanh cấu trúc 1,480 file .npz đa luồng
│   ├── qc_splits_and_stats.py        # Thống kê phân bố tập chia và kích thước VOI
│   ├── run_tv2_step3_qc.py           # Kiểm định chuyên sâu Step 3 & thống kê độ tương phản 3 kênh
│   ├── export_step2_voi.py           # Tái tạo VOI trung gian sau Step 2 và đối chiếu mask
│   └── verify_models_step4_5.py      # Kiểm thử kiến trúc M1, M2, M3 + MultiStageFiLM (Step 4 & 5)
├── src/
│   ├── preprocessing/                # Step 1 (Ingestion), Step 2 (Spatial VOI 96^3), Step 3 (3-Channel Proc)
│   ├── datasets/                     # Step 3.3 & Step 6 (MAMA_MIA_Dataset, DataLoader, 3D Augmentation)
│   ├── models/                       # Step 4 (MultiStageFiLM) & Step 5 (ResUNet3D, AttentionUNet3D, SwinUNETR3D)
│   ├── losses/                       # Step 6 (Deep Supervision DiceBCELoss)
│   ├── engine/                       # Step 6 (Trainer bf16 AMP, LinearWarmupCosineAnnealingLR)
│   ├── postprocessing/               # Step 7 (P0 Raw, P1: 3D CCA + Morphological Closing + Hole Filling)
│   ├── evaluation/                   # Step 8 (DSC, HD95, Recall, Precision, Statistical Tests)
│   ├── visualization/                # Step 8 (2D/3D Contours & Visual Reports)
│   └── utils/                        # Quản lý cấu hình YAML & cố định random seed
└── tests/
    ├── test_tv2_week1_step3.py       # Unit tests công thức toán học Step 3 & QC workflow (TV2)
    ├── test_tv4_week1.py             # Unit tests Dataset, Deep Supervision Loss, Scheduler & Trainer (TV4)
    └── test_week1_real_integration.py # Integration test liên thông TV1 + TV2 + TV3 + TV4 trên dữ liệu thật
```

---

## 2. Tổng kết Công việc Tuần 1 của 4 Thành viên

Trong Tuần 1, cả 4 thành viên đã hoàn thành 100% việc tiền xử lý toàn bộ **1,480 ca bệnh thực tế** của bộ dữ liệu MAMA-MIA và xây dựng xong toàn bộ khung kiến trúc **Steps 1 $\rightarrow$ 6**:

### 2.1. Thành viên 1 (TV1) — Step 1 & Step 2: Lọc dữ liệu, Chia tập Stratified & Chuẩn hóa Không gian 3D VOI
- **Báo cáo chi tiết:** [`docs/tv1_week1.md`](docs/tv1_week1.md)
- **Mã nguồn chính:** [`src/preprocessing/step1_ingestion.py`](src/preprocessing/step1_ingestion.py), [`src/preprocessing/step2_spatial_voi.py`](src/preprocessing/step2_spatial_voi.py), [`scripts/qc_npz_contract.py`](scripts/qc_npz_contract.py), [`scripts/qc_splits_and_stats.py`](scripts/qc_splits_and_stats.py).
- **Kết quả đạt được:**
  - Lọc 26 ca thiếu thông tin phân nhóm (`N/A`) từ 1,506 ca gốc, giữ lại **1,480 ca hợp lệ** thuộc 4 phân nhóm chuẩn (`0: Luminal` — 747 ca, `1: Triple Negative` — 499 ca, `2: HER2-enriched` — 169 ca, `3: HER2-pure` — 65 ca).
  - Bảo toàn tập `Test` chính thức của bài báo (**300 ca**) và chia Stratified 85% `Train` (**1,003 ca**) / 15% `Val` (**177 ca**) theo cặp `(tumor_subtype, dataset)`, đồng thời tính sẵn trọng số `sample_weight` tại [`data/splits/splits_metadata.csv`](data/splits/splits_metadata.csv).
  - Chuẩn hóa hệ trục giải phẫu **RAS**, nội suy đẳng hướng $1.0 \times 1.0 \times 1.0\text{ mm}^3$ (`B-Spline` bậc 3 cho ảnh MRI, `Nearest-Neighbor` cho mặt nạ) và trích xuất cửa sổ VOI tốc độ cao (`_resample_window`) bao trọn khối u + $25\%$ lề với kích thước tối thiểu $96 \times 96 \times 96$ (giữ nguyên 307 ca có ít nhất một trục $> 96$ để không cắt mất mô u).

### 2.2. Thành viên 2 (TV2) — Step 3: Xử lý Ảnh Đa Biểu diễn, Tổng hợp 3 Kênh & Kiểm định Chất lượng
- **Báo cáo chi tiết:** [`docs/tv2_week1.md`](docs/tv2_week1.md) | **Số liệu QC:** [`docs/reports/tv2_step3/`](docs/reports/tv2_step3/) | **Hình mẫu 3 kênh:** [`docs/figures/tv2_step3_samples/`](docs/figures/tv2_step3_samples/)
- **Mã nguồn chính:** [`src/preprocessing/step3_image_proc.py`](src/preprocessing/step3_image_proc.py), [`configs/step3_config.json`](configs/step3_config.json), [`scripts/run_tv2_step3_qc.py`](scripts/run_tv2_step3_qc.py), [`scripts/export_step2_voi.py`](scripts/export_step2_voi.py), [`tests/test_tv2_week1_step3.py`](tests/test_tv2_week1_step3.py).
- **Kết quả đạt được:**
  - Thực thi khử nhiễu giữ biên **3D Non-Local Means** (ước lượng phương sai nhiễu MAD tự động) và cắt ngưỡng phân vị cường độ $[P_{0.5}, P_{99.5}]$ trên vùng foreground ($I > 0$).
  - Tổng hợp tensor 3 kênh `float16` $\mathbf{X} \in \mathbb{R}^{3 \times D \times H \times W}$ gồm:
    1. **Kênh 0 ($C_1 = \hat{I}_{\text{post1}}$):** Ảnh pha sau tiêm đầu tiên chuẩn hóa Z-score.
    2. **Kênh 1 ($C_2 = \hat{I}_{\text{sub}}$):** Ảnh trừ $\max(0, I_{\text{post1}} - I_{\text{pre}})$ chuẩn hóa Z-score nhằm triệt tiêu nền mô mỡ sáng trên T1.
    3. **Kênh 2 ($C_3 = \hat{I}_{\text{PE+CLAHE}}$):** Bản đồ tỷ lệ ngấm thuốc $(I_{\text{post1}} - I_{\text{pre}}) / (I_{\text{pre}} + \tau)$ kết hợp cân bằng histogram cục bộ **3D CLAHE** và chuẩn hóa Z-score.
  - Kiểm định tự động **1,480/1,480 file `.npz` hợp lệ 100%** (`missing: 0`, `invalid: 0`) và thống kê định lượng chứng minh kênh `Subtraction` nâng độ tương phản vùng u so với nền (`tumor_minus_bg`) từ `~3.2` lên **`~4.3`** trên toàn bộ 4 trung tâm (`DUKE`, `ISPY1`, `ISPY2`, `NACT`).

### 2.3. Thành viên 3 (TV3) — Step 4 & Step 5: Điều kiện hóa Đa tầng Multi-Stage FiLM & 3 Kiến trúc Mạng 3D
- **Mã nguồn chính:** [`src/models/film.py`](src/models/film.py), [`src/models/resunet3d.py`](src/models/resunet3d.py), [`src/models/attention_unet3d.py`](src/models/attention_unet3d.py), [`src/models/swin_unetr3d.py`](src/models/swin_unetr3d.py), [`src/models/factory.py`](src/models/factory.py), [`scripts/verify_models_step4_5.py`](scripts/verify_models_step4_5.py).
- **Kết quả đạt được:**
  - Thiết kế module [`MultiStageFiLM`](src/models/film.py) ánh xạ mã phân nhóm `subtype_id` $\in \{0, 1, 2, 3\}$ thành các cặp tham số điều biến $(\gamma^{(s)}, \beta^{(s)})$ ở 3 tầng sâu nhất (Stage 3, Stage 4, Stage 5/Bottleneck) với cơ chế **Zero-Initialization** ($\gamma^{(s)} = \mathbf{0}, \beta^{(s)} = \mathbf{0}$ ở epoch 0 để khởi đầu tương đương mô hình Unconditioned).
  - Xây dựng hoàn chỉnh 3 kiến trúc phân đoạn 3D hỗ trợ bật/tắt `use_film`, tùy chọn số kênh đầu vào (`in_channels` $\in \{1, 2, 3\}$) và tích hợp **Deep Supervision** 3 mức phân giải ($\hat{Y}_1, \hat{Y}_2, \hat{Y}_3$):
    - **`M1` (`ResUNet3D`):** Mạng U-Net 3D 5 tầng với khối Residual, InstanceNorm3d và LeakyReLU (`[32, 64, 128, 256, 320]`).
    - **`M2` (`AttentionUNet3D`):** Tích hợp cổng chú ý không gian 3D (`AttentionGate3D`) tại các nhánh skip connection.
    - **`M3` (`SwinUNETR3D`):** Kiến trúc lai 3D Swin Transformer – CNN với `feature_size=48` và `MultiStageFiLM` gắn tại các tầng đặc trưng sâu.

### 2.4. Thành viên 4 (TV4) — Step 6: Dataset 3 Chế độ Kênh, Deep Supervision Loss & Training Engine
- **Báo cáo chi tiết:** [`docs/tv4_week1.md`](docs/tv4_week1.md) | **Notebook Smoke Test:** [`notebooks/TV4_Week1_Training_SmokeTest.ipynb`](notebooks/TV4_Week1_Training_SmokeTest.ipynb)
- **Mã nguồn chính:** [`src/datasets/mama_mia.py`](src/datasets/mama_mia.py), [`src/losses/dice_bce.py`](src/losses/dice_bce.py), [`src/engine/trainer.py`](src/engine/trainer.py), [`tests/test_tv4_week1.py`](tests/test_tv4_week1.py).
- **Kết quả đạt được:**
  - Xây dựng [`MAMA_MIA_Dataset`](src/datasets/mama_mia.py) trên nền `monai.data.CacheDataset` hỗ trợ trích xuất linh hoạt 3 cấu hình kênh (`IP1_post1` $\rightarrow [0]$, `IP2_sub` $\rightarrow [1]$, `IP3_multi` $\rightarrow [0, 1, 2]$) và chuỗi tăng cường dữ liệu 3D on-the-fly (`RandCropByPosNegLabeld`, `RandFlipd`, `RandRotate90d`, `RandAffined`, `RandGaussianNoised`, `RandScaleIntensityd`).
  - Cài đặt hàm tổn thất đa tỷ lệ [`DiceBCELoss`](src/losses/dice_bce.py) kết hợp trọng số Deep Supervision $(w_1 = 1.0, w_2 = 0.5, w_3 = 0.25)$.
  - Xây dựng [`Trainer`](src/engine/trainer.py) và bộ lập lịch [`LinearWarmupCosineAnnealingLR`](src/engine/trainer.py) hỗ trợ huấn luyện Mixed Precision (`bfloat16` AMP trên GPU CUDA và `float32` trên CPU), lưu trữ tự động `best.pt`, `last.pt` và `history.csv`.

---

## 3. Quy ước Quản lý Nhánh Git & Đóng gói `.zip` cho các Tuần Tiếp theo

Để đảm bảo quá trình hợp nhất code và chia sẻ dữ liệu giữa 4 thành viên trong các tuần tiếp theo luôn sạch sẽ, thống nhất và không xảy ra xung đột:

1. **Giữ nguyên cấu trúc cây thư mục chuẩn trên Git:**
   - Khi làm việc trên nhánh cá nhân (`TV1`, `TV2`, `TV3`, `TV4`), các thành viên thêm/sửa/xóa file trực tiếp trong các thư mục chuẩn (`src/`, `scripts/`, `notebooks/`, `tests/`, `docs/`, `configs/`).
   - **Tuyệt đối không** tạo các thư mục con riêng lẻ theo tên cá nhân ở gốc dự án (như `TV1_Tuan2/`, `TV2_Tuan2/`, `tasks/`).
2. **Quy tắc chia sẻ các file nặng nằm trong `.gitignore` (`data/processed/`, `outputs/checkpoints/`, `outputs/predictions/`, `outputs/logs/`):**
   - Không commit các file nhị phân nặng (`.npz`, `.nii.gz`, `.pt`) lên Git.
   - Khi cần bàn giao dữ liệu đã xử lý, trọng số mô hình (`.pt`), hoặc mặt nạ dự đoán (`outputs/predictions/`) cho thành viên khác: **Nén thành 1 file `.zip` tính từ thư mục gốc `SCB-TumorSeg/`** (sao cho đường dẫn bên trong file `.zip` giữ nguyên cấu trúc tương đối, ví dụ `data/processed/npz_96x96x96/...` hoặc `outputs/checkpoints/...`), tải lên Google Drive và cập nhật link tải vào `README.md`.
   - Thành viên nhận chỉ cần tải file `.zip` về đặt ngay tại thư mục gốc `SCB-TumorSeg/` và giải nén tại chỗ (**Extract Here** hoặc `tar -xf <ten_file>.zip`) là toàn bộ file sẽ tự động nằm đúng vị trí trong cây thư mục local.

---

## 4. Hướng dẫn Tái lập Môi trường & Chạy Kiểm thử (Reproducibility Guide)

### 4.1. Cài đặt Môi trường
```bash
# 1. Clone repository về máy
git clone https://github.com/ngmhoang62/CS406-SCB-TumorSeg.git
cd CS406-SCB-TumorSeg

# 2. Tạo và kích hoạt môi trường ảo Python (Python 3.10 - 3.12)
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 3. Cài đặt PyTorch (chọn bản CPU hoặc CUDA phù hợp với máy tại https://pytorch.org/)
pip install torch

# 4. Cài đặt toàn bộ thư viện phụ thuộc của dự án
pip install -r requirements.txt
```

### 4.2. Chuẩn bị Dữ liệu Tiền xử lý (`.npz` 1,480 Ca)
- **Thư mục Google Drive chung của nhóm:** [Google Drive CS406-IPA](https://drive.google.com/drive/folders/1_R2FBnFZf3N0kXWirW1S9bpbid-Jt14M?usp=sharing)
- **Bộ 1,480 file `.npz` đã qua Steps 1–3 (5.60 GB):** [Thư mục `npz_96x96x96` trên Google Drive](https://drive.google.com/drive/folders/19I0tvr5e0TiXq6oIGmd0bbI6JFaTmwaq?usp=sharing) — tải về và đặt tại `data/processed/npz_96x96x96/`.
- **Bộ dữ liệu thô gốc `mama_mia_raw.zip` (87.8 GB, chỉ cần nếu muốn chạy lại từ Step 1):** [Tải `mama_mia_raw.zip`](https://drive.google.com/file/d/1fvhFojPeb1lm6tvi1hJHUSFyHR4qpsqz/view?usp=sharing) — đặt tại thư mục gốc `SCB-TumorSeg/` và giải nén (`tar -xf mama_mia_raw.zip`).

### 4.3. Chạy Kiểm định Dữ liệu & Kiểm thử Liên thông Toàn Pipeline (Steps 1 $\rightarrow$ 6)
```bash
# 1. Kiểm tra tính toàn vẹn Data Contract trên toàn bộ 1,480 file .npz (Step 1 & 2 - TV1)
python scripts/qc_npz_contract.py --data-root data --threads 16

# 2. Chạy bộ Unit Tests kiểm chứng thuật toán 3 kênh & QC workflow (Step 3 - TV2)
pytest -q tests/test_tv2_week1_step3.py

# 3. Kiểm thử kiến trúc M1, M2, M3 & MultiStageFiLM Zero-Init / Deep Supervision (Step 4 & 5 - TV3)
python scripts/verify_models_step4_5.py

# 4. Chạy bộ Unit Tests cho Dataset, Deep Supervision Loss & Trainer (Step 6 - TV4)
pytest -q tests/test_tv4_week1.py

# 5. Chạy Kiểm thử Tích hợp Liên thông (TV1 + TV2 + TV3 + TV4) trực tiếp trên dữ liệu thật .npz
python tests/test_week1_real_integration.py
```
