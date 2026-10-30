# TV1 – Tuần 1: Thu nhận dữ liệu & chuẩn hóa không gian (Step 1, 2)

Thư mục này gom toàn bộ code, notebook và kết quả TV1 đã làm trong Tuần 1 của đề tài SCB-TumorSeg. Thư mục chạy được độc lập
(có sẵn `src/preprocessing`, `src/utils`, `configs`): chạy lệnh từ chính thư mục `TV1_week1`.

## 1. Đầu ra

Dữ liệu nằm trên Google Drive (`MyDrive/CS406-SCB-TumorSeg/data_real/`), **không** nằm trong thư mục này vì nặng (5,6 GB):

| Đầu ra | Nội dung |
|---|---|
| `processed/npz_96x96x96/{CASE}.npz` ×1.480 | `image` float16 `(3,D,H,W)` (D,H,W ≥ 96), `mask` (expert), `auto_mask` (automatic), `subtype_id` |
| `splits/splits_metadata.csv` | 1.480 dòng: `patient_id, dataset, tumor_subtype, subtype_id, split, sample_weight, …` |
| `raw/` | ảnh Pre/Post1 + mask gốc, `clinical_and_imaging_info.xlsx`, `train_test_splits.csv` (chính thức) |

Bản sao các file nhỏ nằm trong `results/` và `figures/` của thư mục này.

## 2. Kết quả

- **1.480 ca** sau khi bỏ 26 ca thiếu `tumor_subtype`: Luminal 747, Triple Negative 499, HER2-enriched 169, HER2-pure 65.
- **Chia tập** (tập Test lấy từ `train_test_splits.csv` chính thức của bài báo, Val = 15% phần Train+Val, stratified theo (subtype, dataset)):
  Train 1.003 · Val 177 · Test 300 (6 ca Test của bài báo bị bỏ vì thiếu nhãn).
- **Data Contract:** 1.480/1.480 file đạt, 0 vi phạm, 0 mask expert rỗng; 307 ca có một cạnh VOI > 96 (giữ nguyên, không cắt u).
- **Mask automatic:** có thật trong 1.478/1.480 file. Đối chiếu với Dice do tác giả công bố (`results/preliminary_automatic_segmentations_quality_scores.csv`):
  tương quan 0,996, 1.425/1.479 ca lệch ≤ 0,02. Dice automatic vs expert: trung vị ≈ 0,90, trung bình ≈ 0,83, ≈ 120 ca < 0,5.
- Hình kiểm tra trực quan: `figures/qc_preprocessed_samples.png`.

## 3. Cấu trúc thư mục

```
TV1_week1/
├── src/preprocessing/    step1_ingestion.py (Step 1), step2_spatial_voi.py (Step 2, có bản resample nhanh),
│                         step3_image_proc.py (Step 3 – phần của TV2, cần để chạy pipeline)
├── src/utils/            đọc config, cố định seed
├── configs/default_config.yaml
├── scripts/              (xem mục 4)
├── notebooks/            colab_download_preprocess.ipynb, colab_preprocess_from_drive.ipynb, colab_bootstrap_cell.py
├── results/              splits_metadata.csv, train_test_splits.csv, missing_auto_mask_cases.txt,
│                         auto_vs_expert_dice.csv, preliminary_automatic_segmentations_quality_scores.csv
└── figures/              qc_preprocessed_samples.png
```

## 4. Các bước đã làm và script tương ứng

| Bước | Làm gì | Script / hàm |
|---|---|---|
| Step 1 | Lọc ca N/A, gộp nhãn, chia Train/Val/Test, `sample_weight` | `src/preprocessing/step1_ingestion.py` |
| Tải ảnh | Pre, Post1, mask expert từ XNAT (công khai, không cần đăng nhập); tải theo ca, xử lý ngay, xóa ảnh thô | `scripts/download_process_xnat.py` |
| Xử lý từ Drive | Đọc ảnh thô đã nằm trên Drive, xử lý song song | `scripts/preprocess_from_drive.py` |
| Xử lý trên máy | Copy theo lô từ Drive sang ổ cục bộ → xử lý → ghi `.npz` về Drive; có dừng êm và chạy tiếp | `scripts/preprocess_via_local_copy.py`, `run_preprocess.ps1`, `stop_preprocess.ps1`, `status_preprocess.ps1` |
| Step 2 | RAS, resample 1 mm³ (B-Spline ảnh / Nearest mask), cắt VOI 96³ + lề 25% từ mask expert | `step2_spatial_voi.py` (`standardize_and_extract_voi_fast`: chỉ resample vùng VOI, kết quả giống bản gốc từng voxel) |
| Dữ liệu Synapse | Mask automatic, file chia tập chính thức, vài mask expert để đối chiếu | `scripts/download_synapse_extras.py` |
| Mask automatic | Ghi `auto_mask` vào `.npz` | `scripts/add_auto_mask_to_npz.py` |
| Ca lỗi trên XNAT | Xử lý trực tiếp từ Synapse (`ISPY2_601489`) | `scripts/process_case_from_synapse.py` |
| Kiểm tra | Data Contract, chia tập, thống kê, hình mẫu | `scripts/qc_npz_contract.py`, `scripts/qc_splits_and_stats.py` |

### Chạy lại (PowerShell, từ thư mục `TV1_week1`)

```powershell
pip install -r requirements.txt

# A) Tải + xử lý thẳng từ XNAT (cần config trỏ tới thư mục dữ liệu, xem notebooks/colab_download_preprocess.ipynb)
python scripts/download_process_xnat.py --config configs/real_config.yaml --workers 8

# B) Xử lý ảnh thô đã nằm trên Drive (khuyên dùng khi đã có ổ G:)
python scripts/preprocess_via_local_copy.py --data-root "G:\My Drive\CS406-SCB-TumorSeg\data_real" --scratch D:\scb_raw --copy-workers 12 --workers 6 --max-inflight 20

# C) Bổ sung mask automatic (cần SYNAPSE_AUTH_TOKEN, xem mục 6)
python scripts/download_synapse_extras.py --out D:\scb_syn --expert-sample 1506
python scripts/add_auto_mask_to_npz.py --data-root "G:\My Drive\CS406-SCB-TumorSeg\data_real" --auto-dir D:\scb_syn\automatic --workers 3 --force

# D) Kiểm tra
python scripts/qc_npz_contract.py --data-root "G:\My Drive\CS406-SCB-TumorSeg\data_real" --threads 4
python scripts/qc_splits_and_stats.py --data-root "G:\My Drive\CS406-SCB-TumorSeg\data_real"
```

Dừng êm khi đang chạy B: `scripts\stop_preprocess.ps1`; chạy tiếp: `scripts\run_preprocess.ps1` (ca đã có `.npz` được bỏ qua).
Notebook Colab: `notebooks/colab_preprocess_from_drive.ipynb` (tải + xử lý trên Colab, lưu vào Drive).

## 5. Lưu ý & hạn chế

- **Hai nguồn, hai quy ước trục.** Ảnh và mask expert lấy từ XNAT; mask Synapse dùng quy ước trục khác (lật y, z ở DUKE/ISPY2; hoán vị trục ở ISPY1/NACT).
  `add_auto_mask_to_npz.py` tìm phép biến đổi trục khớp mask expert của hai nguồn rồi áp dụng cho mask automatic; không tìm được thì báo lỗi, không đoán.
- **6 ca có `auto_mask` = 0** (`results/missing_auto_mask_cases.txt` chỉ liệt kê 2 ca chưa xác minh được: `DUKE_681`, `DUKE_847`;
  4 ca còn lại – `DUKE_470`, `ISPY1_1239`, `ISPY2_566011`, `ISPY2_831702` – có mask automatic gốc rỗng). Baseline B0.2 cần loại hoặc xử lý riêng.
- **Mask automatic chỉ là bản sơ bộ**: nhiều ca tô thừa (≈ 120 ca Dice < 0,5 so với expert).
- **HER2-pure không có ca nào từ ISPY2** → khi so sánh các subtype có thể bị lẫn với khác biệt giữa các trung tâm; nên báo cáo thêm kết quả theo từng nguồn.
- Vùng VOI được tính từ mask expert (kể cả tập Test); cần nêu rõ khi so với baseline chạy trên toàn ảnh (như B0.3 nnU-Net).
- Mask expert = `_segmentation` của XNAT: đã xác minh trùng với mask expert của Synapse (12/12 ca, chỉ khác trục).

## 6. Bảo mật token Synapse

Script chỉ đọc token từ biến môi trường `SYNAPSE_AUTH_TOKEN`, không in và không ghi ra file. Dùng token chỉ có quyền **View + Download**,
đặt trong cửa sổ PowerShell của bạn, và **thu hồi token sau khi dùng xong** (https://www.synapse.org/#!PersonalAccessTokens:).
