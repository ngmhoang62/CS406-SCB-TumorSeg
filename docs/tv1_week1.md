# TV1 – Tuần 1: Thu nhận dữ liệu & chuẩn hóa không gian (Step 1, 2)

Tài liệu này tổng hợp toàn bộ quy trình, script, notebook và kết quả kiểm định dữ liệu mà TV1 đã thực hiện trong Tuần 1 của đề tài **SCB-TumorSeg** (chạy trực tiếp từ thư mục gốc của repository).

## 1. Đầu ra

Dữ liệu `.npz` đã tiền xử lý nằm trên Google Drive (`MyDrive/CS406-SCB-TumorSeg/data_real/processed/npz_96x96x96/`) và được tải về máy cục bộ tại `data/processed/npz_96x96x96/` (~5.6 GB, được ignore trên Git):

| Đầu ra | Nội dung |
|---|---|
| `data/processed/npz_96x96x96/{CASE}.npz` ×1,480 | `image` float16 `(3,D,H,W)` ($D,H,W \ge 96$), `mask` (expert), `auto_mask` (automatic), `subtype_id` |
| `data/splits/splits_metadata.csv` | 1,480 dòng: `patient_id, dataset, tumor_subtype, subtype_id, split, sample_weight, …` |
| `data/raw/` | Ảnh Pre/Post1 + mask gốc, `clinical_and_imaging_info.xlsx`, `train_test_splits.csv` (chính thức) |

Các bảng thống kê kiểm định nhỏ nằm tại `data/splits/` (`auto_vs_expert_dice.csv`, `missing_auto_mask_cases.txt`) và hình trực quan hóa nằm tại `docs/figures/qc_preprocessed_samples.png`.

## 2. Kết quả Kiểm định (QC)

- **1,480 ca hợp lệ** sau khi bỏ 26 ca thiếu `tumor_subtype`: Luminal 747, Triple Negative 499, HER2-enriched 169, HER2-pure 65.
- **Chia tập** (tập Test lấy từ `train_test_splits.csv` chính thức của bài báo, Val = 15% phần Train+Val, stratified theo `(tumor_subtype, dataset)`):
  Train 1,003 · Val 177 · Test 300 (6 ca Test của bài báo bị bỏ vì thiếu nhãn `tumor_subtype`).
- **Data Contract:** 1,480/1,480 file đạt chuẩn 100%, 0 vi phạm, 0 mask expert rỗng; 307 ca có ít nhất một cạnh VOI $> 96$ (giữ nguyên toàn khối VOI để không cắt cụt khối u).
- **Mask automatic:** Có thật trong 1,478/1,480 file. Đối chiếu với điểm Dice do tác giả công bố (`data/raw/preliminary_automatic_segmentations_quality_scores.csv`):
  tương quan $r = 0.996$, 1,425/1,479 ca lệch $\le 0.02$. Dice automatic vs expert: trung vị $\approx 0.90$, trung bình $\approx 0.83$, $\approx 120$ ca $< 0.5$.
- Hình kiểm tra trực quan: `docs/figures/qc_preprocessed_samples.png`.

## 3. Các bước đã làm và Script tương ứng

| Bước | Làm gì | Script / Hàm |
|---|---|---|
| Step 1 | Lọc ca N/A, gộp nhãn, chia Train/Val/Test, tính `sample_weight` | `src/preprocessing/step1_ingestion.py` |
| Tải ảnh | Pre, Post1, mask expert từ XNAT (công khai, không cần đăng nhập); tải theo ca, xử lý ngay, xóa ảnh thô | `scripts/download_process_xnat.py` |
| Xử lý từ Drive | Đọc ảnh thô đã nằm trên Drive, xử lý song song | `scripts/preprocess_from_drive.py` |
| Xử lý trên máy | Copy theo lô từ Drive sang ổ cục bộ → xử lý → ghi `.npz` về Drive; có dừng êm và chạy tiếp | `scripts/preprocess_via_local_copy.py` |
| Step 2 | RAS, resample $1\text{ mm}^3$ (B-Spline ảnh / Nearest mask), cắt VOI $96^3$ + lề 25% từ mask expert | `src/preprocessing/step2_spatial_voi.py` (`standardize_and_extract_voi_fast`: chỉ resample vùng VOI, kết quả giống bản gốc từng voxel) |
| Dữ liệu Synapse | Mask automatic, file chia tập chính thức, vài mask expert để đối chiếu | `scripts/download_synapse_extras.py` |
| Mask automatic | Ghi `auto_mask` vào `.npz` (đồng bộ hệ trục giữa Synapse và XNAT) | `scripts/add_auto_mask_to_npz.py` |
| Ca lỗi trên XNAT | Xử lý trực tiếp từ Synapse (`ISPY2_601489`) | `scripts/process_case_from_synapse.py` |
| Kiểm tra | Data Contract, chia tập, thống kê, hình mẫu | `scripts/qc_npz_contract.py`, `scripts/qc_splits_and_stats.py` |

### Chạy lại (PowerShell, từ thư mục gốc của repo)

```powershell
pip install -r requirements.txt

# A) Tải + xử lý thẳng từ XNAT (cần config trỏ tới thư mục dữ liệu, xem notebooks/colab_download_preprocess.ipynb)
python scripts/download_process_xnat.py --config configs/real_config.yaml --workers 8

# B) Xử lý ảnh thô đã nằm trên Drive (khuyên dùng khi đã có ổ G:)
python scripts/preprocess_via_local_copy.py --data-root "G:\My Drive\CS406-SCB-TumorSeg\data_real" --scratch D:\scb_raw --copy-workers 12 --workers 6 --max-inflight 20

# C) Bổ sung mask automatic (cần SYNAPSE_AUTH_TOKEN, xem mục 5)
python scripts/download_synapse_extras.py --out D:\scb_syn --expert-sample 1506
python scripts/add_auto_mask_to_npz.py --data-root "G:\My Drive\CS406-SCB-TumorSeg\data_real" --auto-dir D:\scb_syn\automatic --workers 3 --force

# D) Kiểm tra Data Contract và thống kê
python scripts/qc_npz_contract.py --data-root "data" --threads 4
python scripts/qc_splits_and_stats.py --data-root "data"
```

Notebook Colab: `notebooks/colab_preprocess_from_drive.ipynb` và `notebooks/colab_download_preprocess.ipynb`.

## 4. Lưu ý & Hạn chế Kỹ thuật

- **Hai nguồn, hai quy ước trục:** Ảnh và mask expert lấy từ XNAT; mask Synapse dùng quy ước trục khác (lật $y, z$ ở `DUKE`/`ISPY2`; hoán vị trục ở `ISPY1`/`NACT`).
  `scripts/add_auto_mask_to_npz.py` tìm phép biến đổi trục khớp mask expert của hai nguồn rồi áp dụng cho mask automatic; không tìm được thì báo lỗi, không đoán.
- **6 ca có `auto_mask` = 0** (`data/splits/missing_auto_mask_cases.txt` liệt kê 2 ca chưa xác minh được: `DUKE_681`, `DUKE_847`; 4 ca còn lại — `DUKE_470`, `ISPY1_1239`, `ISPY2_566011`, `ISPY2_831702` — có mask automatic gốc rỗng). Baseline B0.2 cần loại hoặc xử lý riêng.
- **Mask automatic chỉ là bản sơ bộ:** Nhiều ca tô thừa ($\approx 120$ ca có $\text{Dice} < 0.5$ so với expert).
- **HER2-pure không có ca nào từ ISPY2:** Khi so sánh các subtype có thể bị lẫn với khác biệt giữa các trung tâm; nên báo cáo thêm kết quả theo từng nguồn.
- Vùng VOI được tính từ mask expert (kể cả tập Test); cần nêu rõ khi so với baseline chạy trên toàn ảnh (như B0.3 nnU-Net).

## 5. Bảo mật Token Synapse

Script chỉ đọc token từ biến môi trường `SYNAPSE_AUTH_TOKEN`, không in và không ghi ra file. Dùng token chỉ có quyền **View + Download**, đặt trong cửa sổ PowerShell của bạn, và **thu hồi token sau khi dùng xong** (https://www.synapse.org/#!PersonalAccessTokens:).
