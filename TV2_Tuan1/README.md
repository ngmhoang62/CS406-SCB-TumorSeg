# TV2 Tuần 1 — Step 3: Xử lý ảnh và tổng hợp 3 kênh

Thư mục này chạy độc lập trên Google Colab, dùng dữ liệu TV1 đã đặt trên Google Drive. Nó **không tải dữ liệu, chia tập, chuẩn hóa RAS hay resample/cắt VOI lại**. Chạy tiền xử lý trên Colab CPU; nếu huấn luyện mô hình sau này, dùng Colab GPU và bộ `.npz` đã QC.

## 1. Chuẩn bị trên Google Drive

Đặt toàn bộ thư mục `TV2_Tuan1` vào:

```text
MyDrive/CS406-SCB-TumorSeg/TV2_Tuan1/
```

Dữ liệu TV1 cần có:

```text
MyDrive/CS406-SCB-TumorSeg/data_real/
├── splits/splits_metadata.csv
└── processed/npz_96x96x96/{CASE}.npz    # 3 kênh TV1 đã tạo
```

Mở [`notebooks/TV2_colab_step3.ipynb`](notebooks/TV2_colab_step3.ipynb) bằng **Open with → Google Colaboratory**, chọn `Runtime → Change runtime type → CPU`, rồi chạy từ trên xuống dưới. Nếu thư mục Drive khác, sửa `PROJECT` trong cell đầu. Cell chạy `pip install`, kiểm thử, xử lý/QC và hiển thị kết quả.

## 2. Hai chế độ đầu vào

`MODE = 'auto'` là mặc định:

- Nếu đủ file Step 2 trung gian `processed/voi_step2/{CASE}.npz`, mỗi file chứa `pre_voi` và `post1_voi` cùng shape `(D,H,W)`, `float32`, trùng không gian với `.npz` TV1, chương trình chạy Step 3 cho toàn bộ 1.480 ca và lưu bộ mới.
- Nếu thiếu VOI trung gian, chương trình kiểm tra bộ 3 kênh TV1 hiện có. Nó **không giả vờ tính lại** từ ba kênh đã z-score/CLAHE, vì không thể khôi phục chính xác Pre/Post1. Xem `inventory.json` để biết có bao nhiêu VOI.

Có thể đổi `MODE` thành `validate` để chỉ QC, hoặc `recompute` để bắt buộc có đủ VOI. Nếu TV1 còn giữ các mảng `pre_voi`, `post1_voi` trong lúc chạy Step 2, TV1 có thể bàn giao bằng cách lưu riêng từng ca:

```python
np.savez_compressed(voi_dir / f"{case_id}.npz",
                    pre_voi=pre_voi.astype(np.float32),
                    post1_voi=post1_voi.astype(np.float32))
```

Hai mảng phải có cùng orientation, spacing, crop và padding như `image` trong `.npz` TV1. File này chỉ chứa VOI đã được TV1 xử lý, không phải ảnh `.nii.gz` gốc.

## 3. Phép xử lý và đầu ra

[`config.json`](config.json) giữ tham số Step 3 hiện tại của TV1: N4 tắt mặc định, 3D non-local means bật, clipping 0,5–99,5% trên voxel dương, rồi:

| Kênh | Cách tạo |
|---|---|
| `image[0]` | Z-score Post1 |
| `image[1]` | Z-score `max(0, Post1 − Pre)` sau xử lý |
| `image[2]` | Z-score của CLAHE 3D trên `max(0, Post1 − Pre) / (Pre + τ)`; `τ = max(P5(Pre>0), 1)` |

Đầu ra có `image` dạng `(3,D,H,W)` và `float16`; `mask`, `auto_mask`, `subtype_id` lấy nguyên từ TV1. VOI có cạnh >96 được giữ nguyên. Bộ TV1 không bị ghi đè.

```text
data_real/
├── processed/npz_tv2_step3/{CASE}.npz   # chỉ khi có đủ VOI trung gian
└── tv2_step3/reports/
    ├── inventory.json
    ├── provenance.json
    ├── pilot_cases.csv
    ├── figures/{CASE}.png
    ├── build_cases.csv                   # chỉ khi tái xử lý
    ├── qc_cases.csv
    ├── channel_summary_by_group.csv
    └── qc_summary.json
```

`qc_cases.csv` ghi lỗi và thống kê từng ca; `channel_summary_by_group.csv` tổng hợp độ nhiễu nền và chênh cường độ u/nền theo nguồn, subtype. `qc_summary.json` ghi số ca hợp lệ, thiếu, lỗi, `auto_mask` rỗng, VOI >96 và số ca theo nguồn/subtype/split. Với bộ TV1 báo cáo, dự kiến metadata có 1.480 ca (Train 1.003, Val 177, Test 300), nhưng notebook phải kiểm tra dữ liệu thật trên Drive. `auto_mask` rỗng được báo riêng, không được thay bằng expert mask.

## 4. Chạy tiếp sau khi Colab ngắt

Chạy lại notebook với cùng `MODE` và `config.json`. File TV2 đã hợp lệ được bỏ qua; file được ghi qua `.tmp.npz` rồi đổi tên. `tv2_build_manifest.json` trong thư mục đầu ra gắn kết mã xử lý với config; nếu đổi mã/config, dùng một **thư mục đầu ra mới** qua `--output-dir`, không trộn phiên bản. Lỗi từng ca nằm trong `build_cases.csv`.

Lệnh tương đương trong Colab:

```bash
python /content/drive/MyDrive/CS406-SCB-TumorSeg/TV2_Tuan1/run_step3.py \
  --data-root /content/drive/MyDrive/CS406-SCB-TumorSeg/data_real --mode auto
```

## 5. Giới hạn hiện tại

- Repo chỉ có bảng metadata và hình QC mẫu; 1.480 file `.npz` thật nằm trên Drive của nhóm, nên chưa thể chạy QC toàn bộ trong môi trường phát triển này.
- Bài báo MAMA-MIA không quy định bộ ba kênh; đây là phương án xử lý ảnh của project. Dữ liệu TV1 đã dùng expert mask để định vị VOI cả ở tập Test; cần nêu điều đó khi đánh giá mô hình sau này.
