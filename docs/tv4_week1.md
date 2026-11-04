# TV4 — Tuần 1: Dataset, Loss & Training Engine

Phạm vi: chỉ Step 6 trong tuần 1 của `docs/task_assignment.md`. Các mô hình và
FiLM dùng trực tiếp từ code TV3 đã có trong repo. Chưa thực hiện TTA, sliding
window, hậu xử lý, benchmark hoặc visualization kết quả của các tuần sau.

## Chạy nhanh trên máy cá nhân

Từ thư mục gốc repo, dùng Python 3.10–3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements-tv4.txt
.\.venv\Scripts\python -m pytest tests/test_tv4_week1.py -q
.\.venv\Scripts\python scripts/execute_tv4_notebook.py
```

Mở `notebooks/TV4_Week1_Training_SmokeTest.ipynb` trong VS Code/Jupyter và chọn
Python của `.venv`, rồi **Run All**. Script trên thực thi notebook bằng kernel mới
và lưu cả output để có thể kiểm tra mà không cần chạy lại. Notebook chứa hướng
dẫn Colab; trên Colab cần tải/clone repo và cài dependencies trước.

## Thành phần được bổ sung

| File | Chức năng |
| --- | --- |
| `src/datasets/mama_mia.py` | `MAMA_MIA_Dataset` dựa trên MONAI `CacheDataset`, kiểm tra contract, 3 IP modes, crop/pad training, augmentation đồng bộ image/mask |
| `src/losses/dice_bce.py` | Dice+BCE trên logits; tổng 3 head có trọng số `1, .5, .25`; resize nhãn nearest-neighbour |
| `src/engine/trainer.py` | AdamW được truyền từ notebook, scheduler warmup–cosine, CUDA bf16 AMP, CPU fp32, kiểm tra NaN/gradient, CSV log và best checkpoint |
| `scripts/generate_synthetic_tv4.py` | Sinh fake NPZ đúng 4 keys cùng CSV đủ 6 cột của nhóm |
| `tests/test_tv4_week1.py` | Kiểm tra dữ liệu, loss, scheduler, gradients và checkpoint |

Cache nằm trước các phép random; augmentation được lấy mẫu lại khi truy cập
training. Validation không augmentation và không cắt bớt VOI. Validation batch=1
để hỗ trợ các ca có kích thước lớn khác nhau. `sample_weight` được giữ trong
metadata; smoke test dùng shuffle đồng đều, chưa bật weighted sampling.

## Dữ liệu giả và cấu hình smoke test

Notebook tạo **12 ca độc lập** (4 train, 4 val, 4 test), mỗi split đủ 4 subtype.
Test split chỉ được tạo để kiểm tra contract; không dùng để chọn checkpoint.
Các file nằm tại `data/synthetic_tv4/`, không ghi đè dữ liệu TV1/TV2.

- NPZ: `image: float16 (3,D,H,W)`, `mask/auto_mask: uint8 (1,D,H,W)`, `subtype_id: int64 scalar`.
- Default `32×32×64` để kiểm tra CPU; đây là ngoại lệ kích thước được ghi rõ
  cho smoke test, không phải dữ liệu production `96³`.
- M3 cần các trục chia hết cho 32 và bottleneck có hơn 1 voxel cho InstanceNorm;
  `32³` gây lỗi ở bottleneck `1³`, vì vậy dùng `32×32×64`.
- M1/M2 dùng channels `(4,8,16,24,32)`, M3 dùng `feature_size=12`. Vẫn chạy
  kiến trúc thật và FiLM thật của TV3, giảm bề rộng để tiết kiệm RAM/thời gian.
- Mỗi mô hình train thật 2 epoch với deep supervision; lưu `best.pt`, `history.csv`
  trong `outputs/tv4_week1/M1`, `M2`, `M3`. Notebook kiểm tra thay đổi trọng số,
  gradient FiLM, số head, finite loss và reload checkpoint.
- Kênh 3 là contrast proxy, không thay thế PE+CLAHE của TV2. Metrics/loss từ
  fake data chỉ chứng minh code chạy, không chứng minh chất lượng phân đoạn.

Sinh fixture đúng kích thước production khi cần:

```powershell
.\.venv\Scripts\python scripts/generate_synthetic_tv4.py --output-dir data/synthetic_tv4_96 --shape 96 96 96
```

## Nối dữ liệu thật và chuyển sang A100

Trong notebook, đặt `SMOKE_TEST=False`, `USE_SYNTHETIC=False`. Cấu hình này dùng:

- `data/processed/npz_96x96x96` và `data/splits/splits_metadata.csv` từ TV1/TV2.
- Patch `96³`, 120 epoch, warmup 5 epoch, LR `3e-4`, weight decay `1e-5`.
- Channels gốc `(32,64,128,256,320)`, Swin feature size 48 và batch size 16
  theo config nhóm. Giảm batch size nếu thiếu VRAM; notebook không cam kết
  cả ba mô hình đều vừa batch 16 trên mọi GPU.
- `ip_mode` chọn `IP1_post1`, `IP2_sub` hoặc `IP3_multi`; `use_film` bật/tắt.
- CUDA hỗ trợ bf16: autocast bf16, không cần GradScaler. CPU/GPU không hỗ trợ:
  fallback fp32. Máy kiểm tra CPU không xác nhận được nhánh AMP trên A100.

Engine tuần 1 validation forward toàn VOI. Với ca lớn hơn 96³, inference cửa sổ
trượt thuộc phần việc tuần 2, cần bổ sung trước khi đánh giá dữ liệu thật quy mô lớn.
Best checkpoint chứa model, optimizer, scheduler, epoch, config và val Dice;
chưa cung cấp cơ chế resume tái lập chính xác RNG/augmentation giữa các phiên.

## Kết quả kiểm tra đã chạy

Môi trường: Windows, Python 3.12.14, PyTorch 2.5.1+cpu, MONAI 1.4.0.
`tests/test_tv4_week1.py`: **10 passed**. Notebook được thực thi từ kernel mới,
đủ 6 code cells, M1/M2/M3 đều **PASS 2 epoch** với IP3 + FiLM, 3 head deep
supervision, gradient FiLM khác 0 và nạp checkpoint thành công. Output và
biểu đồ đã lưu trong notebook. Chưa chạy nhánh CUDA bf16 trên máy này.
