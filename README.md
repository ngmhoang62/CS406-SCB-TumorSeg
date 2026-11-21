# SCB-TumorSeg

Phân đoạn u vú 3D từ DCE-MRI, gồm baseline và hai cải tiến đầu tiên theo [plan của nhóm](docs/scb_tumorseg_image_preprocessing_plan.md) và [hình pipeline](docs/figures/scb_tumorseg_plan_infographic.png).

## Bàn giao cho người chạy A100

Các lệnh dưới đây chạy từ thư mục gốc repo với Python 3.10–3.12. Môi trường GPU cần có PyTorch CUDA tương thích driver. Hướng dẫn chọn bản PyTorch nằm tại [trang cài đặt chính thức](https://pytorch.org/get-started/locally/).

### 1. Cài thư viện và kiểm tra luồng chạy

~~~bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pytest -q -rs
python scripts/run_pipeline.py smoke-test --device cuda
~~~

smoke-test tự tạo NIfTI tổng hợp, chạy tiền xử lý, lấy patch train, forward/backward và một optimizer step, quét toàn ảnh rồi xuất/đọc lại mask NIfTI. Nó kiểm tra M1 với cả IP1/IP2/IP3, M2 và M3 với IP3. Số feature được giảm trong phép thử để kiểm tra luồng chạy nhanh. M1 vẫn có sáu tầng. Kết quả in tên GPU, trạng thái bf16 và từng cấu hình đã PASS. Dữ liệu tổng hợp nằm trong thư mục tạm và được dọn khi kết thúc.

### 2. Đặt dữ liệu và chỉnh đường dẫn

~~~text
data/raw/images/<CASE_ID>/<CASE_ID>_0000.nii.gz
data/raw/images/<CASE_ID>/<CASE_ID>_0001.nii.gz
data/raw/images/<CASE_ID>/<CASE_ID>_0002.nii.gz
...
data/raw/segmentations/expert/<CASE_ID>.nii.gz
data/splits/splits_metadata.csv
~~~

Chỉnh paths trong [configs/default_config.yaml](configs/default_config.yaml) theo nơi đặt dữ liệu. File pha có thể nằm trực tiếp trong raw_images_dir hoặc trong thư mục từng ca. Discovery hỗ trợ tên ca viết hoa/thường. Hai pha bắt buộc là Pre (_0000) và Post1 (_0001). Các pha sau có sẵn của cùng ca góp vào thống kê z-score.

Split đã chốt gồm 1.003 train, 177 validation và 300 test. Luồng mặc định dùng lại CSV này. Có thể tạo lại split từ Excel và split chính thức bằng cờ --rebuild-splits.

### 3. Tiền xử lý từ ảnh gốc

~~~bash
python scripts/run_pipeline.py preprocess
~~~

Kết quả nằm ở data/processed/full_volume_v1. Mỗi file NPZ lưu hai kênh ảnh toàn trường nhìn, expert mask cùng lưới, spacing/origin/direction, thống kê chuẩn hóa và thông tin nguồn. manifest.json ghi các ca hoàn tất, lỗi và SHA-256 từng file kết quả. Chạy lại cùng lệnh để dùng lại các file hợp lệ và thử lại ca thất bại. Khi tiền xử lý lại làm thay đổi dữ liệu, chữ ký dataset cũng thay đổi để phân biệt với các checkpoint đã train.

Bộ npz_96x96x96 trước đây lưu VOI ba kênh. Bộ full_volume_v1 được tạo từ NIfTI theo pipeline hiện tại. Loader kiểm tra contract_version trước khi đọc.

### 4. So IP1/IP2/IP3 trên M1 và chọn bằng validation

~~~bash
python scripts/run_pipeline.py train --model M1 --ip IP1 --device cuda
python scripts/run_pipeline.py train --model M1 --ip IP2 --device cuda
python scripts/run_pipeline.py train --model M1 --ip IP3 --device cuda
python scripts/run_pipeline.py select-input
~~~

Mỗi run giữ nguyên loại đầu vào từ train đến validation/test. select-input kiểm tra ba run M1 đã hoàn tất, dùng cùng dữ liệu/cấu hình và chọn best validation Dice cao nhất. Nếu bằng điểm, thứ tự ưu tiên là IP1, IP2 rồi IP3. Kết quả nằm ở outputs/baseline/selected_input.json, kèm điểm ba cấu hình và hash checkpoint.

### 5. Giữ đầu vào đã chọn để so backbone

~~~bash
python scripts/run_pipeline.py train --model M2 --ip selected --device cuda
python scripts/run_pipeline.py train --model M3 --ip selected --device cuda
~~~

M1 với đầu vào đã chọn được dùng lại từ bước 4. Mỗi thư mục outputs/baseline/<MODEL>_<IP> chứa best.pt, last.pt, history.csv và run_summary.json. best.pt được chọn bằng validation Dice trên toàn ảnh. Chạy lại lệnh train tương ứng để resume.

### 6. Chấm test và xuất kết quả

~~~bash
python scripts/run_pipeline.py evaluate --model M1 --ip selected --split test --device cuda
python scripts/run_pipeline.py evaluate --model M2 --ip selected --split test --device cuda
python scripts/run_pipeline.py evaluate --model M3 --ip selected --split test --device cuda
~~~

Test yêu cầu run đã hoàn tất. So ba backbone dùng đầu vào được chọn bằng validation. Với các thí nghiệm M1, có thể chỉ rõ IP để chấm baseline và cải tiến trên cùng loại đầu vào. Kiến trúc, patch, overlap và ngưỡng được đọc từ checkpoint. evaluation_config.json khóa hash checkpoint và cấu hình của lần chấm.

| File trong thư mục từng run/test | Nội dung |
| --- | --- |
| <CASE_ID>_prob.nii.gz | Bản đồ xác suất u toàn trường nhìn Post1 |
| <CASE_ID>_mask.nii.gz | Mask nhị phân thô trên lưới 1 mm |
| metrics.csv | Dice và HD95 từng ca |
| metrics_by_cohort.csv | Trung bình theo cohort và số ca |
| evaluation_config.json | Checkpoint và cấu hình đã dùng |

Cả probability và mask giữ spacing/origin/direction của lưới Post1 sau resample.

## Đối chiếu với hình pipeline

| Bước | Source hiện tại |
| --- | --- |
| Pre, Post1 và các pha sau | preprocess_case đọc các pha đánh số, bắt buộc có Pre và Post1 |
| RAS | SimpleITK DICOMOrient |
| Resample 1 mm toàn ảnh | Lưới tham chiếu phủ toàn Post1, nội suy tuyến tính cho ảnh |
| Expert mask cùng lưới | Resample nearest-neighbor trên đúng lưới Post1 |
| Z-score chung | Một cặp mean/std từ toàn bộ voxel của tất cả các pha đã resample |
| IP1 | C1 = Z(Post1), một kênh |
| IP2 | C2 = Z(Post1) − Z(Pre), giữ dấu, một kênh |
| IP3 | Ghép [C1, C2], hai kênh |
| Train | Patch 3D, mask hỗ trợ lấy tâm patch ở vùng u hoặc nền và làm nhãn học |
| Validation/test | Sliding window toàn trường nhìn, ghép logits Gaussian, mask dùng để chấm |
| Chọn đầu vào rồi so backbone | Ba run M1 → select-input → M2/M3 với --ip selected |
| M1 | PlainConvUNet sáu tầng của thư viện dùng trong nnU-Net v2 |
| M2 | MONAI Attention U-Net 3D, attention gate tại skip connections |
| M3 | MONAI Swin UNETR, encoder Swin và decoder tích chập |
| Đầu ra tạm | Một đầu dự đoán chính → xác suất → ngưỡng nhị phân → Dice/HD95 |

Mốc triển khai gồm baseline đến mask thô, hai encoder cho IP3 và đầu phụ dự đoán biên u. Plan còn mô tả các bước cSE, ngữ cảnh toàn ảnh và metadata cho giai đoạn tiếp theo.

## Hai cải tiến đầu tiên trên M1

Các thí nghiệm dưới đây dùng lại full_volume_v1 và cùng split. Mỗi cải tiến được train thành một run riêng, với M1 + IP3 baseline làm đối chứng.

| --variant | Cấu trúc |
| --- | --- |
| baseline | Ghép hai kênh IP3 tại đầu vào, một encoder PlainConvUNet |
| dual-add | Hai encoder sáu tầng độc lập cho Post1 và subtraction, cộng feature maps cùng mức, một decoder chung |
| dual-gated | Hai encoder như dual-add, học hai trọng số theo vị trí ở mỗi mức rồi cộng có trọng số |
| boundary | Encoder baseline, đầu phân đoạn chính và đầu biên 1×1×1 sau block decoder cuối |

Ở dual-gated, hai trọng số được tạo bằng Conv3d 1×1×1 và softmax theo nhánh. F = w_Post1 × F_Post1 + w_sub × F_sub, với tổng hai trọng số bằng 1. Cổng được khởi tạo ở mức 0.5 cho mỗi nhánh. Fusion diễn ra tại các skip và bottleneck.

Ở boundary, nhãn biên là morphological gradient từ expert mask toàn ảnh trên lưới 1 mm. Kernel 3×3×3 mặc định tạo một dải cả phía trong và phía ngoài biên u. Nhãn được tạo trước khi cắt patch, rồi crop/flip/rotate đồng bộ với ảnh và mask. Viền scan được xử lý bằng giá trị lân cận để tránh tạo biên do giới hạn ảnh.

L_train = DiceBCE(mask) + λ × DiceBCE(boundary). Giá trị khởi đầu λ = 0.2 và radius = 1 voxel có thể chỉnh bằng --boundary-weight và --boundary-radius. Validation loss, Dice và HD95 dùng đầu phân đoạn chính. Model ở chế độ eval trả tensor phân đoạn chính, và đầu biên được gọi trong chế độ train.

### Kiểm tra và train riêng từng cải tiến

~~~bash
python scripts/run_pipeline.py smoke-test --device cuda --improvements
python scripts/run_pipeline.py train --model M1 --ip IP3 --variant dual-add --device cuda
python scripts/run_pipeline.py train --model M1 --ip IP3 --variant dual-gated --device cuda
python scripts/run_pipeline.py train --model M1 --ip IP3 --variant boundary --boundary-weight 0.2 --boundary-radius 1 --device cuda
~~~

### Chấm cùng IP3 và cùng tập test

~~~bash
python scripts/run_pipeline.py evaluate --model M1 --ip IP3 --variant baseline --split test --device cuda
python scripts/run_pipeline.py evaluate --model M1 --ip IP3 --variant dual-add --split test --device cuda
python scripts/run_pipeline.py evaluate --model M1 --ip IP3 --variant dual-gated --split test --device cuda
python scripts/run_pipeline.py evaluate --model M1 --ip IP3 --variant boundary --split test --device cuda
~~~

Chọn cấu hình và checkpoint bằng validation trước khi chấm test. Các run có tên M1_IP3, M1_IP3_dual-add, M1_IP3_dual-gated và M1_IP3_boundary. Variant, hệ số loss biên và radius được lưu trong checkpoint, hỗ trợ resume và dựng lại model khi evaluate/infer. Thay đổi hệ số hoặc radius của một run đã có yêu cầu outputs_dir riêng như các thay đổi cấu hình thực nghiệm khác.

Suy luận một ca từ run cải tiến bằng cùng tham số --variant.

~~~bash
python scripts/run_pipeline.py infer --model M1 --ip IP3 --variant boundary --case data/processed/full_volume_v1/DUKE_001.npz --output outputs/baseline/predict_boundary_DUKE_001 --device cuda
~~~

## Các lựa chọn triển khai

- M1 có sáu tầng [32, 64, 128, 256, 320, 320], hai convolution mỗi tầng, stride 2 giữa các tầng, khởi tạo He theo PlainConvUNet.
- Ba model dùng trainer chung của nhóm với AdamW, warmup/cosine, Dice+BCE và một đầu ra nhị phân. Khi báo cáo, mô tả M1 là backbone PlainConvUNet từ nnU-Net v2 với giao thức train này.
- Mean/std được tính trên toàn bộ voxel của các pha đã resample, gồm nền và vùng đệm ngoài ảnh. Expert mask được xử lý riêng với các kênh ảnh.
- Validation/test ghép toàn scan trên RAM, các cửa sổ được xử lý trên GPU. Cách bố trí này giới hạn lượng dữ liệu toàn ảnh phải giữ trong VRAM.
- Patch 96³, batch_size 2, một patch mỗi ca, tỷ lệ tâm patch ở vùng u 0.5, overlap 0.5 và 120 epoch là cấu hình khởi đầu trong YAML. Các tham số này có thể hiệu chỉnh theo dữ liệu, VRAM và validation.
- Batch đưa vào model có batch_size × patches_per_case patch. Cache mặc định 0 để tải từng ca. num_workers có thể chỉnh theo CPU/RAM của máy chạy.
- Threshold mặc định 0.5, được dùng nhất quán để chấm validation, chọn checkpoint và tạo mask khi test.

Các thí nghiệm độc lập có thể dùng một bản YAML riêng. Mọi đường dẫn tương đối được tính từ thư mục gốc repo. Truyền --config trước tên lệnh.

~~~bash
python scripts/run_pipeline.py --config configs/default_config.yaml train --model M1 --ip IP1 --device cuda
~~~

Resume yêu cầu cấu hình khớp checkpoint. Đặt outputs_dir mới cho một bộ cấu hình khác để giữ riêng kết quả thực nghiệm.

## Suy luận một ca

~~~bash
python scripts/run_pipeline.py infer --model M1 --ip selected --case data/processed/full_volume_v1/DUKE_001.npz --output outputs/baseline/predict_DUKE_001 --device cuda
~~~

Loader suy luận đọc ảnh và hình học. Có thể tạo NPZ của ca mới bằng preprocess_case với expert_path=None rồi save_case. Khi chuyển riêng một run sang máy khác, chỉ rõ IP1, IP2 hoặc IP3 theo checkpoint.

## Chỉ số và quy ước

Dice tính trên mask nhị phân toàn ảnh. HD95 dùng khoảng cách bề mặt vật lý theo mm và lấy giá trị lớn hơn giữa hai phân vị 95% có hướng. Khi cả hai mask rỗng, Dice=1 và HD95=0. Khi chỉ một mask rỗng, Dice=0 và HD95=inf. Giá trị inf được giữ trong CSV và thống kê.

## Cấu trúc chính

| Đường dẫn | Nội dung |
| --- | --- |
| configs/default_config.yaml | Đường dẫn và tham số |
| scripts/run_pipeline.py | Entry point cho smoke-test, preprocess, train, select-input, evaluate, infer |
| scripts/smoke_baseline.py | Phép thử tổng hợp trên CPU/CUDA |
| src/preprocessing/ | Chia tập và tiền xử lý toàn ảnh |
| src/datasets/ | Đọc dữ liệu và lấy patch train |
| src/models/ | M1, M2, M3 và factory |
| src/engine/ | Train, checkpoint và sliding-window inference |
| src/evaluation/metrics.py | Dice và HD95 |
| tests/test_baseline.py | Kiểm thử baseline và giao thức thực nghiệm |
| tests/test_improvements.py | Hai encoder, nhãn/loss biên, train/resume và đánh giá các variant |
| legacy/ | Script, notebook, cấu hình và báo cáo của các vòng trước |
| MAMA-MIA/ | Mã nguồn tham khảo từ MAMA-MIA |

## Trạng thái kiểm chứng tại máy sửa source

Ngày 21/11/2026, 46 kiểm thử CPU đã đạt, bao gồm công thức chuẩn hóa, chọn kênh, patch train, toàn trường nhìn ở validation/test, sliding window, checkpoint/resume, chọn IP, khóa test và các chỉ số. M1 đã qua forward/backward với dynamic-network-architectures 0.3.1 và sáu tầng. M2 đã qua train/validation/resume cùng các entry point train/evaluate. M3 đã qua forward/backward.

Hai encoder đã được kiểm tra tham số độc lập và gradient ở cả hai nhánh. Nhãn biên đã được kiểm tra vị trí trước/sau crop và augmentation. Đầu biên có gradient khi train, trong khi forward ở chế độ eval và sliding window trả đầu phân đoạn chính. Cả ba variant dual-add, dual-gated và boundary đã qua train, resume, nạp checkpoint và entry point evaluate trên dữ liệu tổng hợp.

Máy sửa source còn thiếu SimpleITK, dữ liệu NIfTI thật và CUDA. Kiểm thử hình học NIfTI và phép thử đầy đủ smoke-test còn chờ môi trường có các thành phần đó. Người chạy A100 có thể dùng các lệnh ở bước 1 để kiểm tra ngay trên môi trường thực thi.
