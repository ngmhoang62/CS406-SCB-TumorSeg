# SCB-TumorSeg

Đồ án phân đoạn khối u vú 3D từ ảnh DCE-MRI của bộ dữ liệu MAMA-MIA. Dự án hỗ trợ ba cấu hình đầu vào, ba backbone và các biến thể M1 với hai encoder hoặc đầu phụ dự đoán biên.

Chi tiết phương pháp nằm trong [đặc tả pipeline](docs/scb_tumorseg_image_preprocessing_plan.md).

## Cài đặt

Sử dụng Python 3.10–3.12 và cài các thư viện cần thiết.

~~~bash
python -m pip install -r requirements.txt
~~~

Chạy các lệnh từ thư mục gốc repo. Chương trình tự chọn thiết bị tính toán có sẵn. Có thể thêm --device cpu hoặc --device cuda vào lệnh train, evaluate, infer và smoke-test để chỉ định thiết bị.

## Chuẩn bị dữ liệu

Đặt ảnh và expert mask theo cấu trúc sau.

~~~text
data/raw/images/<CASE_ID>/<CASE_ID>_0000.nii.gz
data/raw/images/<CASE_ID>/<CASE_ID>_0001.nii.gz
data/raw/images/<CASE_ID>/<CASE_ID>_0002.nii.gz
...
data/raw/segmentations/expert/<CASE_ID>.nii.gz
data/splits/splits_metadata.csv
~~~

Pre tương ứng _0000, Post1 tương ứng _0001. Các pha sau của cùng ca được dùng để tính thống kê chuẩn hóa chung. File pha có thể đặt trực tiếp trong raw_images_dir hoặc trong thư mục từng ca. Tên ca hỗ trợ chữ hoa và chữ thường.

Chỉnh đường dẫn trong [configs/default_config.yaml](configs/default_config.yaml). Các đường dẫn tương đối được tính từ thư mục gốc repo. Split mặc định gồm 1.003 ca train, 177 ca validation và 300 ca test.

## Tiền xử lý

~~~bash
python scripts/run_pipeline.py preprocess
~~~

Pipeline đưa ảnh về hướng RAS và lưới Post1 đẳng hướng 1 mm, chuẩn hóa bằng mean/std chung của các pha và tạo hai kênh Post1, Post1 − Pre có giữ dấu. Expert mask được nội suy nearest-neighbor trên cùng lưới.

Kết quả lưu tại data/processed/full_volume_v1. Mỗi file NPZ chứa ảnh toàn trường nhìn, mask, thông tin hình học và thống kê chuẩn hóa. manifest.json ghi trạng thái các ca và SHA-256 từng file. Chạy lại cùng lệnh để dùng lại file hợp lệ và xử lý các ca thất bại.

Có thể tạo lại split từ bảng clinical và split chính thức bằng lệnh sau.

~~~bash
python scripts/run_pipeline.py preprocess --rebuild-splits
~~~

## Huấn luyện baseline

### So sánh cấu hình đầu vào trên M1

| Cấu hình | Đầu vào |
| --- | --- |
| IP1 | Z(Post1), một kênh |
| IP2 | Z(Post1) − Z(Pre), một kênh |
| IP3 | Ghép [IP1, IP2], hai kênh |

~~~bash
python scripts/run_pipeline.py train --model M1 --ip IP1
python scripts/run_pipeline.py train --model M1 --ip IP2
python scripts/run_pipeline.py train --model M1 --ip IP3
python scripts/run_pipeline.py select-input
~~~

Mỗi run giữ cùng cấu hình đầu vào từ train đến validation/test. select-input chọn cấu hình có best validation Dice cao nhất sau khi cả ba run hoàn tất. Kết quả lưu trong outputs/baseline/selected_input.json. Khi bằng điểm, thứ tự ưu tiên là IP1, IP2 rồi IP3.

### So sánh backbone

| Model | Kiến trúc |
| --- | --- |
| M1 | PlainConvUNet sáu tầng từ thư viện dùng trong nnU-Net v2 |
| M2 | Attention U-Net 3D |
| M3 | Swin UNETR |

Dùng đầu vào đã chọn để huấn luyện M2 và M3. Run M1 tương ứng được dùng lại từ bước trước.

~~~bash
python scripts/run_pipeline.py train --model M2 --ip selected
python scripts/run_pipeline.py train --model M3 --ip selected
~~~

Ba backbone dùng trainer chung với AdamW, warmup/cosine và Dice+BCE. Train lấy patch 3D, validation quét toàn ảnh bằng sliding window. best.pt được chọn theo validation Dice. Chạy lại lệnh train tương ứng để resume.

## Huấn luyện các biến thể M1

Các thí nghiệm dùng M1 + IP3 baseline làm đối chứng.

| --variant | Cấu trúc |
| --- | --- |
| baseline | Ghép hai kênh tại đầu vào, một encoder |
| dual-add | Hai encoder độc lập cho Post1 và subtraction, cộng feature maps cùng mức |
| dual-gated | Hai encoder, hợp nhất feature maps bằng trọng số theo vị trí |
| boundary | Đầu phân đoạn chính và đầu phụ dự đoán biên sau decoder cuối |

~~~bash
python scripts/run_pipeline.py train --model M1 --ip IP3 --variant dual-add
python scripts/run_pipeline.py train --model M1 --ip IP3 --variant dual-gated
python scripts/run_pipeline.py train --model M1 --ip IP3 --variant boundary --boundary-weight 0.2 --boundary-radius 1
~~~

Với boundary, nhãn biên được tạo từ expert mask toàn ảnh trước khi crop và được biến đổi đồng bộ với ảnh. Loss train gồm Dice+BCE cho mask và loss biên có trọng số mặc định 0.2. Radius mặc định là 1 voxel. Đầu phụ được dùng trong train, còn evaluate/infer trả đầu phân đoạn chính.

Mỗi biến thể lưu trong một thư mục riêng, ví dụ M1_IP3_dual-add hoặc M1_IP3_boundary. Variant và tham số loss biên được lưu trong checkpoint.

## Đánh giá

Đánh giá ba backbone với đầu vào đã chọn.

~~~bash
python scripts/run_pipeline.py evaluate --model M1 --ip selected --split test
python scripts/run_pipeline.py evaluate --model M2 --ip selected --split test
python scripts/run_pipeline.py evaluate --model M3 --ip selected --split test
~~~

Đánh giá các cải tiến và baseline đối chứng trên cùng IP3.

~~~bash
python scripts/run_pipeline.py evaluate --model M1 --ip IP3 --variant baseline --split test
python scripts/run_pipeline.py evaluate --model M1 --ip IP3 --variant dual-add --split test
python scripts/run_pipeline.py evaluate --model M1 --ip IP3 --variant dual-gated --split test
python scripts/run_pipeline.py evaluate --model M1 --ip IP3 --variant boundary --split test
~~~

Chọn cấu hình bằng validation trước khi chấm test. Test yêu cầu run đã hoàn tất. Kiến trúc, patch, overlap và ngưỡng được đọc từ checkpoint. Có thể thay --split test bằng --split val để chấm validation.

Kết quả nằm trong outputs/baseline/<MODEL>_<IP>[_<VARIANT>]/<SPLIT>.

| File | Nội dung |
| --- | --- |
| <CASE_ID>_prob.nii.gz | Bản đồ xác suất u toàn trường nhìn Post1 |
| <CASE_ID>_mask.nii.gz | Mask nhị phân thô trên lưới 1 mm |
| metrics.csv | Dice và HD95 từng ca |
| metrics_by_cohort.csv | Trung bình theo cohort và số ca |
| evaluation_config.json | Hash checkpoint và cấu hình đánh giá |

Probability và mask giữ spacing/origin/direction của lưới Post1 sau resample. HD95 được tính theo mm. Khi cả hai mask rỗng, Dice=1 và HD95=0. Khi chỉ một mask rỗng, Dice=0 và HD95=inf.

## Suy luận một ca

~~~bash
python scripts/run_pipeline.py infer --model M1 --ip selected --case data/processed/full_volume_v1/DUKE_001.npz --output outputs/baseline/predict_DUKE_001
~~~

Đối với một biến thể đã train, truyền thêm --variant tương ứng.

~~~bash
python scripts/run_pipeline.py infer --model M1 --ip IP3 --variant boundary --case data/processed/full_volume_v1/DUKE_001.npz --output outputs/baseline/predict_boundary_DUKE_001
~~~

File của ca mới có thể được tạo bằng preprocess_case với expert_path=None rồi save_case. Khi dùng riêng một run đã train, chỉ rõ IP1, IP2 hoặc IP3 theo checkpoint.

## Cấu hình

Patch size, batch size, số epoch, cách lấy patch và overlap được cấu hình trong configs/default_config.yaml. Batch đưa vào model có batch_size × patches_per_case patch.

Có thể dùng một file YAML riêng bằng --config trước tên lệnh.

~~~bash
python scripts/run_pipeline.py --config configs/default_config.yaml train --model M1 --ip IP1
~~~

Resume yêu cầu cấu hình khớp checkpoint. Đặt outputs_dir mới khi chạy một bộ cấu hình khác.

## Kiểm thử

~~~bash
python -m pytest -q -rs
python scripts/run_pipeline.py smoke-test
python scripts/run_pipeline.py smoke-test --improvements
~~~

smoke-test tạo NIfTI tổng hợp để kiểm tra tiền xử lý, train, validation toàn ảnh và xuất/đọc lại mask. Cờ --improvements kiểm tra thêm các biến thể hai encoder và đầu biên.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
| --- | --- |
| configs/ | Cấu hình dữ liệu và huấn luyện |
| scripts/run_pipeline.py | Các lệnh preprocess, train, select-input, evaluate, infer và smoke-test |
| src/preprocessing/ | Chia tập và tiền xử lý toàn ảnh |
| src/datasets/ | Đọc dữ liệu, tạo nhãn biên và lấy patch train |
| src/models/ | Các backbone và biến thể M1 |
| src/engine/ | Huấn luyện, checkpoint và sliding-window inference |
| src/evaluation/ | Chỉ số đánh giá |
| tests/ | Kiểm thử baseline và các cải tiến |
| docs/ | Tài liệu phương pháp |
| legacy/ | Mã nguồn và tài liệu của các vòng thực nghiệm trước |
| MAMA-MIA/ | Mã nguồn tham khảo của bộ dữ liệu |
