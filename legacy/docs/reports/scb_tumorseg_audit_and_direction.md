# SCB-TumorSeg · Audit và hướng thực nghiệm

**17/11/2026 · Audit tại `main@3dbb6d9`.** Dữ liệu nhóm có **1.480 ca** (1.003 train, 177 validation, 300 test). Bằng chứng huấn luyện local gồm **18 run M2** trên ba đợt, cùng seed 42. Audit kiểm tra 1.480 header NPZ, phân tích kênh trên 35 ca validation và chạy hai bộ unit test (**16 pass**). Chưa chạy lại huấn luyện hoặc inference.

## Phát hiện và quyết định

| Phát hiện từ audit | Hướng đã chốt |
| --- | --- |
| **VOI cũ dùng expert mask của chính ca validation/test** để chỉ vị trí u. Dice khoảng 0,85 của M2 và 0,7620 của paper đo trên giao thức khác nhau. | **Train theo patch, validation/test quét toàn trường nhìn.** Expert mask có thể hỗ trợ lấy patch chứa u lúc train và chỉ dùng để chấm lúc validation/test. |
| **IP3 cũ trộn cả Post1, subtraction và CEHS**. Trên mẫu kiểm tra, tương quan C2–CEHS có trung vị khoảng 0,99 theo cohort. Thiếu đối chứng ghép hai kênh để quy mức tăng Dice cho CEHS. | **Ba cấu hình cố định**: IP1 = Post1, IP2 = hiệu có dấu Post1 − Pre, IP3 = hai kênh ghép lại. Dùng RAS, lưới chung 1 mm và z-score theo tất cả các pha của mỗi ca. |
| **FiLM subtype thấp hơn ở 8/9 cặp**. Cặp dương duy nhất tăng 0,013 điểm phần trăm Dice trên validation. Các đợt đổi nhiều tham số cùng lúc và chưa có bằng chứng ổn định qua seed hay test. | Lập mốc chỉ ảnh trước. Nếu thử metadata, bắt đầu bằng **timing Pre → Post1**, cùng một vị trí encoder và so **chỉ ảnh / concat / FiLM**. Timing có ở **1.448/1.480 ca** và chênh rõ giữa các cohort, nên báo riêng theo cohort. |
| **MAMA-MIA có ba mốc khác nhau**. FTV dùng PE/SER, auto mask là nhãn khởi đầu cho chuyên gia, còn vanilla nnU-Net là baseline full-image. Helper FTV của nhóm thiếu SER. Điểm automatic–expert và điểm CV phát hành mang ý nghĩa riêng. | Dùng **nnU-Net v2 `3d_fullres` / `PlainConvUNet`** huấn luyện lại trên split của nhóm làm mốc held-out. Sau đó so với Attention U-Net và Swin UNETR trên cùng bệnh nhân, đầu vào và cách chấm. |

Nguồn: [Paper MAMA-MIA](https://www.ub.edu/mama-mia/wp-content/uploads/2025/03/mamamia_dataset_paper.pdf) · [Code tạo VOI cũ](../../src/preprocessing/step2_spatial_voi.py) · [Code tạo kênh cũ](../../src/preprocessing/step3_image_proc.py) · [Nghiên cứu timing + FiLM](https://pmc.ncbi.nlm.nih.gov/articles/PMC13674782/).

## Triển khai và đối chiếu

- **Ghi rõ khác biệt với paper.** Tác giả dùng nhiều pha sau tiêm và subtraction làm mẫu train một kênh, rồi validation bằng Post1. Code công khai đã xem chưa cho thấy cách dựng và lấy mẫu 6.941 mục train. Paper ghi 1.000 epoch, còn năm checkpoint local ghi 4.000 epoch. Giao thức của nhóm cố định IP1/IP2/IP3 cho từng run.
- **Đo tải trước khi chốt thông số.** Dùng A100 40 GB đo train và quét validation trên một mẫu ca đại diện. Patch size, batch size, cách lấy patch và overlap được hiệu chỉnh theo VRAM, dữ liệu và validation. Mức 33–34 GB của M2 cũ là tham khảo. Bản 80 GB cần giảm hơn 24% thời gian chạy để rẻ hơn với đơn giá đã nêu.
- **Giữ phép so gọn.** So IP1/IP2/IP3 trên M1, chọn đầu vào theo validation rồi so ba backbone. Thử từng module riêng. Báo Dice, HD95 và kết quả theo cohort trên tập test với cấu hình đã khóa. [Đối chứng subtraction của đội FME](https://arxiv.org/abs/2608.29162)

[Plan thực nghiệm đầy đủ](../scb_tumorseg_image_preprocessing_plan.md) · [Infographic](../figures/scb_tumorseg_plan_infographic.png)
