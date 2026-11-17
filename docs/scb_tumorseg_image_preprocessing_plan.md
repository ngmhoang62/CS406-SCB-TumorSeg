# SCB-TumorSeg · Plan từ ảnh gốc đến mask tạm

**Mục tiêu** · Phân đoạn u vú 3D từ toàn trường nhìn DCE-MRI trước điều trị. Dùng cùng bệnh nhân và giao thức đánh giá để đo riêng tác động của kênh ảnh, backbone và từng cải tiến.

[Xem infographic của plan](figures/scb_tumorseg_plan_infographic.png).

## 1. Đầu vào

| Dữ liệu | Vai trò |
| --- | --- |
| **Pre, Post1** | Hai pha dùng để tạo các kênh ảnh đầu vào. |
| **Post2 đến pha cuối** | Chỉ góp vào thống kê z-score của ca, không dùng làm kênh ảnh hay biến thể huấn luyện. |
| **Expert mask** | Nhãn train và tham chiếu đánh giá. Không dùng để chọn vùng ảnh ở validation/test. |
| **Metadata theo ca** | Đầu vào tùy chọn cho thí nghiệm điều kiện hóa. Ưu tiên khoảng thời gian Pre → Post1. |

## 2. Tiền xử lý và cấu hình ảnh

1. **RAS** · Chuẩn hóa hướng ảnh và mask, giữ tọa độ vật lý.
2. **Resample đẳng hướng 1 mm** · Đưa các pha và mask về lưới tham chiếu phủ toàn trường nhìn Post1. Nội suy tuyến tính cho ảnh, nearest-neighbor cho mask. Kiểm tra kích thước, spacing, origin và direction của Pre/Post1 trước khi trừ.
3. **Z-score chung theo ca** · Tính một cặp mean/std từ tất cả các pha, rồi áp cùng thống kê cho Pre và Post1.
4. **Tạo hai kênh** · `C1 = Z(Post1)` và `C2 = Z(Post1) − Z(Pre)`. Giữ dấu ảnh hiệu. Giữ toàn trường nhìn, không crop VOI, NLM, percentile clipping hay CEHS, bỏ qua hết những kỹ thuật tiền xử lý lặt vặt khác.

| Cấu hình | Tensor đầu vào |
| --- | --- |
| **IP1** | `[C1]` · 1 kênh Post1. |
| **IP2** | `[C2]` · 1 kênh ảnh hiệu Post1 − Pre. |
| **IP3 = IP1 + IP2** | `[C1, C2]` · 2 kênh ghép theo chiều kênh, giữ nguyên kích thước không gian. |

**Theo MAMA-MIA** · Resample 1 mm và z-score chung theo pha. Model gốc nhận một kênh mỗi mẫu. Khi train, tác giả dùng nhiều pha sau tiêm và ảnh hiệu làm biến thể. Khi validation, tác giả chỉ dùng Post1. Code công khai đã kiểm tra chưa cho thấy cách tác giả dựng và lấy mẫu các biến thể. [Paper MAMA-MIA](https://www.ub.edu/mama-mia/wp-content/uploads/2025/03/mamamia_dataset_paper.pdf)

**Nhóm tự chốt IP1, IP2 và IP3** · Mỗi run giữ đúng một cấu hình ảnh từ train đến validation/test. IP1 trùng loại ảnh Post1 mà paper dùng khi validation, còn dữ liệu train của IP1 luôn là Post1.

## 3. Mô hình và ý tưởng cải tiến

| Backbone | Vai trò |
| --- | --- |
| **M1 · nnU-Net v2 3D** | `3d_fullres`, `PlainConvUNet` sáu tầng theo cấu hình phát hành MAMA-MIA. Train lại trên split của nhóm. |
| **M2 · Attention U-Net 3D** | U-Net tích chập với attention gate ở các skip connection. |
| **M3 · Swin UNETR** | Encoder Swin Transformer và decoder tích chập. |

**Thứ tự thực nghiệm** · So IP1/IP2/IP3 trên M1 bằng validation. Giữ đầu vào đã chọn để so M1/M2/M3. Vòng nền tắt các module thăm dò. Thử từng cải tiến riêng, ưu tiên M1 + IP3 khi cần cả hai kênh.

| Giai đoạn | Model nhận gì và expert mask dùng ở đâu |
| --- | --- |
| **Train** | Tạo batch từ các patch 3D. Có thể ưu tiên một phần patch chứa u bằng expert mask và lấy thêm patch ngẫu nhiên để model học cả nền. Expert mask làm nhãn học. Nếu bật đầu phụ, tạo thêm nhãn biên từ mask. |
| **Validation** | Quét đủ trường nhìn bằng sliding window, ghép dự đoán các cửa sổ. Expert mask chỉ dùng để chấm. Dùng kết quả để chọn cấu hình và checkpoint. |
| **Test** | Khóa cấu hình và checkpoint đã chọn. Quét toàn trường nhìn và ghép cửa sổ như validation. Expert mask chỉ dùng để tính điểm cuối. |

**Patch và batch** · Patch là một khối không gian 3D được cắt từ một ca. Với IP3, mỗi patch chứa cả hai kênh ở cùng tọa độ. Batch là nhóm một hoặc nhiều patch model xử lý cùng lượt.

**Thông số để mở** · Không đặt sẵn kích thước patch, batch size, quy tắc hoặc tỷ lệ lấy patch chứa u và độ chồng lấn khi quét. Dùng cấu hình phát hành của tác giả làm mốc tham khảo, rồi hiệu chỉnh theo kích thước ảnh/u, VRAM và kết quả validation. **1 mm là khoảng cách giữa các voxel sau resample**, không phải kích thước patch.

### Các nhánh cải tiến để thử

| Ý tưởng | Vị trí và đối chứng |
| --- | --- |
| **Hai nhánh encoder cho IP3** | Cho Post1 và ảnh hiệu qua hai stem riêng. So **ghép ngay đầu vào → cộng feature maps → cộng với cổng theo vị trí**. Concat ở baseline vẫn cho lớp đầu học trọng số riêng cho mỗi kênh. |
| **Đầu phụ dự đoán biên u** | Hai đầu ra sau block decoder cuối. Đầu chính học mask, đầu phụ học dải biên từ expert mask **chỉ lúc train**. So bật/tắt đầu phụ bằng Dice và HD95. |
| **cSE 3D** | Gắn vào đầu ra một block decoder trung gian. Pool đặc trưng theo patch và học một hệ số cho mỗi feature channel. So bật/tắt, xem thêm u nhỏ. |
| **Ngữ cảnh toàn ảnh độ phân giải thấp** | Nhánh thô đọc toàn scan một lần, cung cấp đặc trưng tương ứng vị trí patch cho decoder. Quét đủ trường nhìn. Chỉ cân nhắc nếu lỗi dương tính giả ở xa u đáng kể. Thử patch lớn hơn trước và theo dõi bỏ sót u nhỏ, dương tính giả mỗi ca và thời gian chạy. |
| **Điều kiện hóa bằng metadata** | Tại một block encoder, so **chỉ ảnh / concat metadata vào feature map qua conv 1×1×1 / FiLM**. FiLM dùng MLP nhỏ sinh `γ, β`, áp `F′ = (1 + γ)F + β`. |

**Metadata ưu tiên** · Timing Pre → Post1 là số giây giữa hai pha, có **1.448/1.480 ca**. Chuẩn hóa theo train, điền giá trị trung tính kèm cờ thiếu, giữ cùng tập đánh giá. Báo theo cohort và thử xáo timing trong từng cohort. Ứng viên tiếp theo là **pixel spacing và độ dày lát cắt gốc**. **Subtype/HER2** chỉ xét khi thông tin bệnh học có sẵn lúc suy luận. Hãng máy/site dùng để phân tích nhóm. ER/PR, grade và mật độ vú có độ phủ lệch theo cohort. pCR thuộc giai đoạn sau điều trị.

## 4. Đầu ra tạm

**Bản đồ xác suất u 3D → mask dự đoán thô**, phủ toàn trường nhìn Post1, giữ lưới 1 mm và thông tin hình học. Chỉ dùng đầu dự đoán chính khi suy luận. Đánh giá bước này bằng Dice, HD95 và kết quả theo cohort.

**Post-processing** · Chưa chốt, sẽ bổ sung sau.
