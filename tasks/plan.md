# Kế hoạch TV2: Xử lý ảnh và tổng hợp 3 kênh (Step 3)

## Mục tiêu và phạm vi

Tiếp nhận kết quả Step 1–2 của TV1, kiểm chứng và hoàn thiện Step 3 để tạo đầu vào `IP-1`, `IP-2`, `IP-3` cho các mô hình. Không tải lại dữ liệu, chia tập hay chạy lại chuẩn hóa không gian trong phần việc TV2. Tiền xử lý chạy trên Google Colab CPU, đọc/ghi tại `/content/drive/MyDrive/CS406-SCB-TumorSeg/data_real/`; huấn luyện trên Colab GPU là công việc sau, không thuộc kế hoạch này.

## Căn cứ đã đọc

- [`TV1_Tuan1/README.md`](../TV1_Tuan1/README.md): TV1 báo cáo 1.480 ca, chia train/val/test = 1.003/177/300. File bàn giao trên Drive là `processed/npz_96x96x96/{CASE}.npz`, đã chứa `image` 3 kênh, `mask`, `auto_mask`, `subtype_id`. Thư mục `results/splits_metadata.csv` trong repo có đúng 1.480 dòng. 307 ca có ít nhất một cạnh VOI > 96, cần giữ nguyên.
- [`docs/task_assignment.md`](../docs/task_assignment.md) và [`docs/detailed_pipeline.md`](../docs/detailed_pipeline.md): TV2 phụ trách lọc nhiễu giữ biên, cắt phân vị 0,5–99,5%, tạo Post1, Subtraction, PE + CLAHE và đóng gói `float16`.
- [`TV1_Tuan1/src/preprocessing/step3_image_proc.py`](../TV1_Tuan1/src/preprocessing/step3_image_proc.py): hiện đã có N4 tùy chọn (mặc định tắt), 3D non-local means, clipping, z-score, PE và 3D CLAHE. File này giống hệt `src/preprocessing/step3_image_proc.py` ở root. Pipeline của TV1 đã gọi hàm này khi tạo `.npz`.
- [Bài báo MAMA-MIA](../docs/(MAMA-MIA)%20A%20large-scale%20multicenter%20breast%20cancer%20DCE-MRI%20benchmark%20dataset%20with%20expert%20segmentations.pdf), trang 4–6: dữ liệu gốc có Pre và Post1, nhãn chuyên gia trên Post1; bộ dữ liệu công bố chưa áp dụng N4, z-score hoặc resampling. Tác giả có dùng VOI và resampling 1 mm cho mô hình riêng. **Ba kênh ở đây là thiết kế của project, không phải đầu ra sẵn có của bài báo.**

## Quyết định kỹ thuật

1. Giữ contract đầu ra: `image` có dạng `(3,D,H,W)`, `float16`, `D,H,W >= 96`; thứ tự kênh là `Post1`, `Subtraction`, `PE+CLAHE`. Giữ nguyên `mask`, `auto_mask`, `subtype_id` và `splits_metadata.csv` của TV1.
2. Dùng cấu hình Step 3 hiện có làm mốc tái lập: denoise bật, N4 tắt, clipping trên voxel dương, `tau = max(P5(Pre>0), 1)`, CLAHE rồi z-score. Bất kỳ thay đổi thuật toán nào phải được so sánh bằng kiểm tra định lượng và ảnh mẫu trước khi tạo bộ dữ liệu mới.
3. **Cổng dữ liệu:** file `.npz` TV1 chỉ lưu ba kênh sau xử lý, không lưu `pre_voi` và `post1_voi`. Không thể suy ngược chính xác hai VOI gốc từ z-score, clipping và CLAHE. Nếu Drive còn VOI Pre/Post1 đã qua Step 2, dùng chúng để chạy và kiểm chứng TV2 độc lập. Nếu chỉ còn `.npz`, TV2 có thể kiểm định đầu ra và mã hiện hữu; muốn tái tạo Step 3 phải nhận VOI trung gian từ TV1. Việc dựng lại từ `.nii.gz` thô sẽ kéo theo Step 2 nên không nằm trong phạm vi này.
4. Không ghi đè dữ liệu TV1. Nếu cần tái xử lý, ghi vào thư mục có phiên bản riêng trên Drive, so sánh và chỉ chuyển thành đầu ra dùng chung sau khi QC đạt. Với mã và kết quả giống hệt TV1, dùng bộ `.npz` hiện có thay vì tính lại 1.480 ca.

## Thứ tự thực hiện

### 1. Kiểm kê bàn giao trên Colab CPU

- Mount Drive; kiểm tra số file `.npz`, `splits_metadata.csv`, cấu trúc key/shape/dtype ở mẫu từ cả bốn nguồn DUKE, ISPY1, ISPY2, NACT và các ca có VOI > 96.
- Xác nhận Drive còn VOI Pre/Post1 sau Step 2 hay chỉ còn `.npz`; ghi rõ đường dẫn, dung lượng, trường hợp thiếu file. Không tải lại dữ liệu.
- **Đạt khi:** có manifest số ca và quyết định rõ chế độ làm tiếp: `recompute-from-VOI` hoặc `validate-existing-NPZ`.

### 2. Kiểm chứng và hoàn thiện hàm Step 3

- Đối chiếu từng phép biến đổi với công thức trong `detailed_pipeline.md`. Đóng băng một bản Step 3 trong `TV2_Tuan1` để thư mục chạy độc lập trên Colab; giữ tham số tương ứng TV1 và lưu hash mã/config trong báo cáo để phát hiện lệch phiên bản.
- Kiểm tra bằng dữ liệu tổng hợp: shape/dtype, thứ tự kênh, phép trừ không âm trước z-score, ổn định PE khi Pre gần 0, ảnh hằng/ảnh toàn 0, không có NaN/Inf, không biến đổi mask. Ghi rõ ảnh hưởng của padding đến thống kê z-score.
- **Đạt khi:** kiểm tra tập trung chạy được trên CPU và mọi sai khác có giải thích; không thay thuật toán chỉ vì hình ảnh trực quan có vẻ khác.

### 3. Chạy thử và QC trực quan trên Colab

- Chọn khoảng 8–16 ca đại diện bốn nguồn, bốn subtype, khối u nhỏ/lớn, có ít nhất một ca VOI > 96. Nếu có VOI trung gian, tạo lại ba kênh rồi so với `.npz` TV1 (sai số có tính đến `float16`). Nếu không có, kiểm tra trực tiếp `.npz` TV1 và đánh dấu giới hạn tái lập.
- Xuất hình lát cắt Post1, Subtraction, PE+CLAHE cùng contour mask; thống kê phân bố cường độ, độ tương phản vùng u/nền và mức nhiễu nền của CLAHE.
- **Đạt khi:** thứ tự kênh và căn chỉnh mask được xác nhận; không có kênh rỗng/bão hòa hay bất thường rõ ràng chưa được giải thích.

### 4. Chốt đầu ra 1.480 ca

- Nếu Step 3 hiện hữu đạt kiểm chứng: nhận bộ `.npz` TV1 làm đầu ra Step 3 và lưu manifest phiên bản code/config; không chạy lại hàng loạt.
- Nếu cần thay đổi và đã có VOI trung gian: chạy batch Colab CPU theo từng ca, ghi tệp tạm rồi đổi tên, bỏ qua ca hoàn tất, lưu log lỗi để chạy tiếp; ghi bộ mới ở đường dẫn riêng. Điều chỉnh số worker theo RAM/CPU thực tế, không giả định Colab luôn có A100.
- **Đạt khi:** đúng 1.480 ID theo metadata, không có tệp thiếu/trùng hoặc ghi dở; bản TV1 còn nguyên.

### 5. QC cuối và bàn giao

- Chạy kiểm tra contract toàn bộ: bốn key, `float16`, shape `(3,D,H,W)` với mỗi cạnh >= 96, mask nhị phân cùng kích thước, `subtype_id` khớp metadata, mọi giá trị ảnh hữu hạn. Báo cáo riêng các ca `auto_mask` rỗng; không thay nó bằng expert mask.
- Bàn giao notebook Colab chỉ chạy Step 3/QC, script kiểm tra, hình minh họa và báo cáo CSV/Markdown gồm số ca, lỗi, thống kê kênh theo nguồn và subtype, phiên bản config/code và đường dẫn Drive của bộ dữ liệu được chốt.
- **Đạt khi:** QC 1.480/1.480 ca đạt contract hoặc có danh sách ngoại lệ được giải quyết; một thành viên khác có thể mở notebook và tái chạy kiểm tra.

## Rủi ro và điểm cần xác nhận

| Rủi ro | Cách xử lý |
|---|---|
| Không còn VOI Pre/Post1 sau Step 2 trên Drive | Giới hạn ở kiểm định `.npz` hiện có; nhận VOI trung gian từ TV1 trước khi tuyên bố tái tạo độc lập Step 3. |
| CLAHE có thể làm nổi nhiễu nền | So sánh lát cắt và thống kê trên nhiều nguồn; nếu cần chỉnh, lưu kết quả thành phiên bản mới và nêu rõ thay đổi. |
| Colab CPU/RAM/Drive I/O thay đổi theo phiên | Chạy pilot trước, batch có checkpoint và khả năng chạy tiếp; dữ liệu trên Drive, tạm tính toán ở `/content` khi cần. |
| VOI tập Test của TV1 được định vị bằng expert mask | Ghi rõ giới hạn này khi dùng dữ liệu cho đánh giá mô hình sau này; không xem nó là kết quả suy luận trên toàn ảnh. |

## Ngoài phạm vi hiện tại

Huấn luyện ResUNet/IP-1–3 trên Colab A100, augmentation trong DataLoader, các baseline và đánh giá mô hình thuộc các bước sau. Kế hoạch này chỉ hoàn thiện dữ liệu đầu vào ba kênh cho chúng.
