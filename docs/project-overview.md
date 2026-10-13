# TỔNG QUAN ĐỒ ÁN MÔN HỌC: XỬ LÝ ẢNH VÀ ỨNG DỤNG (CS406)

> **Tên đề tài dự kiến:** Mô hình phân đoạn khối u vú trên ảnh DCE-MRI thích ứng theo phân nhóm bệnh học (Patient/Subtype-Adaptive Tumor Segmentation)  
> **Dataset tham chiếu & Baseline:** MAMA-MIA Benchmark Dataset (*Nature Scientific Data*, 2025)

---

## 1. Đánh giá tính phù hợp với môn học & Định hướng điều chỉnh

### 1.1. Ý tưởng hiện tại có phù hợp với môn Xử lý ảnh không?
* **Hoàn toàn phù hợp:** Ý tưởng giải quyết bài toán **Phân đoạn ảnh (Image Segmentation)** trên ảnh y khoa MRI. Đây là một trong những bài toán trọng tâm và kinh điển nhất của lĩnh vực Xử lý ảnh số & Thị giác máy tính.
* **Đáp ứng chuẩn đầu ra môn học:**
  * **CLO3:** Sử dụng các thư viện và kỹ thuật tiên tiến trong xử lý ảnh.
  * **CLO4:** Xây dựng và phát triển ứng dụng xử lý ảnh nhằm giải quyết một bài toán trong khoa học hoặc thực tiễn.
  * Trực tiếp bám sát nội dung **Chương 4: Phân lớp, nhận dạng và phân đoạn ảnh** trong đề cương CS406 (Trường ĐH Công nghệ Thông tin - UIT).

### 1.2. Những điểm cần điều chỉnh/bổ sung để tối ưu cho môn học
Vì đây là môn **Xử lý ảnh và ứng dụng** (giảng dạy bởi bộ môn Tính toán Đa phương tiện), giảng viên rất coi trọng **vai trò và quy trình xử lý ảnh số** (Image Preprocessing & Enhancement) chứ không chỉ coi mô hình Deep Learning như một chiếc "hộp đen":
1. **Làm nổi bật khâu tiền xử lý ảnh (Image Preprocessing):** 
   * Bài báo gốc (MAMA-MIA) áp dụng nhiều thao tác xử lý ảnh quan trọng: **Phép trừ ảnh số học (Subtraction MRI)**, **Cân bằng/chuẩn hóa lược đồ độ xám (Histogram/Z-score)**, **Lọc khử nhiễu (Denoising)**, và **Hiệu chỉnh trường lệch từ (Bias field correction)**.
   * Cần đưa các bước này thành một phần cốt lõi trong báo cáo và thực nghiệm để thể hiện rõ kiến thức xử lý ảnh cơ bản (Chương 2, 3 của môn học).
2. **Thu hẹp Scope tính toán (Computational Scope):**
   * Dữ liệu DCE-MRI 3D rất nặng. nnU-Net 3D full-resolution huấn luyện 1000 epochs đòi hỏi GPU máy trạm cấu hình cao.
   * Để phù hợp với đồ án môn học (nhẹ nhàng, khả thi trên Google Colab / GPU cá nhân):
     * **Cách 1 (Khuyên dùng):** Cắt theo vùng quan tâm **VOI (Volume of Interest / Bounding Box)** sẵn có trong dataset thay vì chạy toàn bộ ảnh 3D kích thước lớn.
     * **Cách 2:** Trích xuất các lát cắt **2D (slices)** chứa khối u để huấn luyện mô hình 2D U-Net / 2D ResUNet có gắn FiLM, hoặc sử dụng pre-trained weights nnU-Net do bài báo cung cấp để fine-tune / bổ sung module thích ứng.

---

## 2. Mục tiêu bài toán (Problem Statement & Objective)

* **Bối cảnh:** Trong chẩn đoán ung thư vú bằng ảnh cộng hưởng từ tương phản động (DCE-MRI), việc khoanh vùng chính xác ranh giới khối u nguyên phát (Primary Tumor Segmentation) có ý nghĩa quyết định trong việc đo thể tích u, lập kế hoạch phẫu thuật và đánh giá đáp ứng hóa trị tân bổ trợ (NAC).
* **Vấn đề tồn tại:** Các mô hình phân đoạn hiện nay hầu hết được huấn luyện theo dạng tổng quát cho toàn bộ bệnh nhân, dẫn đến việc mô hình thiên lệch về dạng hình thái u phổ biến và hoạt động kém trên các phân nhóm u hiếm hoặc có cấu trúc xâm lấn phức tạp (ví dụ: u phân nhóm Triple Negative, HER2-enriched).
* **Mục tiêu đồ án:**
  * Xây dựng pipeline tiền xử lý ảnh DCE-MRI chuẩn hóa nhằm tối ưu hóa độ tương phản của khối u.
  * Tích hợp cơ chế thích ứng theo ca bệnh thông qua kỹ thuật **FiLM (Feature-wise Linear Modulation)**: sử dụng thông tin phân nhóm ung thư (**Tumor Subtype**) làm điều kiện (conditioning) để tự động điều chỉnh biểu diễn đặc trưng trong mạng phân đoạn, từ đó nâng cao độ chính xác phân đoạn ranh giới khối u.

---

## 3. Đầu vào và Đầu ra của hệ thống (Input & Output)

```
[Ảnh DCE-MRI (Pre-contrast + Post-contrast)]
                     +                         ---> [Pipeline Tiền xử lý & Mô hình U-Net + FiLM] ---> [Mặt nạ phân đoạn nhị phân (Binary Mask)]
[Thông tin Tumor Subtype (Bệnh án)]
```

* **Đầu vào (Input):**
  1. **Dữ liệu hình ảnh:** Chuỗi ảnh T1-weighted DCE-MRI (định dạng NIfTI `.nii.gz`) gồm:
     * Pha trước tiêm chất tương phản (Pre-contrast phase `_0000`).
     * Pha sau tiêm chất tương phản (First post-contrast phase `_0001` / `_0002`).
  2. **Dữ liệu bệnh án đi kèm (Clinical Conditioning):** Phân nhóm ung thư (`tumor_subtype`) của bệnh nhân, gồm 4 nhóm chính:
     * `Luminal` (chiếm ~49.6%)
     * `Triple Negative` (chiếm ~33.1%)
     * `HER2-enriched` (chiếm ~11.2%)
     * `HER2-pure` (chiếm ~4.3%)
* **Đầu ra (Output):**
  * **Mặt nạ phân đoạn nhị phân (Binary Segmentation Mask):** Ma trận cùng kích thước không gian với ảnh gốc (hoặc VOI), trong đó:
    * Giá trị `0`: Nền / mô tuyến vú lành (Background / Normal tissue).
    * Giá trị `1`: Vùng khối u vú nguyên phát (Primary tumor / Malignant lesion).

---

## 4. Quy trình Tiền xử lý và Xử lý ảnh (Image Preprocessing Pipeline)

Quy trình áp dụng trực tiếp các kiến thức nền tảng trong môn học CS406:

1. **Phép trừ ảnh (Subtraction Imaging - Phép toán số học trên ảnh):**
   $$\text{Image}_{\text{sub}} = \text{Image}_{\text{post}} - \text{Image}_{\text{pre}}$$
   * *Ý nghĩa xử lý ảnh:* Triệt tiêu các tín hiệu nền cố định của mô mỡ và cơ ngực, làm nổi bật động học ngấm thuốc cản quang (angiogenesis) tại vùng khối u ác tính.
2. **Hiệu chỉnh trường lệch cường độ (Bias Field Correction):**
   * Sử dụng bộ lọc N4ITK để loại bỏ hiện tượng cường độ sáng không đồng đều do từ trường máy quét MRI (low-frequency intensity non-uniformity).
3. **Khử nhiễu ảnh (Non-linear Denoising Filter):**
   * Áp dụng bộ lọc khử nhiễu phi tuyến giữ biên nhằm loại bỏ nhiễu hạt của máy quét mà không làm mờ ranh giới tổn thương.
4. **Cắt giới hạn cường độ và Chuẩn hóa (Intensity Clipping & Z-score Normalization):**
   * Cắt bỏ các điểm ảnh ngoại lai (intensity outliers) ở phân vị cao/thấp.
   * Chuẩn hóa giá trị voxel theo phân phối chuẩn $(x - \mu) / \sigma$ trên các pha ảnh.
5. **Đồng nhất độ phân giải không gian (Isotropic Resampling):**
   * Resampling các ca ảnh về cùng kích thước voxel đẳng hướng ($1 \times 1 \times 1\text{ mm}^3$) để loại bỏ sai lệch về kích thước vật lý giữa các máy chụp khác nhau.
6. **Trích xuất vùng quan tâm (Volume of Interest - VOI Cropping):**
   * Cắt ảnh theo 3D Bounding Box chứa tổn thương (đã cung cấp sẵn trong dataset) giúp mô hình tập trung vào khối u và giảm tải đáng kể bộ nhớ GPU.

---

## 5. Phương pháp & Mô hình Baselines (Methodology)

### 5.1. Baseline (Mô hình gốc để so sánh)
* **nnU-Net (hoặc U-Net tiêu chuẩn):** Mô hình phân đoạn kinh điển chỉ nhận đầu vào là ảnh DCE-MRI (không có thông tin phân nhóm lâm sàng).
* **Mốc tham chiếu:** Mô hình baseline vanilla nnU-Net công bố trong bài báo MAMA-MIA đạt chỉ số Dice trung bình trên tập kiểm tra là **$\approx 0.762$ ($76.2\%$)**.

### 5.2. Cải tiến đề xuất (FiLM-Conditioned Segmentation)
* **Cơ chế FiLM (Feature-wise Linear Modulation):**
  * Vector nhãn `tumor_subtype` được mã hóa (one-hot hoặc embedding) và đưa qua một mạng nơ-ron nhỏ (MLP) để sinh ra cặp tham số affine $(\gamma, \beta)$.
  * Tại các tầng đặc trưng trung gian của Decoder trong U-Net, feature map $F$ được điều chế theo công thức:
    $$\text{FiLM}(F) = \gamma \odot F + \beta$$
  * *Ý nghĩa:* Tùy thuộc vào loại ung thư của bệnh nhân, mô hình tự động điều chỉnh độ nhạy của các kênh đặc trưng để nhận diện hình thái u đặc trưng của phân nhóm đó tốt hơn.

---

## 6. Thang đo đánh giá (Evaluation Metrics)

Đánh giá độ tương đồng giữa mặt nạ dự đoán ($P$) và mặt nạ chuyên gia ($G$):

1. **Dice Similarity Coefficient (DSC / Dice Score):**
   $$\text{DSC} = \frac{2 |P \cap G|}{|P| + |G|}$$
   * Đo mức độ chồng lấn giữa vùng u dự đoán và nhãn thực tế (giá trị từ $0$ đến $1$, càng cao càng tốt).
2. **95% Hausdorff Distance (HD95):**
   * Đo khoảng cách cực đại giữa biên của hai vùng phân đoạn ở phân vị thứ 95 (loại bỏ nhiễu ngoại lai). Giá trị càng nhỏ thể hiện đường bao u càng chính xác.
3. **Đánh giá phân rã theo Subtype (Subgroup Analysis):**
   * So sánh chi tiết chỉ số DSC trên từng phân nhóm u riêng biệt (`Luminal`, `Triple Negative`, `HER2-enriched`, `HER2-pure`) để kiểm chứng xem FiLM có thực sự cải thiện các nhóm khó/thiểu số hay không.

---

## 7. Kết quả kỳ vọng & Kế hoạch triển khai tinh gọn (Scope)

### 7.1. Kết quả kỳ vọng
* Điểm Dice tổng thể duy trì hoặc vượt mức baseline gốc (~0.76 - 0.78).
* Cải thiện rõ rệt ở các phân nhóm u khó (Triple Negative, HER2-enriched).
* Minh chứng được trực quan tác động của các bước xử lý ảnh (ảnh trừ Subtraction làm rõ u rõ rệt so với ảnh đơn pha).

### 7.2. Lộ trình triển khai khả thi cho đồ án môn học
* **Giai đoạn 1: Khai thác dữ liệu & Pipeline xử lý ảnh**
  * Tải tập con dữ liệu MAMA-MIA (theo file `train_test_splits.csv` đã chia sẵn).
  * Viết script tiền xử lý bằng Python (`SimpleITK`, `nibabel`, `OpenCV`): Trừ ảnh, chuẩn hóa cường độ, crop VOI.
* **Giai đoạn 2: Thiết lập Baseline**
  * Huấn luyện mô hình U-Net cơ bản trên ảnh đã tiền xử lý.
  * Đánh giá điểm Dice và HD95 làm mốc chuẩn.
* **Giai đoạn 3: Bổ sung module FiLM thích ứng**
  * Thêm nhánh đưa thông tin `tumor_subtype` qua FiLM vào các block của U-Net.
  * Huấn luyện và so sánh kết quả trực tiếp với Baseline.
* **Giai đoạn 4: Tổng kết báo cáo & Trực quan hóa**
  * Trực quan hóa kết quả phân đoạn (overlay contour lên ảnh MRI gốc).
  * Viết báo cáo đồ án và chuẩn bị slide thuyết trình theo yêu cầu môn học.
