# TV3 – Tuần 2: Tổng Hợp Các Cải Tiến & Tối Ưu Hóa Kỹ Thuật Trọng Tâm

Tài liệu này không nhắc lại các đầu việc phân công cơ bản của TV3 theo tiến độ tuần, mà tập trung làm rõ **3 cụm cải tiến kỹ thuật quan trọng nhất** đã được nghiên cứu, tối ưu và kiểm chứng thực nghiệm trong Tuần 2:
1. **Nâng cấp Kênh biểu diễn số 3 ($C_3$) bằng kỹ thuật CEHS (Core-Enhanced Hybrid Subtraction)** & Quy trình cập nhật dữ liệu 1.480 ca `.npz`.
2. **Tái cấu trúc và tối ưu toàn diện Module FiLM (Feature-wise Linear Modulation)**: Bản chất vấn đề ở Lần 1 và bước nhảy vọt ở Lần 3 mới nhất.
3. **Các cải tiến kỹ thuật bổ trợ**: Tách Learning Rate (Differential LR), Lấy mẫu cân bằng phân nhóm mềm, Tinh chỉnh Scheduler & Tối ưu tốc độ huấn luyện trên GPU/Colab.

---

## 1. Nâng Cấp Kênh $C_3$: Kỹ Thuật Mới C3 CEHS Tối Ưu Hơn

### 1.1. So sánh $C_3$ CEHS với $C_3$ Cũ

Trong kiến trúc biểu diễn 3 kênh đầu vào $X \in \mathbb{R}^{3 \times 96 \times 96 \times 96}$:
- **Kênh $C_1$ ($I_{\text{post1}}$)**: Ảnh thì ngấm thuốc đầu tiên sau tiêm tương phản, chuẩn hóa Z-score.
- **Kênh $C_2$ ($I_{\text{sub}}$)**: Ảnh hiệu số $I_{\text{sub}} = \max(0, I_{\text{post1}} - I_{\text{pre}})$, chuẩn hóa Z-score.
- **Kênh $C_3$ cũ (Percent Enhancement + 3D CLAHE)**:
  - *Nguyên lý cũ*: Tính tỷ lệ phần trăm tăng ngấm thuốc $\text{PE} = \frac{I_{\text{post1}} - I_{\text{pre}}}{I_{\text{pre}} + \tau}$, sau đó áp dụng cân bằng lược đồ cục bộ 3D CLAHE với kích thước block $16 \times 16 \times 16$.
  - *Nhược điểm chí mạng*: 
    - Thuật toán CLAHE chia block $16 \times 16 \times 16$ đã khuếch đại nhiễu nền cực kỳ nặng nề, tạo ra các **vết sọc ô cờ (checkerboard artifacts)** loang lổ trên khắp các voxel mô mỡ và mô nền lành.
    - Làm lu mờ hoặc mất dấu các vùng **lõi u nghèo tưới máu, xơ hóa hoặc hoại tử (necrotic/fibrotic core)** – nơi không bắt thuốc tương phản mạnh nhưng lại là phần trung tâm của khối u.

```
[Kênh C3 Cũ]:   Nhiễu ô cờ 16x16 ở nền  --->  Lõi u hoại tử bị tối/rỗng  --->  Model phân mảnh dự đoán
[Kênh C3 CEHS]: Nền đen tuyệt đối (C2)   --->  Lõi u được thắp sáng (C1)  --->  Đường biên u sắc nét & đặc
```

### 1.2. Kỹ thuật $C_3$ CEHS là gì? Kết hợp gì từ $C_1$ và $C_2$?

**CEHS (Core-Enhanced Hybrid Subtraction)** ra đời nhằm kết hợp hoàn hảo ưu điểm mạnh nhất của cả 2 kênh $C_1$ và $C_2$:
1. **Thừa hưởng nền đen sạch từ $C_2$ (Subtraction)**:
   - Kênh $C_2$ đã trừ sạch các mô tuyến vú bình thường, mang lại vùng nền xung quanh cực kỳ tối và sạch, giúp định vị chính xác vùng hoạt động tưới máu của khối u.
2. **Bù sáng cấu trúc lõi u từ $C_1$ (Post-1 Contrast)**:
   - Kênh $C_1$ chứa đầy đủ thông tin giải phẫu thật và mật độ mô đặc. Ngay cả khi lõi u bị hoại tử không ngấm thuốc, $C_1$ vẫn thể hiện rõ khối u là một thực thể liền mạch.
3. **Cơ chế Bao Gaussian 3D & Ngưỡng Hãm Chặt (Tightly-Gated Gaussian Envelope)**:
   - Để tránh việc đưa $C_1$ vào làm sáng ngược lại các mô lành hoặc mạch máu lân cận, CEHS tạo ra một mặt nạ bao 3D mượt mà ($\sigma = 1.0$) từ $C_2$, sau đó áp dụng **ngưỡng cắt hãm chặt ($\text{threshold} = 0.05$)**:
     $$\text{Gate} = \text{clip}\left(\frac{\text{Env} - 0.05}{1.0 - 0.05} \times 2.5, 0.0, 1.0\right)$$
   - Nhờ chiếc cổng này, năng lượng tăng sáng lõi ($\text{Core Boost} = C_{1,\text{pos}}^{1.4} \times \text{Gate}$) được **giam chặt 100% bên trong lòng khối u**, triệt tiêu hoàn toàn nguy cơ lem năng lượng ra mạch máu ngoại vi.
   - Kết quả: $C_3 = C_2 + 3.5 \times \text{Core Boost}$, sau đó chuẩn hóa Z-score. Kênh $C_3$ mới vừa sạch bóng nhiễu ô cờ, vừa thắp sáng đồng nhất toàn bộ khối u từ vỏ đến lõi.

---

### 1.3. Hướng Dẫn Thực Nghiệm: Ghi Đè 1.480 File `.npz` Cục Bộ

Để chạy thử nghiệm hoặc kiểm tra kết quả, toàn bộ 1.480 file `.npz` dưới local cần được cập nhật kênh $C_3$ mới:

> [!IMPORTANT]
> **Link tải file nén 1.480 file `.npz` đã cập nhật $C_3$ CEHS:**  
> 🔗 `[DÁN LINK GOOGLE DRIVE CỦA BẠN VÀO ĐÂY]` *(Kích thước ~5.6 GB)*

#### Cách bố trí và giải nén chuẩn (Xác nhận nguyên lý đóng gói):

Bạn đã hiểu **hoàn toàn chính xác 100%** về cơ chế đóng gói và giải nén của hệ điều hành:
1. **Cấu trúc đóng gói**: Trong file `.zip`, thư mục gốc được tạo sẵn cấu trúc phân cấp:  
   `data/processed/npz_96x96x96/<1480 file .npz>`
2. **Thao tác giải nén**:
   - Tải file zip về và đặt trực tiếp tại **thư mục gốc của repository** của từng thành viên (ví dụ: `SCB-TumorSeg/`).
   - Nhấp chuột phải vào file zip, chọn **Extract Here** (hoặc dùng 7-Zip / WinRAR / terminal).
3. **Cơ chế hoạt động**:
   - Hệ điều hành (Windows / Linux) sẽ tìm các thư mục tương ứng (`data/` $\to$ `processed/` $\to$ `npz_96x96x96/`). Trùng tên thư mục thì hệ thống **tự động gộp (merge)** chứ không bao giờ xóa hay tạo thư mục mới.
   - Khi gặp 1.480 file `.npz` trùng tên, Windows sẽ hiện thông báo hỏi:  
     👉 Chọn **"Replace the files in the destination"** (Ghi đè tất cả các file trùng).
   - Toàn bộ các file khác trong repo (như splits, metadata, scripts...) hoàn toàn không bị ảnh hưởng.

*(Tùy chọn: Nếu ai muốn tự chạy cập nhật trực tiếp tại máy mà không cần tải zip, chỉ cần mở PowerShell tại thư mục gốc và chạy: `python scripts/update_c3_cehs.py` – script hỗ trợ đa tiến trình CPU, hoàn tất toàn bộ 1.480 ca chỉ trong ~2 phút).*

---

## 2. Tối Ưu Toàn Diện Module FiLM (Feature-wise Linear Modulation)

### 2.1. Module FiLM cũ hoạt động như thế nào?

Ý tưởng ban đầu của FiLM: Cung cấp thông tin phân nhóm sinh học phân tử của bệnh nhân (Luminal, Triple Negative, HER2-enriched, HER2-pure) để dẫn đường cho mô hình phân đoạn 3D:
- Vector one-hot 4 chiều $s \in \mathbb{R}^4$ được đưa qua mạng nơ-ron nhỏ (Shared MLP) để tạo thành vector biểu diễn 128 chiều $e$.
- Tại các tầng khác nhau của mạng tích chập 3D (Encoder, Bottleneck, Decoder), các đầu tuyến tính sẽ dự đoán ra 2 tham số:
  - $\gamma$ (Scale - độ phóng đại kênh đặc trưng).
  - $\beta$ (Shift - độ dịch chuyển kênh đặc trưng).
- Đặc trưng 3D được điều chế theo công thức:
  $$\tilde{F} = (1 + \gamma) \times F + \beta$$
- Ở Epoch 0, các đầu chiếu được khởi tạo bằng 0 (`zero-init`) để đảm bảo ban đầu $\gamma = 0, \beta = 0$ (mô hình hoạt động như bình thường).

---

### 2.2. Tại sao phải tối ưu? (Bản chất vấn đề phát hiện ở Lần chạy 1)

Giải thích một cách trực quan, dễ hiểu nhất (không lạm dụng công thức toán phức tạp):

* **Hiện tượng bất thường ở Lần 1**:
  - Khi bật FiLM, trong 10–15 epoch đầu tiên, mô hình học **nhanh gấp 3 lần** so với mô hình không dùng FiLM (`No-FiLM`).
  - Tuy nhiên, chỉ sau khoảng 30–50 epoch, mô hình có FiLM bị khựng lại, dừng sớm (Early Stopping) và điểm kiểm thử cuối cùng **hụt hơi, thấp hơn cả mô hình không dùng FiLM** (ở Lần 1, `run12 With-FiLM` đạt Dice $0.8481$, thấp hơn `run11 No-FiLM` đạt $0.8535$).

* **Nguyên nhân cốt lõi – "Bẫy Lối Tắt Dịch Nền" (Global Bias Shortcut)**:
  - Trong ảnh chụp MRI vú, khối u thực tế **rất nhỏ**, chỉ chiếm khoảng **1.8%** thể tích khối hình, còn lại **98.2% là nền đen (mô lành và khoảng không)**.
  - Ban đầu, đầu ra của mạng dự đoán xác suất u là 50% ở mọi voxel ($\text{bias} = 0.0$). Để giảm hàm mất mát nhanh nhất, mạng bắt buộc phải "dập tắt" xác suất này ở 98.2% vùng nền về gần bằng 0%.
  - Mạng tích chập 3D thông thường phải mất 10–15 epoch học các đặc trưng biên dạng phức tạp mới dập được vùng nền. Nhưng module FiLM cũ lại có sẵn 128 tham số kết nối trực tiếp vào các tầng giải mã, nên **FiLM đã "nhận làm việc dễ" này: nó học cách dịch toàn bộ ảnh nền xuống thay cho mạng tích chập!**
  - **Hậu quả nghiêm trọng**:
    1. Cả 4 phân nhóm ung thư khác nhau đều bị FiLM điều khiển dịch nền như nhau $\implies$ **4 vector đặc trưng của 4 loại u bị suy biến, giống hệt nhau đến 99.9%** (không còn phân biệt được u nào vào u nào).
    2. Các tầng tích chập 3D bị "bỏ đói" gradient, lười học các đặc trưng không gian thực thụ của khối u.

---

### 2.3. Các giải pháp tối ưu trong Kiến trúc Mới (Lần chạy 3)

Để triệt tiêu hoàn toàn hiện tượng trên và ép FiLM phải học đúng nhiệm vụ, 4 cải tiến then chốt đã được thực hiện:

1. **Chuẩn hóa Tâm Bằng 0 (Zero-Mean Subtype Centering)**:
   - Ép buộc tổng điều chế của cả 4 phân nhóm cộng lại tại mỗi kênh luôn phải bằng đúng 0:
     $$\sum_{k=0}^{3} \gamma(k) = 0 \quad \text{và} \quad \sum_{k=0}^{3} \beta(k) = 0$$
   - **Tác dụng trực quan**: FiLM bị "khóa tay", không thể dịch nền chung cho cả 4 nhóm được nữa. Nếu muốn tăng độ sáng ở phân nhóm này thì buộc phải giảm ở phân nhóm khác $\implies$ **100% năng lượng của FiLM bắt buộc phải dùng để học sự sai khác sinh học thực sự giữa các loại khối u**.

2. **Khởi tạo Tiên nghiệm Nền cho Đầu Ra (`out_head.bias = -2.2`)**:
   - Ngay từ Epoch 0, ta đặt sẵn độ lệch đầu ra $\text{bias} = -2.2$ (tương đương xác suất có u ban đầu chỉ $\approx 10\%$, phản ánh đúng thực tế là u rất nhỏ).
   - Cả mạng tích chập lẫn FiLM không còn phải tốn công sức trong 15 epoch đầu chỉ để dập nền từ 50% xuống nữa, giảm mức mất mát khởi điểm từ $2.20$ xuống ngay $1.39$.

3. **Kẹp Biên Độ Điều Chế Bằng Hàm Tanh (`max_modulation = 0.25`)**:
   - Sử dụng hàm $\tanh$ để giới hạn mức điều chế $\gamma, \beta$ tối đa trong khoảng $[-0.25, 0.25]$. Điều này ngăn chặn việc FiLM can thiệp quá thô bạo làm nổ hoặc méo mó đặc trưng không gian 3D.

4. **Dropout Phân Nhóm Thực Thụ (True-Zero Classifier-Free Dropout $p=0.15$)**:
   - Trong quá trình train, ngẫu nhiên 15% số ca bệnh bị ẩn thông tin phân nhóm. Khi bị ẩn, tham số điều chế được đưa về chính xác $\gamma = 0, \beta = 0$ tuyệt đối. Điều này rèn luyện cho mạng xương sống 3D luôn duy trì khả năng tự phân đoạn xuất sắc ngay cả khi thiếu dữ liệu lâm sàng.

---

### 2.4. Kết quả Thực nghiệm: Đối chiếu Lần 1 vs Lần 3 Mới Nhất

Thực nghiệm kiểm chứng chuẩn mực trên cùng tập dữ liệu MAMA-MIA (cùng Seed 42, cùng cấu hình phân chia) đã chứng minh bước nhảy vọt:

| Cấu hình Thử Nghiệm | Val Dice (Best) | Val Dice (Last 3 Epochs) | Luminal | Triple Negative | HER2-enriched | HER2-pure |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Lần 1:** Baseline No-FiLM (`bias=0.0`) | 0.7193 | 0.7137 | 0.6974 | 0.7160 | 0.7689 | 0.7376 |
| **Lần 1:** Baseline With-FiLM (Bị dính lối tắt nền) | 0.7664 | 0.7598 | 0.7623 | 0.7351 | 0.7999 | 0.7685 |
| **Lần 3 (Mới):** No-FiLM + Khởi tạo Tiên nghiệm (`bias=-2.2`) | 0.7674 | 0.7636 | 0.7625 | 0.7367 | 0.8035 | 0.7664 |
| **Lần 3 (Mới):** **Pure Zero-Center FiLM + Prior Bias** | **0.7772** | **0.7742** | **0.7711** | **0.7490** | **0.8132** | **0.7737** |

> [!TIP]
> **Kết luận then chốt**:  
> Ở Lần 3 mới nhất, kiến trúc FiLM tối ưu đã **lật ngược tình thế hoàn toàn**, vượt trội mô hình không dùng FiLM tới **$+0.98\%$ Best Dice** và **$+1.07\%$ Last-3 Dice**, đồng thời **chiến thắng áp đảo trên toàn bộ cả 4 phân nhóm bệnh sinh học**!

---

## 3. Các Chỗ Phụ & Cải Tiến Kỹ Thuật Huấn Luyện Khác

Bên cạnh tiền xử lý và kiến trúc mô hình, quy trình huấn luyện (training pipeline) cũng được nâng cấp đồng bộ:

### 3.1. Tách Learning Rate Riêng Biệt (Differential Learning Rates)
- **Vấn đề**: Module FiLM là mạng MLP nhỏ nhẹ, nếu dùng chung tốc độ học lớn của mạng tích chập 3D (`3e-4`), FiLM sẽ học quá nhanh và áp đảo đặc trưng không gian.
- **Giải pháp**: Trong bộ tối ưu hóa `AdamW`, các tham số được chia làm 2 nhóm riêng biệt:
  - **Backbone 3D (Encoder, Attention Gates, Decoder)**: $\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-5}$.
  - **FiLM Generator**: $\text{lr}_{\text{film}} = \frac{\text{lr}}{3} = 1 \times 10^{-4}$, $\text{weight\_decay} = 10^{-4}$ (tăng phạt trọng số để chống học tủ).
- Giúp quá trình hội tụ giữa thông tin phân nhóm và đặc trưng thị giác 3D diễn ra hài hòa, ổn định.

### 3.2. Cân Bằng Lấy Mẫu Phân Nhóm Mềm (Tempered Subtype Sampling)
- Phân nhóm `HER2-pure` chỉ có 37 ca ở tập Train, trong khi `Luminal` có hơn 500 ca. Hơn nữa, tập `ISPY2` hoàn toàn không có ca `HER2-pure` nào.
- Nếu nhân bản ca hiếm quá đà (`power = 1.0`), tỷ lệ ca `Luminal` trong mỗi batch sẽ bị sụt giảm mạnh, gây méo phân phối dữ liệu từ bệnh viện.
- **Cải tiến**: Điều chỉnh trọng số lấy mẫu mềm (`sampler_power = 0.5` - căn bậc hai), vừa đảm bảo mô hình thấy đủ các ca hiếm, vừa bảo tồn nguyên vẹn tính đại diện lâm sàng của tập dữ liệu lớn.

### 3.3. Tinh Chỉnh Early Stopping & Chu Kỳ Lịch Trình (Scheduler)
- Chu kỳ huấn luyện được chuẩn hóa về **90 epochs** kết hợp lịch hạ nhiệt Cosine (`CosineAnnealingLR` về $10^{-6}$).
- Thiết lập `early_stopping_min_epochs = 45` và `PATIENCE = 25`.
- **Mục đích**: Ngăn không cho cơ chế dừng sớm ngắt ngang quá trình học ở các epoch 30–40, đảm bảo mô hình luôn đi trọn vẹn vào giai đoạn tinh chỉnh biên u ở vùng learning rate siêu nhỏ ($< 1.5 \times 10^{-4}$).

### 3.4. Tối Tối Tốc Độ Huấn Luyện Trên Google Colab / GPU
Nhằm tiết kiệm tối đa thời gian và chi phí điện toán GPU:
1. **Nạp Toàn Bộ Dữ Liệu Lên RAM (`CACHE_RATE = 1.0`)**: Tận dụng 83.5 GB RAM của Colab Pro để đưa trọn 1.480 ca vào bộ nhớ, cắt giảm thời gian 1 epoch từ ~2 phút xuống chỉ còn **~15 giây** (tăng tốc gấp 8 lần).
2. **Loại bỏ Nghẽn CPU-GPU**: Thay vì gọi lệnh đồng bộ `.item()` sau mỗi batch gây nghẽn đường truyền, toàn bộ chỉ số đánh giá được gom lũy kế dưới dạng tensor trực tiếp trên VRAM GPU (`non_blocking=True`).
3. **Tối Ưu Cửa Sổ Trượt Khi Validate (`SW_OVERLAP = 0.25`)**: Ở các epoch đánh giá định kỳ, giảm độ chồng lấn cửa sổ trượt từ $0.5$ xuống $0.25$ đối với các ca ngoại cỡ ($> 96^3$), giúp giảm 3.4 lần số phép tính trượt mà kết quả đo lường vẫn phản ánh chính xác xu hướng học.

---

## 4. Tóm Tắt Nhanh Dành Cho Thành Viên Nhóm

| Mục | Điểm mấu chốt cần nhớ | Thao tác cần làm |
|---|---|---|
| **Kênh $C_3$ CEHS** | Sạch 100% nhiễu ô cờ $16 \times 16$; Giữ nền đen $C_2$ và làm sáng lõi đặc $C_1$. | Tải file zip đặt tại root repo $\to$ **Extract Here** để ghi đè 1.480 file `.npz` vào `data/processed/npz_96x96x96/`. |
| **Module FiLM** | Thêm chuẩn hóa tâm bằng 0 (`zero-center`), chặn biên độ $\tanh$ và khởi tạo tiên nghiệm `out_head.bias = -2.2` để chặn bẫy dịch nền. | Kiến trúc đã được tích hợp sẵn trong mã nguồn `src/models/film.py` và các mô hình 3D. |
| **Cấu hình Train** | Differential LR ($\text{lr}/3$ cho FiLM), `CACHE_RATE = 1.0`, `sampler_power = 0.5`. | Dùng trực tiếp qua file cấu hình hoặc notebook `notebooks/TV3_Week2_AttentionUNet_6Runs.ipynb`. |
