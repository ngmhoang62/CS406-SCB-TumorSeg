# TV2 Tuần 1 — Step 3: Xử lý ảnh và tổng hợp 3 kênh (Denoising, Clipping, 3-Channel Synthesis & QC)

Tài liệu này tổng hợp phạm vi công việc, thuật toán, kiểm thử và kết quả kiểm định chất lượng (QC) trên toàn bộ **1,480 ca MAMA-MIA** của **TV2** trong Tuần 1 (Step 3).

---

## 1. Mục tiêu và Phạm vi (Step 3)

Tiếp nhận kết quả chuẩn hóa không gian (Step 1–2) của TV1, hoàn thiện và kiểm định **Step 3: Multi-Representation Image Processing Pipeline** để tạo tensor 3 kênh đầu vào $\mathbf{X} \in \mathbb{R}^{3 \times D \times H \times W}$ phục vụ các cấu hình `IP-1` (1 kênh), `IP-2` (2 kênh), và `IP-3` (3 kênh).

### Công thức xử lý 3 kênh ([`src/preprocessing/step3_image_proc.py`](../src/preprocessing/step3_image_proc.py) & [`configs/step3_config.json`](../configs/step3_config.json))
1. **Lọc nhiễu giữ biên 3D (3D Edge-Preserving Denoising):**
   - Áp dụng `3D Non-Local Means` (`patch_size=3`, `patch_distance=5`, `h_factor=0.8`, `fast_mode=True`) với ước lượng phương sai nhiễu nền tự động theo MAD dọc sai phân không gian ($\hat{\sigma} = \text{median}(|\nabla_x I|) / (0.6745\sqrt{2})$).
   - Tùy chọn `N4ITK Bias Field Correction` (`apply_n4: false` mặc định để tối ưu thời gian xử lý).
2. **Cắt ngưỡng phân vị cường độ (Percentile Intensity Clipping):**
   - Cắt cực trị trong khoảng $[P_{0.5}, P_{99.5}]$ tính trên các voxel vùng mô vú dương ($I > 0$) nhằm loại bỏ nhiễu lóa do ghim kim loại sinh thiết.
3. **Tổng hợp 3 kênh biểu diễn (3-Channel Synthesis):**
   | Kênh | Tên kênh | Công thức tổng hợp | Ý nghĩa lâm sàng / Xử lý ảnh |
   |---|---|---|---|
   | `image[0]` | $C_1$: `Post1` | $\text{Z-score}(I_{\text{post1}})$ | Cấu trúc giải phẫu và hình thái học tuyến vú ở pha ngấm thuốc đầu tiên |
   | `image[1]` | $C_2$: `Subtraction` | $\text{Z-score}(\max(0, I_{\text{post1}} - I_{\text{pre}}))$ | Triệt tiêu mô mỡ nền sáng trên T1, làm nổi bật vùng bắt thuốc tương phản |
   | `image[2]` | $C_3$: `PE + 3D CLAHE` | $\text{Z-score}\left(\text{CLAHE}_{3D}\left(\frac{\max(0, I_{\text{post1}} - I_{\text{pre}})}{I_{\text{pre}} + \tau}\right)\right)$, $\tau = \max(P_5(I_{\text{pre}} > 0), 1)$ | Chuẩn hóa tỷ lệ phần trăm ngấm thuốc (Percent Enhancement) và tăng cường tương phản cục bộ 3D ở rìa khối u / vùng NME |

---

## 2. Kết quả Kiểm định Data Contract & Thống kê Kênh trên 1,480 Ca

Toàn bộ 1,480 file `.npz` đã được kiểm định tự động bằng [`scripts/run_tv2_step3_qc.py`](../scripts/run_tv2_step3_qc.py) (xem báo cáo chi tiết tại [`docs/reports/tv2_step3/qc_summary.json`](reports/tv2_step3/qc_summary.json) và [`docs/reports/tv2_step3/channel_summary_by_group.csv`](reports/tv2_step3/channel_summary_by_group.csv)):

- **Tổng số ca kiểm định:** `1,480 / 1,480` ca hợp lệ (`valid: 1480`, `missing: 0`, `invalid: 0`).
- **Phân bố theo Split:** Train `1,003` | Val `177` | Test `300`.
- **Phân bố theo Trung tâm (Dataset):** `ISPY2` (980), `DUKE` (288), `ISPY1` (166), `NACT` (46).
- **Phân bố theo Phân nhóm (Subtype):** `Luminal` (747), `Triple Negative` (499), `HER2-enriched` (169), `HER2-pure` (65).
- **Các trường hợp đặc biệt được ghi nhận:**
  - `307` ca có ít nhất một chiều VOI $> 96$ (khối u rất lớn được giữ nguyên toàn bộ thể tích u + lề 25%).
  - `6` ca có `auto_mask` rỗng (do thiếu file phân đoạn tự động trong bộ dữ liệu gốc; được giữ nguyên mask 0 và không thay bằng `expert_mask`).
- **Kiểm chứng độ chính xác tái tạo Step 3 từ VOI trung gian:**
  - Khi tái tạo lại 3 kênh từ VOI trung gian (`pre_voi`, `post1_voi`), sai số tuyệt đối trung bình (`mean_abs_diff_vs_tv1`) giữa đầu ra TV2 và `.npz` của TV1 bằng **`0.0` đến `~1e-7`** (trùng khớp hoàn toàn trong phạm vi `float16`).

### Bảng thống kê độ lệch chuẩn nền (`bg_std`) và độ tương phản U – Nền (`tumor_minus_bg`)
Trích từ [`docs/reports/tv2_step3/channel_summary_by_group.csv`](reports/tv2_step3/channel_summary_by_group.csv):

| Nhóm | Phân loại | Số ca | Post1 `bg_std` | Post1 `tumor - bg` | Sub `bg_std` | Sub `tumor - bg` | PE+CLAHE `bg_std` | PE+CLAHE `tumor - bg` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Dataset** | `DUKE` | 288 | 0.9140 | +3.4327 | 0.8479 | **+4.3571** | 0.9496 | +2.1579 |
| **Dataset** | `ISPY1` | 166 | 0.8731 | +3.1975 | 0.7788 | **+4.3895** | 0.9364 | +2.2811 |
| **Dataset** | `ISPY2` | 980 | 0.9113 | +3.1922 | 0.8158 | **+4.2981** | 0.9448 | +2.1316 |
| **Dataset** | `NACT` | 46 | 0.8650 | +3.5008 | 0.8115 | **+4.0119** | 0.9571 | +1.9355 |
| **Subtype** | `Luminal` | 747 | 0.9088 | +3.2860 | 0.8159 | **+4.4070** | 0.9471 | +2.1584 |
| **Subtype** | `Triple Negative` | 499 | 0.9051 | +3.1551 | 0.8224 | **+4.1185** | 0.9425 | +2.0898 |
| **Subtype** | `HER2-enriched` | 169 | 0.9077 | +3.3227 | 0.8184 | **+4.5319** | 0.9484 | +2.2812 |
| **Subtype** | `HER2-pure` | 65 | 0.8793 | +3.3559 | 0.8016 | **+4.1100** | 0.9348 | +2.1148 |

> **Nhận xét chuyên môn:** Kênh 2 (`Subtraction`) vừa giảm độ nhiễu nền (`bg_std` giảm từ `~0.91` xuống `~0.81`), vừa làm tăng vọt độ chênh cường độ giữa vùng u và mô nền (`tumor_minus_bg` tăng từ `+3.19..3.50` lên **`+4.01..4.53`** trên mọi trung tâm và mọi phân nhóm sinh học). Trong khi đó, kênh 3 (`PE + 3D CLAHE`) cân bằng lại dải động cục bộ để làm rõ ranh giới ngoại vi và các vùng ngấm thuốc dạng không phải khối (NME).

---

## 3. Minh họa Trực quan 3 Kênh (`Post1`, `Subtraction`, `PE + CLAHE`)

Các hình minh họa đại diện cho 4 trung tâm và 4 phân nhóm (với đường contour đỏ thể hiện `expert_mask`) được lưu tại [`docs/figures/tv2_step3_samples/`](figures/tv2_step3_samples/):

- `DUKE_873.png` (`DUKE` | `HER2-pure`)
- `ISPY1_1102.png` (`ISPY1` | `HER2-enriched`)
- `ISPY2_783427.png` (`ISPY2` | `Luminal`)
- `NACT_26.png` (`NACT` | `Luminal`)

---

## 4. Hướng dẫn Chạy Kiểm thử & QC Step 3

### Chạy bộ Unit Tests của Step 3:
```bash
pytest -q tests/test_tv2_week1_step3.py
```

### Chạy QC & xuất báo cáo thống kê 3 kênh trên dữ liệu `.npz`:
```bash
python scripts/run_tv2_step3_qc.py --data-root data --mode validate --report-dir docs/reports/tv2_step3
```
