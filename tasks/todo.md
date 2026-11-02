# TV2 Step 3 — danh sách thực hiện

## Task 1 — Kiểm kê dữ liệu TV1 trên Drive

- [ ] Đếm và đối chiếu 1.480 ID `.npz` với `splits_metadata.csv`; ghi số ca theo split và nguồn.
- [ ] Xác nhận có hay không VOI Pre/Post1 đã qua Step 2; ghi đường dẫn và kết luận cổng dữ liệu.
- [ ] Kiểm tra mẫu key, shape, dtype, mask, `auto_mask` và VOI > 96.

**Phụ thuộc:** không. **Xác minh:** manifest và báo cáo mẫu từ Colab CPU. **Phạm vi:** nhỏ.

## Task 2 — Kiểm chứng mã Step 3

- [ ] Đối chiếu code/config với công thức Post1, Subtraction, PE+CLAHE.
- [ ] Kiểm tra tự động các trường hợp số học và shape/dtype/NaN/Inf; ghi sai khác nếu có.

**Phụ thuộc:** Task 1. **Xác minh:** bộ kiểm tra CPU chạy đạt. **Phạm vi:** nhỏ đến vừa.

## Task 3 — Pilot 8–16 ca trên Colab

- [ ] Chọn ca đại diện bốn nguồn, bốn subtype và VOI > 96.
- [ ] Tạo montage có contour mask và thống kê từng kênh; so với TV1 nếu có VOI trung gian.
- [ ] Ghi kết luận dùng lại đầu ra TV1 hay tạo phiên bản mới.

**Phụ thuộc:** Task 1–2. **Xác minh:** notebook chạy hết trên Colab CPU, ảnh và bảng QC đọc được. **Phạm vi:** vừa.

## Checkpoint — chốt thuật toán và nguồn dữ liệu

- [ ] Không còn lỗi số học/chệch mask chưa giải thích.
- [ ] Có quyết định rõ có cần batch tái xử lý hay chỉ chấp nhận bộ TV1 sau QC.

## Task 4 — Chốt bộ 1.480 ca

- [ ] Nếu cần tái xử lý: chạy batch từ VOI trung gian, ghi ra thư mục phiên bản mới, hỗ trợ chạy tiếp và log lỗi.
- [ ] Nếu không cần tái xử lý: lưu manifest mã/config của bộ TV1 hiện hữu.
- [ ] Không ghi đè `.npz` TV1.

**Phụ thuộc:** checkpoint. **Xác minh:** đúng 1.480 ID, không thiếu/trùng/ghi dở. **Phạm vi:** vừa.

## Task 5 — QC toàn bộ và bàn giao

- [ ] Kiểm tra contract trên mọi ca và lập danh sách ngoại lệ, kể cả `auto_mask` rỗng.
- [ ] Bàn giao notebook Colab, script QC, hình minh họa, bảng thống kê và đường dẫn Drive.

**Phụ thuộc:** Task 4. **Xác minh:** 1.480/1.480 ca đạt hoặc mọi ngoại lệ được giải quyết và ghi rõ. **Phạm vi:** vừa.
