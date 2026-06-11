# TÀI LIỆU KHẮC PHỤC LỖI TRUY VẤN VECTOR (TROUBLESHOOTING)

Tài liệu này ghi lại quá trình xử lý lỗi "Không có kết quả trả về khi tìm kiếm giọng nói" mặc dù hệ thống vẫn báo thành công (200 OK) và cơ sở dữ liệu đã có dữ liệu.

---

## 1. Triệu chứng (Symptoms)
- Người dùng upload file âm thanh lên giao diện Web.
- Terminal của Backend hiện thông báo `POST /api/search 200 OK`.
- Tuy nhiên, giao diện không hiển thị kết quả và API trả về danh sách rỗng `[]`.
- Kiểm tra bằng lệnh `SELECT COUNT(*)` cho thấy Database vẫn có đầy đủ 500 bản ghi.

---

## 2. Quá trình truy vết (Investigation)

### Bước 1: Kiểm tra tính hợp lệ của dữ liệu đầu vào
Chúng tôi đã kiểm tra vector được trích xuất từ file âm thanh của người dùng.
- **Kết quả:** Vector sạch, không chứa giá trị `NaN` (Not a Number) hay `Inf` (Infinity).

### Bước 2: Kiểm tra dữ liệu trong Database
Kiểm tra ngẫu nhiên các bản ghi trong bảng `voice_records`.
- **Kết quả:** Các bản ghi đều có vector 99 chiều hợp lệ.

### Bước 3: Kiểm tra cơ chế tìm kiếm của PostgreSQL (pgvector)
Thực hiện truy vấn tìm kiếm trực tiếp trong Database:
- Khi sử dụng Index mặc định: Truy vấn trả về **0 kết quả**.
- Khi ép hệ thống tắt Index (`SET enable_indexscan = off`): Truy vấn trả về **kết quả chính xác**.

**=> Kết luận:** Chỉ mục (Index) của cột vector đang bị lỗi (corrupted) hoặc không tương thích với phân phối dữ liệu hiện tại.

---

## 3. Giải pháp khắc phục (The Fix)

### Vấn đề của Index cũ (`IVFFLAT`)
Hệ thống trước đó sử dụng `IVFFLAT` index. Đây là loại chỉ mục yêu cầu phải xây dựng lại (REINDEX) thường xuyên khi dữ liệu thay đổi và có thể trả về kết quả rỗng nếu cấu hình số lượng danh sách (`lists`) không phù hợp.

### Chuyển sang Index `HNSW`
Chúng tôi đã thay thế bằng **HNSW (Hierarchical Navigable Small World)**. Đây là loại chỉ mục tiên tiến nhất được hỗ trợ bởi `pgvector` với các ưu điểm:
- **Độ ổn định cao:** Không gặp tình trạng trả về kết quả rỗng khi dữ liệu biến động.
- **Tốc độ cực nhanh:** Hiệu năng tìm kiếm vượt trội hơn so với IVFFLAT.
- **Độ chính xác:** Tiệm cận với việc tìm kiếm vét cạn (Exact Search).

### Các lệnh đã thực hiện:
```sql
-- Xóa chỉ mục cũ đang bị lỗi
DROP INDEX IF EXISTS idx_voice_embedding_ivfflat;

-- Tạo chỉ mục HNSW mới để tối ưu hóa tìm kiếm Cosine Similarity
CREATE INDEX idx_voice_embedding_hnsw 
ON voice_records 
USING hnsw (embedding vector_cosine_ops);
```

---

## 4. Bài học rút ra
- Khi gặp lỗi "200 OK nhưng không có dữ liệu" trong các hệ thống tìm kiếm vector, hãy kiểm tra tính toàn vẹn của **Index**.
- Đối với các bài toán Multimedia Retrieval (như Voice Similarity), nên ưu tiên sử dụng chỉ mục **HNSW** thay vì **IVFFLAT** để đảm bảo tính ổn định lâu dài của hệ thống.
