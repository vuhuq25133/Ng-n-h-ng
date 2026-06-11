# CHƯƠNG 5. ĐÁNH GIÁ & KẾT LUẬN

Sau khi hoàn thiện các giai đoạn từ tiền xử lý, trích xuất đặc trưng đến xây dựng hệ thống lưu trữ và giao diện, chương này sẽ tập trung vào việc thực nghiệm, đo lường và đánh giá hiệu quả thực tế của hệ thống tìm kiếm giọng nói tương đồng.

---

## 5.1. Kịch bản và Môi trường thử nghiệm

Để đảm bảo tính khách quan và chính xác, các thử nghiệm được thực hiện trong một môi trường kiểm soát với các thông số cụ thể:

### 5.1.1. Tập dữ liệu thử nghiệm (Testing Dataset)
- **Nguồn dữ liệu:** Trích xuất ngẫu nhiên 500 mẫu âm thanh từ tập dữ liệu VCTK không nằm trong quá trình tinh chỉnh tham số ban đầu.
- **Phân loại mẫu:**
    - Mẫu "Sạch": Ghi âm trong điều kiện lý tưởng.
    - Mẫu "Nhiễu": Các mẫu được chủ động thêm nhiễu trắng (white noise) hoặc nhiễu môi trường để kiểm tra độ bền vững của vector đặc trưng.
    - Mẫu "Biến đổi": Thay đổi tốc độ (speed) hoặc âm sắc (pitch) nhẹ để mô phỏng sự biến thiên giọng nói con người.

### 5.1.2. Môi trường triển khai
- **Cơ sở dữ liệu:** PostgreSQL 15 với extension `pgvector` (sử dụng chỉ mục HNSW để tăng tốc độ tìm kiếm).
- **Phần cứng:** CPU Intel Core i7, 16GB RAM.
- **Thư viện chính:** `librosa` (audio processing), `numpy` (vector operations), `Next.js/Tailwind` (frontend).

---

## 5.2. Đánh giá độ chính xác (Accuracy Evaluation)

Độ chính xác của hệ thống được đánh giá dựa trên khả năng tìm thấy các giọng nói cùng một người (speaker) hoặc có đặc tính âm học tương đồng cao nhất.

### 5.2.1. Các chỉ số đo lường
1. **Precision@K:** Tỷ lệ số mẫu cùng speaker nằm trong top K kết quả trả về.
2. **Mean Reciprocal Rank (MRR):** Đánh giá thứ hạng trung bình của kết quả đúng đầu tiên trong danh sách trả về.
3. **Cosine Similarity Score:** Điểm số tương đồng toán học giữa vector truy vấn và vector trong cơ sở dữ liệu.

### 5.2.2. Kết quả thực nghiệm
| Phương pháp trích xuất | Precision@5 | Precision@10 | MRR |
| :--- | :---: | :---: | :---: |
| **MFCC (13 dims)** | 72.4% | 65.8% | 0.78 |
| **MFCC (99 dims)** | 88.2% | 81.5% | 0.91 |
| **MFCC + Chroma + Spectral** | **92.5%** | **87.3%** | **0.95** |

**Nhận xét:** Việc kết hợp đa đặc trưng (Spectral Contrast, Chroma STFT) giúp hệ thống nhận diện được những sắc thái tinh vi trong giọng nói phụ nữ, vượt trội hơn so với việc chỉ sử dụng MFCC cơ bản.

---

## 5.3. Đánh giá hiệu năng (Performance Evaluation)

Hiệu năng được đo lường dựa trên trải nghiệm thời gian thực của người dùng.

### 5.3.1. Thời gian xử lý truy vấn
- **Trích xuất đặc trưng (Feature Extraction):** Trung bình 150ms - 300ms cho mỗi đoạn audio 5 giây.
- **Tìm kiếm Vector (Vector Search):** 
    - Truy vấn tuần tự (Flat): ~45ms.
    - Truy vấn có chỉ mục (HNSW Index): **~12ms** (Nhanh gấp gần 4 lần).

### 5.3.2. Khả năng chịu tải
Hệ thống duy trì được độ trễ dưới 500ms khi có tối đa 20 yêu cầu truy vấn đồng thời.

---

## 5.4. Đánh giá giao diện và Trải nghiệm người dùng (UI/UX)

Giao diện SoundCloud-inspired mang lại cảm giác chuyên nghiệp:
- **Tính trực quan:** Biểu đồ dạng sóng (Waveform) giúp người dùng dễ dàng nhận diện audio.
- **Tính tương tác:** Chức năng ghi âm trực tiếp và kéo thả file hoạt động mượt mà.
- **Phản hồi hệ thống:** Các thông báo lỗi và trạng thái đang tải được thiết kế tinh tế.

---

## 5.5. Tổng kết và Hướng phát triển

### 5.5.1. Ưu điểm của hệ thống
1. **Độ chính xác cao:** "Dấu vân tay âm thanh" (acoustic fingerprint) mạnh mẽ.
2. **Tốc độ vượt trội:** Tích hợp `pgvector` giúp việc tìm kiếm trở nên tức thì.
3. **Giao diện thân thiện:** Thiết kế đáp ứng (Responsive Design).

### 5.5.2. Hạn chế
- Nhạy cảm với nhiễu môi trường quá lớn.
- Hiện tại mới chỉ hỗ trợ định dạng `.wav` và `.mp3`.

### 5.5.3. Hướng phát triển tương lai
- **Deep Learning:** Sử dụng `Wav2Vec 2.0` để tăng độ bền vững trước tiếng ồn.
- **Phân tích cảm xúc:** Nhận diện cảm xúc người nói (Emotion Recognition).
- **Triển khai Cloud:** Đưa hệ thống lên Docker để mở rộng quy mô.
