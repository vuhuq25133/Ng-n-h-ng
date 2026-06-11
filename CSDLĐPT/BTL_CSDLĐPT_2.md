# BÁO CÁO PHÂN TÍCH: TIỀN XỬ LÝ, ĐẶC TRƯNG & CÂU HỎI PHẢN BIỆN BẢO VỆ

Tài liệu này cung cấp các phân tích chuyên sâu nhằm phân biệt các giai đoạn chuẩn bị dữ liệu, giải cấu trúc toán học của các đặc trưng âm học và chuẩn bị bộ câu hỏi phản biện cận bản chất phục vụ cho việc bảo vệ đồ án **Hệ thống Tìm kiếm Giọng nói Tương đồng (Voice Similarity Search)**.

---

## 1. PHÂN BIỆT "CHUẨN BỊ DỮ LIỆU" VÀ "TIỀN XỬ LÝ DỮ LIỆU"

Dựa trên cấu trúc triển khai thực tế trong chương 3 của báo cáo dự án, hai giai đoạn **Chuẩn bị dữ liệu (Mục 3.2)** và **Tiền xử lý dữ liệu (Mục 3.3)** được phân tách rõ ràng như sau:

| Tiêu chí | 3.2. Chuẩn bị dữ liệu (Data Preparation) | 3.3. Tiền xử lý dữ liệu (Data Preprocessing) |
| :--- | :--- | :--- |
| **Bản chất** | Là quá trình **sàng lọc, phân bổ và lập kế hoạch thiết kế** ở cấp độ quản lý tệp tin (File System) và thuộc tính đối tượng (Speaker Metadata). | Là quá trình **can thiệp sâu vào tín hiệu số (DSP)** và áp dụng các thuật toán toán học để chuẩn hóa cấu trúc mảng dữ liệu. |
| **Mục tiêu** | Trả lời câu hỏi: *"Chúng ta chọn những tệp âm thanh nào, của ai, với số lượng bao nhiêu để đảm bảo tính cân bằng và chất lượng thô?"* | Trả lời câu hỏi: *"Làm thế nào để biến đổi các tệp đã chọn từ trạng thái vật lý ban đầu sang một dạng biểu diễn số học đồng nhất, sạch sẽ để thuật toán trích xuất đặc trưng hoạt động tối ưu?"* |
| **Công việc cụ thể** | 1. Đọc tệp metadata `speaker-info.txt` để **lọc giới tính Nữ** (chọn ra 61 người nói).<br>2. Thiết lập chỉ tiêu **Quota** (lấy mẫu phân tầng - Stratified Sampling) để lấy đúng 500 file sao cho mỗi speaker có từ 7–8 file, tránh thiên lệch hệ thống (systematic bias).<br>3. Kiểm tra chất lượng thô: Loại bỏ file hỏng hoặc thời lượng quá ngắn (< 1s) ở mức hệ thống. | 1. **Làm sạch dữ liệu (Data Cleaning):** Tính toán năng lượng trung bình RMS và loại bỏ các file im lặng hoặc chỉ chứa nhiễu nền (`RMS < 0.0001`).<br>2. **Chuẩn hóa hình thái sóng âm:**<br>&nbsp;&nbsp;&nbsp;&nbsp;- **Resampling:** Hạ tần số lấy mẫu từ 48kHz về 16kHz.<br>&nbsp;&nbsp;&nbsp;&nbsp;- **Mono Conversion:** Chuyển đổi kênh âm thanh từ Stereo sang Mono.<br>&nbsp;&nbsp;&nbsp;&nbsp;- **Length Normalization:** Cắt xén (5 giây đầu) hoặc bù khoảng lặng (Zero-padding ở cuối) để đưa mọi tín hiệu về mảng số thực cố định gồm đúng **80,000 mẫu**.<br>3. Ghi file vật lý dưới định dạng **WAV PCM 16-bit**. |

---

## 2. BỘ CÂU HỎI PHẢN BIỆN CẬN BẢN CHẤT & LỜI GIẢI ĐÁP KHOA HỌC

### Câu hỏi 1: Tại sao lại sử dụng Lược đồ Thời gian - Tần số (Time-Frequency Domain) để so sánh mà không sử dụng miền Thời gian riêng hoặc miền Tần số riêng?
*   **Nếu dùng miền Thời gian riêng (Time Domain - Biên độ theo Thời gian):** Dạng sóng âm thanh (Waveform) cực kỳ nhạy cảm với pha, tiếng ồn môi trường, và tốc độ phát âm (speaking rate). Chỉ cần người nói nói nhanh/chậm một chút, hoặc đứng xa/gần micro là dạng sóng thay đổi hoàn toàn, làm cho việc so khớp trực tiếp (như khoảng cách Euclid) không khả thi.
*   **Nếu dùng miền Tần số riêng (Frequency Domain - Phép biến đổi Fourier FFT toàn phần):** FFT chuyển toàn bộ tín hiệu sang miền tần số để biết "các tần số nào xuất hiện", nhưng nó **mất hoàn toàn thông tin về mặt thời gian** (không biết tần số đó xuất hiện ở giây thứ mấy). Giọng nói con người là tín hiệu phi dừng (non-stationary) thay đổi liên tục theo thời gian; nếu mất trục thời gian, ta sẽ mất đi nhịp điệu, ngữ điệu và sự chuyển dịch âm sắc của người nói.
*   **Lược đồ Thời gian - Tần số (Time-Frequency Domain - STFT/Spectrogram):** Giải quyết bằng cách chia tín hiệu thành các khung cực ngắn (20–40 ms) để giả định tín hiệu là đứng yên (quasi-stationary) trong khung đó, rồi áp dụng FFT trên từng khung. Kết quả thu được là một ma trận 2D (Spectrogram) thể hiện sự biến đổi của phổ tần số theo thời gian. Nó vừa giữ được cấu trúc vật lý thanh quản (tần số cộng hưởng - formants) vừa giữ được phong cách phát âm, tốc độ phát âm (thời gian).

### Câu hỏi 2: Tại sao hệ thống lại kết hợp cả 3 đặc trưng (MFCC, Spectral Contrast, Chroma STFT) thay vì chỉ dùng riêng MFCC?
*   **MFCC** mô phỏng cách tai người nghe và biểu diễn rất tốt hình dáng đường bao phổ trung bình (envelope) - tức cấu trúc vật lý tĩnh của thanh quản. Tuy nhiên, MFCC có xu hướng bỏ lỡ các thông tin chi tiết về mặt nhạc tính (harmonic structure) và độ tương phản năng lượng ở các dải tần số cao.
*   **Spectral Contrast (Tương phản phổ):** Đo lường sự chênh lệch năng lượng giữa đỉnh (peak) và đáy (valley) phổ. Giọng nói rõ ràng, vang, tròn vành rõ chữ sẽ có tương phản phổ cao, ngược lại giọng khàn hoặc thều thào sẽ có phổ phẳng hơn. Đặc trưng này giúp hệ thống nhận diện tốt sắc thái "trong/khàn" của giọng nói (đặc biệt là giọng nữ có dải tần trung-cao rất phong phú).
*   **Chroma STFT (Chroma):** Ánh xạ năng lượng vào 12 nốt nhạc cơ bản trong một quãng tám (bất biến theo quãng tám). Điều này giúp nắm bắt ngữ điệu, cao độ cơ bản và hài âm đặc trưng của giọng nữ, bất kể họ nói ở tông giọng cao hay thấp (do Chroma đã chuẩn hóa năng lượng và cộng dồn các tần số là bội số của nhau).
*   **Kết luận:** Việc kết hợp giúp nâng độ chính xác nhận dạng người nói (Precision@5) lên **92.5%** (so với 88.2% của riêng MFCC nâng cao).

### Câu hỏi 3: Tại sao lại trích xuất cả MFCC Mean (Trung bình) và MFCC Std (Độ lệch chuẩn) theo thời gian?
*   **MFCC Mean (40 chiều):** Phản ánh **đặc tính tĩnh của bộ máy phát âm** (hình dạng vòm họng, dây thanh quản trung bình). Nếu hai người nói có cấu trúc giọng nói nền tảng giống nhau, giá trị Mean của họ sẽ rất sát nhau và dễ gây nhầm lẫn.
*   **MFCC Std (40 chiều):** Đo lường **mức độ biến thiên** của các hệ số phổ theo thời gian. Nó tóm tắt phong cách phát âm, nhịp điệu nói, cách nhấn nhá (prosody & dynamics) của từng người nói (ví dụ cách di chuyển khẩu hình nhanh hay chậm). Sự bổ sung này giúp hệ thống nắm bắt đặc trưng "động" mà không cần dùng đến các mô hình chuỗi thời gian đắt đỏ.

### Câu hỏi 4: Tại sao trong dự án này lại sử dụng chỉ mục HNSW trên pgvector thay vì các chỉ mục không gian truyền thống (K-D Tree, R-Tree) hay IVFFlat?
*   **Sự thất bại của các cây không gian (K-D Tree, R-Tree):** Gặp **"Lời nguyền không gian đa chiều"** (Curse of Dimensionality). Ở không gian 99 chiều, các vùng không gian bị chồng lấp (overlap) gần như 100%. Truy vấn buộc phải duyệt qua toàn bộ các nhánh của cây, làm cho thuật toán bị thoái hóa về quét tuyến tính $O(N)$ nhưng tốn thêm chi phí bộ nhớ để duy trì cây.
*   **IVFFlat:** Là chỉ mục phân cụm không gian (Voronoi cells). Điểm yếu của nó là cần huấn luyện (training) trước và rất dễ trả về kết quả trống hoặc sai lệch nếu các tâm cụm bị lệch khi cập nhật dữ liệu mà không chạy `REINDEX` (như lỗi hệ thống đã gặp phải).
*   **HNSW (Hierarchical Navigable Small World):** Xây dựng đồ thị phân cấp đa tầng. HNSW cho phép tìm kiếm lân cận gần đúng (ANN) với thời gian trễ cực thấp (~12ms so với 45ms quét phẳng) và độ ổn định cao, không yêu cầu training trước và không bị lỗi trả về rỗng khi thay đổi dữ liệu.

---

## 3. CẤU TRÚC VÀ Ý NGHĨA CỦA BỘ VECTOR ĐẶC TRƯNG 99 CHIỀU

Vector đặc trưng tổng hợp đại diện cho mẫu giọng nói là một vector tĩnh **99 chiều**, được cấu tạo bởi:

1.  **MFCC Mean (Chỉ số `[0 : 39]` - 40 chiều):** Giá trị trung bình của 40 hệ số MFCC qua toàn bộ các khung thời gian. *Ý nghĩa:* Đại diện cho hình dạng đường bao phổ, phản ánh **âm sắc vật lý tĩnh** của thanh quản.
2.  **MFCC Std (Chỉ số `[40 : 79]` - 40 chiều):** Độ lệch chuẩn của 40 hệ số MFCC qua các khung thời gian. *Ý nghĩa:* Đại diện cho **biến động phổ, nhịp điệu phát âm** và phong cách nói của cá nhân.
3.  **Spectral Contrast Mean (Chỉ số `[80 : 86]` - 7 chiều):** *Ý nghĩa:* Đại diện cho **độ sắc nét, độ trong/khàn** của giọng nói bằng cách so sánh đỉnh và đáy phổ trên 7 dải octave.
4.  **Chroma STFT Mean (Chỉ số `[87 : 98]` - 12 chiều):** *Ý nghĩa:* Đại diện cho **chữ ký cao độ** của dây thanh quản bằng cách ánh xạ năng lượng phổ vào 12 nốt nhạc cơ bản.

---

## 4. LUỒNG CODE TRÍCH XUẤT ĐẶC TRƯNG & CÁCH THỨC HOẠT ĐỘNG CỦA CÁC HÀM LIBROSA

### Luồng xử lý dữ liệu:
1.  Đọc tín hiệu âm thanh thô bằng `librosa.load`.
2.  Trích xuất song song 3 loại đặc trưng biến thiên theo thời gian: MFCC (40 chiều), Spectral Contrast (7 chiều), Chroma STFT (12 chiều).
3.  Thực hiện **Statistical Pooling** (tính toán `np.mean` cho cả 3 đặc trưng, và `np.std` riêng cho MFCC) theo trục thời gian để thu gọn ma trận 2D về các vector 1D tĩnh.
4.  Ghép nối các vector 1D bằng `np.concatenate` tạo thành vector 99 chiều.

### Cách thức hoạt động của các hàm Librosa:

*   **`librosa.load(file_path, sr=16000, mono=True)`:**
    *   *Cách hoạt động:* Đọc tệp âm thanh vật lý `.wav`. Nếu tần số lấy mẫu gốc khác 16kHz, nó sử dụng thuật toán nội suy (resampling) để chuyển đổi về 16kHz. Nếu là âm thanh Stereo (2 kênh), nó tự động lấy trung bình cộng của 2 kênh để chuyển về Mono (đơn kênh). Đầu ra thu được là mảng NumPy 1 chiều gồm 80,000 số thực nằm trong dải `[-1.0, 1.0]`.
*   **`librosa.feature.mfcc(y, sr, n_mfcc=40, n_fft=2048, hop_length=512)`:**
    *   *Cách hoạt động:*
        1.  **Framing:** Chia mảng 80,000 mẫu thành 157 khung (mỗi khung dài 2048 mẫu ~128ms, bước nhảy 512 mẫu ~32ms).
        2.  **Windowing:** Nhân mỗi khung với cửa sổ Hann để làm mượt biên phổ, tránh rò rỉ phổ (spectral leakage).
        3.  **FFT:** Biến đổi Fourier nhanh trên từng khung để thu được phổ biên độ năng lượng.
        4.  **Mel Filterbank:** Nhân phổ năng lượng với 128 bộ lọc tam giác được phân bố phi tuyến tính theo thang đo Mel (mô phỏng thính giác người).
        5.  **Log-năng lượng:** Lấy logarithm tự nhiên của năng lượng các dải Mel.
        6.  **DCT-II:** Thực hiện phép biến đổi Cosine rời rạc loại II để chuyển đổi sang miền cepstral và giữ lại 40 hệ số đầu tiên. Ma trận đầu ra có kích thước `(40, 157)`.
*   **`librosa.feature.spectral_contrast(y, sr, n_fft=2048, hop_length=512, n_bands=6)`:**
    *   *Cách hoạt động:* Tính STFT của tín hiệu, sau đó chia toàn bộ dải tần số (từ 0 đến 8000 Hz) thành 7 dải con (con số 7 tương ứng với `n_bands + 1`) theo thang đo octave. Trong mỗi dải, thuật toán tìm giá trị lớn nhất (Peak - đỉnh phổ) và nhỏ nhất (Valley - đáy phổ) dựa trên một ngưỡng phần trăm cố định. Trích xuất giá trị tương phản bằng hiệu logarithm giữa Peak và Valley: `log(Peak) - log(Valley)`. Kết quả trả về ma trận kích thước `(7, 157)`.
*   **`librosa.feature.chroma_stft(y, sr, n_fft=2048, hop_length=512)`:**
    *   *Cách hoạt động:* Tính toán STFT để thu được phổ tần số. Sau đó, áp dụng ma trận trọng số Chroma để ánh xạ năng lượng từ các bin tần số Hz vào 12 lớp cao độ âm nhạc (C, C#, D,... B). Phép toán này cộng dồn năng lượng của các tần số cách nhau đúng một quãng tám (ví dụ: cộng dồn năng lượng của 261.6Hz, 523.2Hz, 1046.4Hz... vào nốt C). Kết quả trả về ma trận kích thước `(12, 157)`.

---

## 5. CÁC KỊCH BẢN THỬ NGHIỆM VÀ GIẢI THÍCH KẾT QUẢ

### Kịch bản 1: Tìm kiếm giọng nói của cùng một người nói đã có trong cơ sở dữ liệu
*   **Hiện tượng:** Người dùng tải lên một tệp âm thanh khác của speaker `p225` (ví dụ file `p225_002.wav` - chưa từng dùng để nạp vào DB). Kết quả trả về hiển thị speaker `p225` (thông qua file đã nạp `p225_001.wav`) ở vị trí số 1 (Top-1) với độ tương đồng Cosine rất cao (thường $> 90\%$).
*   **Giải thích tại sao:** Mặc dù nội dung câu nói của hai tệp khác nhau, các đặc trưng vật lý thanh quản tĩnh (MFCC Mean) và nhịp điệu phát âm (MFCC Std) của cùng một người nói là gần như không đổi. Điều này làm cho vector nhúng 99 chiều của hai tệp hướng về cùng một phía trong không gian vector đa chiều, tạo ra góc lệch cực nhỏ và độ tương đồng Cosine tiến gần về 1 (tức 100%).

### Kịch bản 2: Tìm kiếm giọng nói của một người nói hoàn toàn mới (chưa có trong CSDL)
*   **Hiện tượng:** Người dùng tải lên một giọng nữ lạ (không nằm trong 61 người nói ban đầu). Hệ thống vẫn trả về Top-5 kết quả, nhưng điểm tương đồng Cosine thấp hơn rõ rệt (chỉ dao động từ $75\% - 85\%$). Khi nghe các file kết quả, ta thấy các kết quả trả về có tông giọng (cao/trầm), tốc độ nói và mức độ khàn/trong tương tự như giọng nói truy vấn.
*   **Giải thích tại sao:** Phép so sánh khoảng cách Cosine trong Vector Database hoạt động theo cơ chế tìm kiếm lân cận gần nhất (Nearest Neighbor). Vì người nói này hoàn toàn mới, không thể có kết quả khớp 100%. Hệ thống bắt buộc phải quét cơ sở dữ liệu và chọn ra những vector có góc lệch nhỏ nhất - đại diện cho những người nói trong CSDL có cấu trúc thanh quản, cao độ trung bình (Chroma) và đặc tính phổ (Spectral Contrast) gần giống với giọng nói lạ đó nhất.

### Kịch bản 3: Sự cố lỗi chỉ mục Vector trả về danh sách rỗng `[]`
*   **Hiện tượng:** Tải tệp lên, database có đầy đủ 500 bản ghi nhưng danh sách tìm kiếm trả về rỗng `[]`. Khi chạy câu lệnh tắt chỉ mục quét (`SET enable_indexscan = off`), kết quả tìm kiếm lại xuất hiện chính xác.
*   **Giải thích tại sao:** Đây là lỗi phân mảnh/hỏng (corruption) chỉ mục của thuật toán `IVFFlat` cũ. IVFFlat chia không gian vector thành các cụm cố định từ trước. Khi số lượng bản ghi nhỏ hoặc phân phối thay đổi mà chỉ mục không được `REINDEX` thường xuyên, truy vấn tìm kiếm sẽ đi nhầm vào các cụm rỗng hoặc cụm không chứa kết quả tối ưu. Lỗi này được khắc phục triệt để bằng cách chuyển sang chỉ mục **HNSW** (Hierarchical Navigable Small World). HNSW liên kết các điểm dữ liệu dưới dạng đồ thị đa tầng, giúp thuật toán duyệt đồ thị tìm kiếm lân cận gần đúng cực kỳ ổn định, cho phép trả về kết quả chính xác tức thì với độ trễ chỉ **~12ms** (nhanh gấp 4 lần so với quét phẳng).
