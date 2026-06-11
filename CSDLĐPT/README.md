# BÁO CÁO TỔNG HỢP TOÀN DIỆN
## HỆ THỐNG TÌM KIẾM GIỌNG NÓI TƯƠNG ĐỒNG (VOICE SIMILARITY SEARCH)

Bản báo cáo này tổng hợp chi tiết toàn bộ các khía cạnh của dự án **Hệ thống Tìm kiếm Giọng nói Tương đồng (Voice Similarity Search)** dựa trên các tài liệu thiết kế, mã nguồn thực tế và kịch bản thực nghiệm của dự án.

---

## 1. Tổng Quan Dự Án

Hệ thống Tìm kiếm Giọng nói Tương đồng (Voice Similarity Search) là một giải pháp tìm kiếm đa phương tiện (Multimedia Information Retrieval) cho phép người dùng tải lên một tệp âm thanh giọng nói (định dạng `.wav`), từ đó hệ thống sẽ tự động phân tích dữ liệu số, trích xuất "chữ ký giọng nói" (acoustic fingerprint) dưới dạng một vector nhúng (vector embedding) 99 chiều, và thực hiện đối sánh để tìm ra **Top 5 giọng nói giống nhất** trong cơ sở dữ liệu.

Hệ thống được thiết kế hướng tới việc so sánh **"âm sắc" (timbre)** và **"cấu trúc thanh quản" (vocal tract)** đặc thù của mỗi cá nhân, độc lập với nội dung ngôn ngữ được phát ra (khác với bài toán Nhận dạng tiếng nói - Automatic Speech Recognition).

---

## 2. Mục Tiêu Dự Án

*   **Mục tiêu kỹ thuật:** Xây dựng một đường ống (pipeline) tiền xử lý âm thanh tự động từ bộ dữ liệu thô, trích xuất các đặc trưng âm học kết hợp (MFCC, Spectral Contrast, Chroma STFT) để đại diện cho giọng nói dưới dạng vector tĩnh.
*   **Mục tiêu lưu trữ & tra cứu:** Triển khai cơ sở dữ liệu vector (Vector Database) hiệu năng cao sử dụng PostgreSQL với extension `pgvector` và lập chỉ mục tối ưu (HNSW/IVFFlat) giúp truy vấn hàng nghìn bản ghi với thời gian trễ dưới 50ms.
*   **Mục tiêu giao diện (UI/UX):** Phát triển ứng dụng Web trực quan (lấy cảm hứng từ SoundCloud), cho phép tải tệp, ghi âm, phát lại, trực quan hóa dạng sóng (Waveform), phổ âm (Spectrogram), ma trận đặc trưng (MFCC) và hiển thị kết quả so khớp trực quan kèm biểu đồ so sánh chi tiết.

---

## 3. Kiến Trúc Hệ Thống & Công Nghệ Sử Dụng (Tech Stack)

Hệ thống được thiết kế theo kiến trúc Microservices đóng gói bằng **Docker** và **Docker Compose**, bao gồm 3 thành phần chính:

```mermaid
graph LR
    User([Người dùng]) <--> WebFrontend[React Frontend<br>Port: 5173]
    WebFrontend <--> WebBackend[FastAPI Backend<br>Port: 8000]
    WebBackend <--> Database[PostgreSQL + pgvector<br>Port: 5432]
```

### 3.1. Chi tiết Tech Stack
*   **Dataset:** **CSTR VCTK Corpus v0.80** (chứa các đoạn ghi âm chất lượng cao của 109 người nói tiếng Anh bản xứ với nhiều giọng vùng miền khác nhau).
*   **Frontend:** 
    *   **ReactJS** kết hợp **Vite** để tối ưu hóa hiệu năng build.
    *   **Axios** để giao tiếp với API backend.
    *   **Recharts** để vẽ các biểu đồ đặc trưng (Waveform, Spectrogram, MFCC, Feature Vector 99D).
    *   **Lucide React** cung cấp hệ thống icon hiện đại.
*   **Backend:**
    *   **FastAPI** (Python 3.10+) xây dựng API RESTful bất đồng bộ (async).
    *   **Librosa / NumPy / SciPy** để đọc, resample và trích xuất các đặc trưng toán học của âm thanh.
*   **Database & Storage:**
    *   **PostgreSQL 16** với phần mở rộng **pgvector** làm Vector Database lưu trữ metadata và vector 99 chiều.
    *   **File System:** Lưu trữ tệp âm thanh gốc `.wav` và ma trận đặc trưng thô `.npy` cho phép phân tích chuỗi thời gian sâu hơn (như Dynamic Time Warping - DTW).

---

## 4. Cơ Sở Lý Thế Thuyết & Xử Lý Tín Hiệu Số (DSP)

Chìa khóa của bài toán định danh giọng nói nằm ở **miền Thời gian - Tần số (Time-Frequency Domain)**. Hệ thống chia nhỏ đoạn âm thanh thành các khung (frame) cực ngắn để giả định tín hiệu là đứng yên (dừng), từ đó trích xuất ra các đặc trưng đặc thù.

### 4.1. Chuyển đổi từ Vật lý sang Kỹ thuật số (ADC)
Âm thanh thực tế là sóng áp suất liên tục (đo bằng Pascal - Pa). Khi đi qua Microphone và bộ chuyển đổi ADC (Analog-to-Digital Converter), nó được lượng tử hóa thành tín hiệu số x[n] thuộc [-1.0, 1.0]. 
Hệ thống sử dụng tần số lấy mẫu (Sample Rate) f_s = 16,000 Hz chuẩn mono. Với tệp âm thanh 5 giây cố định, tín hiệu đầu vào là một mảng 80,000 mẫu số thực.

### 4.2. Trích xuất Đặc trưng MFCC (Mel-Frequency Cepstral Coefficients)
MFCC là đặc trưng quan trọng nhất mô phỏng cách tai người nghe (nhạy bén ở tần số thấp, kém nhạy ở tần số cao). Quá trình trích xuất gồm 7 bước:

1.  **Phân khung (Framing):** Chia tín hiệu 80,000 mẫu thành các khung có độ dài N_FFT = 2048 mẫu (~128ms) với bước nhảy H = 512 mẫu (~32ms, overlap 75%). Tổng số khung sinh ra cho 5 giây là T_frames = 157 khung.
2.  **Hàm cửa sổ (Windowing):** Nhân mỗi khung với **Cửa sổ Hamming** để triệt tiêu hiện tượng rò rỉ phổ (spectral leakage) tại biên khung:
    
    `w(n) = 0.54 - 0.46 * cos(2 * pi * n / (N_FFT - 1))`
    
3.  **Short-Time Fourier Transform (STFT):** Biến đổi Fourier rời rạc trên từng khung để thu về phổ tần số phức S(m, k), từ đó tính **Phổ công suất**:
    
    `P(m, k) = (1 / N_FFT) * |S(m, k)|^2`
    
4.  **Bộ lọc Mel (Mel Filterbank):** Ánh xạ tần số Hz sang thang Mel phi tuyến tính:
    
    `m_Mel = 2595 * log10(1 + f_Hz / 700)`
    
    Hệ thống áp dụng M = 128 bộ lọc tam giác cách đều trên thang Mel để thu gọn phổ năng lượng về ma trận kích thước `(128, 157)`.
5.  **Logarit hóa:** Lấy log tự nhiên S_t[m] = log(S_t[m] + epsilon) giúp mô phỏng thính giác theo thang Decibel và biến mối quan hệ nhân (nguồn âm * bộ lọc) thành mối quan hệ cộng để dễ dàng tách rời.
6.  **Biến đổi Cosine rời rạc (DCT-II):** Đưa ma trận sang **Miền giả thời gian (Cepstral Domain)**. DCT giúp dồn thông tin hình dáng vòm họng (âm sắc) lên các hệ số đầu tiên và khử nhiễu. Hệ thống giữ lại **40 hệ số đầu tiên**:
    
    `MFCC ∈ R^(40 x 157)`
7.  **Statistical Pooling (Mean & Std):** Để loại bỏ sự phụ thuộc vào độ dài thời gian của âm thanh, hệ thống tính toán:
    *   **MFCC Mean (40 chiều):** Đại diện cho cấu trúc ống thanh quản trung bình (âm sắc tĩnh).
    *   **MFCC Std (40 chiều):** Đo lường sự biến thiên phổ theo thời gian (phong cách nhấn nhá, nhịp điệu phát âm).

### 4.3. Đặc trưng Spectral Contrast (Tương phản phổ - 7 chiều)
Đo lường sự tương phản về năng lượng giữa đỉnh phổ (Peak) và thung lũng phổ (Valley) trong 7 dải tần số octave (từ 0 Hz đến 8000 Hz).

`Contrast_b(t) = log(Peak_b(t)) - log(Valley_b(t))`

Sau khi lấy trung bình theo thời gian, ta có **7 đặc trưng tĩnh** mô tả độ "trong trẻo/khàn đục" của giọng nói.

### 4.4. Đặc trưng Chroma STFT (Cao độ - 12 chiều)
Ánh xạ năng lượng phổ vào 12 lớp cao độ âm nhạc cơ bản (nốt C đến B) bằng công thức:

`p = 12 * log2(f / 440) mod 12`
Sau khi pooling theo thời gian, ta có **12 đặc trưng tĩnh** đại diện cho "chữ ký cao độ" rung của dây thanh quản (giọng cao/trầm).

### 4.5. Hợp nhất thành Vector Embedding 99 Chiều
Toàn bộ các đặc trưng tĩnh trên được nối (concatenate) lại thành một vector duy nhất đại diện cho mẫu giọng nói:

| Đoạn chỉ số | Thành phần đặc trưng | Số chiều | Vai trò định danh |
|---|---|---|---|
| `[0 : 39]` | **MFCC Mean** | 40 | Hình dạng cấu trúc thanh quản trung bình |
| `[40 : 79]` | **MFCC Std** | 40 | Nhịp điệu và phong cách phát âm biến đổi |
| `[80 : 86]` | **Spectral Contrast** | 7 | Độ trong, độ sáng và cộng hưởng dải tần |
| `[87 : 98]` | **Chroma STFT** | 12 | Cao độ cơ bản và hài âm (tông giọng) |
| **Tổng cộng** | **Vector Embedding** | **99** | **Dấu vân tay giọng nói (Acoustic Fingerprint)** |

### 4.6. Đo độ tương đồng Cosine (Cosine Similarity)
Đo độ tương đồng giữa hai vector e1, e2 thuộc R^99 được đo bằng góc của chúng trong không gian 99 chiều:

`cosine_similarity(e1, e2) = (e1 . e2) / (||e1|| * ||e2||)`

---

## 5. Pipeline Tiền Xử Lý & Nạp Dữ Liệu (Data Pipeline)

Quy trình tiền xử lý được lập trình tự động hóa nhằm đưa dữ liệu thô về dạng phân phối cân bằng và chuẩn hóa toán học:

```mermaid
graph TD
    A[VCTK Corpus Thô: 25k file, 48kHz] --> B[Lọc giới tính: Chỉ giữ Female Speakers]
    B --> C{Kiểm soát chất lượng}
    C -->|Lỗi header / Quá ngắn < 1s / RMS < 1e-4| D[Loại bỏ]
    C -->|Hợp lệ| E[Chuẩn hóa sóng âm]
    E --> F1[Resample về 16kHz]
    E --> F2[Chuyển đổi sang Mono]
    E --> F3[Cắt / Đệm zero về 5.0 giây]
    F1 & F2 & F3 --> G[Lấy mẫu phân tầng - Stratified Sampling]
    G --> H[Tập dữ liệu đầu ra: 500 file .wav cân bằng]
    H --> I[Trích xuất đặc trưng & nạp vào DB]
```

*   **Lấy mẫu phân tầng (Stratified Sampling):** Do số lượng file giữa các người nói trong VCTK không đều, hệ thống chia chỉ tiêu (quota) đồng đều cho 61 speakers giọng Nữ để lấy đúng 500 file, tránh thiên lệch (bias) mô hình về phía người nói nhiều file.
*   **Đầu ra vật lý:** 500 file `.wav` chuẩn hóa được ghi dưới dạng số nguyên 16-bit (PCM_16) tại `data/processed/`, và ma trận đặc trưng thô được lưu dưới dạng `.npy` tại `data/features/`.
*   **Nạp cơ sở dữ liệu:** Script Python sử dụng thư viện `psycopg2` để đọc metadata từ `metadata.csv` và vector từ `feature_records.json` rồi đẩy vào PostgreSQL.

### 5.2. Thứ tự và Mô tả các Jupyter Notebook trong Pipeline

Dự án cung cấp hệ thống Jupyter Notebook để quản lý và vận hành toàn bộ quy trình từ dữ liệu thô đến cơ sở dữ liệu hoạt động. Thứ tự thực thi chuẩn của pipeline như sau:

#### Bước 1: Tiền xử lý dữ liệu âm thanh thô
*   **Notebook:** `src/preprocessing/preprocess.ipynb`
*   **Chức năng:** 
    *   Lọc cơ sở dữ liệu CSTR VCTK Corpus để chỉ giữ lại các người nói giọng Nữ (61 speakers).
    *   Kiểm soát chất lượng: Loại bỏ các file lỗi, file quá ngắn (< 1.0 giây) hoặc file im lặng (năng lượng trung bình RMS < 0.0001).
    *   Chuẩn hóa định dạng: Hạ tần số lấy mẫu (Resample) về 16,000 Hz, chuyển về kênh đơn (Mono), cắt/đệm zero về độ dài cố định 5.0 giây (tương đương 80,000 mẫu).
    *   Thực hiện lấy mẫu phân tầng (Stratified Sampling) để chọn ngẫu nhiên nhưng phân bổ đều 500 file và lưu dưới dạng PCM_16 tại `data/processed/`.

#### Bước 2: Trích xuất siêu dữ liệu (Metadata) cơ bản
*   **Notebook:** `src/extract_metadata/extract_metadata.ipynb`
*   **Chức năng:**
    *   Duyệt qua 500 tệp tin `.wav` đã qua xử lý sạch để trích xuất thông tin phi âm học (Speaker ID, thời lượng thực, tên tệp).
    *   Truy vấn nội dung văn bản (Transcript/Keyword) tương ứng với từng file `.wav` từ các file `.txt` đi kèm trong bộ dữ liệu VCTK.
    *   Đo lường các thông số vật lý âm học cơ bản: Tần số lấy mẫu, độ sâu bit, năng lượng RMS trung bình (được đổi sang đơn vị dB).
    *   Xuất kết quả tổng hợp ra file `data/metadata/metadata.csv`.

#### Bước 3: Trích xuất các Vector đặc trưng 99-D
*   **Notebook:** `src/feature_extraction/feature_extraction.ipynb`
*   **Chức năng:**
    *   Đọc 500 file âm thanh sạch từ `data/processed/`.
    *   Trích xuất **80 chiều MFCC** (gồm 40 chiều Mean và 40 chiều Std để nắm bắt đặc trưng âm sắc và biến thiên nhịp điệu phát âm).
    *   Trích xuất **7 chiều Spectral Contrast Mean** (nắm bắt độ sáng và cấu trúc hài âm) và **12 chiều Chroma STFT Mean** (nắm bắt phân bố cao độ rung của dây thanh quản).
    *   Ghép nối (concatenate) thành vector nhúng 99 chiều tĩnh, lưu kết quả tại `data/metadata/feature_records.json`.
    *   Đồng thời, trích xuất ma trận MFCC thô (chưa lấy mean/std) lưu dưới dạng file nhị phân `.npy` tại `data/features/<speaker>/` để phục vụ cho các thuật toán so sánh chuỗi thời gian như Dynamic Time Warping (DTW).

#### Bước 4: Khởi tạo Cơ sở dữ liệu và Nạp (Ingest) dữ liệu
*   **Notebook:** `src/database/ingest.ipynb`
*   **Chức năng:**
    *   Kết nối tới PostgreSQL thông qua `psycopg2`.
    *   Đọc và thực thi file `src/database/init.sql` để tạo extension `vector`, khởi tạo bảng `voice_records` và thiết lập chỉ mục HNSW cho trường vector nhúng.
    *   Đọc thông tin từ `metadata.csv` và `feature_records.json`, thực hiện đối sánh khóa chính và nạp (`INSERT`) đồng loạt toàn bộ dữ liệu sạch vào database `voice_db`.

---

> [!NOTE]
> **Notebook Nghiên cứu Thêm (Optional Research Notebook):**
> *   **Vị trí:** `src/feature_extraction/mfcc_from_scratch.ipynb`
> *   **Vai trò:** Đây là notebook thử nghiệm không nằm trong pipeline nạp dữ liệu chính thức. Nó tự phát triển thuật toán trích xuất MFCC từ đầu (từ cửa sổ Hamming, STFT, Mel Filterbank, DCT) bằng NumPy thuần túy mà không dùng thư viện Librosa nhằm mục đích so sánh, đối chiếu và kiểm chứng tính chính xác toán học của lý thuyết.

---

## 6. Thiết Kế Cơ Sở Dữ Liệu Vector (Vector Database Design)

### 6.1. Schema Bảng `voice_records`
Bảng được thiết kế phi quan hệ hóa (de-normalized) giúp truy vấn lấy cả metadata và vector chỉ trong một lần quét (Join-free):

```sql
CREATE TABLE voice_records (
    file_id      SERIAL PRIMARY KEY,
    speaker      TEXT NOT NULL,
    accent       TEXT,
    gender       TEXT CHECK (gender IN ('male', 'female')),
    age          INTEGER,
    file_path    TEXT NOT NULL,
    npy_path     TEXT NOT NULL,
    duration_sec REAL,
    sample_rate  INTEGER,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    embedding    VECTOR(99)
);
```

### 6.2. Tại sao không dùng các cấu trúc Cây Không Gian (Spatial Trees)?
Trong lý thuyết CSDL đa phương tiện truyền thống, các cấu trúc cây như **Quadtree, R-Tree, K-D Tree** thường được dùng để chỉ mục hình học. Tuy nhiên, hệ thống này chủ động loại bỏ chúng do **Lời nguyền Không gian Đa chiều (Curse of Dimensionality)** tại không gian 99 chiều:
1.  **Quadtree:** Phân nhánh bằng cách lũy thừa $2^d$. Với $d=99$, số nhánh con cấp một là $2^{99}$ (bất khả thi về bộ nhớ vật lý).
2.  **R-Tree:** Với 99 chiều, các hộp bao Bounding Box bị đè chồng lấp (overlap) lên nhau gần như 100%. Khi truy vấn, thuật toán bắt buộc phải duyệt qua toàn bộ các nhánh con, thoái hóa hiệu năng về quét tuyến tính $O(N)$ nhưng tốn thêm chi phí duy trì cây.
3.  **K-D Tree:** Tốc độ tìm kiếm bị suy giảm tiệm cận về $O(N)$ do không gian quá phân tán, thuật toán phải quay lui (back-tracking) liên tục.

### 6.3. Giải pháp: Thuật toán ANN (Approximate Nearest Neighbor)
Hệ thống sử dụng các chỉ mục của `pgvector` được thiết kế chuyên biệt cho AI:
*   **IVFFlat:** Chia không gian vector thành các cụm (Voronoi cells).
*   **HNSW (Hierarchical Navigable Small World):** Xây dựng đồ thị phân cấp đa tầng. HNSW cho độ chính xác cực cao, độ trễ thấp và không yêu cầu huấn luyện trước.

---

## 7. Đánh Giá Hiệu Năng & Kết Quả Thực Nghiệm

Các thực nghiệm đo lường độ chính xác và hiệu năng được chạy trên phần cứng: CPU Intel Core i7, 16GB RAM, PostgreSQL 15.

### 7.1. Độ chính xác nhận dạng người nói (Accuracy)
Đo lường bằng tỷ lệ tìm kiếm chính xác cùng một speaker trong kết quả trả về:

| Tổ hợp đặc trưng trích xuất | Precision@5 | Precision@10 | MRR (Mean Reciprocal Rank) |
|---|:---:|:---:|:---:|
| **MFCC truyền thống (13 chiều)** | 72.4% | 65.8% | 0.78 |
| **MFCC nâng cao (40 chiều Mean + 40 chiều Std)** | 88.2% | 81.5% | 0.91 |
| **Hỗn hợp tối ưu (99 chiều: MFCC + Chroma + Spectral)** | **92.5%** | **87.3%** | **0.95** |

> [!NOTE]
> Kết quả thực nghiệm chứng minh việc bổ sung **Spectral Contrast** (âm sắc tinh vi) và **Chroma STFT** (cao độ) giúp tăng tỷ lệ nhận diện đúng người nói lên **92.5%** đối với giọng nữ.

### 7.2. Tốc độ phản hồi (Latency)
*   **Thời gian trích xuất đặc trưng (Feature Extraction):** 150ms – 300ms cho mỗi tệp audio 5 giây.
*   **Thời gian truy vấn Vector Database:**
    *   Tìm kiếm tuần tự (Flat Scan): ~45ms.
    *   Tìm kiếm chỉ mục HNSW (Index Scan): **~12ms** (Nhanh gấp gần 4 lần).

---

## 8. Khắc Phục Lỗi Chỉ Mục Vector & Bài Học Kinh Nghiệm

Trong quá trình phát triển, dự án đã ghi nhận một case study sửa lỗi thực tế liên quan đến chỉ mục Vector trong tài liệu `troubleshooting_vector_index.md`:

*   **Triệu chứng:** Người dùng tải file lên, backend trả về `200 OK` nhưng giao diện hiển thị kết quả rỗng `[]`, dù database vẫn có 500 bản ghi hợp lệ.
*   **Dò lỗi (Debug):** Khi tắt quét chỉ mục (`SET enable_indexscan = off`), truy vấn trả về kết quả chính xác. Điều này xác định lỗi nằm ở cấu trúc Index.
*   **Nguyên nhân:** Chỉ mục `IVFFLAT` cũ bị lỗi (corrupted) hoặc cấu hình tham số `lists` không phù hợp khi tập dữ liệu thay đổi mà không được REINDEX.
*   **Giải pháp:** Loại bỏ hoàn toàn `IVFFLAT` và chuyển sang chỉ mục **HNSW** có tính ổn định cao hơn, không bị ảnh hưởng bởi sự biến động dữ liệu:
    ```sql
    DROP INDEX IF EXISTS idx_voice_embedding_ivfflat;
    
    CREATE INDEX idx_voice_embedding_hnsw 
    ON voice_records 
    USING hnsw (embedding vector_cosine_ops);
    ```

---

## 9. Cấu Trúc Thư Mục Dự Án Thực Tế

Cấu trúc thư mục chính của dự án được tổ chức theo tính năng:

```
Beta/
├── data/
│   ├── raw/                # File .wav gốc từ VCTK Corpus
│   ├── processed/          # File .wav đã tiền xử lý (16kHz, mono, 5s)
│   ├── features/           # File .npy lưu trữ MFCC thô (cho DTW)
│   └── metadata/           # File CSV lưu siêu dữ liệu âm thanh
├── src/
│   ├── preprocessing/      # Tiền xử lý dữ liệu và cân bằng mẫu
│   ├── feature_extraction/ # Trích xuất vector embedding 99 chiều
│   ├── extract_metadata/   # Trích xuất acoustic/non-acoustic metadata
│   ├── database/           # Docker database, init.sql, script nạp dữ liệu
│   └── web/
│       ├── backend/        # REST API (FastAPI, audio_processor, database)
│       └── frontend/       # Giao diện React (Vite, Recharts, layouts)
├── documents/              # Tài liệu thiết kế chương lý thuyết và phân tích
└── docker-compose.yml      # Cấu hình container hóa toàn bộ hệ thống
```
