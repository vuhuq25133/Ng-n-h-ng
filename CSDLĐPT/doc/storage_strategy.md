# Chiến Lược Lưu Trữ (Metadata + Features Vector)

> Chiến lược lưu trữ tối ưu: tách bạch rõ ràng giữa **Metadata** (PostgreSQL) và **Features Vector** (pgvector + file .npy), đảm bảo truy vấn nhanh, không dư thừa và dễ mở rộng.

---

## 6.1 Tổng quan chiến lược lưu trữ

```
┌─────────────────────────────────────────────────────────┐
│                    NGUỒN DỮ LIỆU                        │
│              (File .wav – CSTR VCTK Corpus)             │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐       ┌──────────────────────┐
│   METADATA      │       │   FEATURES VECTOR    │
│  (PostgreSQL)   │       │  (pgvector + .npy)   │
└─────────────────┘       └──────────────────────┘
```

| Loại dữ liệu | Nơi lưu | Định dạng | Mục đích |
|---|---|---|---|
| Metadata File Âm thanh | File System | `.csv` (metadata.csv) | Phân tích Acoustic/Non-acoustic (rms_db, keyword...) |
| Metadata Database | PostgreSQL | Bảng SQL | Lọc, tra cứu, hiển thị hệ thống |
| Vector embedding tổng hợp | PostgreSQL (pgvector) | `VECTOR(N)` | Tìm kiếm ANN nhanh |
| Đặc trưng thô theo frame | File System | `.npy` (numpy binary) | DTW, so sánh chuỗi thời gian |
| File âm thanh gốc | File System | `.wav` | Phát lại, tái trích xuất |

---

## 6.2 Lưu trữ Metadata

**Vị trí:** PostgreSQL – bảng `voice_records`

**Nguyên tắc:**
- Chỉ lưu metadata **nhẹ, tra cứu nhanh** trong DB.
- Không lưu binary data (file âm thanh, array đặc trưng thô) vào DB để tránh bloat.
- Dùng `TEXT` thay vì `VARCHAR(n)` cho các trường không giới hạn độ dài cứng.

```sql
CREATE TABLE voice_records (
    file_id      SERIAL PRIMARY KEY,
    speaker      TEXT NOT NULL,       -- VD: p225, p226
    accent       TEXT,                -- VD: 'English', 'Scottish'
    gender       TEXT CHECK (gender IN ('male', 'female')),
    age          INTEGER,             -- Tuổi người nói (nếu có)
    file_path    TEXT NOT NULL,       -- Đường dẫn tương đối đến .wav
    npy_path     TEXT NOT NULL,       -- Đường dẫn tương đối đến .npy
    duration_sec REAL,               -- Thời lượng file (giây)
    sample_rate  INTEGER,            -- Sample rate (Hz)
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
```

**Index tối ưu cho Metadata:**
```sql
-- Tìm kiếm theo các trường metadata thường dùng
CREATE INDEX idx_voice_gender  ON voice_records (gender);
CREATE INDEX idx_voice_speaker ON voice_records (speaker);
CREATE INDEX idx_voice_accent  ON voice_records (accent);
```

---

## 6.3 Lưu trữ Features Vector

### 6.3.1 Vector Embedding tổng hợp → pgvector

**Vị trí:** Cột `embedding VECTOR(N)` trong bảng `voice_records`

**Cấu trúc vector (concat theo thứ tự cố định):**

| Đặc trưng | Số chiều | Ghi chú |
|---|---|---|
| MFCC mean | 40 | n_mfcc = 40 |
| MFCC std | 40 | Biến thiên theo thời gian |
| Spectral Contrast mean | 7 | n_bands = 6, + 1 |
| Chroma STFT mean | 12 | 12 nốt nhạc |
| **Tổng cộng** | **99** | `VECTOR(99)` |

```sql
-- Thêm cột embedding vào bảng
ALTER TABLE voice_records
    ADD COLUMN embedding VECTOR(99);
```

**Index ANN (Approximate Nearest Neighbor) – tối ưu tốc độ tìm kiếm:**
```sql
-- IVFFlat: phù hợp cho dataset 500–100k records
-- lists = sqrt(n_records) ≈ sqrt(500) ≈ 22
CREATE INDEX idx_voice_embedding_ivfflat
    ON voice_records
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 22);
```

> **Tại sao IVFFlat thay vì HNSW?**
> - Dataset 500 records → IVFFlat đủ nhanh, tiêu tốn ít bộ nhớ hơn HNSW.
> - HNSW phù hợp hơn khi dataset > 100k records và cần recall rất cao.

**Truy vấn tìm top-5 tương đồng:**
```sql
SELECT
    file_id, speaker, accent, gender,
    file_path, npy_path,
    1 - (embedding <=> $1::vector) AS cosine_similarity
FROM voice_records
WHERE gender = 'female'          -- Lọc metadata trước (optional)
ORDER BY embedding <=> $1::vector
LIMIT 5;
```

---

### 6.3.2 Đặc trưng thô theo frame → File .npy

**Vị trí:** `data/features/<speaker>/<file_id>.npy`

**Nội dung file `.npy`:**
```python
# Shape: (n_frames, feature_dim)
# Lưu MFCC theo từng frame, không rút gọn mean/std
# Dùng cho DTW và so sánh chuỗi thời gian chi tiết

import numpy as np
np.save(npy_path, mfcc_matrix)   # shape: (40, T) hoặc (T, 40)
```

**Cấu trúc thư mục tối ưu:**
```
data/
├── processed/
│   ├── p225/
│   │   ├── p225_001.wav
│   │   └── p225_002.wav
│   └── p226/
│       └── p226_001.wav
└── features/
    ├── p225/
    │   ├── p225_001.npy
    │   └── p225_002.npy
    └── p226/
        └── p226_001.npy
```

> **Tại sao tổ chức theo speaker?**  
> - Tránh thư mục phẳng với 500+ file khó quản lý.  
> - Dễ kiểm tra dữ liệu còn thiếu cho từng speaker.  
> - Đường dẫn `npy_path` lưu trong DB là đường dẫn **tương đối** (`data/features/p225/p225_001.npy`), không hardcode root.

---

## 6.4 Workflow Lưu trữ (Ingest Pipeline)

```
File .wav
    │
    ▼
[1] Tiền xử lý (resample → 16kHz, cắt/pad → 5s)
    │
    ▼
[2] Trích xuất đặc trưng
    ├── MFCC mean/std + Spectral Contrast + Chroma STFT
    │       └──→ concat → vector (99 chiều)
    └── MFCC raw frames
            └──→ lưu file .npy
    │
    ▼
[3] Lưu vào DB (một transaction duy nhất)
    ├── INSERT metadata vào voice_records
    ├── UPDATE embedding VECTOR(99)
    └── Ghi đường dẫn file_path + npy_path
```

**Code mẫu ingest (Python):**
```python
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

def ingest_voice(conn, wav_path, npy_path, metadata: dict, embedding: np.ndarray):
    """Lưu một file giọng nói vào hệ thống."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO voice_records
                (speaker, accent, gender, age, file_path, npy_path,
                 duration_sec, sample_rate, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            RETURNING file_id
        """, (
            metadata['speaker'], metadata['accent'], metadata['gender'],
            metadata.get('age'), wav_path, npy_path,
            metadata['duration_sec'], metadata['sample_rate'],
            embedding.tolist()          # list[float] → pgvector tự parse
        ))
        file_id = cur.fetchone()[0]
    conn.commit()
    return file_id
```

---

## 6.5 Tóm tắt So sánh Lựa chọn Lưu trữ

| Tiêu chí | PostgreSQL + pgvector | File System (.npy) |
|---|---|---|
| Tốc độ tìm kiếm ANN | ✅ Nhanh (index IVFFlat) | ❌ Không hỗ trợ |
| Lọc kết hợp metadata | ✅ WHERE + ORDER BY vector | ❌ Không hỗ trợ |
| Lưu chuỗi thời gian | ❌ Không phù hợp (quá lớn) | ✅ Lưu full frame |
| Backup & restore | ✅ pg_dump | ✅ Rsync/copy |
| Độ phức tạp triển khai | Trung bình | Thấp |

---

## 6.6 Tại sao Không lưu trữ bằng các cấu trúc Cây Không Gian (Spatial Trees)?

Mặc dù trong lý thuyết CSDL Đa phương tiện, các cấu trúc Cây Tìm Kiếm Không Gian như **QuadTree**, **R-Tree**, hay **K-D Tree** thường xuyên được nhắc tới để lập chỉ mục dữ liệu. Tuy nhiên, hệ thống này **chủ động hoàn toàn từ bỏ việc dùng Cây Không Gian truyền thống** để chọn sử dụng index vector chuyên dụng. 

Nguyên nhân cốt lõi đến từ định lý **Lời nguyền Không Gian Đa Chiều (Curse of Dimensionality)**. Đặc trưng vector âm thanh của hệ thống có số chiều rất lớn: **99 chiều (99-D)**. Ở kích thước không gian này, các Cây truyền thống sụp đổ toàn diện về cả logic lẫn hiệu năng:

### 1. Phân mảnh bằng QuadTree / MX-QuadTree (Cây tứ phân)
- **Cơ chế:** Phân chia không gian bằng cách chia đôi mỗi trục tọa độ. 
- **Lý do thất bại ở 99-D:** Ở không gian 2D, một lần chia (split) sinh ra $2^2 = 4$ node con. Ở 3D (Octree) sinh ra $2^3 = 8$ node con. Tuy nhiên, ở không gian 99 chiều, ngay tại Node gốc (Root) khi tiến hành cắt nút, cây sẽ sinh ra liên tiếp **$2^{99}$ nhánh con**. Đây là một con số khổng lồ (vượt mức lưu trữ của bất kỳ siêu máy tính nào) khiến cho việc xây dựng Node là bất khả thi vật lý.

### 2. Bao thể bằng R-Tree / R*-Tree
- **Cơ chế:** Mở rộng bao lẩy các điểm lại bằng các khối hộp (Minimum Bounding Rectangles - MBB).
- **Lý do thất bại ở 99-D:** Phân cấp hộp bao của R-Tree cực kỳ lý tưởng cho không gian $< 10$ chiều. Tuy nhiên, ở mức $99D$, thể tích không gian vỏ lấn át vùng lõi. Hậu quả là khối lượng các hộp Bounding Box của nhánh con **bè lấp lên nhau gần như $100\%$ (Overlap)**. Khi một Query đi xuống, nó sẽ thấy hộp nào cũng chạm vào mốc giao nhau, bắt buộc phải rẽ nhánh đệ quy vào TOÀN BỘ cấu trúc cây. Cấu trúc Cây lúc này bị thoái hóa nặng nề, tốc độ tra cứu chậm đi thành $O(N)$ (thậm chí chi phí maintain cây làm nó chạy chậm hơn cả quét mảng tuần tự).

### 3. Chia tách bằng K-D Tree (K-Dimensional Tree)
- **Cơ chế:** Cây nhị phân, phân tách không gian tuần tự theo từng mặt phẳng của 1 chiều nhất định.
- **Lý do thất bại ở 99-D:** K-D Tree là thuật toán bền bỉ nhất trong các cây kinh điển, nó ổn định cho tới mốc $20-D$. Nhưng ở không gian 99 chiều, không gian trở nên quá phân tán đến mức việc cắt mặt phẳng mất tác dụng loại bỏ nhánh (Pruning). Để tìm ra điểm Vector gần nhất, Cây vẫn chịu cảnh phải dò (Back-tracking) qua đa số các thân cây nhánh. Tốc độ tìm kiếm lại một lần nữa tiệm cận rớt xuống bằng quy mô Linear Search ($O(N)$).

### Kết Luận Kiến Trúc
Đối nghịch với các Cây Không Gian (vẫn cực kỳ bành trướng ở mảng Geographic GIS hoặc hình học 2D/3D), không gian siêu chiều như **99-D** của Vector AI đòi hỏi các công nghệ lập chỉ mục thế hệ mới. Lời giải duy nhất là các thuật toán **Approximate Nearest Neighbor (ANN)** – như thuật toán phân mảnh Vector trung tâm **`ivfflat`** hoặc đồ thị lân cận **`HNSW`** trong PostgreSQL `pgvector`. Đó chính là lý do dự án lựa chọn công nghệ CSDL AI thay vì cấu trúc Đa phương tiện truyền thống.

---

## 6.7 Triển khai Cơ sở dữ liệu với Docker

Để quản lý dễ dàng và tích hợp sẵn `pgvector` phục vụ tìm kiếm sự tương đồng vector (ANN), hệ thống sẽ sử dụng một container Docker cho PostgreSQL.

### 6.7.1 Thành phần triển khai

Các file phục vụ cho quá trình này được đặt tại thư mục `src/database/`:
- `docker-compose.yml`: Định nghĩa service db cho hệ thống, kết nối port và thiết lập volume để dữ liệu được bảo toàn.
- `init.sql`: Kịch bản cung cấp dữ liệu ban đầu, bao gồm tạo extension `vector`, thiết lập bảng `voice_records` và tạo các index tối ưu hóa cho dữ liệu truyền thống cùng các vector siêu chiều.

### 6.7.2 Hướng dẫn khởi tạo

1. Mở terminal, điều hướng đến thư mục khởi tạo cơ sở dữ liệu:
   ```bash
   cd src/database
   ```
2. Chạy lệnh sau để khởi động CSDL ngầm dưới background:
   ```bash
   docker-compose up -d
   ```
3. Sau khi khởi động thành công, cơ sở dữ liệu sẽ lắng nghe kết nối tại `localhost:5432` với:
   - Database name: `voice_db`
   - User: `admin`
   - Password: `admin_password`

*Lưu ý*: Script `init.sql` sẽ tự động thực thi ngay lần đầu khởi tạo giúp quy chuẩn hóa và tạo lập hoàn chỉnh cấu trúc hệ thống bảng mà không cần chạy thủ công.
