# 🎙️ Tài Liệu Triển Khai – Pipeline Tiền Xử Lý Dữ Liệu Âm Thanh VCTK Corpus

> **File nguồn:** `src/preprocessing/preprocess.ipynb`  
> **Ngày tài liệu:** 2026-04-11  
> **Dữ liệu đầu vào:** CSTR VCTK Corpus (48 kHz)  
> **Dữ liệu đầu ra:** 500 file `.wav` chuẩn hóa (16 kHz, 5 giây) tại `data/processed/`

---

## 📌 Tổng Quan Pipeline

Pipeline tiền xử lý được thiết kế để **lọc và chuẩn hóa** 500 file âm thanh giọng nữ từ VCTK Corpus, chuẩn bị cho bước trích xuất đặc trưng (feature extraction) phục vụ bài toán nhận diện giọng nói.

### Sơ đồ luồng dữ liệu

```
VCTK-Corpus/speaker-info.txt
        │
        ▼
① Lọc Female Speakers (61 speakers)
        │
        ▼
② Thu thập .wav từ wav48/ (25,124 file)
        │
        ▼
③ Kiểm tra chất lượng → loại bỏ (25,121 file hợp lệ)
        │
        ▼
④ Áp dụng Resample 48kHz→16kHz + Cắt/Pad → 5 giây
        │
        ▼
⑤ Sampling phân bổ đều 500 file từ 61 speakers
        │
        ▼
💾 Lưu ra data/processed/<speaker>/<file>.wav
        │
        ▼
📊 Báo cáo tổng kết
```

### Kết quả đầu ra thực tế

| Thông số | Giá trị |
|---|---|
| Tổng file xử lý | 500 |
| Số female speakers | 61 |
| File/speaker (min) | 8 |
| File/speaker (max) | 9 |
| File/speaker (avg) | 8.2 |
| Sample rate | 16,000 Hz |
| Độ dài mỗi file | 5.0 giây (80,000 samples) |

---

## ⚙️ Bước 0: Cấu Hình Toàn Cục

### Mã nguồn

```python
from pathlib import Path

# ─── Root project (Beta/) ─────────────────────────────────────────────────────
# notebooks nằm tại src/preprocessing/ → lên 2 cấp
PROJECT_ROOT = Path().resolve().parents[1]

# ─── Nguồn dữ liệu VCTK ──────────────────────────────────────────────────────
VCTK_ROOT    = PROJECT_ROOT / "data" / "raw" / "VCTK-Corpus" / "VCTK-Corpus"
SPEAKER_INFO = VCTK_ROOT / "speaker-info.txt"
WAV48_DIR    = VCTK_ROOT / "wav48"

# ─── Output sau tiền xử lý ───────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"

# ─── Tham số âm thanh ────────────────────────────────────────────────────────
TARGET_SR       = 16_000     # Sample rate mục tiêu (Hz)
TARGET_DURATION = 5.0        # Độ dài cố định (giây)
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)  # = 80,000

MIN_DURATION = 1.0           # Thời lượng tối thiểu để file hợp lệ (giây)
MIN_RMS      = 1e-4          # Ngưỡng năng lượng – loại silence

# ─── Sampling ─────────────────────────────────────────────────────────────────
TOTAL_FILES = 500
RANDOM_SEED = 42
```

### Giải thích chi tiết

#### 1. Xác định thư mục gốc dự án (`PROJECT_ROOT`)

```python
PROJECT_ROOT = Path().resolve().parents[1]
```

- `Path().resolve()` → trả về **đường dẫn tuyệt đối** của thư mục hiện tại khi notebook đang chạy. Vì notebook nằm tại `src/preprocessing/`, đây sẽ là `C:\BTL_CSDLDPT\Beta\src\preprocessing`.
- `.parents[1]` → lên **2 cấp thư mục**:
  - `.parents[0]` = `src/`
  - `.parents[1]` = `Beta/` ← đây là project root
- Kỹ thuật này giúp code **không phụ thuộc vào đường dẫn tuyệt đối** cứng, chạy được trên mọi máy.

#### 2. Cấu trúc đường dẫn dữ liệu

```
Beta/
├── data/
│   ├── raw/
│   │   └── VCTK-Corpus/
│   │       └── VCTK-Corpus/   ← VCTK_ROOT (lồng nhau do cấu trúc bộ dữ liệu gốc)
│   │           ├── speaker-info.txt   ← SPEAKER_INFO
│   │           └── wav48/             ← WAV48_DIR
│   ├── processed/             ← PROCESSED_DIR (output)
│   └── features/              ← FEATURES_DIR (cho bước sau)
```

> **Lý do lồng `VCTK-Corpus/VCTK-Corpus/`:** Đây là cấu trúc khi giải nén file zip gốc của VCTK, thư mục con bên trong trùng tên thư mục ngoài.

#### 3. Tham số âm thanh

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `TARGET_SR` | 16,000 Hz | Sample rate mục tiêu sau khi resample |
| `TARGET_DURATION` | 5.0 giây | Độ dài đồng nhất của mỗi file đầu ra |
| `TARGET_SAMPLES` | 80,000 | Tổng số điểm dữ liệu = SR × Duration |
| `MIN_DURATION` | 1.0 giây | Ngưỡng thời lượng tối thiểu để file hợp lệ |
| `MIN_RMS` | 1e-4 | Ngưỡng năng lượng để loại file im lặng |

---

#### 🔬 Giải thích chi tiết từng tham số

##### `TARGET_SR = 16,000 Hz` – Sample Rate mục tiêu

**Sample rate (tần số lấy mẫu)** là số lần tín hiệu âm thanh được đo mỗi giây.

```
Tín hiệu âm thanh thực (liên tục)
──────────────────────────────────────────────────→ thời gian
  ↑ đo tại đây (16,000 lần/giây)
  [mẫu 1][mẫu 2][mẫu 3]...[mẫu 16000] = 1 giây âm thanh
```

- File VCTK gốc: **48,000 Hz** (chuẩn thu âm studio)
- Sau khi resample: **16,000 Hz** (chuẩn nhận diện giọng nói)
- **Lý do chọn 16kHz:** xem phần Định lý Nyquist bên dưới

---

##### 📐 Định lý Nyquist – Cơ sở khoa học để chọn 16 kHz

> **Định lý Nyquist–Shannon (1948):** Để tái tạo chính xác một tín hiệu có tần số cao nhất là **f_max**, tần số lấy mẫu phải tối thiểu bằng **2 × f_max**.

$$f_{sample} \geq 2 \times f_{max}$$

**Áp dụng vào giọng nói người:**

| Thành phần âm thanh | Dải tần số |
|---|---|
| Giọng nói hội thoại bình thường | 300 Hz – 3,400 Hz |
| Toàn bộ thành phần ngôn ngữ (âm vị, formant) | 80 Hz – 8,000 Hz |
| Âm thanh con người nghe được | 20 Hz – 20,000 Hz |
| Nhạc cụ / âm nhạc chuyên nghiệp | 20 Hz – 20,000 Hz |

> **Kết luận:** Giọng nói chứa tất cả thông tin ngôn ngữ có ý nghĩa trong dải **80 Hz – 8,000 Hz**.

Áp dụng Nyquist:
```
f_max (giọng nói) = 8,000 Hz
f_sample tối thiểu = 2 × 8,000 = 16,000 Hz  ✅
```

**Vì sao 16 kHz là đủ, không cần 48 kHz?**

```
48 kHz có thể tái tạo tần số đến 24,000 Hz
          ↓
Nhưng thông tin giọng nói chỉ đến 8,000 Hz
          ↓
Phần 8,000 Hz – 24,000 Hz là THỪA với bài toán nhận diện giọng
          ↓
16 kHz tái tạo được đến 8,000 Hz → ĐỦ, không mất thông tin
```

**Lợi ích thực tế khi dùng 16 kHz thay vì 48 kHz:**

| Tiêu chí | 48 kHz | 16 kHz | Cải thiện |
|---|---|---|---|
| Số samples/giây | 48,000 | 16,000 | Giảm 3× |
| Dung lượng file 5s (float32) | 960 KB | 320 KB | Giảm 3× |
| Tốc độ tính MFCC | chậm | nhanh 3× | — |
| Bộ nhớ GPU khi training | lớn | nhỏ 3× | — |
| Thông tin giọng nói giữ lại | 100% | 100% | **Không mất** |

> **Tóm lại:** 16 kHz là điểm "ngọt" – đủ để giữ nguyên toàn bộ thông tin giọng nói, đồng thời tối ưu tài nguyên tính toán.

---

##### `TARGET_DURATION = 5.0` – Thời lượng cố định

**Vấn đề:** Các file audio trong VCTK có độ dài khác nhau (1–10 giây).  
**Giải pháp:** Chuẩn hóa về **đúng 5 giây** cho mọi file.

**Tại sao cần độ dài cố định?**

Hầu hết mô hình deep learning (CNN, Transformer, LSTM...) yêu cầu **input tensor có kích thước cố định**. Nếu mỗi file có độ dài khác nhau:

```
File 1: [─────────────────────] 7.2s  → shape (115,200,)
File 2: [──────────] 3.8s             → shape (60,800,)
File 3: [─────────────] 5.0s          → shape (80,000,)
                ↓
Không thể stack thành batch tensor → lỗi khi training
```

Sau khi chuẩn hóa:
```
File 1: [─────────────────────→ CẮT] → shape (80,000,)
File 2: [──────────====PADDING=====] → shape (80,000,)
File 3: [─────────────────────]       → shape (80,000,)
                ↓
Stack được thành batch: (batch_size, 80,000) ✅
```

**Tại sao chọn 5 giây?**
- Đủ dài để chứa một câu nói hoàn chỉnh.
- Không quá dài gây lãng phí bộ nhớ.
- Chuẩn phổ biến trong các nghiên cứu speaker verification (VoxCeleb, GE2E...).

---

##### `TARGET_SAMPLES = 80,000` – Số điểm dữ liệu

Đây là **giá trị tính toán**, không cần chỉnh tay:

```python
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
              = int(16,000 × 5.0)
              = 80,000
```

- Đây là **kích thước vector đầu vào** của mỗi file âm thanh.
- Sau bước tiền xử lý, mọi file đều có shape `(80000,)`.
- Khi trích xuất MFCC: `(80000,)` → `(n_frames, n_mfcc)` (ví dụ: `(157, 40)`).

---

##### `MIN_DURATION = 1.0` – Ngưỡng thời lượng tối thiểu

```python
duration = len(y) / sr
return duration >= MIN_DURATION  # phải >= 1 giây
```

**Tại sao cần?**

Một số file trong dataset bị hỏng hoặc cực ngắn (< 1 giây), ví dụ:
- File chỉ thu được tiếng click của micro (0.05s)
- File bị cắt đột ngột (0.3s)

Những file này **không chứa đủ ngữ âm** để trích xuất đặc trưng có ý nghĩa:
```
File 0.3s ở 16kHz = 4,800 samples
→ MFCC chỉ tính được ~2 frames
→ Vector đặc trưng quá thưa, không đại diện cho giọng
```

→ Ngưỡng 1 giây đảm bảo phải có ít nhất **16,000 samples** làm đầu vào.

---

##### `MIN_RMS = 1e-4` – Ngưỡng phát hiện silence

**RMS (Root Mean Square)** là thước đo năng lượng trung bình của tín hiệu:

$$RMS = \sqrt{\frac{1}{N} \sum_{i=1}^{N} y_i^2}$$

```python
rms = float(np.sqrt(np.mean(y ** 2)))
return rms > MIN_RMS  # phải > 0.0001
```

**Ý nghĩa vật lý:**

| Loại tín hiệu | RMS điển hình | So với MIN_RMS |
|---|---|---|
| File hoàn toàn im lặng (zeros) | 0.0 | << 1e-4 → BỊ LOẠI |
| File chỉ có tiếng ồn nền nhẹ | ~1e-5 đến 1e-4 | ≈ 1e-4 → BỊ LOẠI |
| File có giọng nói | ~0.01 đến 0.3 | >> 1e-4 → HỢP LỆ |
| File có tiếng nói to | ~0.3 đến 1.0 | >> 1e-4 → HỢP LỆ |

**Tại sao chọn `1e-4` (= 0.0001)?**

Đây là ngưỡng phân tách giữa:
- "Tín hiệu vô nghĩa" (noise floor / silence) 
- "Tín hiệu có thông tin thực sự"

Nếu ngưỡng quá cao (ví dụ `0.01`) → loại nhầm giọng nói nhỏ.  
Nếu ngưỡng quá thấp (ví dụ `1e-8`) → không lọc được file silence.  
`1e-4` là ngưỡng **thực nghiệm phổ biến** trong xử lý âm thanh.

---

#### 4. Tham số sampling

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `TOTAL_FILES` | 500 | Tổng số file cần lấy sau cùng |
| `RANDOM_SEED` | 42 | Đảm bảo kết quả **reproducible** (chạy lại cho kết quả giống nhau) |

---

## 📦 Bước Import Thư Viện

### Mã nguồn

```python
import random
import logging
from collections import defaultdict, Counter

import numpy as np
import librosa
import soundfile as sf
from tqdm.notebook import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
```

### Giải thích từng thư viện

| Thư viện | Mục đích |
|---|---|
| `random` | Trộn ngẫu nhiên danh sách file (có seed cố định) |
| `logging` | Ghi log cảnh báo/thông tin trong quá trình chạy |
| `defaultdict` | Dictionary tự tạo key mới nếu chưa tồn tại (dùng cho grouping) |
| `Counter` | Đếm nhanh tần suất xuất hiện của từng speaker |
| `numpy` (np) | Xử lý mảng số – tính RMS, padding/slicing audio |
| `librosa` | Thư viện xử lý âm thanh: load file, resample tự động |
| `soundfile` (sf) | Ghi file `.wav` với encoding PCM_16 |
| `tqdm` | Hiển thị thanh tiến trình trong Jupyter Notebook |

**Cấu hình logging:**
- `level=INFO` → chỉ in thông tin từ mức INFO trở lên (bỏ qua DEBUG).
- `format` → định dạng: `11:20:37 [INFO] Loại bỏ 3 file ...`
- `datefmt="%H:%M:%S"` → chỉ hiển thị giờ:phút:giây, không cần ngày.

---

## ① Đọc & Lọc Female Speakers từ `speaker-info.txt`

### Mã nguồn

```python
def load_female_speakers(speaker_info_path: Path) -> list:
    """Trả về list speaker folder name của giọng nữ: ['p225', 'p228', ...]"""
    female_speakers = []
    with open(speaker_info_path, "r", encoding="utf-8") as f:
        next(f)  # bỏ header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == "F":
                female_speakers.append(f"p{parts[0]}")
    return female_speakers


female_speakers = load_female_speakers(SPEAKER_INFO)
```

**Kết quả:** `✅ Tìm thấy 61 female speakers`  
**Ví dụ:** `['p225', 'p228', 'p229', 'p230', 'p231', 'p233', 'p234', 'p236'] ...`

### Giải thích chi tiết

#### Cấu trúc file `speaker-info.txt`

```
ID   AGE  GENDER  ACCENTS  REGION
225   23     F    English  Southern England
226   22     M    English  Surrey
...
```

- Dòng đầu tiên là **header** → bỏ qua bằng `next(f)`.
- Mỗi dòng được tách bằng khoảng trắng (`.split()` không tham số → tách theo whitespace bất kỳ).

#### Luồng xử lý từng dòng

```
Dòng: "225   23     F    English  Southern England"
         ↓ .strip().split()
parts = ["225", "23", "F", "English", "Southern", "England"]
         ↓ parts[2] == "F"  → True
         ↓ f"p{parts[0]}"
kết quả: "p225"
```

#### Chi tiết từng dòng code

| Code | Giải thích |
|---|---|
| `line.strip()` | Xóa ký tự thừa ở đầu/cuối dòng (space, `\n`) |
| `.split()` | Tách chuỗi theo khoảng trắng → danh sách |
| `len(parts) >= 3` | Guard condition – tránh lỗi nếu dòng bị rỗng/thiếu cột |
| `parts[2] == "F"` | Cột GENDER (index 2) phải là `"F"` (Female) |
| `f"p{parts[0]}"` | Thêm tiền tố `p` vào ID speaker (ví dụ `225` → `p225`) để khớp với tên thư mục trong `wav48/` |

> **Lưu ý:** VCTK đặt tên thư mục theo dạng `p225`, `p226`... nên cần thêm `"p"` vào trước ID số.

---

## ② Duyệt Toàn Bộ File `.wav` của Female Speakers

### Mã nguồn

```python
def collect_wav_files(wav48_dir: Path, speakers: list) -> dict:
    """Trả về dict: { 'p225': [Path, ...], 'p228': [...] }"""
    files_by_speaker = {}
    for speaker in speakers:
        speaker_dir = wav48_dir / speaker
        if not speaker_dir.exists():
            log.warning(f"Không tìm thấy thư mục: {speaker_dir}")
            continue
        wav_list = sorted(speaker_dir.glob("*.wav"))
        if wav_list:
            files_by_speaker[speaker] = wav_list
    return files_by_speaker


files_by_speaker = collect_wav_files(WAV48_DIR, female_speakers)
```

**Kết quả:** `✅ Thu thập: 25124 file .wav từ 61 speakers`  
**Trung bình:** `412 file/speaker`

### Giải thích chi tiết

#### Cấu trúc thư mục `wav48/`

```
wav48/
├── p225/
│   ├── p225_001.wav
│   ├── p225_002.wav
│   └── ...         (~ 400 file)
├── p228/
│   └── ...
└── ...
```

#### Luồng xử lý

```python
for speaker in speakers:           # duyệt: 'p225', 'p228', ...
    speaker_dir = wav48_dir / speaker   # → wav48/p225/
    if not speaker_dir.exists():        # nếu không có thư mục → bỏ qua
        log.warning(...)
        continue
    wav_list = sorted(speaker_dir.glob("*.wav"))  # tìm tất cả .wav, sắp xếp
    if wav_list:                                   # chỉ thêm nếu có file
        files_by_speaker[speaker] = wav_list
```

#### Chi tiết từng thao tác

| Code | Giải thích |
|---|---|
| `wav48_dir / speaker` | Ghép đường dẫn (Path operator `/`), an toàn hơn `os.path.join` |
| `.exists()` | Kiểm tra thư mục tồn tại – bỏ qua speaker bị thiếu dữ liệu |
| `.glob("*.wav")` | Tìm tất cả file có đuôi `.wav` trong thư mục (không đệ quy) |
| `sorted(...)` | Sắp xếp theo tên → đảm bảo thứ tự nhất quán giữa các lần chạy |
| `if wav_list:` | Không thêm speaker rỗng vào dict |

#### Kiểu dữ liệu trả về

```python
{
    'p225': [Path('wav48/p225/p225_001.wav'), Path('wav48/p225/p225_002.wav'), ...],
    'p228': [Path('wav48/p228/p228_001.wav'), ...],
    # 61 speakers tổng cộng
}
```

---

## ③ Kiểm Tra Chất Lượng File Audio

### Mã nguồn

```python
def is_valid_audio(wav_path: Path) -> bool:
    try:
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        duration = len(y) / sr
        rms = float(np.sqrt(np.mean(y ** 2)))
        return duration >= MIN_DURATION and rms > MIN_RMS
    except Exception:
        return False


def filter_valid_files(files_by_speaker: dict) -> dict:
    all_files = [
        (spk, f)
        for spk, files in files_by_speaker.items()
        for f in files
    ]
    valid = defaultdict(list)
    rejected = 0
    for spk, wav_path in tqdm(all_files, desc="Kiểm tra chất lượng"):
        if is_valid_audio(wav_path):
            valid[spk].append(wav_path)
        else:
            rejected += 1
    log.info(f"Loại bỏ {rejected} file (lỗi / silence / quá ngắn)")
    return dict(valid)


valid_files = filter_valid_files(files_by_speaker)
```

**Kết quả:** `✅ Hợp lệ: 25121 / 25124 file` (loại bỏ 3 file)

### Giải thích chi tiết

#### Ba tiêu chí loại bỏ file

```
┌────────────────────────┬──────────────────────────────────┬──────────────────────────────┐
│ Lý do loại             │ Điều kiện                        │ Cách phát hiện               │
├────────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ File bị corrupt        │ librosa raise Exception          │ try/except → return False    │
│ File quá ngắn          │ duration < 1.0 giây              │ len(y) / sr < MIN_DURATION   │
│ File im lặng (silence) │ RMS ≈ 0 (< 1e-4)                │ sqrt(mean(y²)) <= MIN_RMS    │
└────────────────────────┴──────────────────────────────────┴──────────────────────────────┘
```

#### Phân tích hàm `is_valid_audio`

```python
y, sr = librosa.load(str(wav_path), sr=None, mono=True)
```
- `sr=None` → **giữ nguyên sample rate gốc** (48kHz) để tính thời lượng chính xác.
- `mono=True` → chuyển về 1 kênh (mono) nếu file là stereo.
- Nếu file **bị lỗi** (corrupt, thiếu header...) → librosa raise Exception → hàm trả về `False`.

```python
duration = len(y) / sr
```
- Tính thời lượng (giây) = số samples / sample rate.
- Ví dụ: 48,000 samples ở 48kHz = 1 giây.

```python
rms = float(np.sqrt(np.mean(y ** 2)))
```
- **RMS (Root Mean Square)** = căn bậc hai của trung bình bình phương các mẫu.
- Đây là chỉ số **năng lượng trung bình** của tín hiệu âm thanh.
- **File silence:** tất cả mẫu ≈ 0 → RMS ≈ 0 (<< 1e-4).
- **File có tiếng nói:** biên độ dao động lớn → RMS lớn.

```python
return duration >= MIN_DURATION and rms > MIN_RMS
```
- **Cả hai điều kiện** phải thỏa mãn để file được coi là hợp lệ.

```
duration >= 1.0 giây    AND    RMS > 0.0001
```

#### Hàm `filter_valid_files` – làm phẳng và lọc

```python
all_files = [
    (spk, f)
    for spk, files in files_by_speaker.items()
    for f in files
]
```
- **List comprehension lồng nhau** → chuyển dict `{speaker: [files]}` thành list phẳng `[(speaker, file), ...]`.
- Kết quả: 25,124 tuple `(speaker_id, Path)`.

```python
valid = defaultdict(list)
```
- `defaultdict(list)` → khi truy cập key chưa có, tự động tạo `[]`.
- Tránh phải kiểm tra `if key not in dict` trước khi `.append()`.

```python
for spk, wav_path in tqdm(all_files, desc="Kiểm tra chất lượng"):
    if is_valid_audio(wav_path):
        valid[spk].append(wav_path)
    else:
        rejected += 1
```
- Duyệt qua 25,124 file (có thanh tiến trình).
- Mỗi file hợp lệ → thêm vào dict theo speaker.
- File không hợp lệ → tăng bộ đếm.

---

## ④ Resample 48 kHz → 16 kHz & Cắt/Pad về 5 Giây

### Mã nguồn

```python
def preprocess_audio(wav_path: Path) -> np.ndarray:
    """
    Load → resample → cắt/pad → ndarray (80000,) float32.
    librosa.load tự xử lý resample 48kHz → 16kHz.
    """
    y, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)

    if len(y) >= TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]                          # cắt lấy 5 giây đầu
    else:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))     # pad zero ở cuối

    return y  # shape: (80_000,)
```

**Kết quả test:** `Output shape: (80000,)`, `dtype: float32`

### Giải thích chi tiết

#### 1. Load và Resample tự động

```python
y, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
```

- **`sr=TARGET_SR` (16,000)** → librosa tự động resample từ 48kHz xuống 16kHz.
- Thuật toán resampling: librosa dùng **sinc resampling** (chất lượng cao).
- **Tỷ lệ nén:** 48,000 → 16,000 = giảm 3 lần số samples.
- Kết quả `y` là mảng numpy `float32` trong khoảng `[-1.0, 1.0]` (normalized).
- `_` → bỏ qua giá trị sample rate trả về (luôn là 16,000).

**Ví dụ:**
```
File gốc: 48kHz, 6 giây → 288,000 samples (float32)
Sau resample: 16kHz, 6 giây → 96,000 samples (float32)
```

#### 2. Cắt (Truncate) – xử lý file dài hơn 5 giây

```python
if len(y) >= TARGET_SAMPLES:    # TARGET_SAMPLES = 80,000
    y = y[:TARGET_SAMPLES]      # lấy 80,000 mẫu đầu (5 giây đầu tiên)
```

- Nếu file dài hơn 5 giây → **lấy 5 giây đầu** (đầu câu nói thường có nhiều thông tin).
- Ví dụ: file 6 giây (96,000 samples) → cắt còn 80,000 samples.

#### 3. Pad Zero – xử lý file ngắn hơn 5 giây

```python
else:
    y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
```

- `np.pad(array, (before, after))` → thêm giá trị `0` vào đầu/cuối mảng.
- `(0, TARGET_SAMPLES - len(y))` → **không thêm trước, thêm zero ở sau**.
- Ví dụ: file 3 giây (48,000 samples) → thêm 32,000 zero ở cuối.
- **Tại sao pad zero?** Zero tương đương với im lặng → không làm sai lệch tín hiệu.

#### Minh họa trực quan

```
File > 5s (6s = 96,000 samples):
[████████████████████████████████████████████████] 96,000
→ cắt
[████████████████████████████████████] 80,000

File < 5s (3s = 48,000 samples):
[████████████████████████] 48,000
→ pad
[████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 80,000
                           ↑ zero padding

File == 5s (80,000 samples):
[████████████████████████████████████] 80,000 → không đổi
```

#### Output cuối cùng

```
y.shape = (80000,)
y.dtype = float32
Giá trị: [-1.0, 1.0]
```

---

## ⑤ Sampling 500 File – Phân Bổ Đều Theo Speaker

### Mã nguồn

```python
def sample_files(valid_files_by_speaker: dict, total: int = TOTAL_FILES, seed: int = RANDOM_SEED) -> list:
    """
    Trả về list[(speaker, Path)] đủ `total` phần tử, phân bổ đều theo speaker.
    """
    random.seed(seed)
    n_speakers = len(valid_files_by_speaker)
    quota      = total // n_speakers  # ~7-8 file/speaker

    selected     = []
    leftover_pool = []

    for spk, files in valid_files_by_speaker.items():
        shuffled = files[:]
        random.shuffle(shuffled)
        selected.extend((spk, f) for f in shuffled[:quota])
        leftover_pool.extend((spk, f) for f in shuffled[quota:])

    # Bù nếu chưa đủ TOTAL_FILES
    deficit = total - len(selected)
    if deficit > 0 and leftover_pool:
        selected.extend(random.sample(leftover_pool, min(deficit, len(leftover_pool))))

    random.shuffle(selected)  # xáo trộn để không nhóm theo speaker
    return selected[:total]


selected = sample_files(valid_files)
```

**Kết quả:** `✅ Đã chọn 500 file từ 61 speakers`  
**Phân bổ:** `min=8, max=9, avg=8.2 file/speaker`

### Giải thích chi tiết

#### Tại sao cần phân bổ đều (stratified sampling)?

> Nếu lấy ngẫu nhiên hoàn toàn: speaker có 500 file chiếm tỷ lệ cao hơn speaker có 100 file → embedding bị lệch về không gian đặc trưng của speaker nhiều file đó.

| Phương pháp | Kết quả |
|---|---|
| **Random ngẫu nhiên** | Speaker nhiều file chiếm đa số → mất cân bằng |
| **Stratified (đều)** | Mỗi speaker đóng góp ~8 file → mô hình học đồng đều |

#### Tính quota

```python
n_speakers = 61       # số speakers
quota = 500 // 61     # = 8 (phép chia nguyên)
```

- Mỗi speaker được lấy **8 file** (quota).
- 61 × 8 = 488 file → còn thiếu 12 file (deficit).

#### Thuật toán 3 bước

**Bước 1: Lấy quota từ mỗi speaker**

```python
for spk, files in valid_files_by_speaker.items():
    shuffled = files[:]        # copy danh sách (không sửa bản gốc)
    random.shuffle(shuffled)   # trộn ngẫu nhiên
    selected.extend((spk, f) for f in shuffled[:quota])      # lấy 8 file đầu
    leftover_pool.extend((spk, f) for f in shuffled[quota:]) # phần còn lại
```

- **`files[:]`** → copy danh sách để `shuffle` không ảnh hưởng bản gốc.
- Sau shuffle → lấy 8 file đầu (random bởi vì đã shuffle).
- Phần còn lại → pool dự phòng.

**Bước 2: Bù phần còn thiếu (deficit)**

```python
deficit = total - len(selected)   # 500 - 488 = 12
if deficit > 0 and leftover_pool:
    selected.extend(random.sample(leftover_pool, min(deficit, len(leftover_pool))))
```

- `random.sample(pool, k)` → lấy **k phần tử không trùng lặp** từ pool.
- `min(deficit, len(leftover_pool))` → guard nếu pool ít hơn số cần.
- 12 file được bù ngẫu nhiên từ leftover của các speakers.

**Bước 3: Xáo trộn cuối cùng**

```python
random.shuffle(selected)   # trộn để không nhóm theo speaker
return selected[:total]    # đảm bảo đúng 500 file
```

- Nếu không shuffle → batch đầu tiên trong training sẽ chứa toàn p225.
- Sau shuffle → các speaker phân bố đều trong toàn bộ danh sách.

---

## 💾 Lưu File Đã Tiền Xử Lý ra `data/processed/`

### Mã nguồn

```python
def save_processed_files(selected: list, output_dir: Path) -> list:
    """
    Tiền xử lý từng file và lưu ra output_dir/<speaker>/<filename>.wav.
    Trả về list metadata record để dùng cho bước insert DB.
    """
    records = []

    for spk, src_path in tqdm(selected, desc="Xử lý & lưu file"):
        dest_dir  = output_dir / spk
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name

        if dest_path.exists():
            continue  # bỏ qua nếu đã xử lý, chạy lại an toàn

        try:
            y = preprocess_audio(src_path)
            sf.write(str(dest_path), y, TARGET_SR, subtype="PCM_16")

            records.append({
                "speaker"         : spk,
                "file_path"       : str(dest_path.relative_to(output_dir.parent)),
                "src_path"        : str(src_path),
                "sample_rate"     : TARGET_SR,
                "duration_sec"    : TARGET_DURATION,
            })
        except Exception as e:
            log.warning(f"Lỗi khi xử lý {src_path.name}: {e}")

    return records


records = save_processed_files(selected, PROCESSED_DIR)
```

**Kết quả:** `✅ Đã lưu 500 file vào data/processed/`

### Giải thích chi tiết

#### Cấu trúc thư mục output

```
data/processed/
├── p225/
│   ├── p225_001.wav    ← 16kHz, 5s, PCM_16
│   ├── p225_002.wav
│   └── ...
├── p228/
│   └── p228_005.wav
└── ... (61 thư mục speaker)
```

#### Idempotency – An toàn khi chạy lại

```python
if dest_path.exists():
    continue  # bỏ qua nếu đã xử lý
```

- **Idempotent**: Chạy pipeline nhiều lần → kết quả như nhau, không ghi đè.
- Nếu pipeline bị dừng giữa chừng → chạy lại sẽ tiếp tục từ chỗ còn thiếu.
- Đảm bảo an toàn trong môi trường phát triển.

#### Tạo thư mục đích

```python
dest_dir  = output_dir / spk          # data/processed/p225/
dest_dir.mkdir(parents=True, exist_ok=True)
```

- `parents=True` → tạo tất cả thư mục cha nếu chưa có (`data/`, `processed/`).
- `exist_ok=True` → không báo lỗi nếu thư mục đã tồn tại.

#### Ghi file với PCM_16

```python
sf.write(str(dest_path), y, TARGET_SR, subtype="PCM_16")
```

- **`soundfile.write`** → ghi file `.wav` chuẩn.
- **`subtype="PCM_16"`** → mã hóa 16-bit integer (tiết kiệm ~50% dung lượng so với float32).
  - `float32`: 4 bytes/sample × 80,000 = 320 KB/file
  - `PCM_16`: 2 bytes/sample × 80,000 = 160 KB/file → **giảm một nửa**
- Chất lượng nghe vẫn tốt vì 16-bit PCM là chuẩn CD audio.

#### Metadata record cho Database

```python
records.append({
    "speaker"      : spk,           # 'p225'
    "file_path"    : str(dest_path.relative_to(output_dir.parent)),
                                    # 'processed/p225/p225_001.wav'
    "src_path"     : str(src_path), # đường dẫn raw gốc (để trace)
    "sample_rate"  : TARGET_SR,     # 16000
    "duration_sec" : TARGET_DURATION, # 5.0
})
```

- **`file_path`** dùng `relative_to(output_dir.parent)` → lưu đường dẫn **tương đối** so với `data/`.
  - `output_dir.parent` = `data/`
  - Kết quả: `processed/p225/p225_001.wav` (không phụ thuộc máy chủ)
- **`src_path`** lưu đường dẫn tuyệt đối file gốc → dễ trace lại nguồn.
- Các record này được dùng để **insert vào database** ở bước tiếp theo.

#### Error handling

```python
try:
    y = preprocess_audio(src_path)
    sf.write(...)
except Exception as e:
    log.warning(f"Lỗi khi xử lý {src_path.name}: {e}")
```

- Nếu file bị lỗi trong quá trình xử lý → ghi warning log, **không dừng toàn pipeline**.
- Toàn bộ pipeline vẫn tiếp tục xử lý các file còn lại.

---

## 📊 Báo Cáo Tổng Kết

### Mã nguồn

```python
final_counts = Counter(r["speaker"] for r in records)

print("=" * 55)
print("  TỔNG KẾT TIỀN XỬ LÝ")
print("=" * 55)
print(f"  Tổng file đã xử lý  : {len(records)}")
print(f"  Số speakers          : {len(final_counts)}")
if final_counts:
    print(f"  File/speaker (min)   : {min(final_counts.values())}")
    print(f"  File/speaker (max)   : {max(final_counts.values())}")
    print(f"  File/speaker (avg)   : {len(records)/len(final_counts):.1f}")
print(f"  Output directory     : {PROCESSED_DIR}")
print(f"  Sample rate          : {TARGET_SR} Hz")
print(f"  Duration             : {TARGET_DURATION}s ({TARGET_SAMPLES} samples)")
print("=" * 55)
```

### Output thực tế

```
=======================================================
  TỔNG KẾT TIỀN XỬ LÝ
=======================================================
  Tổng file đã xử lý  : 500
  Số speakers          : 61
  File/speaker (min)   : 8
  File/speaker (max)   : 9
  File/speaker (avg)   : 8.2
  Output directory     : C:\BTL_CSDLDPT\Beta\data\processed
  Sample rate          : 16000 Hz
  Duration             : 5.0s (80000 samples)
=======================================================

✅ Tiền xử lý hoàn tất! Sẵn sàng chạy feature extraction.
```

### Phân tích kết quả

- **min=8, max=9**: Phân bổ rất đồng đều (chênh lệch chỉ 1 file/speaker).
- **avg=8.2**: 500 / 61 ≈ 8.2 → đúng với tính toán quota.
- **`Counter(r["speaker"] for r in records)`**: Đếm tần suất speaker trong metadata records (chính xác hơn đếm từ `selected` vì loại bỏ các file xử lý lỗi).

---

## 🔄 Tóm Tắt Toàn Bộ Pipeline

```
Input: VCTK-Corpus/wav48/ (25,124 file, 48kHz, nhiều độ dài)
                │
                │ Bước ①: Lọc 61 female speakers từ speaker-info.txt
                │
                │ Bước ②: Duyệt 25,124 file .wav
                │
                │ Bước ③: Lọc 3 file lỗi/ngắn/silence → 25,121 hợp lệ
                │
                │ Bước ⑤: Stratified sampling → 500 file (8-9/speaker)
                │
                │ Bước ④: Resample 48kHz→16kHz + Cắt/Pad → 5 giây
                │
                │ Lưu: PCM_16 WAV, 160KB/file
                ▼
Output: data/processed/ (500 file, 16kHz, 5 giây cố định)
        → Sẵn sàng cho Feature Extraction (MFCC, Spectral Contrast, Chroma)
```

---

## 📋 Thống Kê Hiệu Năng Thực Tế

| Chỉ số | Giá trị |
|---|---|
| Tổng file đầu vào | 25,124 |
| File hợp lệ sau lọc | 25,121 (99.99%) |
| File bị loại bỏ | 3 (0.01%) |
| File được chọn sau sampling | 500 |
| Dung lượng mỗi file | ~160 KB (PCM_16) |
| Tổng dung lượng output | ~80 MB |
| Tỷ lệ nén âm thanh | 48kHz → 16kHz (giảm 3x samples) |
| Encoding | 32-bit float → 16-bit PCM (giảm 2x dung lượng) |

---

## 🚀 Bước Tiếp Theo

Sau khi chạy xong pipeline này, dữ liệu tại `data/processed/` sẵn sàng cho:

1. **Feature Extraction** (`data/features/`) – Trích xuất MFCC, Spectral Contrast, Chroma STFT.
2. **Database Insertion** – Insert metadata (từ `records`) vào PostgreSQL.
3. **Vector Storage** – Lưu embedding vectors vào `pgvector`.

---

*Tài liệu được tạo tự động từ `src/preprocessing/preprocess.ipynb`.*
