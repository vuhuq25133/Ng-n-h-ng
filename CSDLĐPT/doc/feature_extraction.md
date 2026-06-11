# 🎵 Tài Liệu Triển Khai – Pipeline Trích Xuất Đặc Trưng Âm Thanh

> **File nguồn:** `src/feature_extraction/feature_extraction.ipynb`  
> **Ngày tài liệu:** 2026-04-12  
> **Dữ liệu đầu vào:** 500 file `.wav` chuẩn hóa (16 kHz, 5 giây) tại `data/processed/`  
> **Đầu ra:** Vector 99 chiều (`float32`) + file `.npy` thô tại `data/features/`

---

## 📌 Tổng Quan Pipeline

```
data/processed/<speaker>/<file>.wav   (16kHz, 5s, 80,000 samples)
        │
        ▼
[1] MFCC mean + std         → 40 + 40 = 80 chiều
[2] Spectral Contrast mean  → 7 chiều
[3] Chroma STFT mean        → 12 chiều
        │
        ▼
concat → Vector Embedding (99,) float32
        │
        ├──→ pgvector: VECTOR(99) trong PostgreSQL
        └──→ File: data/features/<speaker>/<file>.npy  (đặc trưng thô MFCC)
```

### Kết quả đầu ra

| Thông số | Giá trị |
|---|---|
| Tổng file xử lý | 500 |
| Chiều vector embedding | 99 |
| Đặc trưng thành phần | MFCC×2, Spectral Contrast, Chroma STFT |
| Kiểu dữ liệu | `float32` |
| File đặc trưng thô | `.npy` (MFCC per-frame) |

---

## 🧠 Phần I: Kiến Thức Nền Tảng

### 1.1 Âm thanh Số là gì?

Âm thanh trong thực tế là **sóng áp suất liên tục** lan truyền qua không khí. Khi ghi âm, micro chuyển sóng âm thành tín hiệu điện, sau đó bộ chuyển đổi ADC (Analog-to-Digital Converter) **lấy mẫu** tín hiệu đó theo từng khoảng thời gian đều nhau.

```
Sóng âm liên tục:
  ╭──╮      ╭──╮
╭╯  ╰──╮╭──╯  ╰╮   →→→  thời gian
         ╰╯

Sau khi lấy mẫu 16,000 lần/giây:
  |    ||  ||    |
  |  | || |||  | |   →→→  16,000 giá trị số / giây
```

Mỗi giá trị số (sample) là biên độ tín hiệu tại thời điểm đó, thường ở dạng `float32` trong khoảng `[-1.0, 1.0]`.

**File WAV sau tiền xử lý của chúng ta:**
- Sample rate: 16,000 Hz (16,000 mẫu/giây)
- Thời lượng: 5 giây
- Tổng mẫu: 16,000 × 5 = **80,000 giá trị float32**
- Shape numpy: `(80000,)`

---

### 1.2 Miền Thời Gian vs Miền Tần Số

Âm thanh có thể được phân tích từ hai góc nhìn:

| Góc nhìn | Miền | Trả lời câu hỏi | Ví dụ |
|---|---|---|---|
| Waveform (dạng sóng) | **Thời gian** | "Biên độ là bao nhiêu tại thời điểm t?" | Xem sóng âm trực tiếp |
| Spectrum (phổ) | **Tần số** | "Tần số nào có mặt và mạnh yếu ra sao?" | Phân tích giọng nói |

**Vấn đề:** Waveform chứa ít thông tin về đặc tính giọng nói.  
**Giải pháp:** Chuyển sang **miền tần số** bằng Fourier Transform.

---

### 1.3 Fourier Transform – Nền tảng của mọi đặc trưng

#### Ý tưởng cốt lõi

> **Mọi tín hiệu phức tạp đều có thể phân tích thành tổng của nhiều sóng sin đơn giản có tần số khác nhau.**

```
Tín hiệu phức tạp = sin(100Hz) × a₁  +  sin(200Hz) × a₂  +  sin(3400Hz) × a₃  + ...
                     ↑ thành phần 1         ↑ thành phần 2        ↑ thành phần 3
```

Fourier Transform tìm ra **biên độ (aᵢ)** của từng thành phần tần số đó.

#### Công thức DFT (Discrete Fourier Transform)

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi kn/N}$$

**Trong đó:**
- $x[n]$ = giá trị tín hiệu tại mẫu thứ $n$ (input)
- $X[k]$ = biên độ phức tại tần số thứ $k$ (output)
- $N$ = tổng số mẫu trong cửa sổ
- $k$ = chỉ số tần số (0 → N-1)
- $e^{-j2\pi kn/N}$ = "nguyên tử" sin+cos tại tần số $k/N$

**Chuyển sang biên độ thực (Power Spectrum):**

$$|X[k]|^2 = \text{Re}(X[k])^2 + \text{Im}(X[k])^2$$

```
Input (thời gian):   [x₀, x₁, x₂, ..., x₂₀₄₇]  (2048 mẫu)
        │
      DFT/FFT
        │
Output (tần số):     [|X₀|², |X₁|², ..., |X₁₀₂₄|²]  (1025 giá trị)
                      ↑ DC    ↑ 7.8Hz       ↑ 8000Hz
```

**FFT (Fast Fourier Transform)** là thuật toán tính DFT nhanh hơn rất nhiều (O(N log N) thay vì O(N²)).

#### Tham số `n_fft = 2048`

```
n_fft = 2048 mẫu  →  cửa sổ thời gian = 2048/16000 = 0.128 giây (128ms)
→ Độ phân giải tần số = 16000/2048 ≈ 7.8 Hz/bin
→ Số bin tần số hữu dụng = n_fft/2 + 1 = 1025 bins
→ Tần số cao nhất = 16000/2 = 8000 Hz (định lý Nyquist)
```

---

### 1.4 Short-Time Fourier Transform (STFT) – FFT theo từng khung

**Vấn đề với DFT thông thường:** Nếu áp dụng cho toàn bộ 80,000 mẫu, ta chỉ biết tần số nào có mặt nhưng **không biết lúc nào** chúng xuất hiện. Giọng nói lại thay đổi theo thời gian (âm "b" khác "a" khác "t").

**Giải pháp: STFT** – Chia audio thành nhiều **khung (frame)** ngắn chồng lấp nhau, áp FFT cho từng khung.

```
Audio 80,000 samples:
┌────────────────────────────────────────────────────┐
│ s₀  s₁  s₂  ... s₇₉₉₉₉                           │
└────────────────────────────────────────────────────┘

Chia thành frames (n_fft=2048, hop_length=512):
Frame 0:   [s₀    ... s₂₀₄₇]     →  FFT  →  X₀[k]
Frame 1:   [s₅₁₂  ... s₂₅₅₉]    →  FFT  →  X₁[k]
Frame 2:   [s₁₀₂₄ ... s₃₀₇₁]    →  FFT  →  X₂[k]
...
Frame 156: [...]                  →  FFT  →  X₁₅₆[k]

Kết quả STFT: shape (1025, 157)  ← (tần số × thời gian)
```

#### Tính số frames

$$n\_frames = \left\lfloor \frac{N - n\_fft}{hop\_length} \right\rfloor + 1 = \left\lfloor \frac{80000 - 2048}{512} \right\rfloor + 1 = 153 + 1 = 154 \approx 157$$

> librosa thêm padding nên thực tế ra **157 frames**.

#### Windowing – Hamming Window

Trước khi FFT, nhân mỗi frame với **hàm cửa sổ Hamming** để tránh hiện tượng "spectral leakage" (rò rỉ phổ) ở biên frame:

$$w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right), \quad 0 \le n \le N-1$$

```
Không có window:  [■■■■■■■■■■■■■■■■]  → biên cắt đột ngột → nhiễu FFT
Có Hamming:       [░▒▓████████▓▒░]   → biên mượt → FFT chính xác hơn
```

---

### 1.5 Thang Mel – Cách Tai Người Nghe

#### Vấn đề với thang Hz tuyến tính

Tai người **không nghe tuyến tính** theo Hz. Sự khác biệt giữa 100 Hz và 200 Hz (chênh 100 Hz) nghe rõ hơn nhiều so với sự khác biệt giữa 8000 Hz và 8100 Hz (cũng chênh 100 Hz). Tai người nhạy hơn ở tần số thấp.

#### Thang Mel – Xấp xỉ cảm nhận của tai người

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

$$f = 700 \cdot \left(10^{m/2595} - 1\right)$$

| Tần số (Hz) | Mel |
|---|---|
| 0 | 0 |
| 100 | 150 |
| 500 | 607 |
| 1,000 | 1,000 |
| 4,000 | 1,789 |
| 8,000 | 2,146 |

**Quan sát:** Từ 0→1000 Hz : tăng 1000 Mel. Nhưng từ 1000→8000 Hz: chỉ tăng ~1146 Mel → Mel nén tần số cao.

```
Hz   :  0   500  1k   2k   4k    8k
        |    |    |    |    |     |
Mel  :  0   600  1k   1.4k 1.8k  2.1k   ← phân bố không đều
        
Mel filter: ≡ ≡ ≡ ≡ ≡ ≡ ≡ ≡ ≡ ≡ ≡ ... ← phân bố đều trên Mel
(filter thưa ở tần số cao, dày ở tần số thấp trên scaleHz)
```

#### Mel Filterbank

Bộ lọc tam giác (triangular filters) phân bố đều trên Mel scale, áp lên phổ FFT:

$$\text{FilterBank}[m] = \sum_{k=f[m]}^{f[m+2]} H_m[k] \cdot |X[k]|^2$$

Với `n_mels = 128`:
- 128 bộ lọc tam giác
- Mỗi bộ lọc "tổng hợp" năng lượng trong dải tần tương ứng → 128 giá trị

Kết quả: **Mel Spectrogram** shape `(128, 157)`.

---

### 1.6 Cepstrum & DCT – Nền tảng của MFCC

#### Cepstrum là gì?

Từ **"cepstrum"** là từ "spectrum" viết ngược một nửa. Đây là Fourier Transform của logarithm của phổ tần số.

Ý tưởng: Giọng nói gồm hai thành phần **nhân** lên nhau:
- **Excitation** (nguồn kích thích): dây thanh âm rung → chuỗi xung tuần hoàn
- **Vocal tract filter** (bộ lọc ống cộng hưởng): hình dạng họng/miệng/mũi → biến dạng phổ

$$S[k] = E[k] \times H[k]$$

Lấy log:
$$\log S[k] = \log E[k] + \log H[k]$$

Giờ hai thành phần **cộng** (không nhân) → có thể tách bằng cepstrum (tương tự low-pass filter trong miền quefrency).

#### DCT (Discrete Cosine Transform)

DCT nén thông tin của 128 Mel bands thành `n_mfcc = 40` hệ số:

$$C[n] = \sum_{m=0}^{M-1} \log(\text{FilterBank}[m]) \cdot \cos\left(\frac{\pi n (m + 0.5)}{M}\right)$$

**Trong đó:**
- $M$ = số Mel bands (128)
- $n$ = chỉ số MFCC (0 → 39)
- $C[n]$ = hệ số MFCC thứ n

**Tại sao DCT quan trọng hơn DFT ở đây?**
- DCT có năng lượng **tập trung ở các hệ số đầu** (số nhỏ) → các hệ số đầu chứa nhiều thông tin nhất
- DCT tạo ra hệ số **không tương quan** (decorrelated) → tốt hơn cho mô hình ML
- Lấy 40 hệ số đầu = giữ thông tin quan trọng nhất, bỏ thành phần nhiễu tần số cao

---

## 🔹 Phần II: MFCC – Mel-Frequency Cepstral Coefficients

### 2.1 Tổng quan

MFCC là đặc trưng âm thanh **phổ biến nhất** trong nhận dạng giọng nói, mô phỏng cách hệ thống thính giác con người xử lý âm thanh.

**Pipeline tính MFCC (7 bước):**

```
Audio waveform (80,000 samples)
    │
    │ [1] Pre-emphasis
    ▼
Tín hiệu tăng cường tần số cao
    │
    │ [2] Framing (n_fft=2048, hop_length=512)
    ▼
157 frames × 2048 samples
    │
    │ [3] Windowing (Hamming)
    ▼
157 frames × 2048 samples (đầu/cuối mượt)
    │
    │ [4] FFT → Power Spectrum
    ▼
shape: (1025, 157)   ← 1025 tần số bins × 157 frames
    │
    │ [5] Mel Filterbank (n_mels=128)
    ▼
shape: (128, 157)    ← 128 Mel bands × 157 frames
    │
    │ [6] Log compression
    ▼
shape: (128, 157)    ← log-Mel spectrogram
    │
    │ [7] DCT → MFCC (n_mfcc=40)
    ▼
shape: (40, 157)     ← 40 hệ số × 157 frames
```

### 2.2 Bước 1 – Pre-emphasis (Tăng cường tần số cao)

```python
y_preemph = np.append(y[0], y[1:] - 0.97 * y[:-1])
```

**Công thức:**

$$y'[n] = y[n] - \alpha \cdot y[n-1], \quad \alpha = 0.97$$

**Ý nghĩa:**
- Giọng nói tự nhiên có **năng lượng tần số cao yếu hơn tần số thấp** (do đặc tính vật lý dây thanh âm).
- Pre-emphasis bù lại bằng cách làm nổi bật tần số cao → cân bằng phổ → tính MFCC chính xác hơn.
- `α = 0.97` là giá trị thực nghiệm phổ biến trong ASR.

```
Trước pre-emphasis:  [████████▆▄▃▂▁] ← năng lượng giảm dần theo tần số
Sau pre-emphasis:    [████████████████] ← phẳng hơn, tần số cao được boost
```

> **Lưu ý:** librosa.feature.mfcc() thực hiện nội bộ, không cần gọi tường minh.

### 2.3 Bước 2–3 – Framing và Windowing

```python
# librosa tự động framing + windowing khi gọi:
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512, n_mels=128)
```

**Tham số:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `n_fft` | 2048 | Kích thước cửa sổ FFT (số mẫu/frame) |
| `hop_length` | 512 | Bước nhảy giữa các frame (số mẫu) |
| `n_mels` | 128 | Số Mel filter banks |
| `n_mfcc` | 40 | Số hệ số MFCC giữ lại |

**Thời gian tương ứng:**
- Mỗi frame: 2048/16000 = **128 ms**
- Bước nhảy: 512/16000 = **32 ms**
- Overlap: (2048-512)/16000 = **96 ms** → ~75% overlap

### 2.4 Bước 4–7 – FFT, Mel, Log, DCT

Đã giải thích đầy đủ ở Phần I. Sau toàn bộ pipeline:

```
Đầu ra MFCC: np.ndarray shape (40, 157)
- Axis 0 (hàng): 40 hệ số MFCC
- Axis 1 (cột): 157 frames theo thời gian

Ví dụ:
         frame0   frame1   frame2   ...  frame156
MFCC 0:  -412.3   -398.7   -401.2   ...   -415.0
MFCC 1:   112.4    118.9    115.3   ...    110.2
MFCC 2:    23.7     19.4     22.1   ...     24.8
...
MFCC 39:   -1.2      0.8     -0.3   ...      1.1
```

### 2.5 Tổng hợp: Mean & Std theo thời gian

**Vấn đề:** Model ML cần input có kích thước cố định, nhưng số frames phụ thuộc vào độ dài audio. Giải pháp: **thống kê theo trục thời gian**.

```python
mfcc_mean = np.mean(mfcc, axis=1)   # shape: (40,)
mfcc_std  = np.std(mfcc,  axis=1)   # shape: (40,)
```

**Ý nghĩa từng chiều:**

$$\text{MFCC\_mean}[i] = \frac{1}{T} \sum_{t=0}^{T-1} C_i[t], \quad i = 0, 1, ..., 39$$

$$\text{MFCC\_std}[i] = \sqrt{\frac{1}{T} \sum_{t=0}^{T-1} \left(C_i[t] - \text{mean}_i\right)^2}$$

**Tại sao cần cả mean VÀ std?**

```
Speaker A (giọng đều):    MFCC[0] = [-400, -402, -401, -399, -400]  → mean=-400.4,  std=1.1
Speaker B (giọng biến động): MFCC[0] = [-430, -380, -410, -370, -420]  → mean=-402.0,  std=23.7

→ Mean gần như giống nhau!
→ Nhưng Std khác biệt rõ ràng (1.1 vs 23.7) → phân biệt được hai giọng
```

- **Mean** → phản ánh "màu sắc âm thanh trung bình" (formant tổng quát)
- **Std** → phản ánh "độ biến thiên" (tốc độ nói, nhịp điệu, cảm xúc)

### 2.6 Ý nghĩa 40 hệ số MFCC

| Hệ số | Ý nghĩa vật lý |
|---|---|
| **C0** | Năng lượng tổng thể (log-energy), thường bỏ hoặc thay bằng log-energy riêng |
| **C1** | Hình dạng phổ tổng quát – âm sắc (timbre) cơ bản |
| **C2–C5** | Hình dạng formant chính (F1, F2, F3) – phân biệt nguyên âm |
| **C6–C12** | Chi tiết cộng hưởng ống cộng hưởng – đặc trưng giọng nói cá nhân |
| **C13–C39** | Chi tiết cao tần – phụ âm, đặc điểm micro, điều kiện thu âm |

**Tại sao 40?** Thực nghiệm cho thấy 13–40 hệ số là optimal cho speaker recognition. Ít hơn → mất thông tin, nhiều hơn → thêm nhiễu. 40 là standard trong speaker verification.

**Tổng MFCC đóng góp vào vector: 40 (mean) + 40 (std) = 80 chiều (chiều 0–79)**

---

## 🔸 Phần III: Spectral Contrast

### 3.1 Tổng quan

Spectral Contrast đo **sự tương phản về năng lượng** giữa đỉnh phổ và thung lũng phổ trong từng dải tần. Phản ánh đặc tính cộng hưởng của ống thanh quản.

**Ý tưởng trực quan:**

```
Giọng "A" (nguyên âm rõ ràng):
Phổ:  ▁▃████▃▁  ← có đỉnh formant rõ, thung lũng sâu → contrast CAO
Peak: ████
Valley: ▁▁

Noise / giọng không rõ:
Phổ:  ▃▄▄▄▄▃▃  ← phẳng, không có đỉnh rõ → contrast THẤP
Peak: ▄▄
Valley: ▃▃
```

### 3.2 Thuật toán tính Spectral Contrast

**Bước 1: Chia phổ thành n_bands+1 dải tần**

Với `n_bands=6`, `fmin=200` Hz:

```
Dải 0 (sub-band):  200  Hz →  400  Hz
Dải 1:             400  Hz →  800  Hz
Dải 2:             800  Hz →  1600 Hz
Dải 3:             1600 Hz →  3200 Hz
Dải 4:             3200 Hz →  6400 Hz
Dải 5:             6400 Hz →  8000 Hz
Dải 6 (valley):    0    Hz →  200  Hz  (bass band riêng biệt)
```

→ Tổng cộng: **6 + 1 = 7 dải**

**Bước 2: Tìm Peak và Valley trong mỗi dải**

Trong mỗi sub-band, sắp xếp các bin FFT theo năng lượng:

$$\text{Peak}[b] = \text{mean}\left(\text{top } \alpha\% \text{ bins trong dải } b\right)$$
$$\text{Valley}[b] = \text{mean}\left(\text{bottom } \alpha\% \text{ bins trong dải } b\right)$$

Mặc định `α = 0.02` (tức top/bottom 2%).

**Bước 3: Tính Contrast**

$$\text{Contrast}[b] = \text{Peak}[b] - \text{Valley}[b]$$

**Đơn vị:** dB (decibel), vì cả Peak và Valley đã ở log scale.

**Bước 4: Tổng hợp theo thời gian**

```python
spectral_contrast = librosa.feature.spectral_contrast(
    y=y, sr=sr, n_bands=6, fmin=200.0
)
# shape: (7, 157)

sc_mean = np.mean(spectral_contrast, axis=1)   # shape: (7,)
```

### 3.3 Ý nghĩa 7 chiều Spectral Contrast

| Chiều | Dải tần | Ý nghĩa |
|---|---|---|
| **SC[0]** | 0 – 200 Hz | Năng lượng bass (cơ âm, "độ trầm" giọng) |
| **SC[1]** | 200 – 400 Hz | Vùng giọng nói thấp, cộng hưởng ngực |
| **SC[2]** | 400 – 800 Hz | Formant F1 – phân biệt nguyên âm |
| **SC[3]** | 800 – 1600 Hz | Formant F2 – nguyên âm trước/sau |
| **SC[4]** | 1600 – 3200 Hz | Formant F3 – đặc trưng giọng cá nhân |
| **SC[5]** | 3200 – 6400 Hz | Phụ âm sibilant (s, sh, f) |
| **SC[6]** | 6400 – 8000 Hz | Tần số cao – tiếng rì, đặc tính micro |

**Tại sao Spectral Contrast bổ sung cho MFCC?**

- MFCC tổng hợp hình dạng phổ → mất thông tin về **tương phản** (âm thanh nhiều năng lượng ở dải nào)
- Spectral Contrast bắt được tính bật ("punchy") hay mượt của giọng
- Hai người có MFCC tương tự nhưng SC khác → phân biệt được tốt hơn

**Tổng Spectral Contrast đóng góp: 7 chiều (chiều 80–86)**

---

## 🔶 Phần IV: Chroma STFT

### 4.1 Tổng quan

Chroma (hay **Chromagram**) phân tích âm thanh theo **12 lớp cao độ âm nhạc** (pitch classes), tương ứng 12 nốt trong một quãng tám: C, C#, D, D#, E, F, F#, G, G#, A, A#, B.

**Ý tưởng:** Tất cả tần số là bội số của nhau cách 2 lần (một quãng tám) đều chia sẻ cùng "màu sắc âm nhạc" (chroma).

```
C4  = 261.63 Hz
C5  = 523.25 Hz  (= 2 × C4)
C6  = 1046.5 Hz  (= 4 × C4)

→ Tất cả đều là "C" về mặt chroma
```

### 4.2 Mối liên hệ Tần số – Pitch Class

**Công thức tính pitch class từ tần số:**

$$p = 12 \cdot \log_2\left(\frac{f}{f_{ref}}\right) \mod 12$$

**Trong đó:**
- $f$ = tần số (Hz)
- $f_{ref}$ = 440 Hz (nốt A4, chuẩn quốc tế)
- $p$ = pitch class (0 = C, 1 = C#, ..., 11 = B)

**Bảng ánh xạ:**

| p | Nốt | Tần số (quãng 4) |
|---|---|---|
| 0 | C | 261.63 Hz |
| 1 | C# / Db | 277.18 Hz |
| 2 | D | 293.66 Hz |
| 3 | D# / Eb | 311.13 Hz |
| 4 | E | 329.63 Hz |
| 5 | F | 349.23 Hz |
| 6 | F# / Gb | 369.99 Hz |
| 7 | G | 392.00 Hz |
| 8 | G# / Ab | 415.30 Hz |
| 9 | A | 440.00 Hz |
| 10 | A# / Bb | 466.16 Hz |
| 11 | B | 493.88 Hz |

### 4.3 Thuật toán Chroma STFT

**Bước 1: Tính STFT** → Ma trận phổ `(1025, 157)`

**Bước 2: Ánh xạ tần số → pitch class**

Mỗi bin FFT thuộc về một pitch class:

```python
chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
# shape: (12, 157)
```

Với mỗi frame, librosa:
1. Tính STFT
2. Với mỗi bin tần số, ánh xạ vào pitch class (fold octaves)
3. Tổng hợp năng lượng của tất cả tần số thuộc cùng pitch class

**Bước 3: Tổng hợp theo thời gian**

```python
chroma_mean = np.mean(chroma, axis=1)   # shape: (12,)
```

### 4.4 Ý nghĩa 12 chiều Chroma

| Chiều | Nốt | Ý nghĩa trong giọng nói |
|---|---|---|
| **Ch[0]** | C | Năng lượng ở các tần số C và bội số |
| **Ch[1]** | C# | — |
| **Ch[2]** | D | — |
| ... | ... | ... |
| **Ch[9]** | A | Thường cao vì A440 nằm trong dải F2-F3 |
| **Ch[11]** | B | — |

**Tại sao Chroma hữu ích cho giọng nói?**

- Cao độ giọng nói (pitch) tự nhiên khác nhau giữa các người → phân bố năng lượng chroma khác nhau
- Giọng cao → chủ yếu năng lượng ở pitch class cao
- Giọng trầm → chủ yếu ở pitch class thấp
- Chroma **bất biến với quãng tám** → robust hơn MFCC khi so sánh hai giọng nói ở tốc độ khác nhau

**Tổng Chroma đóng góp: 12 chiều (chiều 87–98)**

---

## 🔷 Phần V: Vector Embedding 99 Chiều

### 5.1 Cấu trúc Vector

```python
embedding = np.concatenate([mfcc_mean, mfcc_std, sc_mean, chroma_mean])
# shape: (99,)
```

**Chi tiết từng phân đoạn:**

```
Indices   Đặc trưng              Dims  Phản ánh
─────────────────────────────────────────────────────────────────
[0:40]    MFCC mean              40    Hình dạng phổ Mel trung bình
[40:80]   MFCC std               40    Độ biến động phổ theo thời gian
[80:87]   Spectral Contrast mean  7    Tương phản năng lượng các dải tần
[87:99]   Chroma STFT mean       12    Phân bố năng lượng theo pitch class
─────────────────────────────────────────────────────────────────
TOTAL                            99
```

### 5.2 Bảng Tổng Hợp Đầy Đủ

| # | Đặc trưng | Chiều | Thư viện | Tham số chính | Nắm bắt |
|---|---|---|---|---|---|
| 1 | MFCC mean | 40 | `librosa.feature.mfcc` | `n_mfcc=40, n_fft=2048, hop_length=512, n_mels=128` | Màu sắc âm thanh trung bình |
| 2 | MFCC std | 40 | `librosa.feature.mfcc` | (same) | Độ biến thiên giọng nói |
| 3 | Spectral Contrast mean | 7 | `librosa.feature.spectral_contrast` | `n_bands=6, fmin=200` | Tương phản phổ tần số |
| 4 | Chroma STFT mean | 12 | `librosa.feature.chroma_stft` | `n_fft=2048, hop_length=512` | Phân bố pitch class |
| **Tổng** | | **99** | | | |

### 5.3 Cosine Similarity – Đo Độ Tương Đồng

Hai vector embedding được so sánh bằng **Cosine Similarity**:

$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|} = \frac{\sum_{i=0}^{98} A_i B_i}{\sqrt{\sum_{i=0}^{98} A_i^2} \cdot \sqrt{\sum_{i=0}^{98} B_i^2}}$$

| Giá trị | Ý nghĩa |
|---|---|
| 1.0 | Hai giọng giống y hệt nhau |
| 0.9–1.0 | Rất tương đồng |
| 0.7–0.9 | Tương đồng vừa phải |
| < 0.7 | Khác biệt nhiều |

**Tại sao Cosine thay vì Euclidean?**
- Cosine bất biến với **độ lớn** của vector (chỉ quan tâm hướng)
- Hai file cùng giọng nhưng âm lượng khác nhau → cosine similarity vẫn gần 1.0
- Euclidean sẽ bị ảnh hưởng bởi âm lượng → kết quả sai

---

## ⚙️ Phần VI: Cấu Hình và Tham Số

```python
# ─── Audio ────────────────────────────────────────
TARGET_SR = 16_000          # Hz – đã xử lý ở bước tiền xử lý

# ─── MFCC ─────────────────────────────────────────
N_MFCC      = 40            # Số hệ số MFCC giữ lại
N_FFT       = 2048          # Kích thước cửa sổ FFT
HOP_LENGTH  = 512           # Bước nhảy giữa các frame
N_MELS      = 128           # Số Mel filter banks

# ─── Spectral Contrast ────────────────────────────
SC_N_BANDS  = 6             # Số dải tần (→ 7 hệ số kể cả bass band)
SC_FMIN     = 200.0         # Hz – tần số bắt đầu dải đầu tiên

# ─── Chroma ───────────────────────────────────────
# dùng chung N_FFT và HOP_LENGTH

# ─── Output ───────────────────────────────────────
EMBEDDING_DIM = N_MFCC * 2 + (SC_N_BANDS + 1) + 12  # = 99
```

**Lý do chọn các giá trị:**

| Tham số | Lý do chọn |
|---|---|
| `n_mfcc=40` | Standard trong speaker recognition; 13 cho ASR, 40 cho speaker ID |
| `n_fft=2048` | 128ms cửa sổ tại 16kHz – đủ dài để có đủ chu kỳ pitch (~10ms) |
| `hop_length=512` | 75% overlap – cân bằng giữa độ phân giải và tốc độ |
| `n_mels=128` | Đủ Mel bands để capture formant; nhiều hơn không cải thiện đáng kể |
| `n_bands=6` | 6 sub-bands chuẩn trong librosa |
| `fmin=200` | Bỏ qua tần số rất thấp (<200Hz) thường là noise/hum |

---

## 💻 Phần VII: Giải Thích Code Chi Tiết

### 7.1 Cell 0 – Cấu hình

```python
from pathlib import Path

PROJECT_ROOT  = Path().resolve().parents[1]   # Beta/
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"

TARGET_SR  = 16_000
N_MFCC     = 40
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
SC_N_BANDS = 6
SC_FMIN    = 200.0

EMBEDDING_DIM = N_MFCC * 2 + (SC_N_BANDS + 1) + 12  # = 99
```

- `PROCESSED_DIR` → nơi đọc file `.wav` đã tiền xử lý
- `FEATURES_DIR` → nơi lưu file `.npy` đặc trưng thô
- `EMBEDDING_DIM = 99` → kiểm tra tổng chiều đúng với công thức

### 7.2 Cell 1 – Import

```python
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm.notebook import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
```

| Thư viện | Vai trò |
|---|---|
| `librosa` | Tất cả tính toán đặc trưng âm thanh |
| `numpy` | Xử lý mảng, concat vector |
| `soundfile` | Đọc file `.wav` (nhanh hơn librosa.load cho wav thuần túy) |
| `tqdm` | Thanh tiến trình trong notebook |
| `json` | Lưu metadata records |

### 7.3 Cell 2 – Hàm `extract_features()`

```python
def extract_features(wav_path: Path) -> dict:
    """
    Trích xuất MFCC, Spectral Contrast, Chroma từ một file .wav.
    
    Returns:
        dict với keys: 'mfcc_mean', 'mfcc_std', 'sc_mean', 'chroma_mean'
        mỗi key là np.ndarray float32
    """
    # 1. Load audio (đã ở 16kHz, 5s → không cần resample)
    y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    # y.shape = (80000,), dtype=float32

    # 2. MFCC (40 hệ số × 157 frames)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    # shape: (40, 157)
    mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)  # (40,)
    mfcc_std  = np.std(mfcc,  axis=1).astype(np.float32)  # (40,)

    # 3. Spectral Contrast (7 hệ số × 157 frames)
    sc = librosa.feature.spectral_contrast(
        y=y, sr=sr,
        n_bands=SC_N_BANDS,
        fmin=SC_FMIN
    )
    # shape: (7, 157)
    sc_mean = np.mean(sc, axis=1).astype(np.float32)       # (7,)

    # 4. Chroma STFT (12 pitch class × 157 frames)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    # shape: (12, 157)
    chroma_mean = np.mean(chroma, axis=1).astype(np.float32) # (12,)

    return {
        "mfcc_mean"   : mfcc_mean,    # (40,)
        "mfcc_std"    : mfcc_std,     # (40,)
        "sc_mean"     : sc_mean,      # (7,)
        "chroma_mean" : chroma_mean,  # (12,)
        "mfcc_raw"    : mfcc,         # (40, 157) – dùng cho .npy và DTW
    }
```

**Giải thích từng bước:**

| Dòng | Giải thích |
|---|---|
| `librosa.load(..., sr=TARGET_SR)` | Load + resample nếu cần (file đã ở 16kHz → không resample) |
| `librosa.feature.mfcc(...)` | Tự động: pre-emphasis → framing → windowing → FFT → Mel → Log → DCT |
| `np.mean(mfcc, axis=1)` | `axis=1` = trung bình theo cột (thời gian), giữ lại hàng (hệ số) |
| `np.std(mfcc, axis=1)` | Độ lệch chuẩn theo cột |
| `librosa.feature.spectral_contrast(...)` | FFT → chia sub-band → peak - valley mỗi band |
| `librosa.feature.chroma_stft(...)` | STFT → fold frequencies vào 12 pitch classes |
| `.astype(np.float32)` | Đảm bảo kiểu dữ liệu nhất quán, tiết kiệm bộ nhớ |
| `"mfcc_raw": mfcc` | Giữ lại toàn bộ (40,157) để lưu `.npy` cho DTW sau này |

### 7.4 Cell 3 – Hàm `build_embedding()`

```python
def build_embedding(features: dict) -> np.ndarray:
    """
    Concat tất cả đặc trưng thành vector 99 chiều.
    
    Returns:
        np.ndarray shape (99,) dtype=float32
    """
    embedding = np.concatenate([
        features["mfcc_mean"],    # indices [0:40]
        features["mfcc_std"],     # indices [40:80]
        features["sc_mean"],      # indices [80:87]
        features["chroma_mean"],  # indices [87:99]
    ])
    assert embedding.shape == (EMBEDDING_DIM,), \
        f"Embedding shape sai: {embedding.shape}, expected ({EMBEDDING_DIM},)"
    return embedding  # (99,)
```

- `np.concatenate` ghép các mảng 1D thành mảng 1D dài hơn.
- `assert` kiểm tra shape ngay → phát hiện lỗi sớm nếu tham số thay đổi.

### 7.5 Cell 4 – Vòng lặp chính

```python
def process_all(processed_dir: Path, features_dir: Path) -> list:
    """
    Duyệt toàn bộ data/processed/, trích xuất đặc trưng,
    lưu .npy, trả về list records cho DB insert.
    """
    records = []
    wav_files = sorted(processed_dir.rglob("*.wav"))
    # rglob("*.wav") → duyệt đệ quy: processed/p225/p225_001.wav, ...
    
    for wav_path in tqdm(wav_files, desc="Trích xuất đặc trưng"):
        speaker = wav_path.parent.name   # 'p225'
        
        # ── Tạo đường dẫn output .npy ──────────────────────────
        npy_dir  = features_dir / speaker
        npy_dir.mkdir(parents=True, exist_ok=True)
        npy_path = npy_dir / wav_path.with_suffix(".npy").name
        # VD: data/features/p225/p225_001.npy
        
        # ── Idempotency: bỏ qua nếu đã xử lý ──────────────────
        if npy_path.exists():
            # vẫn cần load embedding để build records
            mfcc_raw = np.load(str(npy_path))
            # rebuild features từ npy (chỉ cho records)
        
        try:
            # ── Trích xuất đặc trưng ───────────────────────────
            features  = extract_features(wav_path)
            embedding = build_embedding(features)
            
            # ── Lưu .npy (MFCC raw cho DTW) ────────────────────
            if not npy_path.exists():
                np.save(str(npy_path), features["mfcc_raw"])
                # lưu shape (40, 157) float32
            
            # ── Ghi record metadata ────────────────────────────
            records.append({
                "speaker"     : speaker,
                "file_path"   : str(wav_path.relative_to(processed_dir.parent)),
                "npy_path"    : str(npy_path.relative_to(features_dir.parent)),
                "sample_rate" : TARGET_SR,
                "duration_sec": wav_path.stat().st_size / (TARGET_SR * 2),
                "embedding"   : embedding.tolist(),   # list[float] → JSON-serializable
            })
            
        except Exception as e:
            log.warning(f"Lỗi {wav_path.name}: {e}")
    
    return records
```

**Giải thích từng phần:**

| Code | Giải thích |
|---|---|
| `processed_dir.rglob("*.wav")` | Duyệt đệ quy tìm tất cả `.wav` trong `data/processed/` |
| `wav_path.parent.name` | Lấy tên thư mục cha = speaker ID (`p225`) |
| `npy_dir.mkdir(parents=True, exist_ok=True)` | Tạo `data/features/p225/` nếu chưa tồn tại |
| `wav_path.with_suffix(".npy").name` | Thay `.wav` → `.npy` trong tên file |
| `npy_path.exists()` | Idempotency: bỏ qua nếu đã xử lý, an toàn khi chạy lại |
| `np.save(str(npy_path), features["mfcc_raw"])` | Lưu ma trận MFCC (40,157) cho phân tích DTW sau |
| `embedding.tolist()` | Chuyển numpy array → list để serialize JSON / psycopg2 |

### 7.6 Cell 5 – Báo cáo tổng kết

```python
print("=" * 55)
print("  TỔNG KẾT TRÍCH XUẤT ĐẶC TRƯNG")
print("=" * 55)
print(f"  Tổng file xử lý   : {len(records)}")
print(f"  Chiều embedding   : {EMBEDDING_DIM}")
print(f"  File .npy đã lưu  : {len(list(FEATURES_DIR.rglob('*.npy')))}")
print(f"  MFCC dims         : {N_MFCC * 2} (mean {N_MFCC} + std {N_MFCC})")
print(f"  Spectral Contrast : {SC_N_BANDS + 1} dims")
print(f"  Chroma STFT       : 12 dims")
print("=" * 55)
```

---

## 📁 Phần VIII: Cấu Trúc Output

### File `.npy` – Đặc trưng thô MFCC

```
data/features/
├── p225/
│   ├── p225_001.npy    ← np.ndarray shape (40, 157) float32
│   ├── p225_002.npy
│   └── ...
├── p228/
│   └── ...
└── ... (61 thư mục)
```

**Cách đọc lại:**
```python
mfcc_raw = np.load("data/features/p225/p225_001.npy")
print(mfcc_raw.shape)   # (40, 157)
# Dùng cho DTW: so khớp chuỗi thời gian hai giọng
```

### Records JSON – Metadata cho Database

```python
# Ví dụ một record:
{
    "speaker"     : "p225",
    "file_path"   : "processed/p225/p225_001.wav",
    "npy_path"    : "features/p225/p225_001.npy",
    "sample_rate" : 16000,
    "duration_sec": 5.0,
    "embedding"   : [−412.3, 118.4, ..., 0.32]   # list 99 float
}
```

---

## 🔄 Phần IX: Tóm Tắt Pipeline Hoàn Chỉnh

```
┌─────────────────────────────────────────────────-------┐
│  data/processed/<speaker>/<file>.wav                   │
│  (16kHz, 5s, float32, shape: 80,000 samples)          │
└──────────────────────────┬────────────────────---------┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         MFCC (40,157)  SC (7,157)  Chroma (12,157)
              │            │            │
           mean+std      mean         mean
              │            │            │
           (40,)+(40,)   (7,)         (12,)
              │            │            │
              └────────────┴────────────┘
                           │
                    np.concatenate
                           │
                           ▼
                   Embedding (99,) float32
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   PostgreSQL pgvector            data/features/
   VECTOR(99)                     <speaker>/<file>.npy
   (tìm kiếm KNN)                 (40, 157) – DTW
```

---

## 📋 Phần X: Thống Kê Kỳ Vọng

| Thông số | Giá trị |
|---|---|
| Số file xử lý | 500 |
| Thời gian ước tính (CPU) | ~3–5 phút |
| Dung lượng `.npy` mỗi file | 40 × 157 × 4 bytes ≈ **25 KB** |
| Tổng dung lượng features | 500 × 25 KB ≈ **12.5 MB** |
| Chiều embedding | 99 |
| Dung lượng embedding mỗi file | 99 × 4 bytes = 396 bytes |

---

## 🚀 Phần XI: Bước Tiếp Theo

Sau khi chạy xong feature extraction:

1. **Database Insertion** – Insert `records` vào PostgreSQL:
   - Metadata: `speaker`, `file_path`, `npy_path`, `sample_rate`, `duration_sec`
   - Vector: `embedding` → cột `VECTOR(99)` trong `voice_records`

2. **Index pgvector:**
   ```sql
   CREATE INDEX ON voice_records USING ivfflat (embedding vector_cosine_ops) WITH (lists = 22);
   ```

3. **Query tìm kiếm:**
   ```sql
   SELECT speaker, file_path, 1 - (embedding <=> $1::vector) AS similarity
   FROM voice_records
   ORDER BY embedding <=> $1::vector
   LIMIT 5;
   ```

---

*Tài liệu được tạo song song với `src/feature_extraction/feature_extraction.ipynb`.*  
*Cập nhật lần cuối: 2026-04-12*
