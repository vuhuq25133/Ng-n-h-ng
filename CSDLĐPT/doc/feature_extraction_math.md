# 📐 Toán Học Trích Xuất Đặc Trưng Âm Thanh (Feature Extraction)

> **Dự án:** VCTK Voice Corpus – Speaker Identification  
> **Pipeline:** `feature_extraction.ipynb`  
> **Thư viện:** `librosa 0.11.0`, `numpy 1.26.4`

---

## Tổng quan pipeline

```
Tín hiệu âm thanh x[n]  (16 000 Hz, 5 giây  →  80 000 mẫu)
          │
          ├──► MFCC          →  (40, 157)  →  mean (40,) + std (40,)
          ├──► Spectral Contrast  →  (7, 157)   →  mean  (7,)
          └──► Chroma STFT   →  (12, 157)  →  mean (12,)
                                               ──────────────
                                    concat  →  embedding  (99,)
```

**Tham số toàn cục:**

| Ký hiệu | Giá trị | Ý nghĩa |
|---------|---------|---------|
| $f_s$ | 16 000 Hz | Sample rate |
| $N_{FFT}$ | 2 048 | Kích thước cửa sổ FFT |
| $H$ | 512 | Hop length (bước nhảy) |
| $M$ | 128 | Số Mel filter banks |
| $K$ | 40 | Số hệ số MFCC giữ lại |
| $B$ | 6 | Số dải Spectral Contrast |
| $f_{min}^{SC}$ | 200 Hz | Tần số bắt đầu Spectral Contrast |

---

## 1. Nạp tín hiệu âm thanh

Tín hiệu thô $x[n]$ có:

$$
n \in \{0, 1, \dots, N_{audio} - 1\}, \quad N_{audio} = f_s \times T = 16\,000 \times 5 = 80\,000
$$

$$
x[n] \in [-1.0,\; 1.0], \quad x[n] \in \mathbb{R}^{80\,000}
$$

### Đơn vị của mẫu $x[n]$

$x[n]$ là **số thực không thứ nguyên (dimensionless)**, chuẩn hóa về $[-1.0, +1.0]$. Không có đơn vị Pa hay V.

**Chuỗi chuyển đổi từ vật lý sang số:**

```
Sóng âm (áp suất không khí)
        │  đơn vị: Pascal (Pa)
        ▼
Microphone (chuyển đổi cơ-điện)
        │  đơn vị: Volt (V)  ← tín hiệu analog
        ▼
ADC – Analog-to-Digital Converter
        │  quantize → số nguyên N-bit
        │  VD: 16-bit → [-32768, +32767]
        ▼
Chuẩn hóa (normalize) ÷ 32768
        ▼
  x[n] ∈ [-1.0, +1.0]   ← float32, không đơn vị
```

| Giá trị $x[n]$ | Ý nghĩa vật lý tương đương |
|----------------|---------------------------|
| `+1.0` | Biên độ dương cực đại (âm to nhất ADC đo được) |
| `0.0` | Im lặng |
| `-1.0` | Biên độ âm cực đại |

`librosa.load()` tự động chuẩn hóa về `float32 ∈ [-1.0, 1.0]` khi đọc file `.wav`:

```python
y, sr = librosa.load(wav_path, sr=16_000, mono=True)
# y.dtype  = float32
# y.shape  = (80000,)
# y ∈ [-1.0, 1.0]
```

---

## 2. MFCC – Mel-Frequency Cepstral Coefficients

### 2.1 Cửa sổ Hamming (Hamming Window)

Trước khi biến đổi Fourier, mỗi frame được nhân với **cửa sổ Hamming** để giảm hiệu ứng rò phổ (spectral leakage) tại biên frame:

$$
\boxed{w(n) = 0.54 - 0.46\cos\!\left(\frac{2\pi n}{N - 1}\right), \quad n = 0, 1, \dots, N-1}
$$

trong đó $N = N_{FFT} = 2048$.

**Số frame** sinh ra:

$$
T_{frames} = \left\lfloor \frac{N_{audio} - N_{FFT}}{H} \right\rfloor + 1 = \left\lfloor \frac{80\,000 - 2\,048}{512} \right\rfloor + 1 = \mathbf{157}
$$

### 2.2 Short-Time Fourier Transform (STFT)

STFT phân tích tín hiệu theo cả **tần số** lẫn **thời gian** bằng cách trượt cửa sổ $w(n)$ qua tín hiệu $x[n]$ với bước nhảy $H$:

$$
\boxed{S(m, k) = \sum_{n=0}^{N-1} x(n + m H) \cdot w(n) \cdot e^{-i2\pi n \frac{k}{N}}}
$$

| Ký hiệu | Giá trị | Ý nghĩa |
|---------|---------|--------|
| $m$ | $0, 1, \dots, 156$ | Chỉ số frame (thời gian) |
| $k$ | $0, 1, \dots, N-1$ | Chỉ số bin tần số |
| $N$ | $2048$ | Kích thước cửa sổ ($N_{FFT}$) |
| $H$ | $512$ | Hop length (bước nhảy) |
| $x(n + mH)$ | — | Mẫu tín hiệu tại frame $m$ |
| $w(n)$ | — | Cửa sổ Hamming |
| $e^{-i2\pi n k/N}$ | — | Hạt nhân Fourier |

$$
S \in \mathbb{C}^{(N/2+1)\; \times\; T_{frames}} = \mathbb{C}^{1025 \times 157}
$$

Chỉ cần $N/2 + 1 = 1025$ bin vì $x[n]$ là thực → phổ đối xứng Hermitian: $S(m, k) = S^*(m, N-k)$.

**Phổ biên độ** (magnitude spectrum):

$$
|S(m, k)|, \quad k = 0, 1, \dots, \frac{N}{2}
$$

**Phổ công suất** (power spectrum):

$$
P(m, k) = \frac{1}{N}\,|S(m, k)|^2
$$

### 2.3 Mel Filterbank

Thang Mel mô phỏng cảm nhận tần số của tai người (phi tuyến):

$$
\boxed{f_{mel} = 2595 \cdot \log_{10}\!\left(1 + \frac{f_{Hz}}{700}\right)}
\qquad
\boxed{f_{Hz} = 700\!\left(10^{f_{mel}/2595} - 1\right)}
$$

Tạo $M = 128$ bộ lọc tam giác trải đều trên thang Mel từ $0$ đến $f_s/2 = 8\,000$ Hz. Bộ lọc thứ $m$ có dạng:

$$
H_m[k] = \begin{cases}
0 & k < f[m-1] \\[4pt]
\dfrac{k - f[m-1]}{f[m] - f[m-1]} & f[m-1] \le k \le f[m] \\[4pt]
\dfrac{f[m+1] - k}{f[m+1] - f[m]} & f[m] < k \le f[m+1] \\[4pt]
0 & k > f[m+1]
\end{cases}
$$

trong đó $f[0], f[1], \dots, f[M+1]$ là các tâm bộ lọc trên thang Hz (được tính từ thang Mel đều).

**Năng lượng Mel** của frame $t$:

$$
S_t[m] = \sum_{k=0}^{N_{FFT}/2} P_t[k] \cdot H_m[k], \quad m = 1, \dots, M
$$

**Log Mel Spectrogram:**

$$
\boxed{\tilde{S}_t[m] = \log\!\left(S_t[m] + \epsilon\right)}
$$

với $\epsilon \approx 10^{-10}$ để tránh $\log(0)$.

### 2.4 Discrete Cosine Transform (DCT) → MFCC

Áp dụng DCT-II lên log Mel spectrogram để tách thành các **cepstral coefficient**:

$$
\boxed{c_t[i] = \sum_{m=1}^{M} \tilde{S}_t[m] \cos\!\left(\frac{\pi i (m - 0.5)}{M}\right), \quad i = 0, 1, \dots, K-1}
$$

Giữ lại $K = 40$ hệ số đầu tiên:

$$
\text{MFCC} \in \mathbb{R}^{K \times T_{frames}} = \mathbb{R}^{40 \times 157}
$$

### 2.5 Tổng hợp theo thời gian

$$
\boxed{\mu_{MFCC}[i] = \frac{1}{T_{frames}} \sum_{t=1}^{T_{frames}} c_t[i]} \quad \in \mathbb{R}^{40}
\qquad \text{(MFCC mean — đặc trưng hướng giọng)}
$$

$$
\boxed{\sigma_{MFCC}[i] = \sqrt{\frac{1}{T_{frames}} \sum_{t=1}^{T_{frames}} \left(c_t[i] - \mu_{MFCC}[i]\right)^2}} \quad \in \mathbb{R}^{40}
\qquad \text{(MFCC std — biến động giọng)}
$$

**Kết quả:** `mfcc_mean` $(40,)$ + `mfcc_std` $(40,)$ → **80 chiều**

---

## 3. Spectral Contrast

### 3.1 Ý tưởng

Đo sự tương phản giữa **đỉnh phổ** (peak) và **đáy phổ** (valley) trong mỗi dải tần. Đặc trưng này nắm bắt cấu trúc hài âm (harmonic structure) của giọng nói.

### 3.2 Chia dải tần

Từ $f_{min}^{SC} = 200$ Hz, tạo $B = 6$ dải băng theo octave (nhân đôi tần số):

$$
\text{Dải } b: \quad \left[f_{min}^{SC} \cdot 2^{b-1},\; f_{min}^{SC} \cdot 2^b\right), \quad b = 1, \dots, B
$$

| Dải | Từ (Hz) | Đến (Hz) |
|-----|---------|---------|
| Bass (b=0) | 0 | 200 |
| b=1 | 200 | 400 |
| b=2 | 400 | 800 |
| b=3 | 800 | 1 600 |
| b=4 | 1 600 | 3 200 |
| b=5 | 3 200 | 6 400 |
| b=6 | 6 400 | 8 000 |

Tổng cộng: $B + 1 = 7$ dải (kể cả dải bass $b = 0$).

### 3.3 Tính toán Spectral Contrast

Với mỗi frame $t$ và mỗi dải $b$, sắp xếp các bins biên độ theo thứ tự tăng dần:

$$
\{|X_t[k]| : k \in \text{dải}_b\} \xrightarrow{\text{sort}} \{a_1 \le a_2 \le \dots \le a_{N_b}\}
$$

**Peak** = trung bình $\alpha$% trên cùng (mặc định $\alpha = 0.02$):

$$
\text{peak}_b(t) = \frac{1}{\lceil \alpha N_b \rceil} \sum_{i = N_b - \lceil \alpha N_b \rceil + 1}^{N_b} a_i
$$

**Valley** = trung bình $\alpha$% dưới cùng:

$$
\text{valley}_b(t) = \frac{1}{\lceil \alpha N_b \rceil} \sum_{i=1}^{\lceil \alpha N_b \rceil} a_i
$$

**Spectral Contrast** của dải $b$ tại frame $t$ (đơn vị dB):

$$
\boxed{SC_b(t) = \log\!\left(\text{peak}_b(t) + \epsilon\right) - \log\!\left(\text{valley}_b(t) + \epsilon\right)}
$$

$$
\Rightarrow SC \in \mathbb{R}^{(B+1) \times T_{frames}} = \mathbb{R}^{7 \times 157}
$$

### 3.4 Tổng hợp theo thời gian

$$
\boxed{\mu_{SC}[b] = \frac{1}{T_{frames}} \sum_{t=1}^{T_{frames}} SC_b(t)} \quad \in \mathbb{R}^{7}
$$

**Kết quả:** `sc_mean` $(7,)$ → **7 chiều**

---

## 4. Chroma STFT

### 4.1 Ý tưởng

Ánh xạ phổ tần số sang **12 lớp cao độ** (pitch classes) theo hệ bình quân (equal temperament): C, C#, D, D#, E, F, F#, G, G#, A, A#, B.

### 4.2 Ánh xạ tần số → pitch class

Tần số $f$ (Hz) được ánh xạ sang midi note:

$$
\text{midi}(f) = 12 \cdot \log_2\!\left(\frac{f}{440}\right) + 69
$$

Pitch class (chroma):

$$
\text{chroma}(f) = \text{midi}(f) \bmod 12
$$

### 4.3 Tích lũy năng lượng theo pitch class

Với mỗi frame $t$:

$$
\boxed{C_t[p] = \sum_{\substack{k:\; \text{chroma}(f_k) = p}} |X_t[k]|^2, \quad p = 0, 1, \dots, 11}
$$

trong đó $f_k = k \cdot f_s / N_{FFT}$ là tần số bin $k$.

**Chuẩn hóa** (librosa chuẩn hóa theo $\ell^2$ hoặc $\ell^\infty$ theo từng frame):

$$
\tilde{C}_t[p] = \frac{C_t[p]}{\max_p C_t[p] + \epsilon}
$$

$$
\Rightarrow \text{Chroma} \in \mathbb{R}^{12 \times T_{frames}} = \mathbb{R}^{12 \times 157}
$$

### 4.4 Tổng hợp theo thời gian

$$
\boxed{\mu_{Chroma}[p] = \frac{1}{T_{frames}} \sum_{t=1}^{T_{frames}} \tilde{C}_t[p]} \quad \in \mathbb{R}^{12}
$$

**Kết quả:** `chroma_mean` $(12,)$ → **12 chiều**

---

## 5. Vector Embedding 99 Chiều

### 5.1 Ghép nối (Concatenation)

$$
\boxed{
\mathbf{e} = \bigl[\,\mu_{MFCC} \;\big|\; \sigma_{MFCC} \;\big|\; \mu_{SC} \;\big|\; \mu_{Chroma}\,\bigr]
}
$$

| Đoạn | Đặc trưng | Chiều | Ý nghĩa |
|------|-----------|-------|---------|
| $\mathbf{e}[0:40]$ | MFCC mean | 40 | Đặc trưng hướng giọng trung bình |
| $\mathbf{e}[40:80]$ | MFCC std | 40 | Biến động giọng theo thời gian |
| $\mathbf{e}[80:87]$ | Spectral Contrast mean | 7 | Độ tương phản hài âm |
| $\mathbf{e}[87:99]$ | Chroma mean | 12 | Phân bố cao độ âm nhạc |
| **Tổng** | | **99** | |

$$
\text{Kiểm tra: } 40 + 40 + 7 + 12 = \mathbf{99} \checkmark
$$

### 5.2 Kiểm tra bằng giá trị thực (file `p225_010.wav`)

| Phần tử | Giá trị thực tế |
|---------|----------------|
| $e[0]$ = MFCC$_0$ mean | $-394.19$ |
| $e[1]$ = MFCC$_1$ mean | $48.40$ |
| $e[40]$ = MFCC$_0$ std | $157.79$ |
| $e[80]$ = SC bass mean | $22.02$ dB |
| $e[87]$ = Chroma C mean | $0.48$ |
| $e[98]$ = Chroma B mean | $0.48$ |

---

## 6. MFCC Thô cho DTW

Ngoài embedding, MFCC thô được lưu riêng dưới dạng `.npy`:

$$
\text{mfcc\_raw} \in \mathbb{R}^{40 \times 157}
$$

Được sử dụng cho **Dynamic Time Warping (DTW)** — so sánh chuỗi thời gian có độ co giãn, không cần cùng độ dài.

**Khoảng cách DTW** giữa hai chuỗi $\mathbf{A} \in \mathbb{R}^{K \times T_1}$ và $\mathbf{B} \in \mathbb{R}^{K \times T_2}$:

$$
\text{DTW}(\mathbf{A}, \mathbf{B}) = \min_{\pi} \sum_{(i,j) \in \pi} d(\mathbf{a}_i, \mathbf{b}_j)
$$

trong đó $d(\mathbf{a}_i, \mathbf{b}_j) = \|\mathbf{a}_i - \mathbf{b}_j\|_2$ là khoảng cách Euclidean giữa hai frame.

---

## 7. Cosine Similarity – Đo độ tương đồng giọng nói

Dùng để so sánh hai embedding $\mathbf{e}_1, \mathbf{e}_2 \in \mathbb{R}^{99}$:

$$
\boxed{\text{cosine}(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\|_2 \cdot \|\mathbf{e}_2\|_2} = \frac{\sum_{i=0}^{98} e_1[i]\, e_2[i]}{\sqrt{\sum_{i=0}^{98} e_1[i]^2}\; \sqrt{\sum_{i=0}^{98} e_2[i]^2}}}
$$

**Kết quả sanity check thực tế:**

| So sánh | Cosine Similarity |
|---------|------------------|
| Cùng speaker (p225 vs p225) | **0.9979** |
| Khác speaker (p225 vs p228) | 0.9944 |

$$
0.9979 > 0.9944 \quad \Rightarrow \quad \text{cùng speaker > khác speaker} \checkmark
$$

---

## 8. Sơ đồ chiều dữ liệu tổng hợp

```
x[n]  ∈ ℝ^80000
    │
    ├─ STFT(N_FFT=2048, H=512) ──────────→  X_t[k]  ∈ ℂ^{1025 × 157}
    │                                              │
    │   ┌──── Mel filterbank (M=128) ─────────────┘
    │   │     Log → DCT(K=40)
    │   ▼
    │  MFCC  ∈ ℝ^{40 × 157}
    │   ├── mean(axis=1) → μ_MFCC  ∈ ℝ^40   [e[0:40]]
    │   └── std(axis=1)  → σ_MFCC  ∈ ℝ^40   [e[40:80]]
    │
    ├─ Spectral Contrast(B=6, f_min=200Hz) → SC  ∈ ℝ^{7 × 157}
    │   └── mean(axis=1) → μ_SC    ∈ ℝ^7    [e[80:87]]
    │
    └─ Chroma STFT (N_FFT=2048, H=512) ──→ C   ∈ ℝ^{12 × 157}
        └── mean(axis=1) → μ_Chroma ∈ ℝ^12  [e[87:99]]

                                concat([μ_MFCC, σ_MFCC, μ_SC, μ_Chroma])
                                    ↓
                            e  ∈  ℝ^99   (embedding lưu vào PostgreSQL)
```

---

## 9. Ví dụ tính toán STFT (Mini — $N = 4$)

Để minh họa công thức $S(m, k)$ bằng số cụ thể, ta dùng ví dụ thu nhỏ với $N = 4$ thay vì $2048$.

### Đầu vào

$$
x = [1,\; 1,\; -1,\; -1], \quad m = 0 \text{ (frame đầu tiên, không trượt)}
$$

### Bước 1 — Tính cửa sổ Hamming ($N = 4$)

$$
w(n) = 0.54 - 0.46\cos\!\left(\frac{2\pi n}{N - 1}\right), \quad n = 0, 1, 2, 3
$$

| $n$ | $\cos\!\left(\frac{2\pi n}{3}\right)$ | $w(n)$ |
|-----|--------------------------------------|--------|
| 0 | $\cos(0) = 1.000$ | $0.54 - 0.46 \times 1.000 = \mathbf{0.08}$ |
| 1 | $\cos(2\pi/3) = -0.500$ | $0.54 - 0.46 \times (-0.500) = \mathbf{0.77}$ |
| 2 | $\cos(4\pi/3) = -0.500$ | $0.54 - 0.46 \times (-0.500) = \mathbf{0.77}$ |
| 3 | $\cos(2\pi) = 1.000$ | $0.54 - 0.46 \times 1.000 = \mathbf{0.08}$ |

$$
\mathbf{w} = [0.08,\; 0.77,\; 0.77,\; 0.08]
$$

### Bước 2 — Windowing: $x(n + mH) \cdot w(n)$ với $m=0$

$$
\text{frame}[n] = x[n] \cdot w(n)
$$

| $n$ | $x[n]$ | $w(n)$ | $\text{frame}[n]$ |
|-----|--------|--------|-------------------|
| 0 | $+1$ | $0.08$ | $+0.08$ |
| 1 | $+1$ | $0.77$ | $+0.77$ |
| 2 | $-1$ | $0.77$ | $-0.77$ |
| 3 | $-1$ | $0.08$ | $-0.08$ |

$$
\text{frame} = [+0.08,\; +0.77,\; -0.77,\; -0.08]
$$

### Bước 3 — DFT: $S(0, k) = \sum_{n=0}^{3} \text{frame}[n] \cdot e^{-i2\pi nk/4}$

Hạt nhân $e^{-i2\pi nk/N}$ với $N=4$:

$$
e^{-i2\pi nk/4} = \cos\!\left(\frac{\pi nk}{2}\right) - i\sin\!\left(\frac{\pi nk}{2}\right)
$$

**$k = 0$** (thành phần DC):

$$
S(0,0) = 0.08 \cdot 1 + 0.77 \cdot 1 + (-0.77) \cdot 1 + (-0.08) \cdot 1 = \mathbf{0}
$$

**$k = 1$**:

$$
S(0,1) = 0.08 \cdot e^{0} + 0.77 \cdot e^{-i\pi/2} + (-0.77) \cdot e^{-i\pi} + (-0.08) \cdot e^{-i3\pi/2}
$$

$$
= 0.08(1) + 0.77(-i) + (-0.77)(-1) + (-0.08)(+i)
$$

$$
= (0.08 + 0.77) + (-0.77 - 0.08)i = \mathbf{0.85 - 0.85i}
$$

**$k = 2$**:

$$
S(0,2) = 0.08(1) + 0.77(-1) + (-0.77)(1) + (-0.08)(-1)
$$

$$
= 0.08 - 0.77 - 0.77 + 0.08 = \mathbf{-1.38}
$$

**$k = 3$** (đối xứng Hermitian với $k=1$):

$$
S(0,3) = \overline{S(0,1)} = 0.85 + 0.85i
$$

### Bước 4 — Phổ biên độ $|S(0, k)|$ và phổ công suất $P(0,k)$

Do $x[n]$ thực → chỉ cần $k = 0, 1, 2$ (tức $N/2 + 1 = 3$ bin):

| $k$ | $S(0,k)$ | $|S(0,k)|$ | $P(0,k) = \frac{1}{N}|S|^2$ |
|-----|----------|------------|------------------------------|
| 0 | $0$ | $0$ | $0$ |
| 1 | $0.85 - 0.85i$ | $0.85\sqrt{2} \approx \mathbf{1.202}$ | $\frac{1.202^2}{4} \approx \mathbf{0.361}$ |
| 2 | $-1.38$ | $\mathbf{1.38}$ | $\frac{1.38^2}{4} \approx \mathbf{0.476}$ |

> **Nhận xét:** Bin $k=2$ có năng lượng lớn nhất vì $x = [1,1,-1,-1]$ tương ứng với tần số $f = k \cdot f_s/N = 2 \cdot 4/4 = 2$ Hz (trong ví dụ nhỏ này) — đúng với tín hiệu vuông góc tuần hoàn.

### Ánh xạ sang dự án thực tế ($N = 2048$, $f_s = 16000$ Hz)

$$
f_k = k \cdot \frac{f_s}{N} = k \cdot \frac{16000}{2048} \approx 7.81k \text{ Hz}
$$

| Bin $k$ | Tần số tương ứng |
|---------|-----------------|
| 0 | 0 Hz (DC) |
| 1 | 7.81 Hz |
| 256 | 2 000 Hz |
| 1024 | 8 000 Hz (Nyquist) |

---

## 10. Tóm tắt công thức chính

| STT | Đặc trưng | Công thức cốt lõi | Shape đầu ra |
|-----|-----------|-------------------|-------------|
| 1 | Hamming window | $w[k] = 0.54 - 0.46\cos(2\pi k / (N-1))$ | $(2048,)$ |
| 2 | STFT | $X_t[k] = \sum_n x[n]\,e^{-j2\pi kn/N}$ | $(1025, 157)$ |
| 3 | Mel scale | $f_{mel} = 2595\log_{10}(1+f/700)$ | — |
| 4 | Log Mel | $\tilde{S}_t[m] = \log(S_t[m]+\epsilon)$ | $(128, 157)$ |
| 5 | DCT (MFCC) | $c_t[i] = \sum_m \tilde{S}_t[m]\cos(\pi i(m-0.5)/M)$ | $(40, 157)$ |
| 6 | MFCC mean | $\mu_{MFCC}[i] = \frac{1}{T}\sum_t c_t[i]$ | $(40,)$ |
| 7 | MFCC std | $\sigma_{MFCC}[i] = \sqrt{\frac{1}{T}\sum_t(c_t[i]-\mu)^2}$ | $(40,)$ |
| 8 | Spectral Contrast | $SC_b(t) = \log(\text{peak}) - \log(\text{valley})$ | $(7, 157)$ |
| 9 | SC mean | $\mu_{SC}[b] = \frac{1}{T}\sum_t SC_b(t)$ | $(7,)$ |
| 10 | Chroma | $C_t[p] = \sum_{k:\text{chroma}(f_k)=p} \|X_t[k]\|^2$ | $(12, 157)$ |
| 11 | Chroma mean | $\mu_{Chroma}[p] = \frac{1}{T}\sum_t \tilde{C}_t[p]$ | $(12,)$ |
| 12 | **Embedding** | $\mathbf{e} = [\mu_{MFCC} \mid \sigma_{MFCC} \mid \mu_{SC} \mid \mu_{Chroma}]$ | **(99,)** |
| 13 | Cosine sim | $\cos(\mathbf{e}_1,\mathbf{e}_2) = \frac{\mathbf{e}_1\cdot\mathbf{e}_2}{\|\mathbf{e}_1\|\|\mathbf{e}_2\|}$ | scalar |

---

> **Tham khảo:**  
> - Davis, S. & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition.*  
> - Müller, M. (2015). *Fundamentals of Music Processing.* Springer.  
> - McFee et al. (2015). *librosa: Audio and Music Signal Analysis in Python.*  
> - Jiang et al. (2002). *Music type classification by spectral contrast feature.*
