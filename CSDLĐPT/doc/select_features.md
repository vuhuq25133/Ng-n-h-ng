# 📐 Cơ Sở Toán Học và Nguyên Lý: Lựa Chọn Đặc Trưng MFCC

Tài liệu này giải thích chi tiết về mặt toán học và nguyên lý đằng sau hai trường phái trích xuất đặc trưng âm thanh phổ biến dựa trên MFCC (Mel-Frequency Cepstral Coefficients). Đặc biệt, lý giải sự lựa chọn "Statistical Pooling với 40 MFCCs" cho bài toán Voice Similarity (Tìm kiếm sự tương đồng giọng nói) thay vì dùng "MFCC-39 truyền thống".

---

## 1. Phương Pháp Kinh Điển: MFCC-39 (Hệ Thống ASR / HMM-GMM)

Trong các bài toán Nhận dạng tiếng nói (Automatic Speech Recognition - ASR) truyền thống, việc theo dõi sự thay đổi của khẩu hình miệng theo thời gian thực (ví dụ thời điểm chuyển từ âm "b" sang "a") là cực kỳ quan trọng. Do đó, người ta sử dụng 39 hệ số cho **mỗi khung hình âm thanh (frame)**.

### 1.1. Các hệ số tĩnh (Static Coefficients) - 13 Chiều
Bao gồm năng lượng tĩnh $E$ (hoặc hệ số $C_0$) và 12 hệ số Cepstral ($C_1 \to C_{12}$). 
Sau khi tính toán phổ năng lượng theo thang Mel (Mel-filterbank energies) ký hiệu là $S_k$, hệ số MFCC tĩnh bậc $n$ được tính bằng Phép biến đổi Cosine rời rạc (DCT):

$$ C_n = \sum_{k=1}^{K} \log(S_k) \cos \left[ n \left( k - 0.5 \right) \frac{\pi}{K} \right] $$

Trong đó $K$ là số lượng bộ lọc Mel (Mel filters). 
**Ý nghĩa:** 13 hệ số đầu tiên chứa hầu hết thông tin về đường bao phổ (spectral envelope), đại diện cho dạng ống thanh quản tại một khoảnh khắc tĩnh (khoảng 20-30ms).

### 1.2. Đặc trưng động bậc 1 (Delta Coefficients) $\Delta$ - 13 Chiều
Giọng nói là một quá trình biến đổi liên tục. Hệ số tĩnh không thể hiện được *sự chuyển tiếp* giữa các khung hình. Các hệ số Delta đóng vai trò như **vận tốc** của sự thay đổi tần số theo thời gian.

Về mặt toán học, Delta $\Delta C_n$ không phải là phép trừ đơn giản giữa 2 frame gần nhau, mà thường được tính xấp xỉ bằng **hồi quy tuyến tính (polynomial regression)** trên một cửa sổ (window size) xung quanh frame cần xét (thường $N=2$):

$$ d_t = \frac{\sum_{n=1}^{N} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{N} n^2} $$

Trong đó:
- $d_t$ là hệ số Delta tại thời gian $t$.
- $c_t$ là hệ số MFCC tĩnh tại thời gian $t$.
- $N$ là kích thước cửa sổ phân tích cục bộ.

### 1.3. Đặc trưng động bậc 2 (Delta-Delta) $\Delta\Delta$ - 13 Chiều
Các hệ số Delta-Delta đóng vai trò như **gia tốc** của phổ khung hình âm thanh, đo lường sự thay đổi của các hệ số Delta. Thuật toán tương tự như trên, nhưng đầu vào là biến $d_t$ thay vì $c_t$:

$$ dd_t = \frac{\sum_{n=1}^{N} n (d_{t+n} - d_{t-n})}{2 \sum_{n=1}^{N} n^2} $$

**Tổng kết phương pháp 1:** Kết quả trả về là một chuỗi tuần tự thời gian gồm các ma trận kích thước `(39, T)` với $T$ là số lượng frame.
* **Ưu điểm:** Bắt được mọi diễn biến của tín hiệu thời gian thực.
* **Nhược điểm:** Phải sử dụng Mô hình học sâu chuỗi (LSTM, RNN) hoặc thuật toán chuỗi (Dynamic Time Warping - DTW) cực kỳ tốn chi phí tính toán, không thể lưu tĩnh dạng vector 1D để Index.

---

## 2. Phương Pháp Hiện Đại Trong Diễn Đạt Tĩnh: Statistical Pooling (Voice Similarity Embedding)

Đối với bài toán **Tìm kiếm tương đồng (Voice Similarity)** hoặc **Nhận diện người nói (Speaker Recognition/Verification)** ứng dụng Vector Database (như `pgvector`), ta cần ép một luồng âm thanh có độ dài tùy ý (VD: 5 giây) thành **một vector 1D ($1 \times M$) có kích thước cố định**.

Hệ thống hiện tại chọn `n_mfcc=40` kết hợp thao tác **Statistical Pooling** nhằm tạo ra một Siêu Vector Nhúng Tĩnh.

### 2.1. Tại sao lại là 40 chiều (n_mfcc = 40)?
Trong xử lý nhận dạng ngôn ngữ (nhận dạng chữ mất đi), 13 chiều là đủ để phân biệt các nguyên âm/phụ âm (ví dụ /a/, /i/, /e/).
Tuy nhiên, để phân biệt **người nói** (speaker identity), các đặc điểm cá nhân thường nằm ở các tần số Formant cao hơn và các chi tiết nội tại tinh vi hơn trên phổ âm thanh.
* Nâng $n\_mfcc = 40$ giúp hệ thống giữ lại nhiều cấu trúc Cepstral (thông tin cao tần) hơn, thứ đóng vai trò mấu chốt để phân biệt giọng phổ rộng của người này với người khác. Giúp tăng tính "discriminative" của Embedding.

### 2.2. Gom Cụm Thống Kê (Statistical Pooling)
Thay vì trích xuất mảng biến động (Delta) cho từng frame, mô hình sử dụng hàm thống kê toàn cục trên chiều thời gian $T$.

#### A. Trục Trung Bình (Mean Pooling - $\mu$)
$$ \mu_i = \frac{1}{T} \sum_{t=1}^{T} c_{i, t} \quad \forall i \in [1, 40] $$
* **Ý nghĩa Vật Lý:** Nắm bắt **trung tâm đặc tính âm thanh**. Giọng nói chung của người A thiên về âm trầm hay cao? Độ cộng hưởng trong lồng ngực và thể tích khoang miệng của người A trung bình ra sao trong suốt 5 giây phát âm? Phương trình này tóm tắt "màu sắc cơ bản" của người nói.

#### B. Trục Độ Lệch Chuẩn (Standard Deviation Pooling - $\sigma$)
$$ \sigma_i = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (c_{i, t} - \mu_i)^2} \quad \forall i \in [1, 40] $$
* **Ý nghĩa Vật Lý:** Nó giải quyết chính xác sự thiếu sót của biến Delta! Biên độ của $\sigma_i$ đo lường **độ biến thiên của phổ** trong suốt 5s thu âm. Một người có giọng nói đều đều (monotone) sẽ có $\sigma$ thấp, trong khi một người nói giàu nhịp điệu, tốc độ thay đổi khẩu hình nhanh, âm sắc linh hoạt sẽ có độ lệch chuẩn $\sigma$ cực kỳ cao.
* Đây là cách đóng gói "động lực học thời gian" (temporal dynamics) vô cùng nhẹ và ngắn gọn thay vì phải dùng mảng đạo hàm 2D Delta khổng lồ.

Việc ghép 2 nửa lại với nhau ($[\mu, \sigma]$) sinh ra `40 + 40 = 80` features. 

---

## 3. Tổng Kết Việc Lựa Chọn Giải Pháp

| Đặc Điểm Đo Lường | PP 1: Cổ Điển (MFCC-39 Frame-level) | PP 2: Hiện Tại (Statistical Pooling 40 Mean + Std) |
| :--- | :--- | :--- |
| **Bản chất tính toán** | Vector biến đổi từng mini-giây (20ms/step). | Giá trị thống kê đại diện cho **toàn bộ** đoạn âm thanh. |
| **Kích thước đầu ra** | `39 × T` (Ma trận 2D động). | `80` (Vector 1D tĩnh). (*+19 chiều SC & Chroma = 99 trong code*) |
| **Thông tin đạo hàm (Dynamics)** | Bắt qua Delta và Delta-Delta (Gia tốc & vận tốc). | Bắt qua độ lệch chuẩn (Std Deviation Pooling). |
| **Ứng dụng tối ưu nhất** | Speech-to-Text, Voice Activity Detection, HMM-based ASR. | Speaker Similarity, Voice Biometrics, Vector Database Search. |
| **Độ phức tạp lưu trữ / Tra cứu** | Rất tốn bộ nhớ lưu trữ chuỗi thời gian, cần thuật toán tra cứu phức tạp (Cross-correlation, DTW). | **Cực kỳ tối ưu**, dùng ngay với Dot-Product hoặc **Cosine Similarity** thời gian O(1) qua SQL `pgvector`. |

---

## 4. Danh Mục Chi Tiết Các Features: Phương Pháp Cổ Điển MFCC-39

Trong mỗi frame âm thanh (khoảng 20–30ms), hệ thống cổ điển tạo ra đúng 39 giá trị số thực sau:

### 4.1. Nhóm Static (13 chiều: $C_0 \to C_{12}$)

| Feature | Tên gọi | Ý nghĩa vật lý | Ý nghĩa trong bài toán |
| :---: | :--- | :--- | :--- |
| $C_0$ | Log-energy / Năng lượng ngắn hạn | Tổng năng lượng tín hiệu trong frame (tương đương biên độ sóng). Khung yên lặng có $C_0$ thấp, khung phụ âm bật/khung nguyên âm to có $C_0$ cao. | Phân loại phát âm vs. im lặng trong mô hình ASR. Trong nhận diện người nói, biên độ sóng phản ánh cách người A thường nói to/nhỏ, nhịp cường độ. |
| $C_1$ | Cepstral bậc 1 (Hình dạng phổ tổng quát) | Thành phần chủ yếu của đường bao phổ (spectral envelope). Phân biệt loại âm vị rộng: nguyên âm khác phụ âm, giọng trầm khác giọng cao. | Trong ASR: nhận dạng nguyên âm rộng (/a/, /e/, /o/). Trong nhận diện người nói: phản ánh hình dạng buồng cộng hưởng (vocal tract shape) đặc trưng. |
| $C_2$ | Cepstral bậc 2 | Mã hóa vị trí cộng hưởng Formant 1 (F1) - liên quan đến chiều mở của miệng khi phát âm. | Phân biệt các nguyên âm /a/ (miệng mở to, F1 cao) vs /i/ (miệng khép hẹp, F1 thấp). |
| $C_3$ | Cepstral bậc 3 | Mã hóa vị trí Formant F2 - liên quan đến vị trí lưỡi trước/sau. | Phân biệt nguyên âm trước (/i/, /e/) vs nguyên âm sau (/u/, /o/). |
| $C_4$–$C_6$ | Cepstral bậc 4–6 (Formant F3) | Chi tiết hình dạng ống cộng hưởng, Formant bậc 3. | Đặc trưng giọng nói cá nhân hóa, bắt đầu phân biệt được từng người nói. |
| $C_7$–$C_{12}$ | Cepstral bậc 7–12 (Chi tiết cao tần) | Đặc điểm chi tiết của ống thanh quản — hình dạng môi, míệng, mũi, tốc độ rung thanh quản. | Thông tin nhận diện người nói mịn hơn, phân biệt phụ âm, âm bật hơi. |

> **Ghi chú quan trọng về Formant:**  
> Formant là các đỉnh cộng hưởng (resonance peaks) xuất hiện trên phổ âm thanh khi luồng khí đi qua khoang họng/miệng có hình dạng cụ thể. F1, F2, F3 là 3 Formant đầu tiên và mang nhiều thông tin nhất về âm vị và về đặc trưng giải phẫu của người nói.

### 4.2. Nhóm Delta $\Delta$ (13 chiều: $\Delta C_0 \to \Delta C_{12}$)

> **Ý nghĩa tổng quát:** Đây là **"vận tốc"** biến đổi của mỗi hệ số static qua thời gian.

| Feature | Ý nghĩa vật lý | Ý nghĩa trong bài toán |
| :--- | :--- | :--- |
| $\Delta C_0$ | Tốc độ thay đổi năng lượng âm thanh: nhanh hay chậm? | Nắm bắt "nhịp nói" — người nói nhanh có $\Delta C_0$ biến thiên mạnh hơn người nói chậm. |
| $\Delta C_1$ | Tốc độ biến đổi hình dạng phổ tổng quát. | Bắt được tốc độ chuyển tiếp từ âm vị này sang âm vị khác (phoneme transition). Quan trọng với ASR. |
| $\Delta C_2$–$\Delta C_3$ | Tốc độ di chuyển vị trí Formant F1, F2. | Mã hóa cách miệng/lưỡi dịch chuyển khi phát âm — như di động lưỡi liên tiếp, nhịp nói. |
| $\Delta C_4$–$\Delta C_{12}$ | Tốc độ biến đổi các chi tiết phổ bậc cao. | Bắt tốc độ biến động của đặc điểm cá nhân theo thời gian. |

### 4.3. Nhóm Delta-Delta $\Delta\Delta$ (13 chiều: $\Delta\Delta C_0 \to \Delta\Delta C_{12}$)

> **Ý nghĩa tổng quát:** Đây là **"gia tốc"** biến đổi của mỗi hệ số static. Hay nói cách khác là tốc độ thay đổi của Delta.

| Feature | Ý nghĩa vật lý | Ý nghĩa trong bài toán |
| :--- | :--- | :--- |
| $\Delta\Delta C_0$ | Gia tốc thay đổi năng lượng âm: âm thanh đang tăng hay giảm cường độ đột ngột? | Phát hiện các điểm bắt đầu/kết thúc âm vị (onset/offset detection) trong ASR. |
| $\Delta\Delta C_1$–$C_3$ | Gia tốc biến đổi hình dạng formant. | Bắt được những điểm "bẻ gãy" (inflection points) quan trọng khi chuyển từ phụ âm sang nguyên âm, ví dụ: phụ âm bật /b/ → nguyên âm /a/. |
| $\Delta\Delta C_4$–$C_{12}$ | Gia tốc biến đổi chi tiết phổ cao tần. | Mã hóa tốc độ biến hoá linh hoạt của cách phát âm, nhịp điệu nói, nhấn nhá. |

**→ Tóm kết MFCC-39:** Mỗi frame đầu ra là 1 vector `(39,)`. Tổng 157 frames = ma trận `(39, 157)`. Hệ thống sau đó cần LSTM/HMM để đọc chuỗi này theo thứ tự thời gian và dự đoán ngôn ngữ hoặc nhận ra người nói.

---

## 5. Danh Mục Chi Tiết Các Features: Phương Pháp Hiện Tại – Vector 99 Chiều

Hệ thống dự án lấy ra tổng cộng **99 đặc trưng tĩnh** từ mỗi file âm thanh 5s. Cấu trúc:

```
[0  : 40]  → 40 MFCC Mean       (chiều 0–39)
[40 : 80]  → 40 MFCC Std        (chiều 40–79)
[80 : 87]  → 7  Spectral Contrast Mean  (chiều 80–86)
[87 : 99]  → 12 Chroma STFT Mean        (chiều 87–98)
```

### 5.1. MFCC Mean – 40 Chiều (indices 0–39)

Đây là giá trị **trung bình** của từng hệ số MFCC được tính trên toàn bộ 157 frames của file âm thanh.

| Nhóm chiều | Features cụ thể | Ý nghĩa vật lý | Ý nghĩa trong bài toán Voice Similarity |
| :---: | :--- | :--- | :--- |
| `[0]` | $\mu_{C0}$ – Năng lượng trung bình | Mức âm lượng trung bình của người nói trong 5s. | Giúp phân biệt giọng nói to (phát thanh viên) vs. giọng nói nhỏ (người ít nói). Nhưng thường **bị chuẩn hóa** để bất biến với âm lượng micro. |
| `[1]` | $\mu_{C1}$ – Hình dạng phổ tổng quát TB | Tổ hợp của toàn bộ hình dạng khoang cộng hưởng của người nói trong 5s. |  **Feature phân biệt người nói quan trọng nhất** — 2 người khác nhau sẽ luôn có $\mu_{C1}$ khác nhau rõ rệt vì hình dạng ống thanh quản là đặc điểm cơ thể. |
| `[2]–[5]` | $\mu_{C2} \to \mu_{C5}$ – Formant TB | Vị trí trung bình của Formant F1, F2, F3 trong suốt đoạn nói. | Phân biệt giọng sắc (F2 cao) vs. giọng trầm (F2 thấp), dễ phân biệt giới tính và tuổi tác. |
| `[6]–[12]` | $\mu_{C6} \to \mu_{C12}$ – Chi tiết cộng hưởng TB | Trung bình bề mặt chi tiết của ống thanh quản. | Đặc trưng **độc nhất** của từng người — dùng để phân biệt 2 người có giọng nghe tương tự nhau. |
| `[13]–[39]` | $\mu_{C13} \to \mu_{C39}$ – Bậc Cepstral cao TB | Thông tin về phụ âm, hình dạng môi, phổ cao tần đặc thù. | Ít quan trọng hơn các bậc thấp nhưng giúp **tinh chỉnh** độ chính xác khi phân biệt những người nói giống nhau. |

> **Tại sao Mean đủ tốt để nhận diện người nói?**  
> Vì hình dạng giải phẫu của ống thanh quản (vocal tract) là **đặc điểm cơ thể cố định** — nó không đổi dù người đó đang nói chuyện gì. Do đó, giá trị MFCC trung bình trong 5s đại diện trung thực cho "kiểu giọng" địa sinh học của họ.

### 5.2. MFCC Std – 40 Chiều (indices 40–79)

Đây là **độ lệch chuẩn** của từng hệ số MFCC tính trên 157 frames — thay thế thông tin mà Delta/Delta-Delta cung cấp theo cách thức gọn gàng hơn.

| Nhóm chiều | Features cụ thể | Ý nghĩa vật lý | Ý nghĩa trong bài toán Voice Similarity |
| :---: | :--- | :--- | :--- |
| `[40]` | $\sigma_{C0}$ – Phương sai năng lượng | Người nói có hay thay đổi âm lượng khi nói không? | Phân biệt người nói nhấn nhá (stress pattern) — thường khác nhau giữa các ngôn ngữ và phong cách nói. |
| `[41]` | $\sigma_{C1}$ – Phương sai hình dạng phổ | Người nói có hay thay đổi hình dạng miệng/ống họng không? | Đo **độ linh hoạt khẩu hình** — người nói năng động có $\sigma_{C1}$ cao hơn người nói đều đều. |
| `[42]–[45]` | $\sigma_{C2} \to \sigma_{C5}$ – Phương sai Formant | Mức độ dao động của F1, F2 qua 5s. | Người nói nhiều nguyên âm khác nhau (đa dạng âm tiết) sẽ có Std Formant cao hơn. |
| `[46]–[52]` | $\sigma_{C6} \to \sigma_{C12}$ – Phương sai chi tiết cộng hưởng | Dao động chi tiết của ống thanh quản theo từng âm tiết. | **Nhịp điệu nói** (prosody) cá nhân — mỗi người có thói quen nhấn vần, luyến láy riêng. |
| `[53]–[79]` | $\sigma_{C13} \to \sigma_{C39}$ – Phương sai bậc cao | Biến động phổ cao tần do phụ âm, cách đóng môi, phát âm đuôi câu. | Phong cách phát âm đặc thù của từng người — khó clone nhất. |

> **Tại sao Std quan trọng không kém Mean?**  
> Xét 2 người nói với cùng $\mu_{C1}$ (giọng nghe tương tự): Người A nói đều đều (monotone teacher), $\sigma_{C1} = 2.1$. Người B nói linh hoạt, nhấn nhá (animated speaker), $\sigma_{C1} = 18.7$. Chỉ cần Std là phân biệt được ngay dù Mean gần như bằng nhau!

### 5.3. Spectral Contrast Mean – 7 Chiều (indices 80–86)

Spectral Contrast đo **sự chênh lệch năng lượng** giữa đỉnh phổ (spectral peak) và thung lũng phổ (spectral valley) trong từng dải tần số. Nó bù đắp những thứ MFCC không nắm bắt được.

| Chiều (index) | Dải tần số | Ý nghĩa vật lý | Ý nghĩa trong bài toán Voice Similarity |
| :---: | :--- | :--- | :--- |
| `[80]` SC[0] | 0 – 200 Hz | Năng lượng bass: tần số rất thấp (cơ âm, "độ trầm" nền của giọng). | Phân biệt giọng trầm sâu (bass voice) vs. giọng nhẹ. Liên quan đến tần số cơ bản $F_0$ (pitch). |
| `[81]` SC[1] | 200 – 400 Hz | Vùng giọng nói thấp, cộng hưởng ngực (chest resonance). | Giọng "ngực" vs. giọng "cổ" — đặc điểm cơ thể học. |
| `[82]` SC[2] | 400 – 800 Hz | Dải Formant F1: chiều mở miệng, phân biệt nguyên âm /a/ vs. /i/. | Góc nhìn năng lượng về F1 — bổ sung cho $\mu_{C2}$ của MFCC. |
| `[83]` SC[3] | 800 – 1600 Hz | Dải Formant F2: vị trí lưỡi, phân biệt nguyên âm trước vs. sau. | Mức độ "bật lên" (contrast) của các formant nguyên âm — giọng rõ ràng vs. giọng ú ớ. |
| `[84]` SC[4] | 1600 – 3200 Hz | Dải Formant F3 đặc trưng cá nhân cao. | **Feature phân biệt người nói mạnh nhất** trong Spectral Contrast — F3 gần như độc nhất cho mỗi người. |
| `[85]` SC[5] | 3200 – 6400 Hz | Phụ âm sibilant: /s/, /sh/, /f/ — âm rít cao tần. | Cách phát âm phụ âm "sắc" — thể hiện giọng vùng, thổ ngữ, phát âm chuẩn hay không. |
| `[86]` SC[6] | 6400 – 8000 Hz | Tần số rất cao — tiếng rì, đặc tính micro, hơi thở. | Phân biệt điều kiện thu âm, nhưng cũng chứa thông tin về "âm sắc" đặc thù micro của giọng người. |

> **Tại sao cần Spectral Contrast bên cạnh MFCC?**  
> MFCC tổng hợp *hình dạng đường bao phổ* — nó mô tả "ở dải tần nào có nhiều năng lượng". Nhưng MFCC không nói được "năng lượng phổ có tập trung rõ nét hay dàn trải đều?" (tính "bật" hay "mượt" của phổ). Spectral Contrast lấp đầy khoảng trống này: hai người nói có MFCC tương tự nhưng khác nhau về mức độ rõ ràng của formant (vocal clarity) sẽ bị phân biệt bởi SC.

### 5.4. Chroma STFT Mean – 12 Chiều (indices 87–98)

Chroma phân tích âm thanh theo 12 lớp cao độ âm nhạc (pitch classes): C, C#, D, D#, E, F, F#, G, G#, A, A#, B — bất kể ở quãng tám nào.

| Chiều (index) | Nốt nhạc | Ý nghĩa vật lý | Ý nghĩa trong bài toán Voice Similarity |
| :---: | :--- | :--- | :--- |
| `[87]` Ch[0] | C (261 Hz & bội số) | Năng lượng ở tất cả tần số là bội số của C (261, 523, 1046 Hz...). | Phản ánh tần số cơ bản $F_0$ của người nói, ví dụ giọng dày thường thiên về C thấp. |
| `[88]` Ch[1] | C# / Db | Năng lượng chứa trong nốt C# và bội số của nó. | Cùng nguyên tắc — tổ hợp 12 giá trị tạo thành "chữ ký cao độ" của người nói. |
| `[89]`–`[95]` | D, D#, E, F, F#, G, G# | Phân bố năng lượng qua các tần số trung. | Người nam thường có năng lượng Chroma tập trung ở nốt thấp (C–E), người nữ tập trung ở nốt cao (G–B). |
| `[96]` Ch[9] | A (440 Hz) | Nốt A440 là chuẩn quốc tế — nằm ngay vùng F2-F3 trọng tâm giọng nói. | Thường là chiều có giá trị cao nhất trong phổ Chroma của giọng người — giúp phân biệt giọng cao vs. thấp. |
| `[97]` Ch[10] | A# / Bb | — | Phân bố nhỏ quanh vùng A. |
| `[98]` Ch[11] | B (494 Hz) | Tần số cuối của quãng tám giọng nói phổ biến. | Giọng rất cao thường có Ch[11] lớn hơn trung bình. |

> **Tại sao Chroma hữu ích thêm vào MFCC + Spectral Contrast?**  
> - MFCC mô tả *hình dạng bộ lọc* của ống thanh quản.  
> - Spectral Contrast mô tả *mức độ tương phản* của các vùng năng lượng.  
> - **Chroma mô tả *phân phối năng lượng theo cao độ***, tức là giọng người đó chủ yếu rung ở vùng tần số nào — thứ chặt chẽ liên quan đến dây thanh âm và tần số cơ bản $F_0$.  
> - Ba loại đặc trưng này cùng nhau tạo nên mô tả đa chiều, ít bị overlap thông tin hơn, giúp tăng khả năng phân biệt khi tính Cosine Similarity.

---

## 6. Tổng Hợp: Ý Nghĩa Của Vector 99 Chiều Trong Bài Toán Voice Similarity

Khi 2 file âm thanh được so sánh bằng Cosine Similarity trên vector 99 chiều, việc tính toán thực chất là **so sánh đồng thời 4 khía cạnh độc lập** của giọng nói:

```
Cosine(A, B) phản ánh:
├── [0:40]  MFCC Mean  → "Hình dạng ống thanh quản trung bình có giống nhau không?"
├── [40:80] MFCC Std   → "Phong cách & nhịp điệu nói có tương đồng không?"
├── [80:87] Spectral   → "Độ tương phản phổ năng lượng có giống nhau không?"
│           Contrast     (Giọng rõ hay mờ? Mạnh hay yếu ở từng dải tần?)
└── [87:99] Chroma     → "Tần số cơ bản F0 và phân phối cao độ có gần nhau không?"
```

| Kịch bản | Dự đoán Cosine | Lý giải |
| :--- | :---: | :--- |
| Cùng 1 người nói, 2 file khác nhau | ≥ 0.98 | Tất cả 4 khía cạnh đều gần như y hệt nhau. |
| Cùng giới tính, khác người | 0.90 – 0.96 | MFCC Mean khác (ống thanh quản khác) nhưng phân phối Chroma gần nhau. |
| Khác giới tính | 0.80 – 0.90 | Cả MFCC và Chroma đều khác biệt rõ rệt. |
| Giọng người vs. giọng máy tổng hợp (TTS) | < 0.85 | Std MFCC của TTS thường rất thấp (giọng đều), Spectral Contrast của TTS cũng cứng nhắc hơn giọng người thật. |

**Kết Luận Trong Pipeline Của Dự Án:**
Việc sử dụng 40 Mean + 40 Std + Spectral Contrast (7) + Chroma (12) sinh ra vector 99 chiều tĩnh là chiến lược hoàn toàn có chủ đích. Nó tuân thủ phương luận **d-vector / x-vector global embeddings** trong kiến trúc Machine Learning Audio hiện đại, cho độ chính xác lớn đối với phép đo Cosine Similarity và dễ dàng tích hợp hạ tầng cơ sở dữ liệu như PostgreSQL.
