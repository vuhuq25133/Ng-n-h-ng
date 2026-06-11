# 2.2. Cơ Sở Lý Thuyết Trích Xuất Đặc Trưng Âm Thanh

### Nguyên lý bài toán tìm kiếm tương đồng giọng nói và sự liên hệ với miền Thời gian - Tần số
Bài toán tìm kiếm tương đồng giọng nói (Voice Similarity Search) bản chất là bài toán so sánh **"âm sắc" (timbre)** và **"cấu trúc thanh quản" (vocal tract)** của mỗi cá nhân, chứ không tập trung vào nội dung từ ngữ được phát ra (như trong Speech Recognition). 

Tuy nhiên, giọng nói con người là một tín hiệu không dừng (non-stationary), thay đổi liên tục theo thời gian khi ta phát âm các âm tiết khác nhau. 
* Nếu chỉ phân tích trong **miền thời gian (Waveform)**, ta chủ yếu thu được dao động về biên độ (âm lượng). 
* Nếu áp dụng phân tích **miền tần số (Spectrum)** một lần cho toàn bộ đoạn thu âm, kết quả sẽ giống như việc "xay sinh tố" tất cả các âm thanh lại với nhau. Ta sẽ biết đoạn băng đó có những tần số nào, nhưng hoàn toàn không biết âm nào được phát ra trước, âm nào phát ra sau (mất hoàn toàn thông tin về sự chuyển động của khẩu hình miệng theo thời gian).

Do đó, chìa khóa của bài toán định danh giọng nói nằm ở **miền Thời gian - Tần số (Time-Frequency Domain)**. Bằng cách chia nhỏ đoạn âm thanh thành các khung (frame) cực ngắn (ví dụ 32ms), ta có thể giả định rằng trong khoảnh khắc đó tín hiệu là "dừng" (đứng yên). Việc phân tích phổ tần số liên tục trên từng khung nhỏ này sẽ tạo ra một bản đồ (Spectrogram) biểu diễn chi tiết sự biến thiên của tần số theo thời gian. Chính từ miền Thời gian - Tần số này, các thuật toán mới có khả năng bóc tách được các dải tần số cộng hưởng riêng biệt — được định hình bởi hình dáng thanh quản và vòm họng của người nói — để trích xuất ra "chữ ký giọng nói" (Voiceprint) phục vụ cho đối sánh tương đồng.

---

## 2.2.1. Chuyển đổi dữ liệu âm thanh từ vật lý sang kỹ thuật số (ADC)

Âm thanh trong thực tế là sóng áp suất liên tục lan truyền qua không khí. Khi ghi âm, micro sẽ chuyển đổi sóng âm thành tín hiệu điện (Analog). Tuy nhiên, máy tính không thể xử lý tín hiệu liên tục này, do đó cần phải đi qua một bộ chuyển đổi **ADC (Analog-to-Digital Converter)**.

Quá trình này bao gồm hai bước chính:
1. **Lấy mẫu (Sampling):** Đo lường biên độ tín hiệu điện theo những khoảng thời gian đều đặn. Tần số lấy mẫu (Sample Rate) quyết định số lần đo trong một giây. Trong dự án này, âm thanh được lấy mẫu ở tần số $f_s = 16,000 \text{ Hz}$, tức là có $16,000$ giá trị số được tạo ra mỗi giây.
2. **Lượng tử hóa (Quantization):** Gán mỗi mức biên độ thu được vào một dải giá trị số nguyên (ví dụ 16-bit). Cuối cùng, tín hiệu được chuẩn hóa về một dải biểu diễn số thực không thứ nguyên $[-1.0, 1.0]$. 

Với thời lượng 5 giây cho mỗi file, đầu vào của hệ thống là một mảng $x[n]$ gồm $80,000$ giá trị. Mảng này được gọi là dạng sóng nguyên bản (Waveform).

**Triển khai thực tế trong mã nguồn (`feature_extraction.ipynb` / `audio_processor.py`):**
```python
# Tải tệp âm thanh và ép buộc (resample) về tần số lấy mẫu 16,000 Hz, chuyển thành dạng Mono
TARGET_SR = 16_000
y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)

# Kết quả: y có dạng numpy array, y.shape = (80000,)
```

---

## 2.2.2. Lý thuyết phân loại các nhóm đặc trưng

Để phân tích âm thanh, các thuật toán thường chuyển đổi dạng sóng ban đầu thành các đặc trưng toán học mang tính đại diện cao. Có 3 nhóm đặc trưng chính yếu:

### 2.2.2.a Đặc trưng miền thời gian
Đây là nhóm đặc trưng được trích xuất trực tiếp từ biểu diễn dạng sóng (Waveform). Các phương pháp tiêu biểu gồm: Tốc độ qua điểm 0 (Zero-Crossing Rate), Năng lượng bình phương trung bình (RMS Energy) và Đường bao biên độ (Amplitude Envelope). Nhóm này đo lường các đặc tính vật lý nhưng không mang lại nhiều thông tin về cấu trúc âm sắc.

### 2.2.2.b Đặc trưng miền tần số
Sử dụng công cụ toán học để chuyển đổi toàn bộ âm thanh từ miền thời gian sang phổ tần số (Spectrum). Cách tiếp cận này giúp trả lời câu hỏi "Trong âm thanh này có những tần số nào tồn tại?". Tuy nhiên, nó đánh mất hoàn toàn thông tin về sự thay đổi của tần số theo thời gian (khi nào âm đó được phát ra).

### 2.2.2.c Đặc trưng miền thời gian - tần số
Khắc phục nhược điểm của việc mất trục thời gian, miền Thời gian - Tần số chia âm thanh thành các khung (frame) ngắn xen phủ nhau, sau đó phân tích tần số trên từng khung. Kết quả là một ma trận 2D biểu diễn sự thay đổi của năng lượng các dải tần số liên tục theo chiều thời gian (Spectrogram).

### 2.2.2.d Lựa chọn đặc trưng cho dự án: Tại sao không dùng miền Thời gian/Tần số thuần túy và Triệt tiêu trục thời gian?

Trong dự án nhận dạng giọng nói này, cấu trúc vector đặc trưng 99 chiều được thiết kế dựa trên sự lựa chọn nghiêm ngặt sau:

1. **Loại bỏ Miền Thời gian:** Các đặc trưng miền thời gian bị biến thiên rất mạnh do ảnh hưởng của môi trường thu âm, khoảng cách micro và tạp âm. Đặc biệt, chúng không phản ánh được cấu trúc của ống thanh quản (nguyên nhân cốt lõi tạo ra âm sắc riêng biệt của con người). Do vậy, toàn bộ đặc trưng miền thời gian bị loại bỏ.
2. **Loại bỏ Miền Tần số thuần túy:** Giọng nói là một tín hiệu không dừng (non-stationary). Phân tích tần số trên toàn bộ đoạn thu âm 5 giây sẽ dẫn đến hiện tượng trộn lẫn (smearing effect), phá vỡ toàn bộ cấu trúc lời nói.
3. **Triệt tiêu trục thời gian bằng Thống kê (Statistical Pooling):** Dự án sử dụng nền tảng là miền Thời gian - Tần số (chia khung) để theo dõi biến động giọng nói. Tuy nhiên, để lưu trữ dữ liệu tối ưu trong cơ sở dữ liệu và tìm kiếm bằng Cosine Similarity, hệ thống phải áp dụng hàm tính giá trị trung bình (`Mean`) và độ lệch chuẩn (`Std`) dọc theo trục thời gian.
   Kỹ thuật này "ép phẳng" ma trận đặc trưng ban đầu, tạo thành một vector tĩnh. Quá trình này khử bỏ yếu tố độ dài phát âm, cung cấp một **đặc trưng thống kê tổng quát** về phân bố tần số âm sắc cho một người cụ thể.

---

## 2.2.3. Fourier Transform (FFT) – Công cụ chuyển đổi nền tảng

Fourier Transform là nguyên lý cốt lõi của mọi đặc trưng trong miền tần số, với nền tảng là: *Mọi tín hiệu phức tạp đều có thể được phân tách thành tổng của nhiều sóng lượng giác (sin/cos) ở các tần số khác nhau.*

Công thức Biến đổi Fourier rời rạc (DFT):
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi kn/N}$$

Trong đó $x[n]$ là mẫu tại thời gian thứ $n$, $X[k]$ là phổ phức tại bin tần số thứ $k$. Fast Fourier Transform (FFT) là thuật toán tối ưu giúp giải quyết DFT với độ phức tạp $O(N \log N)$ thay vì $O(N^2)$. Trong dự án, FFT đóng vai trò nền móng để chia tách âm thanh gốc thành các đặc trưng cấu trúc giọng nói phức tạp hơn.

---

## 2.2.4. Đặc trưng MFCC (Mel-Frequency Cepstral Coefficients)

**Đặc trưng MFCC là gì?**
MFCC (Hệ số Cepstral trên thang tần số Mel) là bộ đặc trưng kinh điển và quan trọng nhất trong lĩnh vực nhận dạng giọng nói (Speaker Recognition). Nó được thiết kế dựa trên sự kết hợp giữa toán học và đặc tính sinh học của tai người (nhạy cảm ở dải tần số thấp và kém nhạy dần ở dải tần số cao). Bằng cách sử dụng MFCC, hệ thống có thể tách biệt được "nguồn tạo âm" (dây thanh quản) và "bộ lọc âm" (cấu trúc vòm họng, khoang miệng) - vốn là thứ định hình nên âm sắc độc nhất của mỗi người.

Để trích xuất được bộ hệ số này, sóng âm thanh thô bắt buộc phải trải qua một quá trình biến đổi đa bước từ miền Thời gian $\rightarrow$ miền Tần số $\rightarrow$ Thang tần số Mel (thang nghe của tai người) $\rightarrow$ và cuối cùng dùng phép biến đổi Cosine để đưa về miền Cepstral (miền phổ của phổ).

### 2.2.4.a Ý nghĩa các hệ số MFCC (40 chiều)
Thay vì dùng hàng nghìn giá trị tần số thô để mô tả âm thanh, MFCC nén toàn bộ thông tin đó lại thành một nhóm nhỏ các hệ số. Trong dự án này, chúng ta trích xuất **40 hệ số (n_mfcc = 40)**. Mỗi nhóm hệ số đảm nhận một vai trò cụ thể:

| Nhóm hệ số | Ý nghĩa và Vai trò cốt lõi |
|---|---|
| **Hệ số C0** | Đại diện cho mức năng lượng tổng thể (âm lượng) của toàn bộ khung âm thanh. |
| **Hệ số thấp (C1 - C12)** | Mô tả "đường bao phổ" (spectral envelope), đại diện cho **hình dáng của vòm họng và khoang miệng**. Đây là phần lõi chứa thông tin định danh người nói. |
| **Hệ số cao (C13 - C39)** | Chứa các chi tiết biến thiên phức tạp hơn như tiếng bật, tiếng xát của phụ âm hoặc các đặc tính vi mô của hệ thống phát âm. Việc sử dụng trọn vẹn 40 hệ số thay vì 13 như chuẩn cũ giúp hệ thống đạt độ chính xác tối đa. |

### 2.2.4.b Pipeline tổng quan trích xuất MFCC
Luồng xử lý từ sóng âm thanh tạo ra đặc trưng MFCC bao gồm 7 bước liên tiếp:
*Tín hiệu đầu vào $\rightarrow$ Phân khung $\rightarrow$ Cửa sổ Hamming $\rightarrow$ STFT $\rightarrow$ Bộ lọc Mel $\rightarrow$ Lấy Logarit $\rightarrow$ Biến đổi DCT.*

### 2.2.4.c Phân khung (Framing) & Hàm cửa sổ (Windowing)

**1. Phân khung (Framing) để làm gì?**
Giọng nói là một tín hiệu liên tục thay đổi (non-stationary). Tuy nhiên, để áp dụng các phép biến đổi toán học như Fourier (vốn yêu cầu tín hiệu phải có tính ổn định), ta phải thực hiện một thao tác gọi là Phân khung. Bằng cách chia đoạn âm thanh dài thành các đoạn cắt rất ngắn, ta có thể giả định tín hiệu bên trong mỗi khung đó là "tĩnh" (stationary) để đem đi phân tích phổ.
*Trong dự án, kích thước một khung là $N_{FFT} = 2048$ mẫu (tương đương 128ms), và hệ thống sẽ trượt một khoảng $H = 512$ mẫu (32ms) để sang khung tiếp theo, tạo ra sự chồng lấp (overlap) 75% giữa các khung để bảo toàn thông tin chuyển tiếp.*

**2. Vấn đề Rò rỉ phổ (Spectral Leakage) và Vai trò của Windowing**
* **Vấn đề (Rò rỉ phổ):** Khi ta dùng thuật toán cắt tín hiệu thành các khung, điểm cắt ở hai đầu biên thường rất đột ngột (không mượt mà). Nếu đem nguyên khung bị đứt gãy này đi biến đổi Fourier, thuật toán sẽ sinh ra các dải tần số nhiễu không có thật để bù đắp cho sự đứt gãy đó. Hiện tượng này gọi là **Rò rỉ phổ (Spectral Leakage)**.
* **Cách giải quyết (Windowing):** Ta đưa mỗi khung đi qua một Hàm cửa sổ (Windowing Function). Hàm này giữ nguyên mức tín hiệu ở giữa khung nhưng "bóp" dần tín hiệu ở hai đầu biên về giá trị 0 một cách mượt mà, giúp nối các khung lại với nhau mà không bị gãy góc.

**3. Cửa sổ Hamming**
Trong lĩnh vực xử lý tiếng nói, hàm cửa sổ ưu việt và được sử dụng rộng rãi nhất là **Cửa sổ Hamming**. Mỗi khung âm thanh trước khi đưa vào tính tần số đều được nhân với hàm Hamming theo công thức:
$$w[n] = 0.54 - 0.46\cos\left(\frac{2\pi n}{N - 1}\right)$$
Với $N$ là số mẫu trong khung. Cửa sổ Hamming giúp giữ lại tối đa thông tin ở giữa khung đồng thời triệt tiêu triệt để hiện tượng rò rỉ phổ ở hai biên.

### 2.2.4.d Short-Time Fourier Transform (STFT) và Năng lượng phổ
Sau khi tín hiệu đã được phân khung và nhân với cửa sổ Hamming, ta áp dụng phép Biến đổi Fourier Nhanh (FFT) trên từng khung rời rạc này. Phép toán này được gọi là STFT, đóng vai trò cốt lõi trong việc chuyển đổi từ miền Thời gian thuần túy sang miền Thời gian - Tần số.

Công thức STFT cho khung thứ $m$:
$$S(m, k) = \sum_{n=0}^{N-1} \big[x(n + mH) \cdot w(n)\big] \cdot e^{-j2\pi n k/N}$$
Trong đó:
- $x(n + mH)$: Tín hiệu âm thanh tại khung thứ $m$ (với $H$ là bước nhảy trượt khung).
- $w(n)$: Cửa sổ Hamming đã đề cập ở bước trước.
- $k$: Chỉ số dải tần số (Frequency bin).

Kết quả của $S(m, k)$ là một phổ phức (chứa cả phần thực và ảo). Để biết được cường độ của tần số, ta tính giá trị tuyệt đối để thu được **Phổ biên độ** (Magnitude Spectrum) $|S(m, k)|$.
Cuối cùng, ta tính bình phương phổ biên độ để ra được **Phổ năng lượng** (Power Spectrum):
$$P(m, k) = \frac{1}{N}\,|S(m, k)|^2$$
Ma trận phổ năng lượng $P$ lúc này chính là một biểu diễn Spectrogram, cho biết năng lượng của từng dải tần số biến thiên như thế nào qua từng khung thời gian.

### 2.2.4.e Thang Mel và Bộ lọc Mel (Mel FilterBank)

**1. Vấn đề với thang Hz tuyến tính**
Trong vật lý, tần số âm thanh được đo bằng Hz một cách tuyến tính. Tuy nhiên, tai người lại **không nghe tuyến tính**. Màng nhĩ rất nhạy bén ở những dải tần số thấp nhưng lại kém nhạy ở tần số cao. 
Ví dụ: Sự khác biệt giữa 100 Hz và 200 Hz (chênh 100 Hz) được tai người nhận biết rất rõ ràng; thế nhưng sự khác biệt giữa 8000 Hz và 8100 Hz (cũng chênh 100 Hz) lại gần như không thể phân biệt được. Nếu giữ nguyên phổ tần số theo thang Hz để phân tích, máy tính sẽ dành quá nhiều tài nguyên vô ích cho dải tần số cao, dẫn đến sự sai lệch trong việc định danh.

**2. Thang Mel và Bộ lọc Mel (Mel FilterBank)**
Để máy tính "nghe" giống con người nhất, phổ năng lượng $P(m, k)$ được ánh xạ từ thang Hz sang **thang Mel phi tuyến** bằng công thức logarit:
$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

Bảng so sánh dưới đây minh họa sự nén (bóp méo) của thang Mel so với thang Hz tuyến tính:

| Tần số (Hz) | Giá trị Mel |
|:---:|:---:|
| 0 | 0 |
| 100 | 150 |
| 500 | 607 |
| 1,000 | 1,000 |
| 4,000 | 1,789 |
| 8,000 | 2,146 |

**Quan sát:** Từ $0 \rightarrow 1000$ Hz, thang Mel tăng rất nhanh (xấp xỉ 1:1 với Hz). Tuy nhiên từ $1000 \rightarrow 8000$ Hz (khoảng cách 7000 đơn vị), thang Mel chỉ tăng thêm vỏn vẹn $\sim 1146$ đơn vị. Sự chuyển đổi này đã "nén" toàn bộ các tần số cao không quan trọng lại với nhau.

Sau khi thực hiện phép chuyển đổi, phổ năng lượng được đưa qua một hệ thống gồm $M=128$ **bộ lọc tam giác** (Mel FilterBanks) được phân bố đều đặn trên thang Mel từ $0$ đến $f_s/2 = 8000$ Hz.
* Trên thang Mel, các tam giác lọc này cách đều nhau. 
* Nhưng khi chiếu ngược lại trên thang Hz thực tế, các bộ lọc này sẽ xếp dày đặc ở vùng tần số thấp (để phân tích kỹ) và thưa dần ở vùng tần số cao (để lướt qua). 

Công thức toán học của bộ lọc thứ $m$ ($H_m[k]$):
$$
H_m[k] = \begin{cases}
0 & k < f[m-1] \\
\dfrac{k - f[m-1]}{f[m] - f[m-1]} & f[m-1] \le k \le f[m] \\
\dfrac{f[m+1] - k}{f[m+1] - f[m]} & f[m] < k \le f[m+1] \\
0 & k > f[m+1]
\end{cases}
$$
Trong đó $f[0], f[1], \dots, f[M+1]$ là các tâm bộ lọc (center frequencies) trên thang Hz (được ánh xạ từ thang Mel cách đều).

Áp dụng bộ lọc tam giác này lên phổ năng lượng $P_t[k]$ của khung thứ $t$, ta sẽ "nhóm" và tổng hợp được năng lượng Mel:
$$S_t[m] = \sum_{k=0}^{N_{FFT}/2} P_t[k] \cdot H_m[k], \quad m = 1, \dots, M$$
Kết quả $S_t[m]$ là một ma trận Mel Spectrogram có kích thước đã được thu gọn `(128, số_khung)`.

### 2.2.4.f Áp dụng Logarit (Log Power Spectrum) để làm gì?
Việc áp dụng hàm Logarit tự nhiên lên năng lượng Mel $S_t[m]$ giải quyết cùng lúc hai vấn đề lớn:
1. **Mô phỏng thính giác con người:** Tai người nghe cường độ âm thanh theo hàm logarit (năng lượng vật lý phải tăng gấp 10 lần thì tai mới thấy âm lượng to lên gấp đôi). Việc lấy Log giúp máy tính cảm nhận âm lượng y hệt như cấu tạo sinh lý của tai (tương tự thang đo Decibel).
2. **Hỗ trợ "gỡ rối" toán học:** Về mặt cơ học, âm thanh phát ra là kết quả của việc **nhân** Nguồn âm (Dây thanh quản) với Bộ lọc âm (Hình dáng vòm họng/lưỡi). Khi lấy Logarit, phép nhân $\log(A \times B)$ sẽ biến thành phép cộng $\log(A) + \log(B)$. Việc chuyển đổi này là tiền đề bắt buộc để bước DCT tiếp theo có thể tách rời hai thành phần này bằng phép trừ tuyến tính.

$$\tilde{S}_t[m] = \log\left(S_t[m] + \epsilon\right)$$
Hằng số nhỏ $\epsilon \approx 10^{-10}$ được cộng thêm vào nhằm mục đích tránh lỗi toán học $\log(0)$ tại những dải tần số hoàn toàn câm (năng lượng bằng 0).

### 2.2.4.g Biến đổi Cosine rời rạc (DCT) và Miền giả thời gian (Cepstral Domain)
**Tại sao phải dùng DCT và Miền giả thời gian là gì?**
Sau khi lấy Logarit, thông tin của Dây thanh quản (tạo ra cao độ) và Vòm họng (tạo ra âm sắc đặc trưng) đang bị cộng lẫn lộn vào nhau. Đồng thời, ma trận năng lượng 128 dải Mel đang bị dư thừa thông tin do các bộ lọc tam giác có sự chồng lấp chéo lên nhau.

Để giải quyết, ta áp dụng **Biến đổi Cosine rời rạc (DCT)**. Về bản chất, DCT hoạt động gần giống như phép Biến đổi Fourier nghịch đảo, với công dụng chính:
* Đưa phổ tín hiệu sang một không gian mới gọi là **Miền giả thời gian (Cepstral Domain)** — (chữ *Cepstral* là phép đảo chữ của *Spectral*).
* Tại miền giả thời gian này, phép DCT đóng vai trò như một màng lọc siêu việt: Nó dồn toàn bộ thông tin mô tả "hình dáng vòm họng và khoang miệng" (vốn biến thiên chậm) lên các hệ số đầu tiên; và đẩy các thông tin về "dây thanh quản" cùng "nhiễu" (vốn biến thiên nhanh) xuống các hệ số cuối cùng.

$$c_t[i] = \sum_{m=1}^{M} \tilde{S}_t[m] \cos\left(\frac{\pi i (m - 0.5)}{M}\right)$$

Nhờ tính chất dồn nén dữ liệu cực tốt của DCT, thay vì phải xử lý cả một ma trận khổng lồ, hệ thống chỉ cần trích xuất đúng **40 hệ số đầu tiên** ($c_t[0]$ đến $c_t[39]$) là đã thu thập được trọn vẹn "chữ ký vòm họng" của người nói, đồng thời tự động loại bỏ được các nhiễu môi trường ở phần đuôi mảng. Đây chính là sức mạnh cốt lõi của MFCC!

**Toàn bộ chu trình từ bước 2.2.4.c đến 2.2.4.g được gói gọn trong mã nguồn thông qua hàm của librosa:**
```python
# Trích xuất 40 hệ số MFCC thông qua librosa
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

mfcc = librosa.feature.mfcc(
    y=y, 
    sr=sr,
    n_mfcc=N_MFCC,         # Số hệ số giữ lại
    n_fft=N_FFT,           # Kích thước khung (Framing)
    hop_length=HOP_LENGTH, # Bước nhảy
    n_mels=N_MELS          # Số lượng bộ lọc Mel
)
# Kết quả mfcc có dạng ma trận 2D: (40, 157) - tức 40 hệ số trải qua 157 khung thời gian
```

### 2.2.4.h Tổng hợp vector bằng Thống kê tĩnh (Statistical Pooling): Mean và Std
Kết quả thu được sau bước DCT là một ma trận 2D kích thước `(40, số_khung)` (ví dụ âm thanh 5 giây sẽ có khoảng 157 khung). Nếu để nguyên ma trận này, độ dài vector của mỗi file âm thanh sẽ khác nhau (người nói 3 giây ra ma trận ngắn, người nói 5 giây ra ma trận dài). Sự khác biệt về kích thước này khiến hệ thống không thể lưu trữ ổn định vào cơ sở dữ liệu và không thể dùng các công thức toán học như Cosine Similarity để tính khoảng cách trực tiếp.

Để giải quyết triệt để vấn đề, dự án áp dụng kỹ thuật **Thống kê tĩnh (Statistical Pooling)**. Cơ chế "ép phẳng" này hoạt động bằng cách đi dọc theo từng hàng ngang của ma trận (mỗi hàng đại diện cho 1 trong 40 hệ số đặc trưng). Thuật toán sẽ gom toàn bộ giá trị của hệ số đó trên tất cả các khung thời gian (từ khung đầu tiên đến khung cuối cùng) lại với nhau để tính ra một con số đại diện duy nhất (bằng cách thiết lập `axis=1` trong thư viện NumPy). Nhờ vậy, ma trận 2D có số cột biến động đã được nén chặt thành các vector 1 chiều tĩnh.

**Ví dụ toán học minh họa thao tác "ép phẳng":**
Giả sử thay vì 40 hệ số và 157 khung, ta chỉ trích xuất **2 hệ số MFCC** từ một đoạn âm thanh siêu ngắn gồm **3 khung thời gian**. Ma trận ban đầu sẽ trông như sau:
* Trục thời gian: `[Khung 1]  [Khung 2]  [Khung 3]`
* Hệ số 1 (C1): `[   1.5   ,   2.5   ,   2.0   ]`
* Hệ số 2 (C2): `[   0.1   ,  -0.5   ,   0.4   ]`

Để "gom toàn bộ giá trị" và tính Trung bình (Mean) cho Hệ số 1 (C1), ta đi dọc theo hàng ngang thứ nhất:
- `Mean(C1) = (1.5 + 2.5 + 2.0) / 3 = 2.0`
Làm tương tự với hàng ngang thứ hai của Hệ số 2 (C2):
- `Mean(C2) = (0.1 - 0.5 + 0.4) / 3 = 0.0`
$\Rightarrow$ Kết quả: Ma trận 2D kích thước `(2x3)` ban đầu đã bị ép phẳng thành một vector tĩnh 1 chiều `[2.0, 0.0]`. Dù file âm thanh thực tế có bị kéo dài lên 100 hay 1000 khung, kết quả đầu ra vẫn luôn bị nén lại thành một vector có đúng 2 chiều cố định. Đây chính là cách giải quyết tuyệt đối vấn đề chênh lệch thời lượng file âm thanh.

Trong dự án này, hệ thống kết hợp song song cả phép tính Trung bình (Mean) và Độ lệch chuẩn (Std):

1. **MFCC Mean (Trung bình - 40 chiều):** Phản ánh **"Màu sắc âm thanh tổng quát"**. Phép tính này cộng dồn các khung thời gian lại, tạo ra một cấu trúc vòm họng đại diện của người đó trong suốt đoạn thu âm. Nhược điểm của Mean là nếu có hai người có chất giọng gốc na ná nhau, giá trị Mean của họ sẽ rất sát nhau, dễ gây nhận diện nhầm.
2. **MFCC Std (Độ lệch chuẩn - 40 chiều):** Để giải quyết điểm yếu của Mean, dự án trích xuất thêm Độ lệch chuẩn. Tham số này phản ánh **"Độ biến thiên, nhịp điệu và phong cách phát âm"** (Prosody & Dynamics) của người nói qua thời gian. Nhờ MFCC Std, hệ thống tóm gọn được bức tranh về việc "khẩu hình miệng di chuyển linh hoạt như thế nào" mà không cần phải sử dụng các thuật toán lưu giữ chuỗi thời gian đắt đỏ.

**Ví dụ trực quan về vai trò kết hợp của Mean và Std:**
Giả sử có hai người A và B là anh em sinh đôi, có cấu trúc thanh quản giống hệt nhau (chất giọng bẩm sinh y hệt nhau $\rightarrow$ **MFCC Mean** của A và B sẽ gần như bằng nhau). 
- Người A là biên tập viên thời sự, luôn đọc văn bản với tốc độ chuẩn xác, đều đều, ít ngắt nghỉ thừa $\rightarrow$ Lượng biến thiên theo thời gian rất thấp $\rightarrow$ **MFCC Std** của A sẽ rất thấp.
- Người B là một diễn viên hài kịch, nói chuyện luôn lên bổng xuống trầm, ngắt nhịp liên tục và hay kéo dài các nguyên âm $\rightarrow$ Lượng biến thiên theo thời gian cực kỳ lớn $\rightarrow$ **MFCC Std** của B sẽ rất cao.
$\Rightarrow$ Bằng cách ghép cả Mean và Std lại với nhau (tạo thành 80 chiều vector), máy tính có thể phân biệt thành công A và B dựa vào "phong cách phát âm", dù chất giọng gốc của họ có giống nhau đến đâu.

**Tại sao việc "Triệt tiêu trục thời gian" (tính Mean) lại không làm hỏng bài toán?**
Có một câu hỏi rất lớn đặt ra: *Nếu ta tính Trung bình (Mean), thứ tự các âm thanh phát ra sẽ bị hòa trộn. Một người nói "A rồi B" hay "B rồi A" thì giá trị Trung bình cuối cùng đều bằng nhau. Tại sao dự án vẫn nhận diện được?*
Câu trả lời nằm ở sự khác biệt về mục đích cốt lõi:
- Nếu làm bài toán **Nhận dạng tiếng nói (Speech-to-Text)** để biết người ta đang nói chữ gì, việc ép phẳng làm mất trục thời gian là **thảm họa**.
- Nhưng đồ án này giải quyết bài toán **Định danh người nói (Speaker Identification/Verification)**. Ta hoàn toàn KHÔNG QUAN TÂM người đó đang nói nội dung gì, ta chỉ cần biết **hình dáng vòm họng của họ được cấu tạo như thế nào**. Một đoạn âm thanh 5 giây là đủ dài để người nói huy động hầu hết các khẩu hình cơ bản (A, E, I, O, U, các phụ âm...). Khi lấy Trung bình (Mean) và Độ lệch (Std) của tất cả các khẩu hình này, ta sẽ thu được một "Mô hình sinh học vòm họng tĩnh" cực kỳ chuẩn xác, mang tính định danh cá nhân độc nhất mà hoàn toàn không bị phụ thuộc vào việc họ đang đọc văn bản nào.

**Triển khai thống kê trong mã nguồn:**
```python
# Gộp thống kê dọc theo trục thời gian (axis=1)
mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)  # Vector 40 chiều
mfcc_std  = np.std(mfcc,  axis=1).astype(np.float32)  # Vector 40 chiều
```

---

## 2.2.5. Spectral Contrast (Đặc trưng đối lập phổ - 7 chiều)

Spectral Contrast đo lường **sự tương phản về năng lượng** giữa đỉnh phổ (peak) và thung lũng phổ (valley) trong từng dải tần riêng biệt. Đặc trưng này phản ánh độ sắc nét của cấu trúc cộng hưởng ống thanh quản.

**Ý tưởng trực quan:**
- Giọng phát âm rõ chữ, tiếng vang: Phổ âm thanh sẽ có những "đỉnh" vọt lên rất cao và "thung lũng" sâu $\rightarrow$ Độ tương phản (Contrast) **CAO**.
- Giọng nói thều thào, bị khàn, hoặc âm thanh chứa nhiều tạp âm (Noise): Phổ âm thanh sẽ dàn đều đều, không có đỉnh rõ rệt $\rightarrow$ Độ tương phản (Contrast) **THẤP**.

**Thuật toán trích xuất gồm 2 bước:**
**1. Chia phổ thành 7 dải tần số:**
Với tham số cấu hình `n_bands=6` và `fmin=200`, hệ thống chia toàn bộ tần số thành 7 phân khu trải dài theo quãng tám:
* Dải riêng biệt (Bass band): `0 Hz` $\rightarrow$ `200 Hz`
* Dải 1: `200 Hz` $\rightarrow$ `400 Hz`
* Dải 2: `400 Hz` $\rightarrow$ `800 Hz`
* Dải 3: `800 Hz` $\rightarrow$ `1600 Hz`
* Dải 4: `1600 Hz` $\rightarrow$ `3200 Hz`
* Dải 5: `3200 Hz` $\rightarrow$ `6400 Hz`
* Dải 6: `6400 Hz` $\rightarrow$ `8000 Hz`

**2. Tìm Peak, Valley và tính Contrast:**
Trong mỗi dải phân khu, thuật toán sẽ tính trung bình của 2% những mẫu có năng lượng cao nhất (Peak) và 2% những mẫu có năng lượng thấp nhất (Valley). Từ đó, tính được độ tương phản cho từng dải:
$$\text{Contrast}_b(t) = \text{Peak}_b(t) - \text{Valley}_b(t)$$
*(Vì Peak và Valley đã được lưu ở thang đo Logarit nên phép trừ thực chất chính là phép chia tỉ lệ cường độ, đơn vị sinh ra là Decibel - dB).*

Kết thúc quá trình tính toán, hệ thống áp dụng tiếp Thống kê tĩnh (Tính Mean dọc theo trục thời gian giống MFCC) để nén lại thành một vector **7 chiều tĩnh**, dùng để định danh độ "trong trẻo/rõ ràng" trong chất giọng của người nói.

**Triển khai trong mã nguồn:**
```python
# Trích xuất Spectral Contrast với 6 dải tần (7 hệ số kể cả dải siêu trầm)
SC_N_BANDS = 6
SC_FMIN = 200.0

sc = librosa.feature.spectral_contrast(
    y=y, 
    sr=sr,
    n_bands=SC_N_BANDS,    # 6 dải -> 7 hệ số
    fmin=SC_FMIN           # Bắt đầu tính từ 200 Hz
)
sc_mean = np.mean(sc, axis=1).astype(np.float32)  # Vector 7 chiều
```

---

## 2.2.6. Chroma STFT (Đặc trưng cao độ - 12 chiều)

Đặc trưng Chroma (hay Chromagram) tập trung phân tích âm thanh trực tiếp theo **12 lớp cao độ âm nhạc** (pitch classes), tương ứng với 12 nốt nhạc cơ bản trong một quãng tám: C, C#, D, D#, E, F, F#, G, G#, A, A#, B.

**Ý tưởng trực quan:**
Trong âm thanh học, tất cả các tần số là bội số của nhau (cách nhau đúng 2 lần) đều chia sẻ cùng một "màu sắc" (chroma) và nghe rất giống nhau. Ví dụ:
- Nốt C4 (Đô) có tần số $261.63$ Hz.
- Nốt C5 (Đô cao hơn 1 quãng) có tần số $523.25$ Hz (gấp đôi C4).
- Nốt C6 (Đô cao hơn 2 quãng) có tần số $1046.5$ Hz (gấp 4 lần C4).
$\rightarrow$ Thuật toán Chroma sẽ cộng dồn tất cả các mức năng lượng tại tần số $261.63$, $523.25$, $1046.5$... gộp chung vào một nhóm duy nhất gọi là "Lớp nốt C" (Chỉ số 0).

**Công thức ánh xạ toán học:**
Hệ thống sử dụng công thức sau để ép một tần số $f$ bất kỳ (Hz) vào 1 trong 12 lớp cao độ (chỉ số $p$ từ 0 đến 11):
$$p = 12 \cdot \log_2\left(\frac{f}{440}\right) \mod 12$$
*(Trong đó, mốc tham chiếu $f_{ref} = 440$ Hz là tần số chuẩn quốc tế của nốt La - A4)*. Nhờ phép chia lấy dư (`mod 12`), trục tần số kéo dài vô hạn đã bị cuộn tròn lại thành một vòng tuần hoàn 12 nốt.

**Tại sao dự án dùng Chroma để bổ sung cho MFCC?**
- MFCC và Spectral Contrast tập trung giải mã **Âm sắc** (do hình dáng vòm họng quy định). Quá trình đó đã cố tình nén đi thông tin về **Cao độ** (do tốc độ rung của dây thanh quản quy định).
- Chroma sinh ra để bù đắp chính khoảng trống này. Nó bắt được chính xác cao độ gốc (Pitch) của người nói. Ví dụ: Nó giúp phân biệt lập tức giữa một giọng nam trầm (năng lượng rải ở các nốt C, D) với một giọng nữ cao (năng lượng rải ở các nốt A, B) ngay cả khi họ đọc cùng một chữ.

Tương tự các đặc trưng trước, ma trận Chroma 2D `(12, số_khung)` được ép phẳng bằng Thống kê tĩnh dọc theo trục thời gian (tính Mean) để thu về **vector 12 chiều tĩnh**, ghép lại thành mảnh ghép cuối cùng của quá trình Feature Extraction.

**Triển khai trong mã nguồn:**
```python
# Ánh xạ tần số vào 12 pitch classes
chroma = librosa.feature.chroma_stft(
    y=y, 
    sr=sr,
    n_fft=N_FFT,           # Dùng chung kích thước khung với MFCC
    hop_length=HOP_LENGTH  # Dùng chung bước nhảy
)
chroma_mean = np.mean(chroma, axis=1).astype(np.float32)  # Vector 12 chiều
```

---

## 2.2.7. Ứng dụng Vector Search bằng pgvector & Độ tương đồng Cosine (Cosine Similarity)

Sau khi hoàn thành các bước trích xuất, toàn bộ đặc trưng được nối (concatenate) lại tạo ra một Embedding duy nhất gồm **99 chiều** (40 + 40 + 7 + 12). 

**Mã nguồn gộp vector (trong `feature_extraction.ipynb` và `audio_processor.py`):**
```python
embedding = np.concatenate([
    mfcc_mean,    # [0:40]
    mfcc_std,     # [40:80]
    sc_mean,      # [80:87]
    chroma_mean,  # [87:99]
]) 
# Kết quả: mảng NumPy 1 chiều (99,)
```

**1. Lưu trữ và Tìm kiếm bằng pgvector**
Để xây dựng một hệ thống tìm kiếm giọng nói theo thời gian thực với lượng dữ liệu lớn, việc so sánh từng mảng NumPy bằng các vòng lặp truyền thống trong code Python là bất khả thi về mặt hiệu năng. Do đó, dự án đưa toàn bộ các vector 99 chiều này vào lưu trữ trực tiếp trong cơ sở dữ liệu **PostgreSQL** thông qua phần mở rộng **`pgvector`**.

`pgvector` biến hệ quản trị CSDL truyền thống thành một Vector Database thực thụ. Nó hiểu được kiểu dữ liệu `vector` (thay vì chỉ lưu mảng text) và hỗ trợ quét tìm kiếm đối sánh (Vector Search) dựa trên các thuật toán chỉ mục tiên tiến như **HNSW (Hierarchical Navigable Small World)** đã được tối ưu hóa sâu ở tầng C/C++.

**2. Độ tương đồng Cosine (Cosine Similarity)**
Để xác định hai người nói có giống nhau hay không, `pgvector` sử dụng thuật toán tính khoảng cách Cosine (Ký hiệu toán tử trong SQL là `<=>`). Thay vì đo lường khoảng cách vật lý thông thường, thuật toán này đo lường **góc hợp bởi hai vector** trong không gian 99 chiều:
* Nếu hai vector chỉ sai lệch nhau một góc rất nhỏ $\rightarrow$ Cosine tiến tới 1 $\rightarrow$ Giọng nói cực kỳ giống nhau.
* Nếu hai vector vuông góc (hoặc cách xa nhau) $\rightarrow$ Cosine tiến về 0 $\rightarrow$ Hai giọng nói hoàn toàn khác biệt.

Công thức toán học của Độ tương đồng Cosine:
$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|} = \frac{\sum_{i=0}^{98} A_i B_i}{\sqrt{\sum_{i=0}^{98} A_i^2} \cdot \sqrt{\sum_{i=0}^{98} B_i^2}}$$

**Triển khai trong Backend (`database.py`):**
```python
# Toán tử <=> của pgvector tự động tính toán Khoảng cách Cosine (= 1 - Cosine Similarity)
# Do đó ta dùng công thức `1 - Khoảng cách Cosine` để trả về đúng Độ tương đồng (%)
query = """
    SELECT file_id, speaker, file_path, 
           1 - (embedding <=> %s::vector) AS similarity 
    FROM voice_records 
    ORDER BY embedding <=> %s::vector 
    LIMIT %s
"""
cur.execute(query, (embedding_str, embedding_str, top_k))
```
