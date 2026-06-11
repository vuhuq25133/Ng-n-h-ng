# 🗂️ Tài Liệu Hệ Thống Metadata Âm Thanh

> **Đầu ra:** `data/metadata/metadata.csv` (CSV Format)  
> **Input gốc:** `<speaker>_<file>.wav` trong `data/processed/`  
> **Notebook xử lý:** `src/extract_metadata/extract_metadata.ipynb`  
> **Ngày cập nhật:** 2026-04-13

---

## 📌 1. Mục Đích của Pipeline Xuất Metadata

Khác với **Feature Vectors** (đặc trưng số hóa được lưu dưới mảng numpy hoặc cơ sở dữ liệu `pgvector` phục vụ tính toán khoảng cách cosine, Machine Learning), **Metadata** (Dữ liệu đặc tả) cung cấp **sự biểu hiện dễ hiểu nhất** về file âm thanh đó cho con người hoặc hệ thống phân loại.

Hệ thống Metadata Âm thanh này được trích xuất hoàn toàn riêng biệt so với quy trình Neural Network Extract Feature ở chỗ:
1. Thu thập transcript gốc để dễ truy xuất ngữ nghĩa (Nội dung người đó đọc là gì?).
2. Thống kê chất lượng lọc qua độ lớn âm thầm (RMS). 
3. Xác nhận tính nguyên vẹn chiều dài và sample rate để cảnh báo sớm về Data Drifting.

Mọi kết quả được nén đồng bộ thành file `metadata.csv` gọn nhẹ. Lập trình viên có thể dùng thư viện `pandas` trong hệ sinh thái Python để lọc, plot biểu đồ EDA kiểm nghiệm cực kỳ nhanh chóng.

---

## 📋 2. Phân Loại Metadata

Metadata cho một mẫu âm thanh được chia thành hai mảng thông tin chính:

### I. Non-acoustic Metadata (Siêu dữ liệu hành chính / quản lý)

Liên quan chủ yếu đến cấu trúc file và ngữ liệu phân loại chứ không lấy từ bên trong tín hiệu sóng kỹ thuật số.

| Tên Cột CSV | Loại Dữ Liệu | Giải Thích | Ví Dụ | Nguồn tham chiếu gốc |
|---|---|---|---|---|
| `speaker` | Chuỗi (`str`) | User ID ảo định danh người nói trong Corpus VCTK. | `p225` | Tên của Parent Folder |
| `file_name`| Chuỗi (`str`) | Tên của file âm thanh đuôi `.wav` | `p225_001.wav`| Hệ điều hành |
| `duration_sec`| Số thực (`float`)| Độ dài tuyệt đối trong thời gian thực. Vì đã qua tiền xử lý chuẩn hoá, hầu hết sẽ xấp xỉ `5.0`. | `5.0` | Tính chiều dài array bằng Numpy |
| `keyword` | Chuỗi (`str`) | Phiên bản văn bản (Text Transcript) mà người thu âm đã đọc lên để tạo ra đoạn wav này. | `"Please call Stella."`| `data/raw/VCTK-Corpus/VCTK-Corpus/txt/` |


### II. Acoustic Metadata (Siêu dữ liệu vật lý)

Được đo lường hoặc trích xuất trực tiếp bằng cách quét toàn bộ con trỏ đọc qua tệp tín hiệu âm thanh kỹ thuật số đó.

| Tên Cột CSV | Loại Dữ Liệu | Giải Thích | Ví Dụ | Công thức / Nguồn trích |
|---|---|---|---|---|
| `sample_rate` | Số nguyên (`int`) | Số lượng các "nhát cắt" mẫu rời rạc thu được mỗi 1 giây. Chuẩn chung sau tiền xử lý của dự án là 16kHz. | `16000` | Thuộc tính của File PCM Header |
| `rms_db` | Số thực (`float`)| Mức độ to nhỏ trung bình (năng lượng) của file âm thanh quy về hệ **dBFS (Decibels Full Scale)**. Do là hệ mét số nên nó sẽ mang **giá trị âm**; số càng tiến về 0 thì âm thanh càng to (ngưỡng 0đB là sát trần méo dạng Clipping).| `-25.32` | $dBFS = 20 \times \log_{10}(RMS_{linear} + 1\times 10^{-9})$ với $RMS_{linear}$ trung bình bình phương array numpy |
| `bit_depth`| Số nguyên (`int`) | Số lượng bit dùng để mô tả biên độ tại mỗi mẫu tín hiệu rời rạc. VCTK ghi bằng PCM-16. | `16` | Trích qua `soundfile.info` dựa trên đuôi `Subtype = PCM_16` |

> 💡 **Chú ý**: Tại sao nên dùng `rms_db` thay vì `rms_linear` (vd: `0.0031`)? \
> Do thính giác con người có khả năng cảm nhận độ to theo kiểu hàm số "Logarithm". Cho nên quy đổi ra giá trị decibels (dBFS) sẽ giống trực quan sinh học nhất, và dễ lọc nhiễu âm hơn.

---

## 🛠️ 3. Tương tác với File CSV Bằng Code (Python)

Khi đã xử lý, đường dẫn file là `data/metadata/metadata.csv`.
Đây là cách cơ bản để lập trình viên lấy dữ liệu trong code:

```python
import pandas as pd

# 1. Đọc hệ thống dữ liệu Metadata
df = pd.read_csv("data/metadata/metadata.csv")

# 2. In ra 5 dòng đầu
print(df.head())

# 3. Tra thông tin transcript Text theo mã wav audio. VD: p225_004.wav
keyword = df[df["file_name"] == "p225_004.wav"]["keyword"].values[0]
print(keyword)

# 4. Truy xuất tất cả các đoạn âm thanh mà âm thanh nói ra quá bé (gần như mất tiếng > lọc)
low_volume_samples = df[df["rms_db"] < -40.0]
print("Các file âm thanh bé:", low_volume_samples["file_name"].tolist())
```

---

*Tài liệu được soạn thảo phục vụ chuẩn bị CSDL Audio Vectors Nhận Diện Giọng Nói.*
