# 🌿 Klasifikasi Varietas Daun Menggunakan Machine Learning

> Tugas UTS Mata Kuliah **Computer Vision** — Semester 6 Teknik Informatika  
> Berbasis fitur morfologi, warna (RGB/HSV/LAB), dan tekstur GLCM

---

## 📋 Daftar Isi

- [Deskripsi Proyek](#-deskripsi-proyek)
- [Dataset](#-dataset)
- [Referensi Jurnal](#-referensi-jurnal)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi & Setup](#-instalasi--setup)
- [Pipeline](#-pipeline)
- [Feature Extraction](#-feature-extraction)
- [Model yang Digunakan](#-model-yang-digunakan)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Output yang Dihasilkan](#-output-yang-dihasilkan)
- [Daftar Pustaka](#-daftar-pustaka)

---

## 📌 Deskripsi Proyek

Proyek ini mengimplementasikan sistem klasifikasi varietas daun menggunakan teknik **Machine Learning** berbasis **Computer Vision**. Pendekatan yang digunakan terinspirasi dari 4 jurnal ilmiah tentang klasifikasi varietas beras, yang diadaptasi ke domain klasifikasi daun.

**Pendekatan utama:**
- Ekstraksi fitur morfologi & shape (terinspirasi Jurnal 1 & 4)
- Ekstraksi fitur warna dari multiple color spaces (terinspirasi Jurnal 1)
- Klasifikasi dengan ANN/MLP (terinspirasi Jurnal 2)
- Stacking Ensemble Learning (terinspirasi Jurnal 3)

---

## 📦 Dataset

| Properti | Detail |
|----------|--------|
| **Sumber** | [Kaggle — ichhadhari/leaf-images](https://www.kaggle.com/datasets/ichhadhari/leaf-images) |
| **Jumlah Kelas** | 10 varietas daun |
| **Total Gambar** | 3.003 gambar |
| **Gambar per Kelas** | ~300 gambar |
| **Tipe Dataset** | Balanced (seimbang) |
| **Split** | Training 80% : Testing 20% |

**Kelas yang tersedia:**

| No | Nama Kelas |
|----|------------|
| 1 | Apta |
| 2 | Indian Rubber Tree |
| 3 | Karanj |
| 4 | Kashid |
| 5 | Nilgiri |
| 6 | Pimpal |
| 7 | Sita Ashok |
| 8 | Sonmohar |
| 9 | Vad |
| 10 | Vilayati Chinch |

---

## 📚 Referensi Jurnal

| # | Jurnal | Kontribusi |
|---|--------|------------|
| **J1** | Cinar & Koklu (2021) — *Selcuk J. Agr. Food Sci.* | Ekstraksi 106 fitur: morfologi, shape, warna (RGB, HSV, LAB, YCbCr, XYZ) |
| **J2** | Koklu et al. (2021) — *Comput. Electron. Agric.* | Klasifikasi dengan ANN, DNN, CNN → akurasi 99.87–100% |
| **J3** | Islam et al. (2025) — *Journal of Cereal Science* | Stacking Ensemble + XGBoost meta-learner → akurasi 100% |
| **J4** | Cinar & Koklu (2019) — *IJISAE* | Pipeline LR, MLP, SVM, DT, RF, NB, k-NN → terbaik LR 93.02% |

---

## 🗂 Struktur Proyek

```
leaf-classification/
│
├── leaf_dataset/
│   └── 300_dataset/
│       ├── Apta/
│       ├── Indian Rubber Tree/
│       ├── Karanj/
│       ├── Kashid/
│       ├── Nilgiri/
│       ├── Pimpal/
│       ├── Sita Ashok/
│       ├── Sonmohar/
│       ├── Vad/
│       └── Vilayati Chinch/
│
├── leaf_features.csv              # Hasil ekstraksi 58 fitur
│
├── output/
│   ├── sample_before_preprocessing.png
│   ├── preprocessing_steps.png
│   ├── hasil_klasifikasi.png
│   ├── confusion_matrix_map.png
│   ├── metrics_per_class.png
│   ├── learning_curve_accuracy.png
│   ├── learning_curve_loss.png
│   └── evaluasi_lengkap.png
│
└── README.md
```

---

## ⚙️ Instalasi & Setup

### 1. Buka Google Colab dan setup Kaggle API

```python
# Install kaggle
!pip install kaggle

# Upload kaggle.json
from google.colab import files
files.upload()

# Setup credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download dan ekstrak dataset

```python
!kaggle datasets download -d ichhadhari/leaf-images
!unzip -q leaf-images.zip -d leaf_dataset
```

### 3. Install dependencies

```python
!pip install opencv-python scikit-image scikit-learn xgboost tqdm pandas numpy matplotlib seaborn
```

---

## 🔄 Pipeline

```
Input Gambar (3.003 gambar, 10 kelas)
        │
        ▼
┌─────────────────────────────────┐
│         PREPROCESSING           │
│  • Resize → 256×256 piksel      │
│  • Grayscale (untuk morfologi)  │
│  • Binary Otsu (untuk region)   │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│       FEATURE EXTRACTION        │
│  • 16 Morfologi & Shape         │
│  • 36 Warna (RGB, HSV, LAB)     │
│  • 6 Tekstur GLCM               │
│  • Total: 58 fitur              │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│       PERSIAPAN DATA            │
│  • Label Encoding               │
│  • StandardScaler               │
│  • Train/Test Split 80:20       │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│         KLASIFIKASI             │
│  7 Model ML + Stacking Ensemble │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│          EVALUASI               │
│  • Confusion Matrix             │
│  • Accuracy, Precision,         │
│    Recall, F1-Score             │
│  • Learning Curve               │
│  • Loss Curve                   │
│  • Overfitting Analysis         │
└─────────────────────────────────┘
```

---

## 🔬 Feature Extraction

### Morfologi & Shape — 16 Fitur *(Jurnal 1 & 4)*

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | Area (A) | Jumlah piksel dalam batas daun |
| 2 | Perimeter (P) | Panjang batas keliling daun |
| 3 | Major Axis Length (L) | Panjang sumbu utama |
| 4 | Minor Axis Length (l) | Panjang sumbu minor |
| 5 | Eccentricity (Ec) | Eksentrisitas elips yang sepadan |
| 6 | Equivalent Diameter (ED) | Diameter lingkaran dengan area setara |
| 7 | Solidity (S) | Rasio piksel konveks terhadap area |
| 8 | Convex Area (CA) | Jumlah piksel polygon konveks terkecil |
| 9 | Extent (Ex) | Rasio piksel terhadap bounding box |
| 10 | Aspect Ratio (AR) | L / l |
| 11 | Roundness (Ro) | (4 × A × π) / P² |
| 12 | Compactness (Co) | ED / L |
| 13 | Shape Factor 1 (SF1) | L / A |
| 14 | Shape Factor 2 (SF2) | l / A |
| 15 | Shape Factor 3 (SF3) | A / ((L/2)² × π) |
| 16 | Shape Factor 4 (SF4) | A / ((L/2) × (l/2) × π) |

### Color Features — 36 Fitur *(Jurnal 1)*

Diekstrak dari **3 color spaces**: RGB, HSV, L\*a\*b\*

Untuk setiap channel (3 channel × 3 color space = 9 channel), dihitung:

| Statistik | Deskripsi |
|-----------|-----------|
| Mean | Nilai rata-rata intensitas |
| Std Dev | Standar deviasi piksel |
| Skewness | Kemiringan distribusi |
| Kurtosis | Keruncingan distribusi |

> 9 channel × 4 statistik = **36 fitur warna**

### Texture Features (GLCM) — 6 Fitur

Menggunakan **Gray Level Co-occurrence Matrix** dengan jarak=1, sudut=0°/45°/90°/135°:

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | Contrast | Perbedaan intensitas antar piksel |
| 2 | Dissimilarity | Ketidakmiripan antar piksel |
| 3 | Homogeneity | Keseragaman intensitas |
| 4 | Energy | Keseragaman distribusi (ASM root) |
| 5 | Correlation | Korelasi linier antar piksel |
| 6 | ASM | Angular Second Moment |

---

## 🤖 Model yang Digunakan

### Model Individual

| Model | Parameter | Referensi | Akurasi |
|-------|-----------|-----------|---------|
| Logistic Regression | max_iter=1000 | Jurnal 4 | 75.50% |
| SVM | kernel=RBF, probability=True | Jurnal 4 | 81.67% |
| K-Nearest Neighbor | n_neighbors=5 | Jurnal 4 | 77.17% |
| Decision Tree | default | Jurnal 4 | 68.00% |
| Random Forest | n_estimators=100 | Jurnal 4 | 85.33% |
| **MLP (ANN)** | **hidden=(100,50), max_iter=500** | **Jurnal 2** | **88.50% ⭐** |
| XGBoost | n_estimators=100 | Jurnal 3 | 85.00% |

### Stacking Ensemble *(Jurnal 3)*

```
Base Learners:
  ├── SVM (kernel=RBF)
  ├── Random Forest (100 trees)
  ├── MLP (100, 50)
  └── XGBoost (100 estimators)
         │
         ▼ (5-Fold CV)
  Meta-Learner: XGBoost
         │
         ▼
  Final Prediction
```

---

## 📊 Hasil Evaluasi

### Perbandingan Akurasi Semua Model

```
Decision Tree         ████████████████████░░░░░░  68.00%
Logistic Regression   ████████████████████████░░  75.50%
KNN                   ████████████████████████░░  77.17%
SVM                   ██████████████████████████  81.67%
XGBoost               ███████████████████████████  85.00%
Random Forest         ███████████████████████████  85.33%
MLP (ANN)             ████████████████████████████ 88.50% ⭐
Stacking Ensemble     ████████████████████████████ ~89.00% ⭐
```

### Metrics Model Terbaik — MLP (ANN)

| Metric | Score |
|--------|-------|
| **Accuracy** | 88.50% |
| **Precision** | 88.51% |
| **Recall** | 88.50% |
| **F1-Score** | 88.51% |

### Analisa Overfitting

| Indikator | Nilai | Status |
|-----------|-------|--------|
| Training Accuracy | 100.00% | ⚠️ |
| Validation Accuracy | 84.71% | — |
| **Gap (Train - Val)** | **15.29%** | **⚠️ Overfitting** |
| Training Loss | 0.0014 | ⚠️ |
| Validation Loss | 0.5952 | — |
| Loss Gap | 0.5938 | ⚠️ |

> **Catatan:** Overfitting terjadi karena dataset relatif kecil (~2.400 data training untuk 10 kelas) dengan arsitektur MLP yang kompleks. Solusi: regularisasi L2, data augmentation, atau Dropout (tersedia di Deep Learning semester depan).

---

## ▶️ Cara Menjalankan

Jalankan cell secara berurutan di Google Colab:

```
Cell 1  → Import semua library
Cell 2  → Analisa preprocessing (resize & grayscale)
Cell 3  → Visualisasi sample gambar per kelas
Cell 4  → Definisi fungsi preprocessing
Cell 5  → Definisi fungsi feature extraction
Cell 6  → Ekstraksi fitur semua 3.003 gambar → leaf_features.csv
Cell 7  → Persiapan data (encoding, scaling, split)
Cell 8  → Training & evaluasi 7 model ML
Cell 9  → Stacking Ensemble (Jurnal 3)
Cell 10 → Visualisasi perbandingan akurasi + confusion matrix
Cell 11 → Classification report lengkap
Cell 12 → Evaluasi metrics (accuracy, precision, recall, F1)
Cell 13 → Confusion matrix heatmap (count & normalized)
Cell 14 → Metrics detail per kelas + visualisasi
Cell 15 → Grafik akurasi training vs validasi (learning curve)
Cell 16 → Grafik loss training vs validasi
Cell 17 → Analisa overfitting/underfitting
Cell 18 → Ringkasan semua evaluasi dalam 1 figure
```

> ⏱️ **Estimasi waktu:** Cell 6 (ekstraksi fitur) ±10–15 menit | Cell 9 (Stacking) ±5–10 menit | Cell 15–16 (learning curve) ±5–10 menit

---

## 📁 Output yang Dihasilkan

| File | Deskripsi |
|------|-----------|
| `leaf_features.csv` | Dataset 58 fitur hasil ekstraksi |
| `sample_before_preprocessing.png` | Visualisasi 10 sample gambar per kelas |
| `preprocessing_steps.png` | Tahapan preprocessing (original → resize → grayscale → binary) |
| `hasil_klasifikasi.png` | Perbandingan akurasi + confusion matrix model terbaik |
| `confusion_matrix_map.png` | Confusion matrix (count & normalized %) |
| `metrics_per_class.png` | Precision, Recall, F1 per kelas |
| `learning_curve_accuracy.png` | Grafik akurasi training vs validasi + overfitting gap |
| `learning_curve_loss.png` | Grafik loss training vs validasi + loss gap |
| `evaluasi_lengkap.png` | Ringkasan 6 grafik evaluasi dalam 1 figure |

---

## 📖 Daftar Pustaka

1. Cinar, I., & Koklu, M. (2021). Determination of Effective and Specific Physical Features of Rice Varieties by Computer Vision In Exterior Quality Inspection. *Selcuk Journal of Agriculture and Food Sciences*, 35(3), 229–243. https://doi.org/10.15316/SJAFS.2021.252

2. Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. *Computers and Electronics in Agriculture*, 187, 106285. https://doi.org/10.1016/j.compag.2021.106285

3. Islam, Md. M., Himel, G. M. S., Moazzam, G., & Uddin, M. S. (2025). Precision in Rice Variety Classification using Stacking-based Ensemble Learning. *Journal of Cereal Science*, 122, 104128. https://doi.org/10.1016/j.jcs.2025.104128

4. Cinar, I., & Koklu, M. (2019). Classification of Rice Varieties Using Artificial Intelligence Methods. *International Journal of Intelligent Systems and Applications in Engineering*, 7(3), 188–194. https://doi.org/10.18201/ijisae.2019355381

5. Ichha Dhari. (2024). *Leaf Images Dataset*. Kaggle. https://www.kaggle.com/datasets/ichhadhari/leaf-images

---

<div align="center">

**Mata Kuliah Computer Vision | Semester 6 | Teknik Informatika**

*Terinspirasi dari domain klasifikasi beras → diadaptasi ke klasifikasi daun*

</div>
