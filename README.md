# EstateX — Prediksi & Analisis Harga Properti Residensial

EstateX adalah aplikasi dashboard interaktif berbasis web yang dirancang untuk melakukan Analisis Eksploratif Data (EDA) dan prediksi harga properti residensial di Indonesia menggunakan metode **Regresi Linier Berganda (Multiple Linear Regression)**. Proyek ini dikembangkan sebagai bagian dari Tugas Akhir/Final Project mata kuliah **Big Data**.

Aplikasi ini dibangun menggunakan **Streamlit** dengan menerapkan desain visual modern berbasis **Glassmorphic Dark UI** untuk memberikan pengalaman pengguna yang estetis dan interaktif.

---

## 🚀 Fitur Utama

### 1. Ringkasan & Analisis Eksploratif Data (EDA)
Dashboard interaktif yang menampilkan karakteristik fisik dan geografis properti residensial melalui 4 tab visualisasi:
*   **Distribusi Harga:** Visualisasi sebaran harga properti menggunakan histogram interaktif dan analisis kuartil (*box plot*).
*   **Korelasi:** Matriks Korelasi Pearson yang menunjukkan hubungan antara variabel numerik (harga, luas tanah, luas bangunan, jumlah kamar, kapasitas listrik, dan jumlah lantai).
*   **Hubungan Fisik:** *Scatter plot* interaktif yang memetakan hubungan antara Luas Bangunan vs Harga dan Luas Tanah vs Harga dengan dimensi tambahan (jumlah kamar tidur/mandi).
*   **Analisis Geografis:** Grafik perbandingan rata-rata harga dan jumlah listing properti antar provinsi di Indonesia.

### 2. Kalkulator Prediksi Harga Properti
Estimasi nilai pasar properti secara *real-time* berdasarkan spesifikasi masukan pengguna:
*   **Input Lokasi:** Provinsi dan Kota/Kabupaten yang terintegrasi secara dinamis.
*   **Input Fisik:** Luas tanah, luas bangunan, jumlah kamar tidur, jumlah kamar mandi, dan daya listrik.
*   **Statistik Pasar Lokal:** Menampilkan jumlah unit terdaftar, harga tertinggi, terendah, dan rata-rata di Kota/Kabupaten yang dipilih.
*   **Histogram Distribusi Lokal:** Grafik sebaran harga lokal dilengkapi dengan garis penanda (*vline*) estimasi harga prediksi untuk membandingkan posisi harga rumah dengan pasar sekitarnya.

---

## 🛠️ Teknologi yang Digunakan

*   **Bahasa Pemrograman:** Python 3.x
*   **Antarmuka Web (UI/UX):** Streamlit (dikustomisasi dengan Glassmorphism Custom CSS)
*   **Pemrosesan Data:** Pandas, NumPy
*   **Visualisasi Data:** Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
*   **Machine Learning:** Scikit-learn (StandardScaler, LinearRegression, Train-Test Split)
*   **Statistik:** Statsmodels

---

## 📊 Pipeline Pembersihan Data (`clean_data.py`)

Sebelum data digunakan oleh model prediktif, dataset mentah (`dataset_rumah99_raw.csv`) diproses terlebih dahulu untuk menjamin kualitas data. Langkah pembersihan meliputi:
1.  **Penanganan Nilai Kosong (*Missing Values*):** Menghapus baris yang memiliki nilai NaN pada kolom kritis (`Harga_Rupiah`, `Kamar_Tidur`, `Kamar_Mandi`, `Luas_Tanah_m2`, `Luas_Bangunan_m2`, `Jumlah_Lantai`, `Kota`).
2.  **Filter Nilai Tidak Valid:** Memastikan jumlah kamar tidur, kamar mandi, dan daya listrik lebih besar dari 0.
3.  **Penanganan Outlier:** membatasi data pada rentang logis:
    *   **Harga Properti:** Rp 100 Juta s.d. Rp 50 Miliar.
    *   **Luas Tanah & Bangunan:** Maksimum 2.000 m².
    *   **Jumlah Kamar Tidur & Mandi:** Maksimum 15 kamar.

---

## 📂 Struktur Proyek

```text
├── .devcontainer/          # Konfigurasi container development (opsional)
├── datasets/
│   ├── dataset_rumah99_raw.csv    # Dataset mentah sebelum dibersihkan
│   └── dataset_rumah99_clean.csv  # Dataset hasil pembersihan (digunakan aplikasi)
├── clean_data.py           # Skrip Python untuk pembersihan data
├── dashboard.py           # Kode utama aplikasi dashboard Streamlit & Model MLR
├── requirements.txt       # Daftar pustaka/dependencies proyek
└── README.md              # Dokumentasi proyek (file ini)
```

---

## ⚡ Cara Menjalankan Proyek

### 1. Kloning Proyek & Masuk ke Direktori
```bash
cd "03_KULIAH/01. BIG DATA/FP"
```

### 2. Buat dan Aktifkan Virtual Environment (Direkomendasikan)
*   **macOS/Linux:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
*   **Windows:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

### 3. Instalasi Dependencies
Instal pustaka-pustaka yang diperlukan menggunakan file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Pembersihan Data (Opsional)
Jika file `dataset_rumah99_clean.csv` belum ada di folder `datasets/`, jalankan skrip berikut:
```bash
python clean_data.py
```

### 5. Jalankan Dashboard Streamlit
Jalankan aplikasi utama dengan perintah:
```bash
streamlit run dashboard.py
```
Aplikasi akan secara otomatis terbuka di peramban (browser) Anda pada alamat `http://localhost:8501`.

---

## 📝 Catatan Model Prediksi
Model prediksi menggunakan algoritma **Multiple Linear Regression (MLR)** dengan fitur numerik yang diskalakan menggunakan `StandardScaler` dan fitur kategori (`Provinsi`, `Kota`) yang diubah menggunakan teknik *One-Hot Encoding*. Akurasi model (skor $R^2$) ditampilkan secara dinamis pada sidebar dashboard setelah model selesai dilatih secara instan saat aplikasi dijalankan.
