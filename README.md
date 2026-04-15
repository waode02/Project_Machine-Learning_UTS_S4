# ❤️ Klasifikasi Penyakit Jantung — Streamlit App

Aplikasi prediksi penyakit jantung berbasis Machine Learning (SVM, Random Forest, XGBoost)
dengan penanganan imbalanced data menggunakan SMOTE.

---

## 📁 Struktur File

```
├── app.py                          ← File utama Streamlit
├── requirements.txt                ← Daftar library
├── heart_disease_prediction.csv    ← Dataset (letakkan di sini)
└── README.md
```

---

## 🚀 Cara Menjalankan Lokal

### 1. Install library
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

### 3. Buka browser
Otomatis terbuka di `http://localhost:8501`

---

## ☁️ Deploy ke Streamlit Cloud (Gratis)

1. **Push ke GitHub**
   ```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "Heart disease classifier app"
   git remote add origin https://github.com/USERNAME/REPO.git
   git push -u origin main
   ```

2. **Buka** https://share.streamlit.io

3. **Klik** "New app" → pilih repo → pilih `app.py` → Deploy

4. **Upload dataset** melalui sidebar aplikasi saat pertama kali digunakan,
   atau letakkan `heart_disease_prediction.csv` di root repository.

---

## 📊 Fitur Aplikasi

| Halaman | Fitur |
|---------|-------|
| 🏠 Beranda | Info proyek, pipeline metodologi, hasil terbaik |
| 📊 Eksplorasi Data | Info dataset, distribusi fitur, distribusi target, heatmap korelasi |
| 🤖 Performa Model | Perbandingan metrik, confusion matrix, classification report, feature importance |
| 🔮 Prediksi Pasien | Input data pasien → prediksi dari 3 model + voting mayoritas |

---

## 🏆 Hasil Model (GridSearchCV, Split 80:20)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | 95.14% | 95.14% |
| XGBoost | 94.50% | 94.50% |
| SVM | 93.95% | 93.95% |

**Model terbaik:** Random Forest pada split 90:10 dengan akurasi **95.54%**

---

## 📦 Library yang Digunakan

- `streamlit` — Framework web app
- `scikit-learn` — ML models & preprocessing
- `xgboost` — XGBoost classifier
- `imbalanced-learn` — SMOTE
- `pandas`, `numpy` — Data processing
- `matplotlib`, `seaborn` — Visualisasi
