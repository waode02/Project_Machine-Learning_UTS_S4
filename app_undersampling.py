import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #C62828, #1565C0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #555; font-size: 1rem; margin-top: 0; }
    .result-card {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        font-size: 1.4rem; font-weight: 700; margin: 1rem 0;
    }
    .berisiko  { background: #FFEBEE; color: #C62828; border: 2px solid #C62828; }
    .aman      { background: #E8F5E9; color: #2E7D32; border: 2px solid #2E7D32; }
    .metric-box {
        background: #F5F5F5; border-radius: 10px; padding: 1rem;
        text-align: center; border: 1px solid #E0E0E0;
    }
    .metric-box h3 { margin: 0; font-size: 1.8rem; }
    .metric-box p  { margin: 0; color: #777; font-size: 0.85rem; }
    .info-box {
        background: #E3F2FD; border-left: 4px solid #1565C0;
        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 1rem;
    }
    .warn-box {
        background: #FFF8E1; border-left: 4px solid #F9A825;
        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 1rem;
    }
    hr { border: 0; border-top: 1px solid #E0E0E0; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    model_path = "best_model_rf_7030.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


bundle = load_bundle()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
    st.markdown("## ❤️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi prediksi risiko **penyakit jantung** menggunakan 3 model Machine Learning:
    - 🔵 **SVM** (Support Vector Machine)
    - 🟢 **Random Forest**
    - 🟠 **XGBoost**

    **Dataset:** Heart Disease Prediction (Kaggle)  
    **Jumlah Data:** 5.568 baris × 19 kolom  
    **Teknik:** Random Undersampling untuk menangani data tidak seimbang
    """)
    st.markdown("---")
    st.markdown("### 📊 Performa Model (Split 80:20)")
    st.markdown("""
    | Model | Akurasi |
    |---|---|
    | SVM | ~82% |
    | Random Forest | ~80% |
    | XGBoost | ~83% |
    """)
    st.markdown("---")
    st.caption("Mata Kuliah: Pembelajaran Mesin | Kelas A Informatika")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🫀 Klasifikasi Risiko Penyakit Jantung</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Prediksi risiko penyakit jantung menggunakan SVM, Random Forest, dan XGBoost dengan Voting Mayoritas</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Model Status ─────────────────────────────────────────────────────────────
if bundle is None:
    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>File model tidak ditemukan!</strong><br>
    Pastikan file <code>best_model_rf_7030.pkl</code> berada di direktori yang sama dengan <code>app.py</code>.<br>
    Jalankan notebook terlebih dahulu untuk menghasilkan file model tersebut.
    </div>
    """, unsafe_allow_html=True)

    st.info("**Cara mendapatkan file model:** Jalankan semua sel di notebook, lalu pastikan cell pada Section 10 (Simpan Model) sudah dijalankan sehingga file `best_model_rf_7030.pkl` terbentuk.")
    st.stop()
else:
    st.markdown(f"""
    <div class="info-box">
    ✅ Model berhasil dimuat: <strong>{bundle.get('model_name', 'Random Forest')}</strong>
    &nbsp;|&nbsp; Split: <strong>{bundle.get('split_label', '70:30')}</strong>
    &nbsp;|&nbsp; Accuracy: <strong>{bundle.get('accuracy', '-')}%</strong>
    &nbsp;|&nbsp; F1-Score: <strong>{bundle.get('f1_score', '-')}%</strong>
    </div>
    """, unsafe_allow_html=True)


# ── Feature Names ─────────────────────────────────────────────────────────────
FEAT_NAMES = bundle['feat_names']
IQR_BOUNDS = bundle['iqr_bounds']
_model     = bundle['model']
_scaler    = bundle['scaler']


# ── Prediksi Single ────────────────────────────────────────────────────────────
def predict_single(data_pasien: dict):
    df_inp = pd.DataFrame([data_pasien], columns=FEAT_NAMES)
    df_inp_c = df_inp.copy()
    for col, bounds in IQR_BOUNDS.items():
        if col in df_inp_c.columns:
            df_inp_c[col] = df_inp_c[col].clip(bounds['lower'], bounds['upper'])
    inp_scaled = _scaler.transform(df_inp_c)
    pred = _model.predict(inp_scaled)[0]
    prob = _model.predict_proba(inp_scaled)[0]
    return {
        'prediksi'    : int(pred),
        'label'       : '🔴 BERISIKO SAKIT JANTUNG' if pred == 1 else '🟢 TIDAK BERISIKO',
        'prob_positif': round(prob[1] * 100, 2),
        'prob_negatif': round(prob[0] * 100, 2),
    }


# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Pasien", "📋 Prediksi Batch (CSV)", "ℹ️ Panduan Fitur"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI SATU PASIEN
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Input Data Pasien")
    st.markdown("Isi semua kolom di bawah sesuai kondisi pasien, lalu klik tombol **Prediksi**.")

    with st.form("form_pasien"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Data Demografis")
            age   = st.number_input("Usia (tahun)", min_value=1.0, max_value=120.0, value=45.0, step=1.0)
            sex   = st.selectbox("Jenis Kelamin", options=[(1.0, "Laki-laki"), (2.0, "Perempuan")],
                                 format_func=lambda x: x[1])
            race  = st.selectbox("Ras/Etnis (kode)", options=[(1.0,"1"), (2.0,"2"), (3.0,"3"),
                                                               (4.0,"4"), (5.0,"5"), (6.0,"6")],
                                 format_func=lambda x: x[1])
            edu   = st.selectbox("Pendidikan", options=[(1.0,"1 - Rendah"),(2.0,"2"),(3.0,"3 - Menengah"),
                                                        (4.0,"4"),(5.0,"5 - Tinggi")],
                                 format_func=lambda x: x[1])
            pir   = st.number_input("Rasio Pendapatan (Poverty Income Ratio)", min_value=0.0, max_value=10.0,
                                    value=2.5, step=0.1)

        with col2:
            st.markdown("##### 💊 Riwayat Medis & Obat")
            bp_med   = st.radio("Minum obat tekanan darah?",   [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            chol_med = st.radio("Minum obat kolesterol?",      [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            insulin  = st.radio("Pakai insulin?",              [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            diab_pil = st.radio("Minum obat diabetes?",        [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            prediab  = st.radio("Pernah didiagnosis prediabetes?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            diabetes = st.radio("Terdiagnosis diabetes?",      [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)

        with col3:
            st.markdown("##### ❤️ Faktor Risiko Jantung")
            stroke   = st.radio("Pernah stroke?",              [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            fam_hist = st.radio("Riwayat keluarga serangan jantung?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            smoking_e= st.radio("Pernah merokok?",             [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            smoking_c= st.radio("Merokok saat ini?",           [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            active   = st.radio("Aktif secara fisik?",         [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            hyperten = st.radio("Terdiagnosis hipertensi?",    [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
            high_ch  = st.radio("Kolesterol tinggi?",          [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)

        submitted = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True, type="primary")

    if submitted:
        data_pasien = {
            'taking_bp_meds'             : bp_med,
            'taking_cholesterol_meds'    : chol_med,
            'age'                        : age,
            'sex'                        : sex[0],
            'race_ethnicity'             : race[0],
            'education'                  : edu[0],
            'poverty_income_ratio'       : pir,
            'taking_insulin'             : insulin,
            'taking_diabetes_pills'      : diab_pil,
            'told_prediabetes'           : prediab,
            'told_stroke'                : stroke,
            'family_history_heart_attack': fam_hist,
            'diabetes'                   : diabetes,
            'smoking_ever'               : smoking_e,
            'smoking_current'            : smoking_c,
            'physically_active'          : active,
            'hypertension'               : hyperten,
            'high_cholesterol'           : high_ch,
        }

        hasil = predict_single(data_pasien)

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        # Kartu Hasil Utama
        css_class = "berisiko" if hasil['prediksi'] == 1 else "aman"
        st.markdown(f'<div class="result-card {css_class}">{hasil["label"]}</div>', unsafe_allow_html=True)

        # Metrik Probabilitas
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <p>Probabilitas Sakit Jantung</p>
                <h3 style="color:#C62828">{hasil['prob_positif']}%</h3>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <p>Probabilitas Tidak Sakit</p>
                <h3 style="color:#2E7D32">{hasil['prob_negatif']}%</h3>
            </div>""", unsafe_allow_html=True)
        with c3:
            verdict = "BERISIKO" if hasil['prediksi'] == 1 else "AMAN"
            color   = "#C62828" if hasil['prediksi'] == 1 else "#2E7D32"
            st.markdown(f"""
            <div class="metric-box">
                <p>Keputusan Model</p>
                <h3 style="color:{color}">{verdict}</h3>
            </div>""", unsafe_allow_html=True)

        # Progress bar probabilitas
        st.markdown("#### Distribusi Probabilitas")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("🔴 **Risiko Sakit Jantung**")
            st.progress(hasil['prob_positif'] / 100)
            st.caption(f"{hasil['prob_positif']}%")
        with col_b:
            st.markdown("🟢 **Tidak Berisiko**")
            st.progress(hasil['prob_negatif'] / 100)
            st.caption(f"{hasil['prob_negatif']}%")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PANDUAN FITUR
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ Panduan Fitur Input")
    st.markdown("Berikut penjelasan lengkap untuk setiap fitur yang digunakan dalam prediksi:")

    fitur_info = {
        "taking_bp_meds"             : ("0 / 1", "Apakah pasien sedang mengonsumsi obat tekanan darah? (0=Tidak, 1=Ya)"),
        "taking_cholesterol_meds"    : ("0 / 1", "Apakah pasien sedang mengonsumsi obat kolesterol? (0=Tidak, 1=Ya)"),
        "age"                        : ("float", "Usia pasien dalam tahun (contoh: 45.0)"),
        "sex"                        : ("1.0 / 2.0", "Jenis kelamin pasien (1.0=Laki-laki, 2.0=Perempuan)"),
        "race_ethnicity"             : ("1.0 – 6.0", "Kode ras/etnis pasien (kategori numerik 1–6)"),
        "education"                  : ("1.0 – 5.0", "Tingkat pendidikan pasien (1=paling rendah, 5=paling tinggi)"),
        "poverty_income_ratio"       : ("float", "Rasio pendapatan terhadap garis kemiskinan (contoh: 2.5)"),
        "taking_insulin"             : ("0 / 1", "Apakah pasien sedang menggunakan insulin? (0=Tidak, 1=Ya)"),
        "taking_diabetes_pills"      : ("0 / 1", "Apakah pasien mengonsumsi obat diabetes? (0=Tidak, 1=Ya)"),
        "told_prediabetes"           : ("0 / 1", "Apakah pasien pernah didiagnosis prediabetes? (0=Tidak, 1=Ya)"),
        "told_stroke"                : ("0 / 1", "Apakah pasien pernah didiagnosis stroke? (0=Tidak, 1=Ya)"),
        "family_history_heart_attack": ("0 / 1", "Apakah ada riwayat serangan jantung dalam keluarga? (0=Tidak, 1=Ya)"),
        "diabetes"                   : ("0 / 1", "Apakah pasien terdiagnosis diabetes? (0=Tidak, 1=Ya)"),
        "smoking_ever"               : ("0 / 1", "Apakah pasien pernah merokok? (0=Tidak, 1=Ya)"),
        "smoking_current"            : ("0 / 1", "Apakah pasien saat ini merokok? (0=Tidak, 1=Ya)"),
        "physically_active"          : ("0 / 1", "Apakah pasien aktif secara fisik? (0=Tidak, 1=Ya)"),
        "hypertension"               : ("0 / 1", "Apakah pasien terdiagnosis hipertensi? (0=Tidak, 1=Ya)"),
        "high_cholesterol"           : ("0 / 1", "Apakah pasien memiliki kolesterol tinggi? (0=Tidak, 1=Ya)"),
    }

    df_panduan = pd.DataFrame([
        {"Fitur": k, "Nilai": v[0], "Keterangan": v[1]}
        for k, v in fitur_info.items()
    ])
    st.dataframe(df_panduan, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔬 Tentang Preprocessing")
    st.markdown("""
    Sebelum prediksi dilakukan, data pasien akan melalui preprocessing otomatis:
    1. **IQR Capping** pada fitur `age` dan `poverty_income_ratio` untuk menangani outlier
    2. **StandardScaler** — normalisasi menggunakan scaler yang sudah di-fit pada data training (split 70:30)
    """)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("❤️ Aplikasi Klasifikasi Penyakit Jantung | Pembelajaran Mesin — Kelas A Informatika | Dataset: Kaggle Heart Disease Prediction")
