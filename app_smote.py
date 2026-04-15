import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(21,101,192,0.3);
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #1565C0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-card h3 { color: #1565C0; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #555; margin: 0; font-size: 0.9rem; }
    .result-positive {
        background: linear-gradient(135deg, #C62828, #E53935);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.3rem; font-weight: bold;
        box-shadow: 0 4px 15px rgba(198,40,40,0.4);
    }
    .result-negative {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.3rem; font-weight: bold;
        box-shadow: 0 4px 15px rgba(27,94,32,0.4);
    }
    .info-box {
        background: #E3F2FD; border-left: 4px solid #1565C0;
        padding: 1rem; border-radius: 6px; margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        color: white; border: none; padding: 0.7rem 2rem;
        border-radius: 8px; font-size: 1rem; font-weight: bold;
        width: 100%; transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); opacity: 0.9; }
    .sidebar-header {
        background: #1565C0; color: white; padding: 0.8rem;
        border-radius: 8px; text-align: center; margin-bottom: 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'taking_bp_meds', 'taking_cholesterol_meds', 'age', 'sex',
    'race_ethnicity', 'education', 'poverty_income_ratio', 'taking_insulin',
    'taking_diabetes_pills', 'told_prediabetes', 'told_stroke',
    'family_history_heart_attack', 'diabetes', 'smoking_ever',
    'smoking_current', 'physically_active', 'hypertension', 'high_cholesterol'
]

BEST_SVM_PARAMS = {'C': 50, 'gamma': 'auto', 'kernel': 'rbf'}
BEST_RF_PARAMS  = {
    'max_depth': None, 'max_features': 'sqrt',
    'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300
}
BEST_XGB_PARAMS = {
    'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1,
    'max_depth': 7, 'n_estimators': 100, 'subsample': 0.8
}

MODEL_RESULTS = {
    'SVM':           {'70:30': 94.02, '80:20': 93.95, '90:10': 94.95, 'avg': 94.30},
    'Random Forest': {'70:30': 94.98, '80:20': 95.14, '90:10': 95.54, 'avg': 95.22},
    'XGBoost':       {'70:30': 94.51, '80:20': 94.50, '90:10': 94.35, 'avg': 94.45},
}

# ─── TRAIN & CACHE MODEL ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(df):
    """
    Pipeline: IQR Capping → Split → Normalize (fit train) → SMOTE (train only) → Train
    Model terbaik: Random Forest, split 80:20
    """
    df_clean = df.copy()
    for col in ['age', 'poverty_income_ratio']:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        df_clean[col] = df_clean[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))

    X = df_clean.drop('heart_disease', axis=1)
    y = df_clean['heart_disease']

    # Split 80:20
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Normalize — fit only on train
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_te_scaled = scaler.transform(X_te_raw)

    # SMOTE — only on train
    smote = SMOTE(random_state=42)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)

    # Train all 3 models
    models = {
        'SVM':           SVC(**BEST_SVM_PARAMS, random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(**BEST_RF_PARAMS, random_state=42),
        'XGBoost':       XGBClassifier(**BEST_XGB_PARAMS, random_state=42,
                                       eval_metric='logloss', use_label_encoder=False),
    }
    trained = {}
    metrics = {}
    for name, model in models.items():
        model.fit(X_tr_res, y_tr_res)
        y_pred = model.predict(X_te_scaled)
        trained[name] = model
        metrics[name] = {
            'accuracy' : accuracy_score(y_te, y_pred),
            'precision': precision_score(y_te, y_pred, average='weighted', zero_division=0),
            'recall'   : recall_score(y_te, y_pred, average='weighted', zero_division=0),
            'f1'       : f1_score(y_te, y_pred, average='weighted', zero_division=0),
            'cm'       : confusion_matrix(y_te, y_pred),
            'y_pred'   : y_pred,
            'y_test'   : y_te,
        }

    return trained, scaler, metrics, df_clean, X_te_scaled, y_te

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">❤️ Heart Disease Classifier</div>',
                unsafe_allow_html=True)
    st.markdown("**Navigasi**")
    page = st.radio("", [
        "🏠 Beranda",
        "📊 Eksplorasi Data",
        "🤖 Performa Model",
        "🔮 Prediksi Pasien",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Upload Dataset**")
    uploaded_file = st.file_uploader("Upload file CSV", type=['csv'],
                                     help="Upload heart_disease_prediction.csv")

    st.markdown("---")
    st.markdown("""
    **Info Proyek**
    - 📚 Mata Kuliah: Pembelajaran Mesin
    - 🎓 Universitas Haluoleo
    - 📅 2026
    - 👤 Wa Ode Yurismawati
    - 🔖 NIM: E1E124080
    """)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Try loading from local path
    local_paths = ['heart_disease_prediction.csv', 'data/heart_disease_prediction.csv']
    for path in local_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break

# ─── PAGE: BERANDA ────────────────────────────────────────────────────────────
if page == "🏠 Beranda":
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Klasifikasi Penyakit Jantung</h1>
        <p style="font-size:1.1rem; opacity:0.9;">
            Berbasis Machine Learning dengan Teknik SMOTE 
        </p>
        <p style="opacity:0.75; font-size:0.9rem;">
            SVM · Random Forest · XGBoost
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <h3>5.568</h3><p>Total Sampel</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <h3>18</h3><p>Jumlah Fitur</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <h3>3</h3><p>Algoritma ML</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <h3>95.54%</h3><p>Akurasi Terbaik</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📌 Tentang Proyek")
        st.markdown("""
        Proyek ini membangun sistem klasifikasi penyakit jantung menggunakan
        algoritma machine learning dengan menangani masalah **imbalanced data**
        melalui  pendekatan:

        - **SMOTE** — Synthetic Minority Over-sampling Technique
        
        Dataset berasal dari **Kaggle — Heart Disease Prediction** (NHANES 2017–2018)
        dengan distribusi kelas yang sangat tidak seimbang: **90,54% tidak sakit
        vs 9,46% sakit jantung**.
        """)
    with col_r:
        st.subheader("🏆 Hasil Terbaik")
        res_data = {
            'Model': ['Random Forest', 'XGBoost', 'SVM'],
            'Avg Accuracy': ['95.22%', '94.45%', '94.30%'],
            'Best Split': ['90:10 → 95.54%', '70:30 → 94.51%', '90:10 → 94.95%'],
            'Status': ['🥇 Terbaik', '🥈 2nd', '🥉 3rd'],
        }
        st.dataframe(pd.DataFrame(res_data), use_container_width=True, hide_index=True)

    st.subheader("🔄 Pipeline Metodologi")
    steps = [
        ("1️⃣", "Load Dataset", "5.568 sampel × 19 kolom dari Kaggle"),
        ("2️⃣", "Preprocessing", "Missing value check, IQR Capping, Pemisahan fitur"),
        ("3️⃣", "EDA", "Distribusi data, heatmap korelasi, analisis fitur penting"),
        ("4️⃣", "Split Data", "70:30 / 80:20 / 90:10 (Stratified)"),
        ("5️⃣", "Normalize", "StandardScaler — fit hanya pada training set"),
        ("6️⃣", "SMOTE", "Oversampling hanya pada training set (anti data leakage)"),
        ("7️⃣", "Tuning", "GridSearchCV + Stratified 5-Fold CV"),
        ("8️⃣", "Evaluasi", "Accuracy, Precision, Recall, F1-Score, Confusion Matrix"),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#f0f4ff; border-radius:10px; padding:0.8rem;
                        margin-bottom:0.5rem; border-top:3px solid #1565C0; text-align:center;">
                <div style="font-size:1.5rem">{icon}</div>
                <strong style="color:#1565C0">{title}</strong>
                <p style="font-size:0.8rem; color:#555; margin:0.3rem 0 0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    if df is None:
        st.info("⬆️ Upload file **heart_disease_prediction.csv** di sidebar untuk mengaktifkan semua fitur.")

# ─── PAGE: EKSPLORASI DATA ────────────────────────────────────────────────────
elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data (EDA)")

    if df is None:
        st.warning("⬆️ Upload dataset terlebih dahulu di sidebar.")
        st.stop()

    df_clean = df.copy()
    for col in ['age', 'poverty_income_ratio']:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        df_clean[col] = df_clean[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Info Dataset", "📈 Distribusi Fitur",
        "🎯 Distribusi Target", "🔥 Korelasi"
    ])

    # ── Tab 1: Info Dataset ──────────────────────────────────────────────────
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sampel", f"{df.shape[0]:,}")
        col2.metric("Total Kolom", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.subheader("5 Data Pertama")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe().T.round(3), use_container_width=True)

        st.subheader("Tipe Data & Missing Values")
        info_df = pd.DataFrame({
            'Tipe Data'     : df.dtypes.astype(str),
            'Missing Values': df.isnull().sum(),
            'Missing (%)'   : (df.isnull().sum() / len(df) * 100).round(2),
            'Nilai Unik'    : df.nunique(),
        })
        st.dataframe(info_df, use_container_width=True)

    # ── Tab 2: Distribusi Fitur ──────────────────────────────────────────────
    with tab2:
        feat_cols = [c for c in df_clean.columns if c != 'heart_disease']
        selected  = st.selectbox("Pilih fitur:", feat_cols)

        fig, ax = plt.subplots(figsize=(8, 4))
        if df_clean[selected].nunique() <= 6:
            vc = df_clean[selected].value_counts().sort_index()
            ax.bar(vc.index.astype(str), vc.values, color='#1565C0',
                   edgecolor='black', alpha=0.85)
            ax.set_xlabel('Nilai')
        else:
            ax.hist(df_clean[selected], bins=30, color='#1565C0',
                    edgecolor='white', alpha=0.85)
            ax.set_xlabel(selected)
        ax.set_ylabel('Frekuensi')
        ax.set_title(f'Distribusi Fitur: {selected}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()

        st.subheader("Distribusi Semua Fitur Sekaligus")
        binary_cols = [c for c in feat_cols if df_clean[c].nunique() <= 2]
        cont_cols   = [c for c in feat_cols if df_clean[c].nunique() > 2]

        fig2, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        for i, col in enumerate(feat_cols):
            if df_clean[col].nunique() <= 6:
                vc = df_clean[col].value_counts().sort_index()
                axes[i].bar(vc.index.astype(str), vc.values,
                            color='#1565C0', edgecolor='black', alpha=0.85)
            else:
                axes[i].hist(df_clean[col], bins=20, color='#1565C0',
                             edgecolor='white', alpha=0.85)
            axes[i].set_title(col, fontsize=9, fontweight='bold')
            axes[i].set_xlabel('')
        for j in range(len(feat_cols), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Distribusi Semua Fitur', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Tab 3: Distribusi Target ─────────────────────────────────────────────
    with tab3:
        counts = df['heart_disease'].value_counts()
        col1, col2 = st.columns(2)
        col1.metric("Kelas 0 (Tidak Sakit)", f"{counts[0]:,} ({counts[0]/len(df)*100:.2f}%)")
        col2.metric("Kelas 1 (Sakit Jantung)", f"{counts[1]:,} ({counts[1]/len(df)*100:.2f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels  = ['Tidak Sakit (0)', 'Sakit Jantung (1)']
        colors  = ['#1565C0', '#C62828']

        bars = axes[0].bar(labels, counts.values, color=colors,
                            edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, counts.values):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 30, str(val),
                         ha='center', fontweight='bold', fontsize=12)
        axes[0].set_title('Distribusi Kelas Target', fontweight='bold')
        axes[0].set_ylabel('Jumlah Sampel')
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%',
                    colors=colors, startangle=90,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})
        axes[1].set_title('Proporsi Kelas Target', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="info-box">
        ⚠️ <strong>Imbalanced Data</strong>: Rasio kelas sangat tidak seimbang (9,56:1).
        Tanpa penanganan khusus, model akan bias memprediksi semua sampel sebagai kelas 0.
        Oleh karena itu diterapkan <strong>SMOTE</strong> untuk menyeimbangkan distribusi kelas.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Korelasi ──────────────────────────────────────────────────────
    with tab4:
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots(figsize=(14, 10))
        corr = df_clean.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    mask=mask, linewidths=0.5, square=True,
                    cbar_kws={'shrink': 0.7}, annot_kws={'size': 7}, ax=ax)
        ax.set_title('Heatmap Korelasi Antar Fitur', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Korelasi Fitur terhadap Target")
        target_corr = df_clean.corr()['heart_disease'].drop('heart_disease')\
                               .sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        colors_bar = ['#C62828' if v > 0 else '#1565C0' for v in target_corr.values]
        ax2.barh(target_corr.index, target_corr.values,
                 color=colors_bar, edgecolor='black', linewidth=0.7)
        ax2.axvline(x=0, color='black', linewidth=1.2)
        ax2.set_title('Korelasi Fitur vs Target (heart_disease)', fontweight='bold')
        ax2.set_xlabel('Nilai Korelasi Pearson')
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 Top 5 Korelasi Positif**")
            top_pos = target_corr.head(5).reset_index()
            top_pos.columns = ['Fitur', 'Korelasi']
            st.dataframe(top_pos.round(4), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**🔵 Top 5 Korelasi Negatif**")
            top_neg = target_corr.tail(5).reset_index()
            top_neg.columns = ['Fitur', 'Korelasi']
            st.dataframe(top_neg.round(4), use_container_width=True, hide_index=True)

# ─── PAGE: PERFORMA MODEL ─────────────────────────────────────────────────────
elif page == "🤖 Performa Model":
    st.title("🤖 Performa Model")

    if df is None:
        st.warning("⬆️ Upload dataset terlebih dahulu di sidebar.")
        st.stop()

    with st.spinner("⏳ Melatih model... (mungkin butuh 1-2 menit)"):
        trained_models, scaler, live_metrics, df_clean, X_te, y_te = train_models(df)
    st.success("✅ Model berhasil dilatih!")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Perbandingan Metrik", "🗂️ Confusion Matrix",
        "📋 Classification Report", "🌿 Feature Importance"
    ])

    # ── Tab 1: Perbandingan Metrik ───────────────────────────────────────────
    with tab1:
        st.subheader("Hasil Evaluasi Model (Split 80:20, Pipeline Anti Leakage)")
        rows = []
        for name, m in live_metrics.items():
            rows.append({
                'Model'    : name,
                'Accuracy' : f"{m['accuracy']*100:.2f}%",
                'Precision': f"{m['precision']*100:.2f}%",
                'Recall'   : f"{m['recall']*100:.2f}%",
                'F1-Score' : f"{m['f1']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Perbandingan Akurasi (Hasil GridSearchCV — semua split)")
        results_all = {
            'Model': ['SVM', 'Random Forest', 'XGBoost'],
            '70:30': [94.02, 94.98, 94.51],
            '80:20': [93.95, 95.14, 94.50],
            '90:10': [94.95, 95.54, 94.35],
            'Rata-rata': [94.30, 95.22, 94.45],
        }
        df_res = pd.DataFrame(results_all)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        models  = ['SVM', 'Random Forest', 'XGBoost']
        splits  = ['70:30', '80:20', '90:10']
        colors3 = ['#1565C0', '#2E7D32', '#E65100']
        accs    = [[94.02, 93.95, 94.95], [94.98, 95.14, 95.54], [94.51, 94.50, 94.35]]
        x, w = np.arange(len(splits)), 0.25
        for i, (model, color, acc) in enumerate(zip(models, colors3, accs)):
            bars = axes[0].bar(x + i*w, acc, w, label=model, color=color,
                               edgecolor='black', linewidth=0.7, alpha=0.9)
            for bar in bars:
                axes[0].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.1,
                             f'{bar.get_height():.1f}%',
                             ha='center', fontsize=8, fontweight='bold')
        axes[0].axhline(y=90, color='red', linestyle='--', linewidth=1.5,
                        label='Target ≥90%', alpha=0.7)
        axes[0].set_xticks(x + w)
        axes[0].set_xticklabels(splits)
        axes[0].set_ylim(85, 100)
        axes[0].set_title('Perbandingan Akurasi per Rasio Split', fontweight='bold')
        axes[0].set_ylabel('Akurasi (%)')
        axes[0].legend(fontsize=9)
        axes[0].grid(axis='y', alpha=0.3)

        markers3 = ['o', 's', '^']
        for model, color, marker, acc in zip(models, colors3, markers3, accs):
            axes[1].plot(splits, acc, color=color, marker=marker,
                         linewidth=2.5, markersize=10, label=model)
            for xi, yi in zip(splits, acc):
                axes[1].annotate(f'{yi:.1f}%', (xi, yi),
                                 textcoords='offset points', xytext=(0, 10),
                                 ha='center', fontsize=9, color=color, fontweight='bold')
        axes[1].axhline(y=90, color='red', linestyle='--', linewidth=1.5,
                        label='Target ≥90%', alpha=0.7)
        axes[1].set_ylim(85, 100)
        axes[1].set_title('Tren Akurasi per Rasio Split', fontweight='bold')
        axes[1].set_ylabel('Akurasi (%)')
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Confusion Matrix ──────────────────────────────────────────────
    with tab2:
        st.subheader("Confusion Matrix — Split 80:20")
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        cmaps3 = ['Blues', 'Greens', 'Oranges']
        for ax, (name, m), cmap in zip(axes, live_metrics.items(), cmaps3):
            disp = ConfusionMatrixDisplay(
                confusion_matrix=m['cm'],
                display_labels=['Tidak Sakit (0)', 'Sakit (1)']
            )
            disp.plot(ax=ax, cmap=cmap, colorbar=False)
            ax.set_title(f'{name}\nAcc: {m["accuracy"]*100:.2f}%',
                         fontweight='bold', fontsize=11)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        plt.suptitle('Confusion Matrix Semua Model (Split 80:20)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="info-box">
        <strong>Interpretasi Confusion Matrix:</strong><br>
        • <strong>True Positive (TP)</strong>: Pasien sakit diprediksi sakit ✅<br>
        • <strong>True Negative (TN)</strong>: Pasien sehat diprediksi sehat ✅<br>
        • <strong>False Positive (FP)</strong>: Pasien sehat diprediksi sakit ⚠️<br>
        • <strong>False Negative (FN)</strong>: Pasien sakit diprediksi sehat ❌ (paling berbahaya)
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Classification Report ─────────────────────────────────────────
    with tab3:
        st.subheader("Classification Report — Split 80:20")
        selected_model = st.selectbox("Pilih Model:", list(trained_models.keys()))
        m = live_metrics[selected_model]
        report = classification_report(
            m['y_test'], m['y_pred'],
            target_names=['Tidak Sakit Jantung', 'Sakit Jantung'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).T.round(4)
        st.dataframe(report_df, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{m['precision']*100:.2f}%")
        col3.metric("Recall",    f"{m['recall']*100:.2f}%")
        col4.metric("F1-Score",  f"{m['f1']*100:.2f}%")

    # ── Tab 4: Feature Importance ─────────────────────────────────────────────
    with tab4:
        st.subheader("Feature Importance — Random Forest & XGBoost")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, name, color in zip(axes,
                                   ['Random Forest', 'XGBoost'],
                                   ['#2E7D32', '#E65100']):
            fi = pd.Series(
                trained_models[name].feature_importances_,
                index=FEATURE_NAMES
            ).sort_values(ascending=True)
            ax.barh(fi.index, fi.values, color=color,
                    edgecolor='black', linewidth=0.6, alpha=0.9)
            ax.set_title(f'Feature Importance — {name}',
                         fontweight='bold', fontsize=12)
            ax.set_xlabel('Importance Score')
            ax.grid(axis='x', alpha=0.3)
        plt.suptitle('Feature Importance (Split 80:20)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─── PAGE: PREDIKSI PASIEN ────────────────────────────────────────────────────
elif page == "🔮 Prediksi Pasien":
    st.title("🔮 Prediksi Risiko Penyakit Jantung")
    st.markdown("""
    <div class="info-box">
    Masukkan data pasien di bawah ini untuk mendapatkan prediksi risiko penyakit jantung
    menggunakan ketiga model yang telah dilatih (SVM, Random Forest, XGBoost).
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.warning("⬆️ Upload dataset terlebih dahulu di sidebar.")
        st.stop()

    with st.spinner("⏳ Mempersiapkan model..."):
        trained_models, scaler, live_metrics, df_clean, X_te, y_te = train_models(df)

    st.subheader("📝 Input Data Pasien")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏥 Informasi Klinis**")
        age = st.slider("Usia (tahun)", 20, 80, 50)
        sex = st.selectbox("Jenis Kelamin", [1, 2],
                           format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
        hypertension = st.selectbox("Hipertensi", [0, 1],
                                    format_func=lambda x: "Ya" if x == 1 else "Tidak")
        high_cholesterol = st.selectbox("Kolesterol Tinggi", [0, 1],
                                        format_func=lambda x: "Ya" if x == 1 else "Tidak")
        diabetes = st.selectbox("Diabetes", [0, 1],
                                format_func=lambda x: "Ya" if x == 1 else "Tidak")
        told_stroke = st.selectbox("Pernah Stroke", [0, 1],
                                   format_func=lambda x: "Ya" if x == 1 else "Tidak")

    with col2:
        st.markdown("**💊 Riwayat Obat & Kondisi**")
        taking_bp_meds = st.selectbox("Minum Obat Tekanan Darah", [0, 1],
                                      format_func=lambda x: "Ya" if x == 1 else "Tidak")
        taking_cholesterol_meds = st.selectbox("Minum Obat Kolesterol", [0, 1],
                                               format_func=lambda x: "Ya" if x == 1 else "Tidak")
        taking_insulin = st.selectbox("Menggunakan Insulin", [0, 1],
                                      format_func=lambda x: "Ya" if x == 1 else "Tidak")
        taking_diabetes_pills = st.selectbox("Minum Obat Diabetes", [0, 1],
                                             format_func=lambda x: "Ya" if x == 1 else "Tidak")
        told_prediabetes = st.selectbox("Pernah Prediabetes", [0, 1],
                                        format_func=lambda x: "Ya" if x == 1 else "Tidak")
        family_history_heart_attack = st.selectbox("Riwayat Jantung dalam Keluarga", [0, 1],
                                                   format_func=lambda x: "Ya" if x == 1 else "Tidak")

    with col3:
        st.markdown("**🧬 Gaya Hidup & Sosial**")
        smoking_ever = st.selectbox("Pernah Merokok", [0, 1],
                                    format_func=lambda x: "Ya" if x == 1 else "Tidak")
        smoking_current = st.selectbox("Merokok Saat Ini", [0, 1],
                                       format_func=lambda x: "Ya" if x == 1 else "Tidak")
        physically_active = st.selectbox("Aktif Fisik / Olahraga", [0, 1],
                                         format_func=lambda x: "Ya" if x == 1 else "Tidak")
        race_ethnicity = st.selectbox("Ras/Etnis",
                                      [1, 2, 3, 4, 5, 6, 7],
                                      format_func=lambda x: {
                                          1: "1-Mexican American",
                                          2: "2-Other Hispanic",
                                          3: "3-Non-Hispanic White",
                                          4: "4-Non-Hispanic Black",
                                          5: "5-Non-Hispanic Asian",
                                          6: "6-Other/Multi-Racial",
                                          7: "7-Lainnya"
                                      }[x])
        education = st.selectbox("Tingkat Pendidikan",
                                 [1, 2, 3, 4, 5],
                                 format_func=lambda x: {
                                     1: "1-Tidak Tamat SD",
                                     2: "2-SD/SMP",
                                     3: "3-SMA",
                                     4: "4-D3/S1",
                                     5: "5-S2/S3"
                                 }[x])
        poverty_income_ratio = st.slider("Rasio Pendapatan (0–5)", 0.0, 5.0, 2.5, 0.1)

    st.markdown("---")
    predict_btn = st.button("🔮 PREDIKSI SEKARANG", use_container_width=True)

    if predict_btn:
        input_data = {
            'taking_bp_meds'            : taking_bp_meds,
            'taking_cholesterol_meds'   : taking_cholesterol_meds,
            'age'                       : float(age),
            'sex'                       : float(sex),
            'race_ethnicity'            : float(race_ethnicity),
            'education'                 : float(education),
            'poverty_income_ratio'      : poverty_income_ratio,
            'taking_insulin'            : taking_insulin,
            'taking_diabetes_pills'     : taking_diabetes_pills,
            'told_prediabetes'          : told_prediabetes,
            'told_stroke'               : told_stroke,
            'family_history_heart_attack': family_history_heart_attack,
            'diabetes'                  : diabetes,
            'smoking_ever'              : smoking_ever,
            'smoking_current'           : smoking_current,
            'physically_active'         : physically_active,
            'hypertension'              : hypertension,
            'high_cholesterol'          : high_cholesterol,
        }

        # Preprocessing input
        df_input = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        for col in ['age', 'poverty_income_ratio']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            df_input[col] = df_input[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))
        input_scaled = scaler.transform(df_input)

        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")

        results = {}
        for name, model in trained_models.items():
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]
            results[name] = {'pred': pred, 'prob': prob}

        # Voting
        votes = [r['pred'] for r in results.values()]
        final_pred = 1 if sum(votes) >= 2 else 0

        # Tampilkan hasil voting
        if final_pred == 1:
            st.markdown("""
            <div class="result-positive">
                ❗ HASIL: BERISIKO SAKIT JANTUNG<br>
                <span style="font-size:1rem; font-weight:normal">
                Mayoritas model memprediksi pasien ini berisiko penyakit jantung.
                Disarankan segera berkonsultasi dengan dokter spesialis jantung.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-negative">
                ✅ HASIL: TIDAK BERISIKO SAKIT JANTUNG<br>
                <span style="font-size:1rem; font-weight:normal">
                Mayoritas model memprediksi pasien ini tidak berisiko penyakit jantung.
                Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Detail Prediksi Per Model")

        cols = st.columns(3)
        model_colors = {'SVM': '#1565C0', 'Random Forest': '#2E7D32', 'XGBoost': '#E65100'}
        for col, (name, res) in zip(cols, results.items()):
            with col:
                label = "🔴 Berisiko" if res['pred'] == 1 else "🟢 Tidak Berisiko"
                prob_pos = res['prob'][1] * 100
                prob_neg = res['prob'][0] * 100
                st.markdown(f"""
                <div style="background:white; border-radius:10px; padding:1rem;
                            border-top:4px solid {model_colors[name]};
                            box-shadow:0 2px 8px rgba(0,0,0,0.1); text-align:center;">
                    <h4 style="color:{model_colors[name]}; margin:0">{name}</h4>
                    <h3 style="margin:0.5rem 0">{label}</h3>
                    <p style="margin:0.2rem 0">
                        🔴 Sakit: <strong>{prob_pos:.1f}%</strong>
                    </p>
                    <p style="margin:0.2rem 0">
                        🟢 Sehat: <strong>{prob_neg:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Visualisasi probabilitas
        st.subheader("📈 Visualisasi Probabilitas")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, res), color in zip(axes, results.items(),
                                          ['#1565C0', '#2E7D32', '#E65100']):
            probs  = [res['prob'][1]*100, res['prob'][0]*100]
            labels = ['Sakit Jantung', 'Tidak Sakit']
            bar_colors = ['#C62828', '#1565C0']
            bars = ax.bar(labels, probs, color=bar_colors, edgecolor='black',
                          linewidth=0.9, alpha=0.9, width=0.5)
            for bar, val in zip(bars, probs):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1, f'{val:.1f}%',
                        ha='center', fontweight='bold', fontsize=12)
            ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.2,
                       label='Threshold 50%')
            ax.set_ylim(0, 120)
            ax.set_title(f'{name}\n{"🔴 Berisiko" if res["pred"]==1 else "🟢 Tidak Berisiko"}',
                         fontweight='bold')
            ax.set_ylabel('Probabilitas (%)')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        plt.suptitle('Probabilitas Prediksi — Semua Model', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="info-box">
        ⚠️ <strong>Disclaimer</strong>: Prediksi ini bersifat indikatif dan berbasis model machine learning.
        Hasil ini <strong>bukan diagnosis medis</strong>. Selalu konsultasikan dengan dokter atau
        tenaga medis profesional untuk diagnosis dan penanganan yang tepat.
        </div>
        """, unsafe_allow_html=True)
