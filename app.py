import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Konfigurasi halaman ──────────────────────────────
st.set_page_config(
    page_title="Prediksi Personal Loan",
    layout="wide"
)

# ── Custom CSS Neobrutalism ──────────────────
st.markdown("""
<style>
/* Sidebar Background & Text */
[data-testid="stSidebar"] {
    background-color: #4ade80 !important; /* Green */
    border-right: 4px solid #000000;
}
[data-testid="stSidebar"] .stRadio label p {
    font-size: 20px !important;
    font-weight: 900 !important;
    padding: 8px 0px;
    color: #000000;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-weight: 900 !important;
    color: #000000 !important;
}

/* Metrics (Blue theme) */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 3px solid #000000;
    padding: 1.25rem;
    border-radius: 0px;
    box-shadow: 6px 6px 0px 0px #4287f5; /* Blue */
    transition: transform 0.1s;
    color: #000000;
}
div[data-testid="metric-container"] label {
    font-weight: 900 !important;
}
div[data-testid="metric-container"]:hover {
    transform: translate(-3px, -3px);
    box-shadow: 9px 9px 0px 0px #4287f5;
}

/* Charts (Orange theme) */
div[data-testid="stPlotlyChart"] {
    background-color: #ffffff;
    border: 3px solid #000000;
    border-radius: 0px;
    padding: 1rem;
    box-shadow: 6px 6px 0px 0px #ff9c2a; /* Orange */
}

/* DataFrames (Red theme) */
div[data-testid="stDataFrame"] {
    border: 3px solid #000000;
    border-radius: 0px;
    box-shadow: 6px 6px 0px 0px #ff4747; /* Red */
}

/* Main App Background */
.stApp {
    background-color: #f5f5f5;
}

/* Buttons (Neobrutalism Style) */
.stButton button {
    background-color: #ff9c2a !important; /* Orange */
    color: #000000 !important;
    border: 3px solid #000000 !important;
    border-radius: 0px !important;
    box-shadow: 5px 5px 0px 0px #000000 !important;
    font-weight: 900 !important;
    transition: all 0.1s !important;
}
.stButton button:active {
    transform: translate(5px, 5px) !important;
    box-shadow: 0px 0px 0px 0px #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/preposesing_personal_loan.csv")
    return df

# ── Train model ──────────────────────────────────────
@st.cache_resource
def train_models(df):
    df_m = df.copy()
    df_m['target'] = (df_m['Personal Loan'] == 'Loan').astype(int)

    FEATURES = ['Age', 'Experience', 'Income', 'Family',
                'CCAvg', 'Education', 'Mortgage']
    X = df_m[FEATURES]
    y = df_m['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    dt.fit(Xtr, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr, y_train)

    return dt, lr, scaler, FEATURES, Xte, y_test

df = load_data()

# ── Sidebar navigasi ─────────────────────────────────
st.sidebar.markdown("""
<div style='background-color: #ff4747; border: 3px solid #000000; box-shadow: 4px 4px 0px 0px #000000; padding: 15px; margin-bottom: 25px;'>
    <h2 style='font-size: 22px; font-weight: 900; color: #ffffff !important; text-align: center; margin: 0; text-transform: uppercase;'>Personal Loan<br>Data Bank</h2>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigasi", [
    "Overview",
    "EDA",
    "Loan Insight",
    "Model Evaluation",
    "Predictor"
])

CLR = ['#4C72B0', '#DD8452']

# ══════════════════════════════════════════════════════
# HALAMAN 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "Overview":
    st.title("Executive Overview")

    total   = len(df)
    loan    = len(df[df['Personal Loan'] == 'Loan'])
    no_loan = total - loan
    pct     = loan / total * 100
    avg_inc = df['Income'].mean()
    avg_age = df['Age'].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Nasabah", f"{total:,}")
    c2.metric("Nasabah Loan", f"{loan:,}")
    c3.metric("Nasabah Non-Loan", f"{no_loan:,}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Persentase Loan", f"{pct:.1f}%")
    c5.metric("Rata-rata Income", f"${avg_inc:.1f}K")
    c6.metric("Rata-rata Usia", f"{avg_age:.1f} tahun")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(
            values=[loan, no_loan],
            names=['Loan', 'No Loan'],
            title='Distribusi Status Loan',
            color_discrete_sequence=CLR
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Statistik Ringkas Dataset")
        st.dataframe(
            df[['Age', 'Income', 'CCAvg', 'Mortgage']].describe().round(2)
        )

    st.markdown("---")
    st.subheader("Tren Data")
    df_trend = df.groupby(['Age', 'Personal Loan'])['Income'].mean().reset_index()
    fig_line = px.line(
        df_trend, x='Age', y='Income', color='Personal Loan',
        title='Rata-rata Pendapatan per Usia berdasarkan Status Loan',
        color_discrete_sequence=CLR,
        markers=True
    )
    fig_line.update_layout(xaxis_title='Usia', yaxis_title='Rata-rata Income ($K)')
    st.plotly_chart(fig_line, use_container_width=True)

# ══════════════════════════════════════════════════════
# HALAMAN 2 — EDA
# ══════════════════════════════════════════════════════
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x='Income', color='Personal Loan',
            title='Distribusi Income', nbins=40,
            color_discrete_sequence=CLR), use_container_width=True)

        st.plotly_chart(px.histogram(df, x='CCAvg', color='Personal Loan',
            title='Distribusi CCAvg', nbins=40,
            color_discrete_sequence=CLR), use_container_width=True)

    with c2:
        st.plotly_chart(px.histogram(df, x='Age', color='Personal Loan',
            title='Distribusi Usia', nbins=30,
            color_discrete_sequence=CLR), use_container_width=True)

        df_edu = df.copy()
        df_edu['Pendidikan'] = df_edu['Education'].map(
            {1: 'Undergrad', 2: 'Graduate', 3: 'Professional'})
        edu_cnt = df_edu.groupby(
            ['Pendidikan', 'Personal Loan']).size().reset_index(name='Jumlah')
        st.plotly_chart(px.bar(edu_cnt, x='Pendidikan', y='Jumlah',
            color='Personal Loan', barmode='group',
            title='Distribusi Pendidikan',
            color_discrete_sequence=CLR), use_container_width=True)

# ══════════════════════════════════════════════════════
# HALAMAN 3 — LOAN INSIGHT
# ══════════════════════════════════════════════════════
elif page == "Loan Insight":
    st.title("Loan Insight — Faktor Penentu")

    avg_grp = df.groupby('Personal Loan')[
        ['Income', 'CCAvg', 'Age', 'Mortgage']].mean().reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(avg_grp, x='Personal Loan', y='Income',
            color='Personal Loan', title='Rata-rata Income vs Status Loan',
            color_discrete_sequence=CLR), use_container_width=True)

        st.plotly_chart(px.box(df, x='Personal Loan', y='Income',
            color='Personal Loan', title='Sebaran Income per Status Loan',
            color_discrete_sequence=CLR), use_container_width=True)

    with c2:
        st.plotly_chart(px.bar(avg_grp, x='Personal Loan', y='CCAvg',
            color='Personal Loan', title='Rata-rata CCAvg vs Status Loan',
            color_discrete_sequence=CLR), use_container_width=True)

        df_edu = df.copy()
        df_edu['Pendidikan'] = df_edu['Education'].map(
            {1: 'Undergrad', 2: 'Graduate', 3: 'Professional'})
        edu_loan = df_edu.groupby(
            ['Pendidikan', 'Personal Loan']).size().reset_index(name='Jumlah')
        st.plotly_chart(px.bar(edu_loan, x='Pendidikan', y='Jumlah',
            color='Personal Loan', barmode='group',
            title='Pendidikan vs Status Loan',
            color_discrete_sequence=CLR), use_container_width=True)

# ══════════════════════════════════════════════════════
# HALAMAN 4 — MODEL EVALUATION
# ══════════════════════════════════════════════════════
elif page == "Model Evaluation":
    st.title("Evaluasi & Perbandingan Model")

    def load_eval_metrics(csv_path):
        df_eval = pd.read_csv(csv_path)
        accuracy = df_eval['Accuracy'].iloc[2]
        precision = df_eval['Precision'].iloc[1]
        recall = df_eval['Recall'].iloc[1]
        f1 = df_eval['F-measure'].iloc[1]
        
        return {
            'Accuracy':  round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall':    round(recall, 4),
            'F1-Score':  round(f1, 4),
        }

    def load_confusion_matrix(csv_path):
        df_cm = pd.read_csv(csv_path)
        return df_cm.values
        
    m_dt = load_eval_metrics('data/eval_DT.csv')
    m_lr = load_eval_metrics('data/eval_LR.csv')
    cm = load_confusion_matrix('data/eval_confusion_DT.csv')
    cm2 = load_confusion_matrix('data/eval_Confusion_LR.csv')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Decision Tree")
        for k, v in m_dt.items():
            st.metric(k, v, delta=round(v - m_lr.get(k, 0), 4))
        st.plotly_chart(px.imshow(cm, text_auto=True,
            x=['Pred No Loan', 'Pred Loan'],
            y=['Actual No Loan', 'Actual Loan'],
            title='Confusion Matrix — Decision Tree',
            color_continuous_scale='Blues'), use_container_width=True)

    with c2:
        st.subheader("Logistic Regression")
        for k, v in m_lr.items():
            st.metric(k, v, delta=round(v - m_dt.get(k, 0), 4))
        st.plotly_chart(px.imshow(cm2, text_auto=True,
            x=['Pred No Loan', 'Pred Loan'],
            y=['Actual No Loan', 'Actual Loan'],
            title='Confusion Matrix — Logistic Regression',
            color_continuous_scale='Oranges'), use_container_width=True)

    st.markdown("---")
    st.subheader("Visualisasi Decision Tree")
    dt_vis, _, _, FEATURES_VIS, _, _ = train_models(df)
    fig_tree, ax_tree = plt.subplots(figsize=(24, 10))
    plot_tree(
        dt_vis,
        feature_names=FEATURES_VIS,
        class_names=['No Loan', 'Loan'],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax_tree,
        max_depth=3
    )
    ax_tree.set_title('Struktur Decision Tree (max_depth=3)', fontsize=16, fontweight='bold')
    st.pyplot(fig_tree)

# ══════════════════════════════════════════════════════
# HALAMAN 5 — PREDICTOR
# ══════════════════════════════════════════════════════
elif page == "Predictor":
    st.title("Personal Loan Predictor")
    st.markdown("Masukkan data nasabah untuk mendapatkan prediksi kelayakan personal loan.")

    dt, lr, scaler, FEATURES, _, _ = train_models(df)

    c1, c2 = st.columns(2)
    with c1:
        age        = st.slider("Usia", 18, 70, 35)
        experience = st.slider("Pengalaman Kerja (tahun)", 0, 45, 10)
        income     = st.slider("Pendapatan Tahunan ($K)", 8, 250, 60)
        family     = st.selectbox("Jumlah Anggota Keluarga", [1, 2, 3, 4])

    with c2:
        ccavg    = st.slider("Pengeluaran Kartu Kredit/bulan ($K)", 0.0, 10.0, 1.5, 0.1)
        edu_opt  = st.selectbox("Pendidikan",
                       [(1, "Undergrad"), (2, "Graduate"), (3, "Professional")],
                       format_func=lambda x: x[1])
        mortgage = st.slider("Nilai Mortgage ($K)", 0, 700, 0)

    if st.button("Prediksi Sekarang", type="primary", use_container_width=True):
        inp = np.array([[age, experience, income, family,
                         ccavg, edu_opt[0], mortgage]])
        inp_scaled = scaler.transform(inp)

        pred_dt = dt.predict(inp_scaled)[0]
        prob_dt = dt.predict_proba(inp_scaled)[0][1]
        pred_lr = lr.predict(inp_scaled)[0]
        prob_lr = lr.predict_proba(inp_scaled)[0][1]

        label = lambda p: ("LOAN DIREKOMENDASIKAN" if p == 1 else "LOAN TIDAK DIREKOMENDASIKAN")

        st.markdown("---")
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Decision Tree")
            st.markdown(f"### {label(pred_dt)}")
            st.metric("Probabilitas Disetujui", f"{prob_dt * 100:.1f}%")
            st.progress(float(prob_dt))
        with r2:
            st.subheader("Logistic Regression")
            st.markdown(f"### {label(pred_lr)}")
            st.metric("Probabilitas Disetujui", f"{prob_lr * 100:.1f}%")
            st.progress(float(prob_lr))
