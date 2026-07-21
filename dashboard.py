import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EstateX — Prediksi Harga Properti",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Modern Glassmorphism Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ===== Import Google Font ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ===== Root Variables ===== */
    :root {
        --bg-primary: #06080d;
        --bg-card: rgba(15, 18, 30, 0.65);
        --bg-card-hover: rgba(20, 24, 40, 0.8);
        --glass-border: rgba(255, 255, 255, 0.06);
        --glass-border-hover: rgba(100, 180, 255, 0.2);
        --accent-primary: #4f8fff;
        --accent-secondary: #38bdf8;
        --accent-gradient: linear-gradient(135deg, #4f8fff 0%, #38bdf8 50%, #818cf8 100%);
        --accent-gradient-warm: linear-gradient(135deg, #f97316 0%, #ef4444 100%);
        --text-primary: #e8eaf0;
        --text-secondary: #8a90a5;
        --text-muted: #5a6078;
        --radius-sm: 10px;
        --radius-md: 14px;
        --radius-lg: 20px;
        --shadow-card: 0 4px 30px rgba(0, 0, 0, 0.3);
        --shadow-glow: 0 0 40px rgba(79, 143, 255, 0.08);
    }

    /* ===== Global ===== */
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    html, body, [data-testid="stAppViewContainer"], .main .block-container {
        color: var(--text-primary) !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1280px !important;
    }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(79, 143, 255, 0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(79, 143, 255, 0.4); }

    /* ===== Typography ===== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: var(--text-primary) !important;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
        line-height: 1.15;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 2.2rem;
        line-height: 1.6;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--accent-primary);
        margin-bottom: 6px;
        display: block;
    }

    /* ===== Glass Card ===== */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 28px 28px 24px;
        box-shadow: var(--shadow-card);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .glass-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(79,143,255,0.3), transparent);
        opacity: 0;
        transition: opacity 0.35s ease;
    }

    .glass-card:hover {
        border-color: var(--glass-border-hover);
        box-shadow: var(--shadow-card), var(--shadow-glow);
        transform: translateY(-3px);
    }

    .glass-card:hover::after { opacity: 1; }

    /* ===== Metric Card ===== */
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(24px);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 24px 24px 20px;
        box-shadow: var(--shadow-card);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
        background: var(--accent-gradient);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover {
        border-color: var(--glass-border-hover);
        transform: translateY(-4px);
        box-shadow: var(--shadow-card), var(--shadow-glow);
    }

    .metric-card:hover::before { opacity: 1; }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--text-secondary);
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.2;
        letter-spacing: -0.03em;
    }

    .metric-sub {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 8px;
        font-weight: 400;
    }

    /* ===== Accent Metric (Prediction) ===== */
    .metric-value-accent {
        font-size: 2.1rem;
        font-weight: 900;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        letter-spacing: -0.03em;
        margin: 8px 0;
    }

    /* ===== Stat Chip ===== */
    .stat-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(79, 143, 255, 0.08);
        border: 1px solid rgba(79, 143, 255, 0.15);
        border-radius: 99px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--accent-secondary);
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background: rgba(8, 10, 18, 0.95) !important;
        border-right: 1px solid var(--glass-border) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary) !important;
    }

    .sidebar-brand {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }

    .sidebar-tagline {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 400;
        margin-bottom: 1.6rem;
    }

    /* ===== Tab Styling ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 18, 30, 0.5);
        border-radius: var(--radius-md);
        padding: 4px;
        border: 1px solid var(--glass-border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm) !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.25s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(79, 143, 255, 0.08) !important;
        color: var(--text-primary) !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(79, 143, 255, 0.12) !important;
        color: var(--accent-primary) !important;
        border-bottom: none !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ===== Input Fields ===== */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: rgba(15, 18, 30, 0.7) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        transition: all 0.25s ease !important;
    }

    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: var(--glass-border-hover) !important;
    }

    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(79, 143, 255, 0.15) !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background-color: var(--accent-primary) !important;
    }

    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--accent-primary) !important;
        border: 2px solid var(--bg-primary) !important;
        box-shadow: 0 0 8px rgba(79, 143, 255, 0.4) !important;
    }

    /* ===== DataFrame ===== */
    .stDataFrame {
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        border: 1px solid var(--glass-border) !important;
    }

    /* ===== Plotly Charts ===== */
    .js-plotly-plot, .stPlotlyChart {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--glass-border);
        background: var(--bg-card);
    }

    /* ===== Divider ===== */
    hr {
        border-color: var(--glass-border) !important;
        margin: 1.8rem 0 !important;
    }

    /* ===== Expander ===== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    /* ===== Hide default Streamlit elements but keep sidebar toggle button ===== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    header [data-testid="stSidebarCollapseButton"],
    header [data-testid="collapsedControl"] {
        visibility: visible !important;
    }

    /* ===== Responsive ===== */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.8rem; }
        .metric-value { font-size: 1.3rem; }
        .metric-value-accent { font-size: 1.6rem; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly Theme
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8a90a5", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#161a2a",
        bordercolor="rgba(79,143,255,0.3)",
        font=dict(family="Inter", size=13, color="#e8eaf0"),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    colorway=["#4f8fff", "#38bdf8", "#818cf8", "#a78bfa", "#f97316", "#34d399"],
)

# ---------------------------------------------------------------------------
# Data & Model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_data_and_model():
    clean_path = "datasets/dataset_rumah99_clean.csv"
    if not os.path.exists(clean_path):
        raw_path = "datasets/dataset_rumah99_raw.csv"
        df = pd.read_csv(raw_path).dropna(
            subset=[
                "Harga_Rupiah", "Kamar_Tidur", "Kamar_Mandi",
                "Luas_Tanah_m2", "Luas_Bangunan_m2", "Jumlah_Lantai", "Kota",
            ]
        )
        df = df[(df["Kamar_Tidur"] > 0) & (df["Kamar_Mandi"] > 0) & (df["Daya_Listrik_Watt"] > 0)]
        df = df[
            (df["Harga_Rupiah"] >= 100_000_000)
            & (df["Harga_Rupiah"] <= 50_000_000_000)
            & (df["Luas_Tanah_m2"] <= 2000)
            & (df["Luas_Bangunan_m2"] <= 2000)
            & (df["Kamar_Tidur"] <= 15)
            & (df["Kamar_Mandi"] <= 15)
        ]
    else:
        df = pd.read_csv(clean_path)

    # Multiple Linear Regression
    X_features = [
        "Luas_Tanah_m2", "Luas_Bangunan_m2", "Kamar_Tidur",
        "Kamar_Mandi", "Daya_Listrik_Watt", "Provinsi", "Kota",
    ]
    X_raw = df[X_features]
    Y = df["Harga_Rupiah"]

    # One-Hot Encoding
    X_encoded = pd.get_dummies(X_raw, columns=["Provinsi", "Kota"], drop_first=True)
    dummy_cols = [
        c for c in X_encoded.columns
        if c not in ["Luas_Tanah_m2", "Luas_Bangunan_m2", "Kamar_Tidur", "Kamar_Mandi", "Daya_Listrik_Watt"]
    ]
    X_encoded[dummy_cols] = X_encoded[dummy_cols].astype(int)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_encoded, Y, test_size=0.2, random_state=42
    )

    # Standard Scaling
    num_cols = ["Luas_Tanah_m2", "Luas_Bangunan_m2", "Kamar_Tidur", "Kamar_Mandi", "Daya_Listrik_Watt"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])

    # Fit
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)

    # Test score
    X_test_scaled = X_test.copy()
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    r2_train = model.score(X_train_scaled, Y_train)
    r2_test = model.score(X_test_scaled, Y_test)

    # Geo mappings
    prov_kota_map = df.groupby("Provinsi")["Kota"].unique().to_dict()
    prov_kota_map = {k: sorted(list(v)) for k, v in sorted(prov_kota_map.items())}

    return df, model, scaler, X_encoded.columns.tolist(), num_cols, prov_kota_map, r2_train, r2_test


df, model, scaler, encoded_cols, num_cols, prov_kota_map, r2_train, r2_test = load_data_and_model()

# ---------------------------------------------------------------------------
# Helper: format Rupiah
# ---------------------------------------------------------------------------
def fmt_rp(value, short=False):
    """Pretty-format Rupiah."""
    if short:
        if value >= 1e12:
            return f"Rp {value/1e12:,.1f} T"
        if value >= 1e9:
            return f"Rp {value/1e9:,.2f} M"
        if value >= 1e6:
            return f"Rp {value/1e6:,.0f} Jt"
        return f"Rp {value:,.0f}"
    return f"Rp {value:,.0f}"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">EstateX</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Prediksi harga properti berbasis data</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["Ringkasan & EDA", "Kalkulator Prediksi"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.7;">
        <span class="section-label">Tentang Model</span>
        <b style="color:var(--text-secondary);">Metode</b> — Regresi Linier Berganda<br>
        <b style="color:var(--text-secondary);">Target</b> — Harga Properti (Rp)<br>
        <b style="color:var(--text-secondary);">Sampel</b> — {df.shape[0]:,} baris bersih<br>
        <b style="color:var(--text-secondary);">R² Train</b> — {r2_train*100:.1f}%<br>
        <b style="color:var(--text-secondary);">R² Test</b> — {r2_test*100:.1f}%
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW & EDA
# ═══════════════════════════════════════════════════════════════════════════
if page == "Ringkasan & EDA":
    st.markdown('<span class="section-label">Overview</span>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Ringkasan Data & Analisis Eksploratif</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        "Eksplorasi interaktif karakteristik fisik dan geografis listing properti residensial."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Key Metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Data Bersih</div>
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-sub">baris tervalidasi</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Harga</div>
            <div class="metric-value">{fmt_rp(df['Harga_Rupiah'].mean(), short=True)}</div>
            <div class="metric-sub">median {fmt_rp(df['Harga_Rupiah'].median(), short=True)}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Luas Tanah</div>
            <div class="metric-value">{df['Luas_Tanah_m2'].mean():.0f} m²</div>
            <div class="metric-sub">median {df['Luas_Tanah_m2'].median():.0f} m²</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Luas Bangunan</div>
            <div class="metric-value">{df['Luas_Bangunan_m2'].mean():.0f} m²</div>
            <div class="metric-sub">median {df['Luas_Bangunan_m2'].median():.0f} m²</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data Preview ──────────────────────────────────────────────────────
    with st.expander("Lihat Sampel Data Properti", expanded=False):
        preview_cols = [
            "Judul", "Harga_Rupiah", "Provinsi", "Kota",
            "Luas_Tanah_m2", "Luas_Bangunan_m2",
            "Kamar_Tidur", "Kamar_Mandi", "Daya_Listrik_Watt",
        ]
        st.dataframe(
            df[preview_cols].head(15).style.format({"Harga_Rupiah": "Rp {:,.0f}"}),
            use_container_width=True,
            height=420,
        )

    st.markdown("---")

    # ── Visualizations ────────────────────────────────────────────────────
    st.markdown('<span class="section-label">Visualisasi</span>', unsafe_allow_html=True)
    st.markdown("### Analisis Eksploratif")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribusi Harga",
        "Korelasi",
        "Hubungan Fisik",
        "Analisis Geografis",
    ])

    # ── Tab 1: Distribusi ─────────────────────────────────────────────────
    with tab1:
        fig = px.histogram(
            df,
            x=df["Harga_Rupiah"] / 1e9,
            nbins=50,
            labels={"x": "Harga (Miliar Rp)", "count": "Jumlah"},
            title="Distribusi Harga Properti",
            color_discrete_sequence=["#4f8fff"],
            opacity=0.85,
        )
        fig.update_layout(**PLOTLY_LAYOUT, bargap=0.06)
        fig.update_traces(
            hovertemplate="Rp %{x:.2f} Miliar<br>Jumlah: %{y}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Box-plot complement
        fig_box = px.box(
            df,
            x=df["Harga_Rupiah"] / 1e9,
            labels={"x": "Harga (Miliar Rp)"},
            title="Sebaran Kuartil Harga",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_box.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Tab 2: Korelasi ───────────────────────────────────────────────────
    with tab2:
        corr_cols = [
            "Harga_Rupiah", "Luas_Tanah_m2", "Luas_Bangunan_m2",
            "Kamar_Tidur", "Kamar_Mandi", "Daya_Listrik_Watt", "Jumlah_Lantai",
        ]
        corr_labels = [
            "Harga", "L. Tanah", "L. Bangunan",
            "K. Tidur", "K. Mandi", "Listrik", "Lantai",
        ]
        corr_matrix = df[corr_cols].corr()

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_labels,
                y=corr_labels,
                colorscale=[
                    [0, "#1e1b4b"],
                    [0.25, "#312e81"],
                    [0.5, "#1e293b"],
                    [0.75, "#0e4f8f"],
                    [1, "#38bdf8"],
                ],
                zmin=-1, zmax=1,
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                textfont=dict(size=12, color="#e8eaf0"),
                hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
                colorbar=dict(
                    title=dict(text="r", font=dict(color="#8a90a5")),
                    tickfont=dict(color="#8a90a5"),
                ),
            )
        )
        fig_corr.update_layout(
            **PLOTLY_LAYOUT,
            title="Matriks Korelasi Pearson",
            height=480,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tab 3: Hubungan Fisik ─────────────────────────────────────────────
    with tab3:
        c1, c2 = st.columns(2)

        with c1:
            fig_sc = px.scatter(
                df,
                x="Luas_Bangunan_m2",
                y=df["Harga_Rupiah"] / 1e9,
                color="Kamar_Tidur",
                labels={
                    "Luas_Bangunan_m2": "Luas Bangunan (m²)",
                    "y": "Harga (Miliar Rp)",
                    "Kamar_Tidur": "K. Tidur",
                },
                title="Luas Bangunan vs Harga",
                color_continuous_scale="Viridis",
                opacity=0.65,
            )
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=440)
            fig_sc.update_traces(
                marker=dict(size=6, line=dict(width=0)),
                hovertemplate=(
                    "Luas: %{x} m²<br>"
                    "Harga: Rp %{y:.2f} M<br>"
                    "K. Tidur: %{marker.color}<extra></extra>"
                ),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with c2:
            fig_lt = px.scatter(
                df,
                x="Luas_Tanah_m2",
                y=df["Harga_Rupiah"] / 1e9,
                color="Kamar_Mandi",
                labels={
                    "Luas_Tanah_m2": "Luas Tanah (m²)",
                    "y": "Harga (Miliar Rp)",
                    "Kamar_Mandi": "K. Mandi",
                },
                title="Luas Tanah vs Harga",
                color_continuous_scale="Cividis",
                opacity=0.65,
            )
            fig_lt.update_layout(**PLOTLY_LAYOUT, height=440)
            fig_lt.update_traces(
                marker=dict(size=6, line=dict(width=0)),
                hovertemplate=(
                    "Luas: %{x} m²<br>"
                    "Harga: Rp %{y:.2f} M<br>"
                    "K. Mandi: %{marker.color}<extra></extra>"
                ),
            )
            st.plotly_chart(fig_lt, use_container_width=True)

    # ── Tab 4: Geografis ──────────────────────────────────────────────────
    with tab4:
        # Top provinces bar chart
        top_prov = df["Provinsi"].value_counts().head(8).index
        df_top = df[df["Provinsi"].isin(top_prov)]
        avg_price = (
            df_top.groupby("Provinsi")["Harga_Rupiah"]
            .mean()
            .sort_values(ascending=True) / 1e9
        )

        fig_bar = px.bar(
            x=avg_price.values,
            y=avg_price.index,
            orientation="h",
            labels={"x": "Rata-rata Harga (Miliar Rp)", "y": "Provinsi"},
            title="Rata-rata Harga Properti per Provinsi (Top 8)",
            color=avg_price.values,
            color_continuous_scale=["#1e3a5f", "#4f8fff", "#38bdf8"],
        )
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False)
        fig_bar.update_traces(
            hovertemplate="<b>%{y}</b><br>Rata-rata: Rp %{x:.2f} Miliar<extra></extra>"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Listing count per province
        count_data = df["Provinsi"].value_counts().head(10).reset_index()
        count_data.columns = ["Provinsi", "Jumlah"]

        fig_count = px.bar(
            count_data,
            x="Jumlah",
            y="Provinsi",
            orientation="h",
            labels={"Jumlah": "Jumlah Listing", "Provinsi": "Provinsi"},
            title="Jumlah Listing per Provinsi (Top 10)",
            color="Jumlah",
            color_continuous_scale=["#312e81", "#818cf8", "#c4b5fd"],
        )
        fig_count.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        fig_count.update_traces(
            hovertemplate="<b>%{y}</b><br>Listing: %{x}<extra></extra>"
        )
        st.plotly_chart(fig_count, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICTION CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Kalkulator Prediksi":
    st.markdown('<span class="section-label">Prediksi</span>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Kalkulator Harga Properti</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        "Masukkan spesifikasi rumah untuk mengestimasi harga pasar menggunakan model Regresi Linier Berganda."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Input Form ────────────────────────────────────────────────────────
    col_form, col_spacer, col_result = st.columns([1.4, 0.1, 1])

    with col_form:
        st.markdown('<span class="section-label">Spesifikasi</span>', unsafe_allow_html=True)
        st.markdown("#### Lokasi")

        g1, g2 = st.columns(2)
        with g1:
            selected_prov = st.selectbox("Provinsi", list(prov_kota_map.keys()))
        with g2:
            available_kotas = prov_kota_map[selected_prov]
            selected_kota = st.selectbox("Kota/Kabupaten", available_kotas)

        st.markdown("#### Dimensi & Fasilitas")

        p1, p2 = st.columns(2)
        with p1:
            luas_tanah = st.number_input(
                "Luas Tanah (m²)", min_value=10, max_value=2000, value=120, step=10
            )
            kamar_tidur = st.slider("Kamar Tidur", min_value=1, max_value=15, value=3)
        with p2:
            luas_bangunan = st.number_input(
                "Luas Bangunan (m²)", min_value=10, max_value=2000, value=100, step=10
            )
            kamar_mandi = st.slider("Kamar Mandi", min_value=1, max_value=15, value=2)

        daya_listrik = st.selectbox(
            "Daya Listrik (Watt)",
            [450, 900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 11000, 13200, 16500, 22000, 33000, 41500, 53000],
            index=3,
        )

    # ── Prediction ────────────────────────────────────────────────────────
    with col_result:
        # Build input vector
        input_data = pd.DataFrame(0.0, index=[0], columns=encoded_cols)

        input_data.loc[0, "Luas_Tanah_m2"] = luas_tanah
        input_data.loc[0, "Luas_Bangunan_m2"] = luas_bangunan
        input_data.loc[0, "Kamar_Tidur"] = kamar_tidur
        input_data.loc[0, "Kamar_Mandi"] = kamar_mandi
        input_data.loc[0, "Daya_Listrik_Watt"] = daya_listrik

        scaled_nums = scaler.transform(
            [[luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, daya_listrik]]
        )
        input_data.loc[0, num_cols] = scaled_nums[0]

        prov_col = f"Provinsi_{selected_prov}"
        kota_col = f"Kota_{selected_kota}"
        if prov_col in input_data.columns:
            input_data.loc[0, prov_col] = 1
        if kota_col in input_data.columns:
            input_data.loc[0, kota_col] = 1

        predicted_price = model.predict(input_data)[0]
        if predicted_price < 0:
            predicted_price = 100_000_000

        # ── Result Card ───────────────────────────────────────────────────
        st.markdown('<span class="section-label">Hasil Estimasi</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:36px 28px;">
            <div class="metric-label" style="text-align:center;">Prediksi Harga Pasar</div>
            <div class="metric-value-accent">{fmt_rp(predicted_price)}</div>
            <div class="metric-sub" style="text-align:center; margin-top:12px;">
                {fmt_rp(predicted_price, short=True)} — berdasarkan model MLR
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Local Market Stats ────────────────────────────────────────────
        kota_data = df[df["Kota"] == selected_kota]
        if not kota_data.empty:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Statistik Pasar — {selected_kota}</div>
                <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:14px;">
                    <span class="stat-chip">Listing: {kota_data.shape[0]} unit</span>
                    <span class="stat-chip">Tertinggi: {fmt_rp(kota_data['Harga_Rupiah'].max(), short=True)}</span>
                    <span class="stat-chip">Terendah: {fmt_rp(kota_data['Harga_Rupiah'].min(), short=True)}</span>
                    <span class="stat-chip">Rata-rata: {fmt_rp(kota_data['Harga_Rupiah'].mean(), short=True)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Price distribution mini chart
            fig_mini = px.histogram(
                kota_data,
                x=kota_data["Harga_Rupiah"] / 1e9,
                nbins=20,
                labels={"x": "Harga (Miliar Rp)", "count": "Jumlah"},
                title=f"Distribusi Harga di {selected_kota}",
                color_discrete_sequence=["#818cf8"],
                opacity=0.8,
            )
            fig_mini.update_layout(
                **PLOTLY_LAYOUT,
                height=280,
                bargap=0.08,
                title_font_size=14,
            )
            # Add prediction line
            fig_mini.add_vline(
                x=predicted_price / 1e9,
                line_dash="dash",
                line_color="#f97316",
                line_width=2,
                annotation_text="Prediksi",
                annotation_font_color="#f97316",
                annotation_font_size=11,
            )
            st.plotly_chart(fig_mini, use_container_width=True)
        else:
            st.info(f"Belum tersedia data listing untuk {selected_kota}.")
