import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

# Set page configuration
st.set_page_config(
    page_title="Rumah99 Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design and premium look
st.markdown("""
    <style>
        .reportview-container {
            background: #0a0a0f;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #e1e1e6;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0a0a0f;
        }
        ::-webkit-scrollbar-thumb {
            background: #2a2a3a;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3a3a4a;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        
        .main-header {
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(135deg, #00B4DB, #0083B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 25px;
            text-shadow: 0 4px 20px rgba(0, 180, 219, 0.3);
        }
        
        .subheader {
            font-size: 18px;
            color: #a0a0b0;
            margin-bottom: 35px;
            font-weight: 400;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #1e222b, #171a23);
            padding: 28px;
            border-radius: 16px;
            border: 1px solid rgba(0, 180, 219, 0.1);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            margin-bottom: 25px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, #00B4DB, #0083B0);
            transition: width 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0, 180, 219, 0.15);
            border-color: rgba(0, 180, 219, 0.3);
        }
        
        .metric-card:hover::before {
            width: 8px;
        }
        
        .prediction-title {
            font-size: 14px;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        
        .prediction-value {
            font-size: 42px;
            font-weight: 800;
            color: #00B4DB;
            margin: 12px 0;
            background: linear-gradient(135deg, #00B4DB, #0083B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: #0a0a0f;
            padding: 20px 15px;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .sidebar h2 {
            color: #00B4DB !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            margin-bottom: 20px !important;
            letter-spacing: 1px;
        }
        
        .sidebar .sidebar-radio > label > div {
            background: #1e222b;
            border-radius: 10px;
            padding: 12px 15px;
            margin-bottom: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }
        
        .sidebar .sidebar-radio > label > div:hover {
            background: rgba(0, 180, 219, 0.1);
            border-color: rgba(0, 180, 219, 0.3);
            transform: translateX(5px);
        }
        
        .sidebar .sidebar-radio > label > div[data-baseweb="radio"] > div:first-child {
            margin-right: 12px;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            margin-right: 5px;
            font-weight: 600;
            color: #a0a0b0 !important;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(0, 180, 219, 0.1);
            color: #00B4DB !important;
        }
        
        .stTabs [aria-selected="true"] [data-baseweb="tab"] {
            background-color: rgba(0, 180, 219, 0.15) !important;
            color: #00B4DB !important;
            border-bottom: 3px solid #00B4DB !important;
        }
        
        /* Input fields */
        .stSelectbox > div > div {
            background-color: #1e222b;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 8px 12px;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div:hover {
            border-color: rgba(0, 180, 219, 0.3);
        }
        
        .stNumberInput > div > div > input, .stSlider > div > div > div > input {
            background-color: #1e222b;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 10px 15px;
            transition: all 0.3s ease;
            color: white;
        }
        
        .stSlider > div > div > div > div {
            background-color: #00B4DB !important;
        }
        
        /* Tables */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .stDataFrame table {
            background-color: #1e222b;
        }
        
        .stDataFrame th {
            background-color: #171a23;
            color: #00B4DB;
            font-weight: 600;
            padding: 15px;
        }
        
        .stDataFrame td {
            padding: 12px 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Cards in stats section */
        .stMetric {
            background-color: #1e222b;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stMetric:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 180, 219, 0.1);
            border-color: rgba(0, 180, 219, 0.2);
        }
        
        .stMetric label {
            color: #8892b0;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stMetric [data-testid="metric-container"] {
            color: #00B4DB !important;
            font-size: 28px !important;
            font-weight: 700 !important;
        }
        
        /* Container spacing */
        .element-container {
            margin-bottom: 25px;
        }
        
        /* Chart containers */
        .js-plotly-plot, .stPlotlyChart {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            background: #1e222b;
            padding: 15px;
        }
        
        /* Dark theme text colors */
        p, div, span, li, td, th {
            color: #e1e1e6;
        }
        
        a {
            color: #00B4DB;
            text-decoration: none;
            transition: all 0.3s ease;
        }
        
        a:hover {
            color: #0083B0;
            text-decoration: underline;
        }
        
        /* File upload and other inputs */
        .stFileUploader > div {
            background-color: #1e222b;
            border-radius: 12px;
            border: 2px dashed rgba(0, 180, 219, 0.3);
            padding: 30px;
            transition: all 0.3s ease;
        }
        
        .stFileUploader > div:hover {
            border-color: rgba(0, 180, 219, 0.5);
            background-color: rgba(0, 180, 219, 0.05);
        }
        
        /* Descriptions and help text */
        .element-container .stAlert {
            border-radius: 10px;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Cache data loading and model fitting
@st.cache_resource
def load_data_and_model():
    clean_path = 'datasets/dataset_rumah99_clean.csv'
    if not os.path.exists(clean_path):
        # Fallback if cleaner hasn't run in context
        raw_path = 'datasets/dataset_rumah99_raw.csv'
        df = pd.read_csv(raw_path).dropna(subset=['Harga_Rupiah', 'Kamar_Tidur', 'Kamar_Mandi', 'Luas_Tanah_m2', 'Luas_Bangunan_m2', 'Jumlah_Lantai', 'Kota'])
        df = df[(df['Kamar_Tidur'] > 0) & (df['Kamar_Mandi'] > 0) & (df['Daya_Listrik_Watt'] > 0)]
        df = df[
            (df['Harga_Rupiah'] >= 100_000_000) & (df['Harga_Rupiah'] <= 50_000_000_000) &
            (df['Luas_Tanah_m2'] <= 2000) & (df['Luas_Bangunan_m2'] <= 2000) &
            (df['Kamar_Tidur'] <= 15) & (df['Kamar_Mandi'] <= 15)
        ]
    else:
        df = pd.read_csv(clean_path)
        
    # Fit the Multiple Linear Regression model
    X_features = ['Luas_Tanah_m2', 'Luas_Bangunan_m2', 'Kamar_Tidur', 'Kamar_Mandi', 'Daya_Listrik_Watt', 'Provinsi', 'Kota']
    X_raw = df[X_features]
    Y = df['Harga_Rupiah']
    
    # One-Hot Encoding
    X_encoded = pd.get_dummies(X_raw, columns=['Provinsi', 'Kota'], drop_first=True)
    dummy_cols = [c for c in X_encoded.columns if c not in ['Luas_Tanah_m2', 'Luas_Bangunan_m2', 'Kamar_Tidur', 'Kamar_Mandi', 'Daya_Listrik_Watt']]
    X_encoded[dummy_cols] = X_encoded[dummy_cols].astype(int)
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)
    
    # Standard Scaling
    num_cols = ['Luas_Tanah_m2', 'Luas_Bangunan_m2', 'Kamar_Tidur', 'Kamar_Mandi', 'Daya_Listrik_Watt']
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    
    # Fit Model
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)
    
    # Calculate training score
    r2_score = model.score(X_train_scaled, Y_train)
    
    # Geolocation mappings for forms
    # Create mapping of Provinsi -> list of unique Kotas
    prov_kota_map = df.groupby('Provinsi')['Kota'].unique().to_dict()
    # Sort keys and values
    prov_kota_map = {k: sorted(list(v)) for k, v in sorted(prov_kota_map.items())}
    
    return df, model, scaler, X_encoded.columns.tolist(), num_cols, prov_kota_map, r2_score

# Load data and models
df, model, scaler, encoded_cols, num_cols, prov_kota_map, r2_score = load_data_and_model()

# Sidebar Navigation
st.sidebar.markdown("<h2 style='color:#00B4DB;'>🧭 Navigasi</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Pilih Halaman:", ["📈 Ringkasan & EDA", "🏠 Kalkulator Prediksi Harga"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Detail Proyek Akhir:**
*   **Metode**: Regresi Linier Berganda
*   **Target ($Y$)**: Harga Properti (Rupiah)
*   **Jumlah Sampel**: 1.345 Baris Data Bersih
*   **Model $R^2$**: {:.2f}%
""".format(r2_score * 100))

# ----------------- PAGE 1: OVERVIEW & EDA -----------------
if page == "Ringkasan & EDA":
    st.markdown("<h1 class='main-header'>Ringkasan Data & Analisis Eksploratif</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Visualisasi karakteristik fisik dan geografis listing rumah tinggal</p>", unsafe_allow_html=True)
    
    # Key Metrics Rows
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="prediction-title">Total Data Bersih</div>
            <div style="font-size: 32px; font-weight: 700; color: #e1e1e6;">{df.shape[0]} Baris</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="prediction-title">Rata-rata Harga</div>
            <div style="font-size: 32px; font-weight: 700; color: #e1e1e6;">Rp {df['Harga_Rupiah'].mean()/1e9:.2f} Miliar</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="prediction-title">Rata-rata Luas Tanah</div>
            <div style="font-size: 32px; font-weight: 700; color: #e1e1e6;">{df['Luas_Tanah_m2'].mean():.1f} m²</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="prediction-title">Rata-rata Luas Bangunan</div>
            <div style="font-size: 32px; font-weight: 700; color: #e1e1e6;">{df['Luas_Bangunan_m2'].mean():.1f} m²</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("### Sampel Data Properti")
    st.dataframe(df[['Judul', 'Harga_Rupiah', 'Provinsi', 'Kota', 'Luas_Tanah_m2', 'Luas_Bangunan_m2', 'Kamar_Tidur', 'Kamar_Mandi', 'Daya_Listrik_Watt']].head(10), use_container_width=True)
    
    # Visualizations
    st.write("### Visualisasi Eksperimen Proyek")
    tab1, tab2, tab3 = st.tabs(["📊 Distribusi Harga", "🌡️ Korelasi", "📈 Hubungan Fisik & Geografis"])
    
    with tab1:
        st.write("#### Distribusi Harga Rumah Properti (Rupiah)")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['Harga_Rupiah'] / 1e9, kde=True, color='teal', ax=ax)
        ax.set_xlabel('Harga Rumah (Miliar Rupiah)')
        ax.set_ylabel('Frekuensi / Jumlah Rumah')
        ax.set_title('Sebaran Harga Rumah Bersih')
        st.pyplot(fig)
        
    with tab2:
        st.write("#### Matriks Korelasi Pearson")
        fig, ax = plt.subplots(figsize=(8, 5))
        numeric_df = df[['Harga_Rupiah', 'Luas_Tanah_m2', 'Luas_Bangunan_m2', 'Kamar_Tidur', 'Kamar_Mandi', 'Daya_Listrik_Watt', 'Jumlah_Lantai']]
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, linewidths=0.5)
        ax.set_title('Korelasi Antara Variabel Fisik vs Harga')
        st.pyplot(fig)
        
    with tab3:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write("#### Hubungan Luas Bangunan vs Harga Rumah")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df, x='Luas_Bangunan_m2', y=df['Harga_Rupiah']/1e9, alpha=0.5, color='forestgreen', ax=ax)
            ax.set_xlabel('Luas Bangunan (m²)')
            ax.set_ylabel('Harga Rumah (Miliar Rupiah)')
            st.pyplot(fig)
        with col_t2:
            st.write("#### Rata-rata Harga Rumah per Provinsi")
            top_prov = df['Provinsi'].value_counts().head(5).index
            df_top = df[df['Provinsi'].isin(top_prov)]
            avg_price = df_top.groupby('Provinsi')['Harga_Rupiah'].mean().sort_values(ascending=False) / 1e9
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=avg_price.values, y=avg_price.index, palette='crest', ax=ax)
            ax.set_xlabel('Rata-rata Harga (Miliar Rupiah)')
            st.pyplot(fig)

# ----------------- PAGE 2: PREDICTION CALCULATOR -----------------
elif page == "Kalkulator Prediksi Harga":
    st.markdown("<h1 class='main-header'>Kalkulator Prediksi Harga Rumah</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Masukkan spesifikasi rumah untuk mengestimasi harga pasar menggunakan Regresi Linier Berganda</p>", unsafe_allow_html=True)
    
    # Split input form and prediction result
    col_input, col_result = st.columns([1.5, 1])
    
    with col_input:
        st.write("### Spesifikasi Rumah")
        
        # 1. Geolocation Fields
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            selected_prov = st.selectbox("Pilih Provinsi:", list(prov_kota_map.keys()))
        with col_g2:
            available_kotas = prov_kota_map[selected_prov]
            selected_kota = st.selectbox("Pilih Kota:", available_kotas)
            
        # 2. Physical Characteristics
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            luas_tanah = st.number_input("Luas Tanah (m²):", min_value=10, max_value=2000, value=120, step=10)
            luas_bangunan = st.number_input("Luas Bangunan (m²):", min_value=10, max_value=2000, value=100, step=10)
        with col_p2:
            kamar_tidur = st.slider("Jumlah Kamar Tidur:", min_value=1, max_value=15, value=3)
            kamar_mandi = st.slider("Jumlah Kamar Mandi:", min_value=1, max_value=15, value=2)
            
        daya_listrik = st.selectbox("Kapasitas Listrik (Watt):", [450, 900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 11000, 13200, 16500, 22000, 33000, 41500, 53000], index=3)
        
    with col_result:
        st.write("### Estimasi Harga Pasar")
        
        # Trigger prediction logic
        # 1. Build input vector matching one-hot encoded columns
        input_data = pd.DataFrame(0.0, index=[0], columns=encoded_cols)
        
        # Fill numeric values
        input_data.loc[0, 'Luas_Tanah_m2'] = luas_tanah
        input_data.loc[0, 'Luas_Bangunan_m2'] = luas_bangunan
        input_data.loc[0, 'Kamar_Tidur'] = kamar_tidur
        input_data.loc[0, 'Kamar_Mandi'] = kamar_mandi
        input_data.loc[0, 'Daya_Listrik_Watt'] = daya_listrik
        
        # Scale numeric features
        scaled_nums = scaler.transform([[luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, daya_listrik]])
        input_data.loc[0, num_cols] = scaled_nums[0]
        
        # Fill dummy variables
        prov_col = f"Provinsi_{selected_prov}"
        kota_col = f"Kota_{selected_kota}"
        
        if prov_col in input_data.columns:
            input_data.loc[0, prov_col] = 1
        if kota_col in input_data.columns:
            input_data.loc[0, kota_col] = 1
            
        # Predict
        predicted_price = model.predict(input_data)[0]
        
        # Ensure price is not negative (boundary constraint)
        if predicted_price < 0:
            predicted_price = 100_000_000 # Floor value
            
        # Render clean result card
        st.markdown(f"""
        <div class="metric-card">
            <div class="prediction-title">Prediksi Harga Rumah</div>
            <div class="prediction-value">Rp {predicted_price:,.0f}</div>
            <div style="font-size: 14px; color: #8892b0; margin-top: 10px;">
                Estimasi didasarkan pada model Regresi Linier Berganda. Rata-rata harga rumah di <b>{selected_kota}, {selected_prov}</b> adalah sekitar 
                <b>Rp {df[df['Kota'] == selected_kota]['Harga_Rupiah'].mean()/1e9:.2f} Miliar</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional comparison with database averages
        kota_data = df[df['Kota'] == selected_kota]
        if not kota_data.empty:
            st.write("#### Statistik Pasar Lokal:")
            st.write(f"- **Harga Tertinggi di {selected_kota}**: Rp {kota_data['Harga_Rupiah'].max()/1e9:.2f} Miliar")
            st.write(f"- **Harga Terendah di {selected_kota}**: Rp {kota_data['Harga_Rupiah'].min()/1e6:.1f} Juta")
            st.write(f"- **Jumlah Listing di {selected_kota}**: {kota_data.shape[0]} unit")
