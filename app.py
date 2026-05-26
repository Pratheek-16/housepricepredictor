import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ── Load model artifacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load("model.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = json.load(open("feature_columns.json"))
    cities  = json.load(open("city_avg.json"))
    zips    = json.load(open("zip_avg.json"))
    return model, scaler, columns, cities, zips

model, scaler, columns, city_avg, zip_avg = load_artifacts()

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#2c3e50;'>🏠 House Price Predictor</h1>
    <p style='text-align:center; color:#7f8c8d; font-size:16px;'>
        King County, Washington — Enter house details to get an instant price estimate
    </p>
    <hr style='border:1px solid #ecf0f1;'>
""", unsafe_allow_html=True)

# ── Layout: 3 columns ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("🛏️ Basic Details")
    bedrooms  = st.slider("Bedrooms",  1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 6,  2, step=1)
    floors    = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0], index=1)
    condition = st.select_slider(
        "Condition",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: {1:"Poor", 2:"Fair", 3:"Average", 4:"Good", 5:"Excellent"}[x]
    )
    view = st.select_slider(
        "View Quality",
        options=[0, 1, 2, 3, 4],
        value=0,
        format_func=lambda x: {0:"None", 1:"Fair", 2:"Average", 3:"Good", 4:"Excellent"}[x]
    )

with col2:
    st.subheader("📐 Size & Structure")
    sqft_living   = st.slider("Living Area (sqft)",   300,  10000, 1800, step=50)
    sqft_lot      = st.slider("Lot Size (sqft)",      500,  50000, 5000, step=100)
    sqft_basement = st.slider("Basement Area (sqft)", 0,    3000,  0,    step=50)
    sqft_above    = sqft_living - sqft_basement
    st.info(f"Above-ground area: **{sqft_above} sqft**")
    waterfront = st.checkbox("🌊 Waterfront Property")

with col3:
    st.subheader("📍 Location & Age")
    city = st.selectbox("City", sorted(city_avg.keys()), index=list(sorted(city_avg.keys())).index("Seattle"))
    zip_code = st.selectbox("ZIP Code", sorted(zip_avg.keys()))
    yr_built = st.slider("Year Built", 1900, 2015, 1995)
    renovated = st.checkbox("🔧 Has been renovated?")
    yr_renovated = 0
    if renovated:
        yr_renovated = st.slider("Year Renovated", yr_built, 2015, min(yr_built + 10, 2015))

# ── Compute derived features ────────────────────────────────────────────────────
sale_year  = 2014
sale_month = 5
house_age  = sale_year - yr_built
effective_reno_year = yr_renovated if yr_renovated > 0 else yr_built
years_since_reno = sale_year - effective_reno_year
total_sqft = sqft_living + sqft_lot
bath_bed_ratio = bathrooms / (bedrooms + 0.1)
city_avg_price = city_avg.get(city, np.mean(list(city_avg.values())))
zip_avg_price  = zip_avg.get(zip_code, np.mean(list(zip_avg.values())))

# ── Build feature row ───────────────────────────────────────────────────────────
house = pd.DataFrame([{
    "bedrooms":          bedrooms,
    "bathrooms":         bathrooms,
    "sqft_living":       sqft_living,
    "sqft_lot":          sqft_lot,
    "floors":            floors,
    "waterfront":        int(waterfront),
    "view":              view,
    "condition":         condition,
    "sqft_above":        sqft_above,
    "sqft_basement":     sqft_basement,
    "yr_built":          yr_built,
    "sale_year":         sale_year,
    "sale_month":        sale_month,
    "house_age":         house_age,
    "years_since_reno":  years_since_reno,
    "total_sqft":        total_sqft,
    "bath_bed_ratio":    bath_bed_ratio,
    "city_avg_price":    city_avg_price,
    "zip_avg_price":     zip_avg_price,
}]).reindex(columns=columns, fill_value=0)

house_scaled = scaler.transform(house)
log_pred     = model.predict(house_scaled)[0]
predicted    = np.expm1(log_pred)

# ── Prediction display ──────────────────────────────────────────────────────────
st.markdown("<hr style='border:1px solid #ecf0f1;'>", unsafe_allow_html=True)

res_col1, res_col2, res_col3, res_col4 = st.columns([1.5, 1, 1, 1])

with res_col1:
    st.markdown(f"""
        <div style='background:linear-gradient(135deg,#2ecc71,#27ae60);
                    border-radius:16px; padding:28px; text-align:center;'>
            <p style='color:white; font-size:15px; margin:0; opacity:0.9;'>Estimated Price</p>
            <h1 style='color:white; font-size:42px; margin:6px 0;'>${predicted:,.0f}</h1>
            <p style='color:white; font-size:13px; margin:0; opacity:0.75;'>Random Forest Model</p>
        </div>
    """, unsafe_allow_html=True)

with res_col2:
    st.metric("Price per sqft", f"${predicted/sqft_living:,.0f}")
    st.metric("City Avg Price", f"${city_avg_price:,.0f}")

with res_col3:
    st.metric("House Age", f"{house_age} yrs")
    st.metric("Total Area", f"{total_sqft:,} sqft")

with res_col4:
    st.metric("Bed / Bath", f"{bedrooms} / {bathrooms}")
    st.metric("Condition", {1:"Poor",2:"Fair",3:"Average",4:"Good",5:"Excellent"}[condition])

# ── Feature summary expander ────────────────────────────────────────────────────
with st.expander("📋 View all input features"):
    display_df = house.copy()
    display_df.insert(0, "Predicted Price ($)", f"{predicted:,.0f}")
    st.dataframe(display_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
    <hr style='border:1px solid #ecf0f1;'>
    <p style='text-align:center; color:#bdc3c7; font-size:13px;'>
        Trained on King County House Sales dataset · Random Forest Regressor · R² ≈ 0.72
    </p>
""", unsafe_allow_html=True)
