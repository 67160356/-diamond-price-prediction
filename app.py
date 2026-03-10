import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="💎 Diamond Price Predictor",
    page_icon="💎",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    import pickle
    with open("diamond_model_new.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ==========================================
# HEADER
# ==========================================
st.title("💎 Diamond Price Predictor")
st.markdown("ทำนายราคาเพชรจากคุณสมบัติต่างๆ โดยใช้ **Gradient Boosting Regression**")
st.divider()

# ==========================================
# SIDEBAR — INPUT
# ==========================================
st.sidebar.header("🔧 กรอกข้อมูลเพชร")
st.sidebar.markdown("กรอกข้อมูลเพชรที่ต้องการประเมินราคา")

# Carat
carat = st.sidebar.slider(
    "⚖️ Carat (น้ำหนัก)",
    min_value=0.20, max_value=5.01, value=1.00, step=0.01,
    help="น้ำหนักของเพชร 1 carat = 0.2 กรัม"
)

# Cut
cut = st.sidebar.selectbox(
    "✂️ Cut (การเจียระไน)",
    options=["Fair", "Good", "Very Good", "Premium", "Ideal"],
    index=4,
    help="คุณภาพการเจียระไน Ideal = ดีที่สุด"
)

# Color
color = st.sidebar.selectbox(
    "🎨 Color (สี)",
    options=["J", "I", "H", "G", "F", "E", "D"],
    index=3,
    help="สีของเพชร D = ไม่มีสี (ดีที่สุด), J = มีสีเหลืองเล็กน้อย"
)

# Clarity
clarity = st.sidebar.selectbox(
    "🔍 Clarity (ความใส)",
    options=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
    index=3,
    help="ความใสของเพชร IF = Internally Flawless (ดีที่สุด)"
)

st.sidebar.subheader("📐 ขนาดเพชร (mm)")

depth_pct = st.sidebar.slider(
    "Depth %",
    min_value=43.0, max_value=79.0, value=61.5, step=0.1,
    help="ความลึกเป็น % ของเส้นผ่าศูนย์กลาง (ปกติ 56-65%)"
)

table_pct = st.sidebar.slider(
    "Table %",
    min_value=43.0, max_value=95.0, value=57.0, step=1.0,
    help="ความกว้างของหน้าบนสุดเป็น % (ปกติ 53-65%)"
)

x_mm = st.sidebar.number_input("X — ความยาว (mm)", min_value=0.1, max_value=12.0, value=4.50, step=0.01)
y_mm = st.sidebar.number_input("Y — ความกว้าง (mm)", min_value=0.1, max_value=60.0, value=4.52, step=0.01)
z_mm = st.sidebar.number_input("Z — ความลึก (mm)", min_value=0.1, max_value=32.0, value=2.77, step=0.01)

# ==========================================
# INPUT VALIDATION
# ==========================================
warnings = []

if depth_pct < 50 or depth_pct > 75:
    warnings.append(f"⚠️ Depth% = {depth_pct:.1f} อยู่นอกช่วงปกติ (50–75%)")

if table_pct < 50 or table_pct > 80:
    warnings.append(f"⚠️ Table% = {table_pct:.1f} อยู่นอกช่วงปกติ (50–80%)")

if x_mm <= 0 or y_mm <= 0 or z_mm <= 0:
    warnings.append("❌ ขนาด X, Y, Z ต้องมากกว่า 0")

ratio = z_mm / ((x_mm + y_mm) / 2) * 100 if (x_mm + y_mm) > 0 else 0
if abs(ratio - depth_pct) > 5:
    warnings.append(f"⚠️ Depth% ({depth_pct:.1f}) ไม่สอดคล้องกับ Z/(X+Y)/2 ({ratio:.1f}) — อาจเป็นข้อมูลผิดพลาด")

# ==========================================
# PREDICTION
# ==========================================
input_df = pd.DataFrame([{
    'carat': carat,
    'cut': cut,
    'color': color,
    'clarity': clarity,
    'depth': depth_pct,
    'table': table_pct,
    'x': x_mm,
    'y': y_mm,
    'z': z_mm
}])

pred_log = model.predict(input_df)[0]
predicted_price = np.expm1(pred_log)

# Confidence range ±8% (model MAPE ~6-8%)
lower = predicted_price * 0.92
upper = predicted_price * 1.08

# ==========================================
# MAIN CONTENT
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 สรุปข้อมูลเพชรที่กรอก")

    summary_data = {
        "คุณสมบัติ": ["⚖️ Carat", "✂️ Cut", "🎨 Color", "🔍 Clarity",
                     "Depth %", "Table %", "X (mm)", "Y (mm)", "Z (mm)"],
        "ค่า": [f"{carat:.2f}", cut, color, clarity,
               f"{depth_pct:.1f}%", f"{table_pct:.0f}%",
               f"{x_mm:.2f}", f"{y_mm:.2f}", f"{z_mm:.2f}"]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # Warnings
    if warnings:
        st.warning("**ข้อควรระวัง:**\n" + "\n".join(warnings))

with col2:
    st.subheader("💰 ผลการทำนายราคา")

    if not any("❌" in w for w in warnings):
        st.metric(
            label="ราคาที่ทำนาย",
            value=f"${predicted_price:,.0f}",
            help="ราคาเป็น USD"
        )

        st.markdown(f"""
        **ช่วงความเชื่อมั่น (±8%):**  
        `${lower:,.0f}` — `${upper:,.0f}`
        """)

        # Price gauge visualization
        fig, ax = plt.subplots(figsize=(7, 3))
        price_ranges = [
            (0, 1000, '#2ecc71', 'Budget\n(<$1K)'),
            (1000, 5000, '#3498db', 'Mid-range\n($1K-5K)'),
            (5000, 15000, '#9b59b6', 'Premium\n($5K-15K)'),
            (15000, 20000, '#e74c3c', 'Luxury\n(>$15K)')
        ]

        for start, end, color_hex, label in price_ranges:
            ax.barh(0, end - start, left=start, height=0.5,
                    color=color_hex, alpha=0.7)
            ax.text((start + end) / 2, -0.45, label,
                    ha='center', va='top', fontsize=8.5)

        # Marker
        ax.axvline(x=predicted_price, color='black', linewidth=3, linestyle='-')
        ax.text(predicted_price, 0.32, f'${predicted_price:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

        ax.set_xlim(0, 20000)
        ax.set_ylim(-0.8, 0.6)
        ax.axis('off')
        ax.set_title('ระดับราคา', fontsize=11, pad=5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.error("ไม่สามารถทำนายได้ กรุณาแก้ไขข้อมูลที่ผิดพลาดก่อน")

# ==========================================
# FEATURE GUIDE
# ==========================================
st.divider()
st.subheader("📚 คำอธิบาย Features")

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("""
    **✂️ Cut (การเจียระไน)**
    | Grade | ความหมาย |
    |-------|---------|
    | Ideal | เจียระไนสมบูรณ์แบบ |
    | Premium | คุณภาพสูง |
    | Very Good | ดีมาก |
    | Good | ดี |
    | Fair | พอใช้ |
    """)

with col4:
    st.markdown("""
    **🎨 Color (สี)**
    | Grade | ความหมาย |
    |-------|---------|
    | D | ไม่มีสี (ดีที่สุด) |
    | E–F | เกือบไม่มีสี |
    | G–H | สีจางมาก |
    | I–J | มีสีเหลืองเล็กน้อย |
    """)

with col5:
    st.markdown("""
    **🔍 Clarity (ความใส)**
    | Grade | ความหมาย |
    |-------|---------|
    | IF | ไม่มีตำหนิ |
    | VVS1–VVS2 | ตำหนิน้อยมาก |
    | VS1–VS2 | ตำหนิน้อย |
    | SI1–SI2 | ตำหนิเล็กน้อย |
    | I1 | มีตำหนิ |
    """)

# ==========================================
# DISCLAIMER
# ==========================================
st.divider()
st.info("""
⚠️ **Disclaimer:** ราคาที่แสดงเป็นเพียงการประมาณจากโมเดล Machine Learning เท่านั้น 
ราคาจริงขึ้นอยู่กับปัจจัยอื่นๆ เช่น ยี่ห้อ ใบเซอร์ GIA ภาษี และสภาพตลาด 
**ไม่ควรใช้เป็นราคาอ้างอิงในการซื้อขายจริง**

🤖 โมเดล: Gradient Boosting Regressor | R² ≈ 0.98 | MAPE ≈ 6-8%  
📊 ข้อมูล: Diamonds Dataset, Kaggle (53,940 ตัวอย่าง)
""")
