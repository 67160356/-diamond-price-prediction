# 💎 Diamond Price Predictor

ทำนายราคาเพชรด้วย Machine Learning (Gradient Boosting Regression)

## 🎯 ปัญหาที่แก้ไข
ราคาเพชรถูกกำหนดจากปัจจัยหลายอย่างที่ซับซ้อน โปรเจคนี้สร้างโมเดลที่ช่วยให้ผู้บริโภคและร้านค้าสามารถประเมินราคาเพชรเบื้องต้นได้

## 📊 Dataset
- **ที่มา:** [Diamonds Dataset - Kaggle](https://www.kaggle.com/datasets/shivam2503/diamonds)
- **จำนวน:** 53,940 แถว
- **Features:** carat, cut, color, clarity, depth, table, x, y, z
- **Target:** price (USD)

## 🚀 Demo App
[🔗 Streamlit App](YOUR_STREAMLIT_URL_HERE)

## 📁 โครงสร้างไฟล์
```
├── diamonds.csv          # Dataset
├── diamond_project.ipynb # Notebook หลัก (EDA + Model)
├── app.py                # Streamlit web app
├── diamond_model.pkl     # โมเดลที่ train แล้ว
├── requirements.txt      # Dependencies
└── README.md
```

## 📈 ผลลัพธ์โมเดล
| Metric | Value |
|--------|-------|
| R² | ~0.98 |
| MAE | ~$350 |
| MAPE | ~6-8% |

## 🛠️ วิธีรันบนเครื่อง
```bash
pip install -r requirements.txt

# Train โมเดล (รัน notebook ก่อน)
jupyter notebook diamond_project.ipynb

# รัน Streamlit app
streamlit run app.py
```

## 🔍 สิ่งที่ค้นพบจาก EDA
1. **Carat** มีผลต่อราคามากที่สุด (feature importance ~60%)
2. Price มี right-skewed distribution → ใช้ log transform
3. x, y, z มี multicollinearity สูง (ล้วนวัดขนาด)
4. Gradient Boosting ทำงานดีกว่า Linear Regression และ Random Forest
