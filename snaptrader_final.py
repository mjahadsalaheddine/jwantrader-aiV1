import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# إعداد الواجهة
st.set_page_config(page_title="AI Trader", layout="centered")
st.title("📈 توصيات التداول بالذكاء الاصطناعي - SnapTrader")

# الشريط الجانبي
st.sidebar.title("⚙️ إعدادات")
symbol = st.sidebar.selectbox("اختر الأصل المالي", ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "TSLA"])
start_date = st.sidebar.date_input("📅 من", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("📅 إلى", pd.to_datetime("2024-12-31"))

# تحميل البيانات
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, interval="1d")
    data.dropna(inplace=True)
    return data

data = load_data(symbol, start_date, end_date)

# تحقق من تحميل البيانات بنجاح
if data is None or data.empty or "Close" not in data.columns:
    st.error("❌ فشل تحميل بيانات الأصل المالي المحدد. الرجاء تجربة تاريخ مختلف أو أصل آخر.")
    st.stop()

# المؤشرات الفنية
if data.empty or "Close" not in data.columns or data["Close"].isnull().all():
    st.error("⚠️ البيانات غير متوفرة أو غير صالحة لحساب المؤشرات الفنية.")
    st.stop()
data["rsi"] = RSIIndicator(data["Close"]).rsi()
data["macd"] = MACD(data["Close"]).macd_diff()
bb = BollingerBands(data["Close"])
data["bb_width"] = bb.bollinger_wband()
data["obv"] = OnBalanceVolumeIndicator(data["Close"], data["Volume"]).on_balance_volume()

# توليد الإشارة
data["signal"] = np.where(
    (data["rsi"] < 30) & (data["macd"] > 0), 1,
    np.where((data["rsi"] > 70) & (data["macd"] < 0), -1, 0)
)

# تحضير النموذج
features = ["rsi", "macd", "bb_width", "obv"]
df = data.dropna()
X = df[features]
y = df["signal"]
X = X[y != 0]
y = y[y != 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# نموذج XGBoost
model = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# دقة النموذج
accuracy = accuracy_score(y_test, model.predict(X_test))
st.markdown(f"### 🎯 دقة النموذج: {accuracy * 100:.2f}%")

# توصية اليوم
latest_data = X.iloc[-1:]
prediction = model.predict(latest_data)[0]

if prediction == 1:
    st.success("✅ توصية اليوم: شراء (Buy Signal)")
elif prediction == -1:
    st.error("❌ توصية اليوم: بيع (Sell Signal)")
else:
    st.warning("⏸️ لا توجد توصية واضحة.")

# الشارت الرئيسي
st.subheader("🔍 شارت السعر")
if len(data) >= 100:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index[-100:], data["Close"].tail(100), label="Close Price", color='blue')
    ax.set_title(f"السعر - {symbol}")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("🔎 البيانات غير كافية لعرض الشارت (تحتاج على الأقل 100 يوم من البيانات).")

# إشارات البيع والشراء
st.subheader("📊 إشارات البيع والشراء")
if len(df) >= 100:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    colors = df["signal"].map({1: "green", -1: "red", 0: "gray"})
    ax2.scatter(df.index, df["Close"], c=colors, label="Signals", alpha=0.6)
    ax2.plot(df["Close"], label="Close Price", color='black')
    ax2.set_title("إشارات التداول")
    st.pyplot(fig2)
else:
    st.warning("🚦 البيانات غير كافية لعرض إشارات التداول.")

# عرض البيانات
st.subheader("📄 بيانات التحليل")
st.dataframe(df.tail(5)[features + ["signal"]])

# زر تحميل البيانات
st.download_button("📥 تحميل البيانات كـ CSV", df.to_csv().encode(), file_name=f"{symbol}_data.csv", mime="text/csv")

# توقيع
st.caption("تطبيق تجريبي - SnapTrader AI Scalping V2")
