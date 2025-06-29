import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Trader", layout="centered")
st.title("📈 توصيات التداول بالذكاء الاصطناعي")

suggested_assets = {
    "البيتكوين (BTC)": "BTC-USD",
    "الإيثريوم (ETH)": "ETH-USD",
    "الذهب (XAU/USD)": "XAUUSD=X",
    "سهم آبل (AAPL)": "AAPL",
    "سهم تيسلا (TSLA)": "TSLA"
}

st.sidebar.header("⚙️ إعدادات")
asset_label = st.sidebar.selectbox("اختر الأصل المالي", list(suggested_assets.keys()))
symbol = suggested_assets[asset_label]
start_date = st.sidebar.date_input("من", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("إلى", pd.to_datetime("2024-12-31"))

@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, interval="1d")
    data.dropna(inplace=True)
    return data

data = load_data(symbol, start_date, end_date)

if data.empty:
    st.error("⚠️ لا توجد بيانات متاحة في الفترة الزمنية المحددة أو للأصل المالي المحدد.")
    st.stop()

data["rsi"] = RSIIndicator(data["Close"]).rsi()
data["macd"] = MACD(data["Close"]).macd_diff()
bb = BollingerBands(data["Close"])
data["bb_width"] = bb.bollinger_wband()
data["obv"] = OnBalanceVolumeIndicator(data["Close"], data["Volume"]).on_balance_volume()

data["signal"] = np.where((data["rsi"] < 30) & (data["macd"] > 0), 1,
                  np.where((data["rsi"] > 70) & (data["macd"] < 0), -1, 0))

features = ["rsi", "macd", "bb_width", "obv"]
df = data.dropna()
X = df[features]
y = df["signal"]
X = X[y != 0]
y = y[y != 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.markdown(f"### 🎯 دقة النموذج: `{accuracy * 100:.2f}%`")

latest_data = X.iloc[-1:]
prediction = model.predict(latest_data)[0]
if prediction == 1:
    st.success("✅ توصية اليوم: شراء (Buy Signal)")
elif prediction == -1:
    st.error("❌ توصية اليوم: بيع (Sell Signal)")
else:
    st.warning("⏸️ لا توجد توصية واضحة.")

st.subheader("🔍 شارت السعر")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.index[-100:], data["Close"].tail(100), label="Close Price", color='blue')
ax.set_title(f"السعر - {symbol}")
ax.legend()
st.pyplot(fig)

st.caption("تطبيق تجريبي - AI Scalping V2")
