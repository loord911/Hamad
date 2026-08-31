import os, json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="US Stock AI Agent Desk", page_icon="📈", layout="wide")
st.title("📈 US Stock AI Agent Desk")
st.caption("نسخة تحليلية: وكلاء فنيون + مدير قرار. لا يرسل أوامر تداول.")

@st.cache_data(ttl=300)
def load_data(ticker, period="6mo", interval="1d"):
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=True, progress=False
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    x = df.copy()
    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()
    # SMA200 is optional: 6 months normally has fewer than 200 sessions.
    x["SMA200"] = x["Close"].rolling(200).mean()
    x["RSI14"] = rsi(x["Close"])
    x["ATR14"] = atr(x)
    x["ATR_PCT"] = x["ATR14"] / x["Close"] * 100
    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / x["VOL20"]
    fast = x["Close"].ewm(span=12, adjust=False).mean()
    slow = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = fast - slow
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]
    x["RET5"] = x["Close"].pct_change(5) * 100
    x["RET20"] = x["Close"].pct_change(20) * 100

    required = [
        "Close","High","Low","Open","Volume","SMA20","SMA50",
        "RSI14","ATR14","ATR_PCT","VOL20","VOL_RATIO",
        "MACD","MACD_SIGNAL","MACD_HIST","RET5","RET20"
    ]
    return x.dropna(subset=required)

def clamp(v, lo=-100, hi=100):
    return float(max(lo, min(hi, v)))

def agents(x):
    r = x.iloc[-1]
    out = []

    score = 30 if r.Close > r.SMA20 else -30
    score += 30 if r.SMA20 > r.SMA50 else -30
    if pd.notna(r.SMA200):
        score += 30 if r.SMA50 > r.SMA200 else -30
    score += 10 if r.RET20 > 0 else -10
    out.append(("Trend Agent", score, f"Close {r.Close:.2f}; SMA20 {r.SMA20:.2f}; SMA50 {r.SMA50:.2f}"))

    score = clamp((r.RSI14 - 50) * 1.4) + (20 if r.RET5 > 0 else -20)
    out.append(("Momentum Agent", clamp(score), f"RSI {r.RSI14:.1f}; 5D return {r.RET5:.2f}%"))

    score = (70 if r.MACD > r.MACD_SIGNAL else -70) + (20 if r.MACD_HIST > 0 else -20)
    out.append(("MACD Agent", clamp(score), f"MACD {r.MACD:.3f}; signal {r.MACD_SIGNAL:.3f}"))

    score = 50 if r.VOL_RATIO >= 1.2 and r.RET5 > 0 else (-50 if r.VOL_RATIO >= 1.2 and r.RET5 < 0 else 0)
    out.append(("Volume Agent", score, f"Volume ratio {r.VOL_RATIO:.2f}x 20D average"))

    score = 15 if r.ATR_PCT < 3 else (0 if r.ATR_PCT < 6 else -15)
    out.append(("Volatility Agent", score, f"ATR {r.ATR_PCT:.2f}% of price"))

    body = abs(r.Close-r.Open)
    rng = max(r.High-r.Low, 1e-9)
    loc = (r.Close-r.Low)/rng
    score = 60 if loc > .7 and r.Close > r.Open else (-60 if loc < .3 and r.Close < r.Open else 0)
    out.append(("Price Action Agent", score, f"Close location {loc:.2f}; body/range {body/rng:.2f}"))

    prior_high = x["High"].iloc[-21:-1].max()
    prior_low = x["Low"].iloc[-21:-1].min()
    if r.Close > prior_high:
        out.append(("Breakout Agent", 85, f"Breakout above 20D high {prior_high:.2f}"))
    elif r.Close < prior_low:
        out.append(("Breakout Agent", -85, f"Breakdown below 20D low {prior_low:.2f}"))
    else:
        out.append(("Breakout Agent", 0, f"Inside 20D range {prior_low:.2f}-{prior_high:.2f}"))

    lo = x["Low"].tail(60).min()
    hi = x["High"].tail(60).max()
    dl = (r.Close-lo)/r.Close
    dh = (hi-r.Close)/r.Close
    if dl < .025:
        out.append(("Support/Resistance Agent", 55, f"Near support {lo:.2f}"))
    elif dh < .025:
        out.append(("Support/Resistance Agent", -55, f"Near resistance {hi:.2f}"))
    else:
        out.append(("Support/Resistance Agent", 0, f"Support {lo:.2f}; resistance {hi:.2f}"))

    std20 = x["Close"].rolling(20).std().iloc[-1]
    z = (r.Close-r.SMA20) / (std20 if pd.notna(std20) and std20 != 0 else 1)
    out.append(("Mean Reversion Agent", clamp(-z*35), f"20D z-score {z:.2f}"))

    score = 30 if r.SMA20 > r.SMA50 and r.RET20 > 0 else (-30 if r.SMA20 < r.SMA50 and r.RET20 < 0 else 0)
    regime = "bullish" if score > 0 else ("bearish" if score < 0 else "mixed")
    out.append(("Market Regime Agent", score, f"Regime: {regime}"))

    score = 20 if r.ATR_PCT < 4 else (0 if r.ATR_PCT < 7 else -30)
    out.append(("Risk Agent", score, f"ATR% {r.ATR_PCT:.2f}"))

    dist = (r.Close-r.SMA20)/r.SMA20*100
    out.append(("Overextension Agent", -clamp(dist*12), f"Distance from SMA20 {dist:.2f}%"))

    out.append(("Relative Strength Agent", clamp(r.RET20*5), f"20D return {r.RET20:.2f}%"))
    out.append(("Data Quality Agent", 25, "Data quality acceptable"))

    return out

ticker = st.text_input("رمز السهم الأمريكي", value="AAPL").strip().upper()
period = st.selectbox("الفترة", ["3mo","6mo","1y","2y"], index=1)

if st.button("🔎 حلّل السهم", type="primary"):
    try:
        raw = load_data(ticker, period)
        if raw.empty:
            st.error("لم أستطع جلب بيانات السهم. تحقق من الرمز.")
            st.stop()

        x = add_indicators(raw)

        # FIX: do not require SMA200 for a 6-month analysis.
        if len(x) < 40:
            st.error(f"البيانات غير كافية بعد تجهيز المؤشرات ({len(x)} شمعة). اختر 1y أو 2y.")
            st.stop()

        results = agents(x)
        rdf = pd.DataFrame(results, columns=["Agent","Score","Reason"])

        weighted = float(rdf["Score"].mean())
        bullish = int((rdf["Score"] > 20).sum())
        bearish = int((rdf["Score"] < -20).sum())
        agreement = float((rdf["Score"].abs() > 20).mean() * 100)

        if weighted >= 25:
            decision = "LONG BIAS"
        elif weighted <= -25:
            decision = "SHORT BIAS"
        else:
            decision = "NO TRADE"

        r = x.iloc[-1]
        entry = float(r.Close)
        atrv = float(r.ATR14)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("السعر", f"${entry:.2f}")
        c2.metric("القرار", decision)
        c3.metric("متوسط الوكلاء", f"{weighted:.1f}/100")
        c4.metric("توافق الإشارات", f"{agreement:.0f}%")

        if decision == "LONG BIAS":
            stop = entry - 1.5*atrv
            tp1 = entry + 1.5*atrv
            tp2 = entry + 3*atrv
            st.info(f"دخول مرجعي: ${entry:.2f} | وقف: ${stop:.2f} | TP1: ${tp1:.2f} | TP2: ${tp2:.2f}")
        elif decision == "SHORT BIAS":
            stop = entry + 1.5*atrv
            tp1 = entry - 1.5*atrv
            tp2 = entry - 3*atrv
            st.info(f"دخول مرجعي: ${entry:.2f} | وقف: ${stop:.2f} | TP1: ${tp1:.2f} | TP2: ${tp2:.2f}")
        else:
            st.warning("النظام لا يرى أفضلية كافية للدخول حالياً.")

        st.subheader("نتائج الوكلاء")
        st.dataframe(rdf, use_container_width=True, hide_index=True)

        st.subheader("السعر والمؤشرات")
        cols = ["Close","SMA20","SMA50"]
        if x["SMA200"].notna().any():
            cols.append("SMA200")
        st.line_chart(x[cols].tail(120))

        st.subheader("ملخص القرار")
        st.write({
            "ticker": ticker,
            "last_bar": str(x.index[-1]),
            "decision": decision,
            "bullish_agents": bullish,
            "bearish_agents": bearish,
            "mean_score": round(weighted,1),
            "agreement_pct": round(agreement,1),
            "RSI14": round(float(r.RSI14),1),
            "ATR%": round(float(r.ATR_PCT),2),
            "volume_ratio": round(float(r.VOL_RATIO),2),
        })

        st.caption("نموذج بحثي/تحليلي وليس توصية استثمارية. اختبره تاريخياً واحتسب العمولات والانزلاق قبل الاعتماد عليه.")

    except Exception as e:
        st.exception(e)
else:
    st.markdown("""
### ماذا يفعل النظام؟
1. يجلب بيانات السهم من Yahoo Finance.
2. يحسب الاتجاه والزخم وMACD والحجم والتذبذب والدعم والمقاومة وغيرها.
3. كل وكيل يعطي درجة من -100 إلى +100.
4. يجمع النظام الدرجات في قرار LONG BIAS / SHORT BIAS / NO TRADE.
5. يحسب وقفاً وأهدافاً مرجعية باستخدام ATR.

**الإصلاح المهم:** فترة 6mo لم تعد تفشل بسبب SMA200؛ SMA200 أصبح اختيارياً، لذلك FIGR وغيره سيُحلّل بالبيانات المتاحة.
""")
