
import os, json, math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Optional LLM layer
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="US Stock AI Agent Desk", page_icon="📈", layout="wide")

st.title("📈 US Stock AI Agent Desk")
st.caption("نسخة أولية: 15 وكيل متخصص + مدير قرار. النظام تحليلي فقط ولا يرسل أوامر تداول.")

# ---------- Data ----------
@st.cache_data(ttl=300)
def load_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

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

def macd(close):
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line-signal

def add_indicators(df):
    x = df.copy()
    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()
    x["SMA200"] = x["Close"].rolling(200).mean()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["RSI14"] = rsi(x["Close"])
    x["ATR14"] = atr(x)
    x["ATR_PCT"] = x["ATR14"] / x["Close"] * 100
    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / x["VOL20"]
    m, s, h = macd(x["Close"])
    x["MACD"] = m
    x["MACD_SIGNAL"] = s
    x["MACD_HIST"] = h
    x["RET5"] = x["Close"].pct_change(5) * 100
    x["RET20"] = x["Close"].pct_change(20) * 100
    return x.dropna()

def clamp(v, lo=-100, hi=100):
    return float(max(lo, min(hi, v)))

# ---------- Deterministic specialist agents ----------
def agent_trend(x):
    r = x.iloc[-1]
    score = 0
    score += 30 if r.Close > r.SMA20 else -30
    score += 30 if r.SMA20 > r.SMA50 else -30
    if not pd.isna(r.SMA200):
        score += 30 if r.SMA50 > r.SMA200 else -30
    score += 10 if r.RET20 > 0 else -10
    return score, f"Close {r.Close:.2f}; SMA20 {r.SMA20:.2f}; SMA50 {r.SMA50:.2f}"

def agent_momentum(x):
    r=x.iloc[-1]
    score = clamp((r.RSI14-50)*1.4)
    if r.RET5 > 0: score += 20
    else: score -= 20
    return clamp(score), f"RSI {r.RSI14:.1f}; 5D return {r.RET5:.2f}%"

def agent_macd(x):
    r=x.iloc[-1]
    score = 70 if r.MACD > r.MACD_SIGNAL else -70
    score += 20 if r.MACD_HIST > 0 else -20
    return clamp(score), f"MACD {r.MACD:.3f}; signal {r.MACD_SIGNAL:.3f}"

def agent_volume(x):
    r=x.iloc[-1]
    score = 50 if r.VOL_RATIO >= 1.2 and r.RET5 > 0 else 0
    score += -50 if r.VOL_RATIO >= 1.2 and r.RET5 < 0 else 0
    return score, f"Volume ratio {r.VOL_RATIO:.2f}x 20D average"

def agent_volatility(x):
    r=x.iloc[-1]
    # High volatility reduces confidence rather than directly predicting direction.
    score = 15 if r.ATR_PCT < 3 else (0 if r.ATR_PCT < 6 else -15)
    return score, f"ATR {r.ATR_PCT:.2f}% of price"

def agent_price_action(x):
    r=x.iloc[-1]
    body = abs(r.Close-r.Open)
    rng = max(r.High-r.Low, 1e-9)
    close_location = (r.Close-r.Low)/rng
    score = 60 if close_location > .7 and r.Close > r.Open else -60 if close_location < .3 and r.Close < r.Open else 0
    return score, f"Close location {close_location:.2f}; body/range {body/rng:.2f}"

def agent_breakout(x):
    r=x.iloc[-1]
    prior_high=x["High"].iloc[-21:-1].max()
    prior_low=x["Low"].iloc[-21:-1].min()
    if r.Close > prior_high: score=85; msg=f"Breakout above 20D high {prior_high:.2f}"
    elif r.Close < prior_low: score=-85; msg=f"Breakdown below 20D low {prior_low:.2f}"
    else: score=0; msg=f"Inside 20D range {prior_low:.2f}-{prior_high:.2f}"
    return score,msg

def agent_support_resistance(x):
    r=x.iloc[-1]
    lo=x["Low"].tail(60).min()
    hi=x["High"].tail(60).max()
    d_lo=(r.Close-lo)/r.Close
    d_hi=(hi-r.Close)/r.Close
    if d_lo < .025: score=55; msg=f"Near support {lo:.2f}"
    elif d_hi < .025: score=-55; msg=f"Near resistance {hi:.2f}"
    else: score=0; msg=f"Range support {lo:.2f}, resistance {hi:.2f}"
    return score,msg

def agent_mean_reversion(x):
    r=x.iloc[-1]
    z=(r.Close-r.SMA20)/(x["Close"].rolling(20).std().iloc[-1] or 1)
    score=clamp(-z*35)
    return score,f"20D z-score {z:.2f}"

def agent_regime(x):
    r=x.iloc[-1]
    score=30 if r.SMA20 > r.SMA50 and r.RET20 > 0 else -30 if r.SMA20 < r.SMA50 and r.RET20 < 0 else 0
    return score,f"Regime: {'bullish' if score>0 else 'bearish' if score<0 else 'mixed'}"

def agent_risk(x):
    r=x.iloc[-1]
    # Risk agent penalizes excessive volatility and gives a small positive score to controlled volatility.
    score=20 if r.ATR_PCT < 4 else 0 if r.ATR_PCT < 7 else -30
    return score,f"ATR% {r.ATR_PCT:.2f}; risk {'controlled' if score>0 else 'elevated' if score<0 else 'moderate'}"

def agent_overextension(x):
    r=x.iloc[-1]
    dist=(r.Close-r.SMA20)/r.SMA20*100
    score=-clamp(dist*12)
    return score,f"Distance from SMA20 {dist:.2f}%"

def agent_relative_strength(x):
    r=x.iloc[-1]
    score=clamp(r.RET20*5)
    return score,f"20D return {r.RET20:.2f}%"

def agent_quality_gate(x):
    r=x.iloc[-1]
    ok = r.Volume > 0 and not any(pd.isna(r[c]) for c in ["SMA20","SMA50","RSI14","ATR14"])
    return (25 if ok else -80), ("Data quality acceptable" if ok else "Insufficient data")

AGENTS = [
    ("Trend Agent", agent_trend),
    ("Momentum Agent", agent_momentum),
    ("MACD Agent", agent_macd),
    ("Volume Agent", agent_volume),
    ("Volatility Agent", agent_volatility),
    ("Price Action Agent", agent_price_action),
    ("Breakout Agent", agent_breakout),
    ("Support/Resistance Agent", agent_support_resistance),
    ("Mean Reversion Agent", agent_mean_reversion),
    ("Market Regime Agent", agent_regime),
    ("Risk Agent", agent_risk),
    ("Overextension Agent", agent_overextension),
    ("Relative Strength Agent", agent_relative_strength),
    ("Data Quality Agent", agent_quality_gate),
]

# ---------- LLM agents ----------
def llm_call(client, system, payload):
    resp = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        input=[
            {"role":"system","content":system},
            {"role":"user","content":payload}
        ],
    )
    return resp.output_text

def run_llm_panel(ticker, snapshot, news_text=""):
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return []
    client=OpenAI()
    specs=[
        ("LLM Technical Reviewer","Review technical evidence only. Return a score from -100 to +100 and explain contradictions."),
        ("LLM Risk Reviewer","Focus on downside, invalidation, volatility, and risk/reward. Return a score from -100 to +100."),
        ("LLM News Reviewer","Assess only the supplied news/context. Do not invent news. Return -100 to +100."),
    ]
    out=[]
    for name, role in specs:
        prompt=f"""Ticker: {ticker}
Market snapshot:
{json.dumps(snapshot, ensure_ascii=False, default=str, indent=2)}
News/context:
{news_text[:6000]}
Task: {role}
Output exactly:
SCORE: number
RATIONALE: short text
Do not claim certainty and do not fabricate missing information."""
        try:
            txt=llm_call(client, role, prompt)
            score=0
            for line in txt.splitlines():
                if line.upper().startswith("SCORE:"):
                    score=float(line.split(":",1)[1].strip())
            out.append((name, clamp(score), txt))
        except Exception as e:
            out.append((name,0,f"LLM unavailable: {e}"))
    return out

# ---------- UI ----------
ticker=st.text_input("رمز السهم الأمريكي", value="AAPL").strip().upper()
period=st.selectbox("الفترة", ["3mo","6mo","1y","2y"], index=1)

if st.button("🔎 حلّل السهم", type="primary"):
    try:
        raw=load_data(ticker, period=period)
        if raw.empty:
            st.error("لم أستطع جلب بيانات السهم. تحقق من الرمز.")
            st.stop()
        x=add_indicators(raw)
        if len(x)<60:
            st.error("البيانات غير كافية للتحليل.")
            st.stop()

        results=[]
        for name, fn in AGENTS:
            s,reason=fn(x)
            results.append({"Agent":name,"Score":round(s,1),"Reason":reason})

        snap=x.iloc[-1].to_dict()
        news=""
        llm_results=run_llm_panel(ticker, snap, news)
        for name,score,reason in llm_results:
            results.append({"Agent":name,"Score":round(score,1),"Reason":reason})

        rdf=pd.DataFrame(results)
        # Quality gate: if data quality is bad, block.
        quality=float(rdf.loc[rdf.Agent=="Data Quality Agent","Score"].iloc[0])
        weighted=rdf["Score"].mean()
        bullish=(rdf["Score"]>20).sum()
        bearish=(rdf["Score"]<-20).sum()
        agreement=(rdf["Score"].abs()>20).mean()*100

        if quality < 0:
            decision="NO TRADE"
        elif weighted >= 25:
            decision="LONG BIAS"
        elif weighted <= -25:
            decision="SHORT BIAS"
        else:
            decision="NO TRADE"

        r=x.iloc[-1]
        atrv=float(r.ATR14)
        entry=float(r.Close)
        if decision=="LONG BIAS":
            stop=entry-1.5*atrv
            tp1=entry+1.5*atrv
            tp2=entry+3.0*atrv
        elif decision=="SHORT BIAS":
            stop=entry+1.5*atrv
            tp1=entry-1.5*atrv
            tp2=entry-3.0*atrv
        else:
            stop=tp1=tp2=float("nan")

        c1,c2,c3,c4=st.columns(4)
        c1.metric("السعر",f"${entry:.2f}")
        c2.metric("القرار",decision)
        c3.metric("متوسط الوكلاء",f"{weighted:.1f}/100")
        c4.metric("توافق الإشارات",f"{agreement:.0f}%")

        if decision!="NO TRADE":
            st.info(f"دخول مرجعي: ${entry:.2f} | وقف مبدئي: ${stop:.2f} | TP1: ${tp1:.2f} | TP2: ${tp2:.2f}")
        else:
            st.warning("النظام لا يرى أفضلية كافية للدخول حالياً.")

        st.subheader("نتائج الوكلاء")
        st.dataframe(rdf, use_container_width=True, hide_index=True)

        st.subheader("السعر والمؤشرات")
        chart=x[["Close","SMA20","SMA50","SMA200"]].tail(120)
        st.line_chart(chart)

        st.subheader("ملخص القرار")
        st.write({
            "ticker": ticker,
            "last_bar": str(x.index[-1]),
            "decision": decision,
            "bullish_agents": int(bullish),
            "bearish_agents": int(bearish),
            "mean_score": round(float(weighted),1),
            "agreement_pct": round(float(agreement),1),
            "RSI14": round(float(r.RSI14),1),
            "ATR%": round(float(r.ATR_PCT),2),
            "volume_ratio": round(float(r.VOL_RATIO),2),
        })

        st.caption("تنبيه: هذا نموذج بحثي/تحليلي. لا يعتمد عليه وحده لفتح صفقة حقيقية. قبل التنفيذ يجب إجراء backtest وforward test واحتساب الانزلاق والعمولات.")
    except Exception as e:
        st.exception(e)
else:
    st.markdown("""
### ماذا يفعل النظام؟
1. يجلب بيانات السهم من Yahoo Finance.
2. يشغّل مجموعة من الوكلاء المتخصصين.
3. كل وكيل يعطي درجة من -100 إلى +100 وسبباً واضحاً.
4. يجمع النتائج في **Master Decision**.
5. يعطي Bias للدخول، ووقفاً وأهدافاً مبنية على ATR عندما تتوفر أفضلية كافية.
6. يمكن تفعيل طبقة LLM بإضافة `OPENAI_API_KEY`، وعندها تُضاف مراجعات LLM مستقلة.

**النسخة الحالية لا تفتح صفقات تلقائياً.** هذه نقطة مقصودة إلى أن نثبت الأداء تاريخياً.
""")
