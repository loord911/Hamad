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
st.caption("60+ وكلاء تحليليين مستقلين + مدير قرار. النظام تحليلي ولا يرسل أوامر تداول.")

@st.cache_data(ttl=300)
def load_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
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
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-prev).abs(), (df["Low"]-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    x = df.copy()
    c, h, l, o, v = x["Close"], x["High"], x["Low"], x["Open"], x["Volume"]
    for n in (5, 10, 20, 50, 100, 200):
        x[f"SMA{n}"] = c.rolling(n).mean()
        x[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    x["RSI14"] = rsi(c)
    x["RSI7"] = rsi(c, 7)
    x["ATR14"] = atr(x)
    x["ATR_PCT"] = x["ATR14"] / c * 100
    x["VOL20"] = v.rolling(20).mean()
    x["VOL_RATIO"] = v / x["VOL20"]
    x["VOL_Z"] = (v - x["VOL20"]) / v.rolling(20).std()
    fast = c.ewm(span=12, adjust=False).mean()
    slow = c.ewm(span=26, adjust=False).mean()
    x["MACD"] = fast - slow
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]
    for n in (1, 3, 5, 10, 20, 60):
        x[f"RET{n}"] = c.pct_change(n) * 100
    mid = c.rolling(20).mean(); sd = c.rolling(20).std()
    x["BB_MID"] = mid; x["BB_UP"] = mid + 2*sd; x["BB_LOW"] = mid - 2*sd
    x["BB_PCT"] = (c-x["BB_LOW"])/(x["BB_UP"]-x["BB_LOW"]).replace(0,np.nan)
    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    x["STO_K"] = 100*(c-ll14)/(hh14-ll14).replace(0,np.nan)
    x["STO_D"] = x["STO_K"].rolling(3).mean()
    x["WILLR"] = -100*(hh14-c)/(hh14-ll14).replace(0,np.nan)
    tp = (h+l+c)/3
    mad = tp.rolling(20).apply(lambda z: np.mean(np.abs(z-z.mean())), raw=True)
    x["CCI20"] = (tp-tp.rolling(20).mean())/(0.015*mad).replace(0,np.nan)
    x["ROC10"] = c.pct_change(10)*100; x["ROC20"] = c.pct_change(20)*100
    direction = np.sign(c.diff()).fillna(0)
    x["OBV"] = (direction*v).cumsum(); x["OBV_EMA20"] = x["OBV"].ewm(span=20, adjust=False).mean()
    pos = (tp*v).where(c.diff()>0, 0).rolling(14).sum()
    neg = (tp*v).where(c.diff()<0, 0).abs().rolling(14).sum().replace(0,np.nan)
    x["MFI14"] = 100 - 100/(1 + pos/neg)
    x["GAP_PCT"] = (o-c.shift(1))/c.shift(1)*100
    x["HIGH20"] = h.shift(1).rolling(20).max(); x["LOW20"] = l.shift(1).rolling(20).min()
    x["HIGH60"] = h.shift(1).rolling(60).max(); x["LOW60"] = l.shift(1).rolling(60).min()
    x["RANGE_PCT"] = (h-l)/c*100; x["BODY_PCT"] = (c-o)/o*100
    x["CLOSE_LOC"] = (c-l)/(h-l).replace(0,np.nan)
    x["SLOPE20"] = c.rolling(20).apply(lambda z: np.polyfit(np.arange(len(z)), z, 1)[0], raw=True)
    x["SLOPE50"] = c.rolling(50).apply(lambda z: np.polyfit(np.arange(len(z)), z, 1)[0], raw=True)
    x["DIST_SMA20"] = (c-x["SMA20"])/x["SMA20"]*100
    x["DIST_SMA50"] = (c-x["SMA50"])/x["SMA50"]*100
    x["DIST_SMA200"] = (c-x["SMA200"])/x["SMA200"]*100
    x["ADX_PROXY"] = ((h.diff().clip(lower=0).rolling(14).mean() + (-l.diff()).clip(lower=0).rolling(14).mean()) / x["ATR14"])
    x = x.replace([np.inf,-np.inf], np.nan)
    return x.dropna(subset=["Close","SMA20","SMA50","RSI14","ATR14","MACD","VOL20","BB_PCT","STO_K","CCI20","OBV","SLOPE20"])

def clamp(v, lo=-100, hi=100):
    try: return float(max(lo, min(hi, v)))
    except Exception: return 0.0

def specialist_agents(x):
    r=x.iloc[-1]; p=x.iloc[-2]; out=[]
    def add(name, score, reason): out.append((name, clamp(score), reason))
    std20=x["Close"].rolling(20).std().iloc[-1]
    range_avg=x["RANGE_PCT"].rolling(20).mean().iloc[-1]
    atr_avg=x["ATR_PCT"].rolling(20).mean().iloc[-1]
    z=(r.Close-r.SMA20)/(std20 if pd.notna(std20) and std20!=0 else 1)
    add("Trend Agent",30*(r.Close>r.SMA20)-30*(r.Close<=r.SMA20)+30*(r.SMA20>r.SMA50)-30*(r.SMA20<=r.SMA50)+20*(r.RET20>0)-20*(r.RET20<=0),f"Close {r.Close:.2f}; SMA20 {r.SMA20:.2f}; SMA50 {r.SMA50:.2f}")
    add("EMA Trend Agent",45*(r.EMA20>r.EMA50)-45*(r.EMA20<=r.EMA50)+30*(r.Close>r.EMA20)-30*(r.Close<=r.EMA20),f"EMA20 {r.EMA20:.2f}; EMA50 {r.EMA50:.2f}")
    add("Long Trend Agent",50*(r.SMA50>r.SMA200)-50*(r.SMA50<=r.SMA200)+30*(r.Close>r.SMA200)-30*(r.Close<=r.SMA200) if pd.notna(r.SMA200) else 0,f"SMA50 {r.SMA50:.2f}; SMA200 {r.SMA200:.2f}")
    add("Momentum Agent",(r.RSI14-50)*1.4+(20 if r.RET5>0 else -20),f"RSI14 {r.RSI14:.1f}; 5D {r.RET5:.2f}%")
    add("Fast Momentum Agent",(r.RSI7-50)*1.2+25*np.sign(r.RET3),f"RSI7 {r.RSI7:.1f}; 3D {r.RET3:.2f}%")
    add("MACD Agent",65*np.sign(r.MACD-r.MACD_SIGNAL)+25*np.sign(r.MACD_HIST),f"MACD {r.MACD:.3f}; signal {r.MACD_SIGNAL:.3f}")
    add("MACD Cross Agent",80 if r.MACD>r.MACD_SIGNAL and p.MACD<=p.MACD_SIGNAL else (-80 if r.MACD<p.MACD_SIGNAL and p.MACD>=p.MACD_SIGNAL else 10*np.sign(r.MACD-r.MACD_SIGNAL)),"MACD cross state")
    add("RSI Regime Agent",55 if 50<r.RSI14<70 else (-55 if 30<r.RSI14<50 else (25 if r.RSI14<=30 else -25)),f"RSI {r.RSI14:.1f}")
    add("RSI Reversal Agent",70 if p.RSI14<30 and r.RSI14>p.RSI14 else (-70 if p.RSI14>70 and r.RSI14<p.RSI14 else 0),f"RSI {p.RSI14:.1f}->{r.RSI14:.1f}")
    add("Stochastic Agent",55 if r.STO_K>r.STO_D and r.STO_K<80 else (-55 if r.STO_K<r.STO_D and r.STO_K>20 else 0),f"K {r.STO_K:.1f}; D {r.STO_D:.1f}")
    add("Stochastic Reversal Agent",75 if p.STO_K<20 and r.STO_K>p.STO_K else (-75 if p.STO_K>80 and r.STO_K<p.STO_K else 0),f"K {r.STO_K:.1f}")
    add("CCI Agent",60 if r.CCI20>100 else (-60 if r.CCI20<-100 else 20*np.sign(r.CCI20)),f"CCI {r.CCI20:.1f}")
    add("Williams Agent",55 if -80<r.WILLR<-20 and r.WILLR>p.WILLR else (-55 if -80<r.WILLR<-20 and r.WILLR<p.WILLR else 0),f"Williams %R {r.WILLR:.1f}")
    add("ROC10 Agent",r.ROC10*8,f"ROC10 {r.ROC10:.2f}%")
    add("ROC20 Agent",r.ROC20*6,f"ROC20 {r.ROC20:.2f}%")
    add("5D Return Agent",r.RET5*10,f"5D {r.RET5:.2f}%")
    add("20D Return Agent",r.RET20*5,f"20D {r.RET20:.2f}%")
    add("60D Return Agent",r.RET60*3,f"60D {r.RET60:.2f}%")
    add("Bollinger Position Agent",(r.BB_PCT-0.5)*120,f"BB position {r.BB_PCT:.2f}")
    add("Bollinger Break Agent",75 if r.Close>r.BB_UP else (-75 if r.Close<r.BB_LOW else 0),f"Bands {r.BB_LOW:.2f}-{r.BB_UP:.2f}")
    add("Bollinger Mean Reversion Agent",-(r.BB_PCT-0.5)*90,f"BB mean reversion {r.BB_PCT:.2f}")
    add("Volume Agent",60*np.sign(r.RET5) if r.VOL_RATIO>1.2 else 0,f"Volume {r.VOL_RATIO:.2f}x")
    add("Volume Surge Agent",70*np.sign(r.BODY_PCT) if r.VOL_Z>2 else (-50*np.sign(r.BODY_PCT) if r.VOL_Z<-2 else 0),f"Volume z {r.VOL_Z:.2f}")
    add("OBV Agent",65*np.sign(r.OBV-r.OBV_EMA20),"OBV vs EMA20")
    add("MFI Agent",60 if 50<r.MFI14<80 else (-60 if 20<r.MFI14<50 else (30 if r.MFI14<=20 else -30)),f"MFI {r.MFI14:.1f}")
    add("ATR Risk Agent",25 if r.ATR_PCT<4 else (0 if r.ATR_PCT<7 else -35),f"ATR {r.ATR_PCT:.2f}%")
    add("Volatility Regime Agent",40 if r.ATR_PCT<5 else (-30 if r.ATR_PCT>9 else 0),f"ATR regime {r.ATR_PCT:.2f}%")
    add("Range Expansion Agent",55*np.sign(r.BODY_PCT) if r.RANGE_PCT>range_avg*1.5 else 0,f"Range {r.RANGE_PCT:.2f}% vs {range_avg:.2f}%")
    add("Gap Agent",35*np.sign(r.GAP_PCT) if abs(r.GAP_PCT)>1 else 0,f"Gap {r.GAP_PCT:.2f}%")
    add("Candle Body Agent",55 if r.BODY_PCT>1 and r.CLOSE_LOC>.65 else (-55 if r.BODY_PCT<-1 and r.CLOSE_LOC<.35 else 0),f"Body {r.BODY_PCT:.2f}%")
    add("Candle Close Location Agent",45 if r.CLOSE_LOC>.75 else (-45 if r.CLOSE_LOC<.25 else 0),f"Close location {r.CLOSE_LOC:.2f}")
    add("Support Agent",60 if abs(r.Close-r.LOW20)/r.Close<.025 else 0,f"20D low {r.LOW20:.2f}")
    add("Resistance Agent",-60 if abs(r.HIGH20-r.Close)/r.Close<.025 else 0,f"20D high {r.HIGH20:.2f}")
    add("60D Support Agent",55 if abs(r.Close-r.LOW60)/r.Close<.035 else 0,f"60D low {r.LOW60:.2f}")
    add("60D Resistance Agent",-55 if abs(r.HIGH60-r.Close)/r.Close<.035 else 0,f"60D high {r.HIGH60:.2f}")
    add("20D Breakout Agent",85 if r.Close>r.HIGH20 else (-85 if r.Close<r.LOW20 else 0),f"20D {r.LOW20:.2f}-{r.HIGH20:.2f}")
    add("60D Breakout Agent",90 if r.Close>r.HIGH60 else (-90 if r.Close<r.LOW60 else 0),f"60D {r.LOW60:.2f}-{r.HIGH60:.2f}")
    add("SMA20 Distance Agent",r.DIST_SMA20*12,f"Distance SMA20 {r.DIST_SMA20:.2f}%")
    add("SMA50 Distance Agent",r.DIST_SMA50*8,f"Distance SMA50 {r.DIST_SMA50:.2f}%")
    add("SMA200 Distance Agent",r.DIST_SMA200*4 if pd.notna(r.DIST_SMA200) else 0,f"Distance SMA200 {r.DIST_SMA200:.2f}%")
    add("EMA Alignment Agent",70 if r.EMA5>r.EMA10>r.EMA20>r.EMA50 else (-70 if r.EMA5<r.EMA10<r.EMA20<r.EMA50 else 0),"EMA5/10/20/50 alignment")
    add("Golden Cross Agent",75 if r.SMA20>r.SMA50 and p.SMA20<=p.SMA50 else (-75 if r.SMA20<r.SMA50 and p.SMA20>=p.SMA50 else 0),"SMA20/SMA50 cross")
    add("Death Cross Agent",-75 if pd.notna(r.SMA200) and r.SMA50<r.SMA200 and p.SMA50>=p.SMA200 else 0,"SMA50/SMA200 cross")
    add("Trend Slope20 Agent",r.SLOPE20/r.Close*10000,f"20D slope {r.SLOPE20:.4f}")
    add("Trend Slope50 Agent",r.SLOPE50/r.Close*10000,f"50D slope {r.SLOPE50:.4f}")
    add("Higher High Agent",55 if r.High>p.High and r.Low>=p.Low else (-55 if r.High<p.High and r.Low<=p.Low else 0),"Latest high/low structure")
    add("Higher Low Agent",60 if r.Low>p.Low and r.Close>p.Close else (-60 if r.Low<p.Low and r.Close<p.Close else 0),"Latest swing structure")
    add("Price Structure Agent",60 if r.Close>r.SMA20 and r.SMA20>r.SMA50 else (-60 if r.Close<r.SMA20 and r.SMA20<r.SMA50 else 0),"Price/SMA structure")
    add("Regime Agent",45 if r.SMA20>r.SMA50 and r.RET20>0 else (-45 if r.SMA20<r.SMA50 and r.RET20<0 else 0),"Bullish/bearish regime")
    add("Trend Strength Proxy Agent",45 if r.ADX_PROXY>1.2 and r.SLOPE20>0 else (-45 if r.ADX_PROXY>1.2 and r.SLOPE20<0 else 0),f"ADX proxy {r.ADX_PROXY:.2f}")
    add("Mean Reversion Z Agent",-z*35,f"20D z-score {z:.2f}")
    add("Oversold Bounce Agent",65 if r.RSI14<35 and r.Close>p.Close else (-65 if r.RSI14>65 and r.Close<p.Close else 0),"Oversold/overbought bounce")
    add("Overbought Exhaustion Agent",-65 if r.RSI14>72 and r.CLOSE_LOC<.5 else 0,f"RSI {r.RSI14:.1f}")
    add("Momentum Divergence Proxy",55 if r.Close<p.Close and r.RSI14>p.RSI14 else (-55 if r.Close>p.Close and r.RSI14<p.RSI14 else 0),"Price vs RSI divergence")
    add("MACD Histogram Acceleration Agent",55 if r.MACD_HIST>p.MACD_HIST else (-55 if r.MACD_HIST<p.MACD_HIST else 0),f"Hist {p.MACD_HIST:.3f}->{r.MACD_HIST:.3f}")
    add("Volume Confirmation Agent",50 if r.RET5>0 and r.VOL_RATIO>1 else (-50 if r.RET5<0 and r.VOL_RATIO>1 else 0),"Return + volume confirmation")
    add("Volatility Breakout Agent",60*np.sign(r.RET5) if r.ATR_PCT>atr_avg*1.25 else 0,f"ATR {r.ATR_PCT:.2f}% vs avg {atr_avg:.2f}%")
    add("Risk/Reward Bias Agent",40 if r.Close>r.SMA20 and r.ATR_PCT<8 else (-40 if r.Close<r.SMA20 and r.ATR_PCT>8 else 0),"Trend vs volatility")
    add("Liquidity Proxy Agent",35 if r.VOL_RATIO>=1 else -15,f"Volume ratio {r.VOL_RATIO:.2f}")
    add("Data Quality Agent",25 if len(x)>=60 else 10,f"Usable bars {len(x)}")
    add("Ensemble Consistency Agent",50 if r.Close>r.SMA20 and r.MACD>r.MACD_SIGNAL and r.RSI14>50 else (-50 if r.Close<r.SMA20 and r.MACD<r.MACD_SIGNAL and r.RSI14<50 else 0),"Trend + MACD + RSI alignment")
    return out[:60]

def get_secret(name):
    try: return st.secrets.get(name)
    except Exception: return os.getenv(name)

def llm_review(ticker,r,rdf):
    key=get_secret("OPENAI_API_KEY")
    if not key or OpenAI is None: return None
    try:
        client=OpenAI(api_key=key)
        strongest=rdf.sort_values("Score",ascending=False).head(8)[["Agent","Score","Reason"]].to_dict("records")
        weakest=rdf.sort_values("Score",ascending=True).head(8)[["Agent","Score","Reason"]].to_dict("records")
        prompt=f"حلل {ticker} كمراجع كمي مستقل. السعر {r.Close:.2f}. RSI {r.RSI14:.1f}. ATR% {r.ATR_PCT:.2f}. 20D return {r.RET20:.2f}%. أقوى الإشارات: {json.dumps(strongest,ensure_ascii=False)}. أضعف الإشارات: {json.dumps(weakest,ensure_ascii=False)}. أعطني تحيزاً صاعداً/محايداً/هابطاً، أهم 3 أسباب، أهم خطرين، وهل توجد أفضلية دخول واضحة الآن. لا تعطِ يقيناً أو ضماناً."
        resp=client.responses.create(model="gpt-4.1-mini",input=prompt)
        return resp.output_text
    except Exception as e: return f"تعذر تشغيل مراجع LLM: {e}"

ticker=st.text_input("رمز السهم الأمريكي",value="FIGR").strip().upper()

interval_options = {
    "1 دقيقة": ("1m", ["1d","5d","7d"]),
    "5 دقائق": ("5m", ["5d","1mo","3mo"]),
    "15 دقيقة": ("15m", ["1mo","3mo","6mo"]),
    "30 دقيقة": ("30m", ["1mo","3mo","6mo"]),
    "ساعة": ("1h", ["1mo","3mo","6mo","1y"]),
    "يومي": ("1d", ["3mo","6mo","1y","2y","5y"]),
    "أسبوعي": ("1wk", ["6mo","1y","2y","5y","10y"]),
}
interval_label = st.selectbox("الفاصل الزمني", list(interval_options.keys()), index=5)
interval, period_options = interval_options[interval_label]
period = st.selectbox("الفترة التاريخية", period_options, index=min(1, len(period_options)-1))

st.caption("الفواصل من الدقيقة إلى الأسبوعية متاحة. بعض الفواصل القصيرة لها حدود تاريخية من مصدر البيانات (Yahoo Finance).")

if st.button("🔎 حلّل السهم بـ 60 وكيل",type="primary"):
    try:
        raw=load_data(ticker,period,interval)
        if raw.empty:
            st.error("لم أستطع جلب بيانات السهم. تحقق من الرمز."); st.stop()
        x=add_indicators(raw)
        if len(x)<60:
            st.error(f"البيانات غير كافية بعد تجهيز المؤشرات ({len(x)} شمعة). اختر فترة أطول أو فاصلًا زمنيًا أكبر."); st.stop()
        results=specialist_agents(x)
        rdf=pd.DataFrame(results,columns=["Agent","Score","Reason"])
        weighted=float(rdf.Score.mean()); bullish=int((rdf.Score>20).sum()); bearish=int((rdf.Score<-20).sum()); neutral=len(rdf)-bullish-bearish
        agreement=float((rdf.Score.abs()>20).mean()*100)
        if weighted>=25 and bullish>=30: decision="LONG BIAS"
        elif weighted<=-25 and bearish>=30: decision="SHORT BIAS"
        else: decision="NO TRADE"
        r=x.iloc[-1]; entry=float(r.Close); atrv=float(r.ATR14)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("السعر",f"${entry:.2f}"); c2.metric("قرار 60 وكيل",decision); c3.metric("متوسط الوكلاء",f"{weighted:.1f}/100"); c4.metric("التوافق",f"{agreement:.0f}%")
        b1,b2,b3=st.columns(3); b1.metric("صاعد",bullish); b2.metric("محايد",neutral); b3.metric("هابط",bearish)
        if decision=="LONG BIAS":
            st.success(f"تحيز شراء: دخول مرجعي ${entry:.2f} | وقف ${entry-1.5*atrv:.2f} | TP1 ${entry+1.5*atrv:.2f} | TP2 ${entry+3*atrv:.2f}")
        elif decision=="SHORT BIAS":
            st.error(f"تحيز بيع: دخول مرجعي ${entry:.2f} | وقف ${entry+1.5*atrv:.2f} | TP1 ${entry-1.5*atrv:.2f} | TP2 ${entry-3*atrv:.2f}")
        else: st.warning("60 وكيل لم يتفقوا على أفضلية دخول قوية حالياً.")
        st.subheader("🤖 مراجعة الذكاء الاصطناعي")
        review=llm_review(ticker,r,rdf)
        if review: st.info(review)
        else: st.caption("مراجع LLM اختياري: أضف OPENAI_API_KEY في Streamlit Secrets لتشغيله. الوكلاء الـ60 كميّون ولا يحتاجون API.")
        st.subheader("نتائج 60 وكيل")
        st.dataframe(rdf.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)
        st.subheader("السعر والمؤشرات")
        cols=["Close","SMA20","SMA50"]
        if x["SMA200"].notna().any(): cols.append("SMA200")
        st.line_chart(x[cols].tail(180))
        st.subheader("ملخص القرار")
        st.json({"ticker":ticker,"last_bar":str(x.index[-1]),"decision":decision,"agents":len(rdf),"bullish_agents":bullish,"neutral_agents":neutral,"bearish_agents":bearish,"mean_score":round(weighted,1),"agreement_pct":round(agreement,1),"RSI14":round(float(r.RSI14),1),"ATR%":round(float(r.ATR_PCT),2),"volume_ratio":round(float(r.VOL_RATIO),2)})
        st.caption("نموذج بحثي/تحليلي وليس توصية استثمارية. كثرة الوكلاء لا تعني دقة أعلى تلقائياً؛ اختبر الأداء خارج العينة واحتسب العمولات والانزلاق.")
    except Exception as e: st.exception(e)
else:
    st.markdown("""
### ماذا يفعل الإصدار الجديد؟
- يشغّل **60 وكيلًا كميًا متخصصًا** في الاتجاه، المتوسطات، RSI، MACD، Stochastic، CCI، Williams %R، ROC، Bollinger، الحجم، OBV، MFI، ATR، الدعم والمقاومة، الاختراقات، هيكل السعر والتذبذب.
- كل وكيل يعطي درجة من **-100 إلى +100** وسبباً مختصراً.
- مدير القرار يجمع النتائج ويشترط توافقاً قبل LONG/SHORT.
- يوجد **مراجع LLM اختياري واحد** إذا وضعت OPENAI_API_KEY في Secrets؛ لا يشغّل 60 مكالمة API مكلفة.
""")
