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
st.title("📈 US Stock AI Agent Desk — Re-Engineered")
st.caption("60 محللًا كميًا متخصصًا + 10 مجموعات مستقلة + مدير قرار. لا توجد أوامر تداول تلقائية.")

INTERVAL_PERIODS = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["5d", "1mo", "3mo"],
    "15m": ["1mo", "3mo", "6mo"],
    "30m": ["1mo", "3mo", "6mo"],
    "1h": ["1mo", "3mo", "6mo", "1y"],
    "1d": ["3mo", "6mo", "1y", "2y", "5y"],
}

@st.cache_data(ttl=180)
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
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-prev).abs(), (df["Low"]-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    x = df.copy()
    c, h, l, o, v = x["Close"], x["High"], x["Low"], x["Open"], x["Volume"]
    for n in (5, 8, 10, 20, 50, 100, 200):
        x[f"SMA{n}"] = c.rolling(n).mean()
        x[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    x["RSI14"] = rsi(c, 14); x["RSI7"] = rsi(c, 7); x["RSI21"] = rsi(c, 21)
    x["ATR14"] = atr(x); x["ATR_PCT"] = x["ATR14"] / c * 100
    x["ATR20_AVG"] = x["ATR_PCT"].rolling(20).mean()
    x["VOL20"] = v.rolling(20).mean(); x["VOL_RATIO"] = v / x["VOL20"]
    x["VOL_Z"] = (v-x["VOL20"]) / x["Volume"].rolling(20).std().replace(0, np.nan)
    fast = c.ewm(span=12, adjust=False).mean(); slow = c.ewm(span=26, adjust=False).mean()
    x["MACD"] = fast-slow; x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean(); x["MACD_HIST"] = x["MACD"]-x["MACD_SIGNAL"]
    for n in (1, 2, 3, 5, 10, 20, 60): x[f"RET{n}"] = c.pct_change(n)*100
    mid=c.rolling(20).mean(); sd=c.rolling(20).std()
    x["BB_MID"]=mid; x["BB_UP"]=mid+2*sd; x["BB_LOW"]=mid-2*sd
    x["BB_PCT"]=(c-x["BB_LOW"])/(x["BB_UP"]-x["BB_LOW"]).replace(0,np.nan)
    ll14, hh14=l.rolling(14).min(), h.rolling(14).max()
    x["STO_K"]=100*(c-ll14)/(hh14-ll14).replace(0,np.nan); x["STO_D"]=x["STO_K"].rolling(3).mean()
    x["WILLR"]=-100*(hh14-c)/(hh14-ll14).replace(0,np.nan)
    tp=(h+l+c)/3; mad=tp.rolling(20).apply(lambda z: np.mean(np.abs(z-z.mean())), raw=True)
    x["CCI20"]=(tp-tp.rolling(20).mean())/(0.015*mad).replace(0,np.nan)
    x["ROC10"]=c.pct_change(10)*100; x["ROC20"]=c.pct_change(20)*100
    direction=np.sign(c.diff()).fillna(0); x["OBV"]=(direction*v).cumsum(); x["OBV_EMA20"]=x["OBV"].ewm(span=20,adjust=False).mean()
    pos=(tp*v).where(c.diff()>0,0).rolling(14).sum(); neg=(tp*v).where(c.diff()<0,0).abs().rolling(14).sum().replace(0,np.nan)
    x["MFI14"]=100-100/(1+pos/neg)
    x["GAP_PCT"]=(o-c.shift(1))/c.shift(1)*100
    for n in (10,20,50,60):
        x[f"HIGH{n}"]=h.shift(1).rolling(n).max(); x[f"LOW{n}"]=l.shift(1).rolling(n).min()
    x["RANGE_PCT"]=(h-l)/c*100; x["BODY_PCT"]=(c-o)/o*100; x["CLOSE_LOC"]=(c-l)/(h-l).replace(0,np.nan)
    x["SLOPE20"]=c.rolling(20).apply(lambda z: np.polyfit(np.arange(len(z)),z,1)[0],raw=True)
    x["SLOPE50"]=c.rolling(50).apply(lambda z: np.polyfit(np.arange(len(z)),z,1)[0],raw=True)
    x["DIST_SMA20"]=(c-x["SMA20"])/x["SMA20"]*100; x["DIST_SMA50"]=(c-x["SMA50"])/x["SMA50"]*100; x["DIST_SMA200"]=(c-x["SMA200"])/x["SMA200"]*100
    x["ADX_PROXY"]=((h.diff().clip(lower=0).rolling(14).mean()+(-l.diff()).clip(lower=0).rolling(14).mean())/x["ATR14"])
    x["HIGHER_HIGH"]=(h>h.shift(1)).astype(int); x["HIGHER_LOW"]=(l>l.shift(1)).astype(int)
    x=x.replace([np.inf,-np.inf],np.nan)
    required=["Close","SMA20","SMA50","RSI14","ATR14","MACD","VOL20","BB_PCT","STO_K","CCI20","OBV","SLOPE20"]
    return x.dropna(subset=required)

def clamp_series(s):
    return s.replace([np.inf,-np.inf],np.nan).fillna(0).clip(-100,100)

def sgn(v): return np.sign(v).fillna(0)

def build_agents(x):
    r=x
    specs=[]
    def add(group,name,score,reason): specs.append({"Group":group,"Agent":name,"Score":clamp_series(score),"Reason":reason})
    # 1) Trend — 6
    add("Trend","SMA Structure",35*sgn(r.Close-r.SMA20)+35*sgn(r.SMA20-r.SMA50)+20*sgn(r.SMA50-r.SMA200),"السعر وSMA20/50/200")
    add("Trend","EMA Structure",40*sgn(r.EMA20-r.EMA50)+30*sgn(r.Close-r.EMA20)+20*sgn(r.EMA10-r.EMA20),"EMA10/20/50 + السعر")
    add("Trend","Trend Slope 20",sgn(r.SLOPE20)*(np.abs(r.SLOPE20)/r.Close*10000),"ميل 20 شمعة")
    add("Trend","Trend Slope 50",sgn(r.SLOPE50)*(np.abs(r.SLOPE50)/r.Close*10000),"ميل 50 شمعة")
    add("Trend","Long-Term Position",30*sgn(r.Close-r.SMA200)+30*sgn(r.SMA50-r.SMA200)+20*sgn(r.RET60),"الموقع طويل الأجل")
    add("Trend","EMA Alignment",70*((r.EMA5>r.EMA10)&(r.EMA10>r.EMA20)&(r.EMA20>r.EMA50)).astype(float)-70*((r.EMA5<r.EMA10)&(r.EMA10<r.EMA20)&(r.EMA20<r.EMA50)).astype(float),"اصطفاف EMA")
    # 2) Momentum — 6
    add("Momentum","RSI14 Momentum",(r.RSI14-50)*1.5,"RSI14")
    add("Momentum","RSI7 Momentum",(r.RSI7-50)*1.2,"RSI7")
    add("Momentum","ROC10",r.ROC10*7,"عائد 10 شموع")
    add("Momentum","ROC20",r.ROC20*5,"عائد 20 شمعة")
    add("Momentum","Short Return",r.RET5*10,"عائد قصير")
    add("Momentum","Medium Return",r.RET20*5,"عائد متوسط")
    # 3) Mean Reversion — 6
    z=(r.Close-r.SMA20)/r.Close.rolling(20).std().replace(0,np.nan)
    add("Mean Reversion","Z-Score Reversion",-z*35,"بعد السعر عن SMA20")
    add("Mean Reversion","Bollinger Reversion",-(r.BB_PCT-.5)*90,"موضع Bollinger")
    add("Mean Reversion","RSI Reversal",70*((r.RSI14.shift(1)<30)&(r.RSI14>r.RSI14.shift(1))).astype(float)-70*((r.RSI14.shift(1)>70)&(r.RSI14<r.RSI14.shift(1))).astype(float),"انعكاس RSI")
    add("Mean Reversion","Stochastic Reversal",75*((r.STO_K.shift(1)<20)&(r.STO_K>r.STO_K.shift(1))).astype(float)-75*((r.STO_K.shift(1)>80)&(r.STO_K<r.STO_K.shift(1))).astype(float),"انعكاس Stochastic")
    add("Mean Reversion","Oversold Bounce",65*((r.RSI14<35)&(r.RET1>0)).astype(float)-65*((r.RSI14>65)&(r.RET1<0)).astype(float),"ارتداد من التشبع")
    add("Mean Reversion","Overextension",-np.clip(r.DIST_SMA20*8,-100,100),"تمدد السعر عن المتوسط")
    # 4) MACD / oscillator — 6
    add("Oscillators","MACD Direction",65*sgn(r.MACD-r.MACD_SIGNAL)+25*sgn(r.MACD_HIST),"MACD مقابل Signal")
    add("Oscillators","MACD Cross",80*((r.MACD>r.MACD_SIGNAL)&(r.MACD.shift(1)<=r.MACD_SIGNAL.shift(1))).astype(float)-80*((r.MACD<r.MACD_SIGNAL)&(r.MACD.shift(1)>=r.MACD_SIGNAL.shift(1))).astype(float),"تقاطع MACD")
    add("Oscillators","MACD Acceleration",55*sgn(r.MACD_HIST-r.MACD_HIST.shift(1)),"تسارع Histogram")
    add("Oscillators","Stochastic",55*sgn(r.STO_K-r.STO_D)*(1-(r.STO_K>80).astype(float)-(r.STO_K<20).astype(float)*0),"K مقابل D")
    add("Oscillators","CCI",np.clip(r.CCI20/2,-100,100),"CCI20")
    add("Oscillators","Williams",45*sgn(r.WILLR-r.WILLR.shift(1)),"Williams %R")
    # 5) Volume — 6
    add("Volume","Volume Confirmation",50*sgn(r.RET5)*np.clip(r.VOL_RATIO,0,2)/2,"الحجم مع العائد")
    add("Volume","Volume Surge",70*sgn(r.BODY_PCT)*(r.VOL_Z>2).astype(float),"طفرة حجم")
    add("Volume","OBV Trend",65*sgn(r.OBV-r.OBV_EMA20),"OBV مقابل EMA")
    add("Volume","MFI",np.where((r.MFI14>50)&(r.MFI14<80),45,np.where((r.MFI14<50)&(r.MFI14>20),-45,0)),"MFI14")
    add("Volume","Liquidity Proxy",np.clip((r.VOL_RATIO-1)*60,-60,60),"حجم مقابل متوسط 20")
    add("Volume","Volume Price Agreement",50*sgn(r.RET1)*np.clip(r.VOL_RATIO,0,2)/2,"اتفاق السعر والحجم")
    # 6) Volatility / risk — 6
    add("Risk","ATR Regime",np.where(r.ATR_PCT<5,30,np.where(r.ATR_PCT>9,-35,0)),"ATR%")
    add("Risk","ATR Relative",np.clip((r.ATR20_AVG-r.ATR_PCT)*15,-60,60),"ATR مقابل متوسطه")
    add("Risk","Range Expansion",55*sgn(r.BODY_PCT)*(r.RANGE_PCT>r.RANGE_PCT.rolling(20).mean()*1.5).astype(float),"اتساع النطاق")
    add("Risk","Volatility Breakout",60*sgn(r.RET5)*(r.ATR_PCT>r.ATR20_AVG*1.25).astype(float),"ارتفاع التذبذب مع الاتجاه")
    add("Risk","Candle Risk",-45*(r.ATR_PCT>10).astype(float),"مخاطر التذبذب")
    add("Risk","Risk/Trend Balance",np.where((r.Close>r.SMA20)&(r.ATR_PCT<8),40,np.where((r.Close<r.SMA20)&(r.ATR_PCT>8),-40,0)),"اتزان الاتجاه والمخاطر")
    # 7) Price Action — 6
    add("Price Action","Candle Body",55*((r.BODY_PCT>1)&(r.CLOSE_LOC>.65)).astype(float)-55*((r.BODY_PCT<-1)&(r.CLOSE_LOC<.35)).astype(float),"جسم الشمعة")
    add("Price Action","Close Location",45*sgn(r.CLOSE_LOC-.5),"موقع الإغلاق داخل الشمعة")
    add("Price Action","Gap",35*sgn(r.GAP_PCT)*(r.GAP_PCT.abs()>1).astype(float),"الفجوة")
    add("Price Action","Higher High",55*((r.High>r.High.shift(1))&(r.Low>=r.Low.shift(1))).astype(float)-55*((r.High<r.High.shift(1))&(r.Low<=r.Low.shift(1))).astype(float),"Higher High/Lower Low")
    add("Price Action","Higher Low",60*((r.Low>r.Low.shift(1))&(r.Close>r.Close.shift(1))).astype(float)-60*((r.Low<r.Low.shift(1))&(r.Close<r.Close.shift(1))).astype(float),"Higher Low/Lower High")
    add("Price Action","Structure vs SMA",60*((r.Close>r.SMA20)&(r.SMA20>r.SMA50)).astype(float)-60*((r.Close<r.SMA20)&(r.SMA20<r.SMA50)).astype(float),"هيكل السعر")
    # 8) Breakout / S-R — 6
    add("Breakout","20D Breakout",85*sgn(r.Close-r.HIGH20)*(r.Close>r.HIGH20).astype(float)-85*sgn(r.Close-r.LOW20)*(r.Close<r.LOW20).astype(float),"اختراق 20 يوم")
    add("Breakout","60D Breakout",90*sgn(r.Close-r.HIGH60)*(r.Close>r.HIGH60).astype(float)-90*sgn(r.Close-r.LOW60)*(r.Close<r.LOW60).astype(float),"اختراق 60 يوم")
    add("Breakout","20D Support",60*((r.Close-r.LOW20).abs()/r.Close<.025).astype(float),"قرب دعم 20 يوم")
    add("Breakout","20D Resistance",-60*((r.HIGH20-r.Close).abs()/r.Close<.025).astype(float),"قرب مقاومة 20 يوم")
    add("Breakout","60D Support",55*((r.Close-r.LOW60).abs()/r.Close<.035).astype(float),"قرب دعم 60 يوم")
    add("Breakout","60D Resistance",-55*((r.HIGH60-r.Close).abs()/r.Close<.035).astype(float),"قرب مقاومة 60 يوم")
    # 9) Multi-timeframe / confirmation — 6
    add("Confirmation","SMA20 Distance",np.clip(r.DIST_SMA20*10,-70,70),"المسافة عن SMA20")
    add("Confirmation","SMA50 Distance",np.clip(r.DIST_SMA50*7,-60,60),"المسافة عن SMA50")
    add("Confirmation","SMA200 Distance",np.clip(r.DIST_SMA200*3,-50,50),"المسافة عن SMA200")
    add("Confirmation","Trend + MACD",50*(((r.Close>r.SMA20)&(r.MACD>r.MACD_SIGNAL)).astype(float)-((r.Close<r.SMA20)&(r.MACD<r.MACD_SIGNAL)).astype(float)),"اتفاق الاتجاه وMACD")
    add("Confirmation","Trend + RSI",45*(((r.Close>r.SMA20)&(r.RSI14>50)).astype(float)-((r.Close<r.SMA20)&(r.RSI14<50)).astype(float)),"اتفاق الاتجاه وRSI")
    add("Confirmation","Triple Confirmation",70*(((r.Close>r.SMA20)&(r.MACD>r.MACD_SIGNAL)&(r.RSI14>50)).astype(float)-((r.Close<r.SMA20)&(r.MACD<r.MACD_SIGNAL)&(r.RSI14<50)).astype(float)),"سعر + MACD + RSI")
    # 10) Regime / quality — 6
    add("Regime","Trend Regime",45*(((r.SMA20>r.SMA50)&(r.RET20>0)).astype(float)-((r.SMA20<r.SMA50)&(r.RET20<0)).astype(float)),"نظام صاعد/هابط")
    add("Regime","Trend Strength",45*sgn(r.SLOPE20)*(r.ADX_PROXY>1.2).astype(float),"قوة الاتجاه")
    add("Regime","Momentum Regime",50*(((r.RSI14>50)&(r.RET20>0)).astype(float)-((r.RSI14<50)&(r.RET20<0)).astype(float)),"نظام الزخم")
    add("Regime","Volatility Regime",40*(((r.ATR_PCT<7)&(r.RET20>0)).astype(float)-((r.ATR_PCT>9)&(r.RET20<0)).astype(float)),"نظام التذبذب")
    add("Regime","Data Quality",np.where(r.notna().mean(axis=1)>.98,25,10),"جودة البيانات")
    add("Regime","Ensemble Consistency",50*(((r.Close>r.SMA20)&(r.MACD>r.MACD_SIGNAL)&(r.RSI14>50)).astype(float)-((r.Close<r.SMA20)&(r.MACD<r.MACD_SIGNAL)&(r.RSI14<50)).astype(float)),"اتساق الإشارات")
    return specs

GROUP_WEIGHTS={"Trend":0.16,"Momentum":0.13,"Mean Reversion":0.09,"Oscillators":0.10,"Volume":0.10,"Risk":0.10,"Price Action":0.10,"Breakout":0.09,"Confirmation":0.08,"Regime":0.05}

def evaluate_agents(specs, x):
    rows=[]
    last=x.iloc[-1]
    for a in specs:
        s=float(a["Score"].iloc[-1])
        reason=a["Reason"]
        if not isinstance(reason,str): reason=str(reason.iloc[-1]) if hasattr(reason,'iloc') else str(reason)
        rows.append({"Group":a["Group"],"Agent":a["Agent"],"Score":s,"Reason":reason})
    df=pd.DataFrame(rows)
    group=df.groupby("Group")["Score"].mean().to_dict()
    weighted=sum(GROUP_WEIGHTS[g]*group.get(g,0) for g in GROUP_WEIGHTS)
    group_agreement=sum(GROUP_WEIGHTS[g] for g in GROUP_WEIGHTS if abs(group.get(g,0))>=20)*100
    return df,group,weighted,group_agreement

def validation(specs,x,horizon):
    future=x.Close.shift(-horizon)/x.Close-1
    out=[]
    for a in specs:
        sig=a["Score"].shift(1)
        mask=sig.notna() & future.notna() & (sig.abs()>=20)
        if mask.sum()<25: hit=np.nan; n=int(mask.sum())
        else:
            hit=float((np.sign(sig[mask])==np.sign(future[mask])).mean()*100); n=int(mask.sum())
        out.append({"Group":a["Group"],"Agent":a["Agent"],"Hit rate %":hit,"Samples":n})
    v=pd.DataFrame(out)
    vg=v.groupby("Group")["Hit rate %"].mean().sort_values(ascending=False)
    return v,vg

def llm_review(ticker,r,group_df,decision):
    key=None
    try: key=st.secrets.get("OPENAI_API_KEY")
    except Exception: key=os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None: return None
    try:
        client=OpenAI(api_key=key)
        groups=group_df.to_dict("records")
        prompt=(f"راجع سهم {ticker} كمراجع كمي مستقل. السعر {r.Close:.2f}. القرار الآلي {decision}. "
                f"نتائج المجموعات: {json.dumps(groups,ensure_ascii=False)}. "
                "أعطني تحيزاً صاعداً/محايداً/هابطاً، 3 أسباب، خطرين، وهل توجد أفضلية دخول واضحة. "
                "لا تعط يقيناً ولا توصية مضمونة.")
        resp=client.responses.create(model="gpt-4.1-mini",input=prompt)
        return resp.output_text
    except Exception as e: return f"تعذر تشغيل مراجع LLM: {e}"

def decision_engine(weighted, group, agreement, atr_pct):
    bullish=sum(v>=20 for v in group.values()); bearish=sum(v<=-20 for v in group.values())
    # يمنع قراراً قوياً إذا كانت المخاطرة مرتفعة جداً أو المجموعات منقسمة.
    if atr_pct>=15: return "HIGH RISK / NO TRADE"
    if weighted>=28 and bullish>=6 and agreement>=60: return "LONG BIAS"
    if weighted<=-28 and bearish>=6 and agreement>=60: return "SHORT BIAS"
    return "NO TRADE"

# UI
with st.sidebar:
    st.header("إعدادات التحليل")
    ticker=st.text_input("رمز السهم الأمريكي",value="FIGR").strip().upper()
    interval=st.selectbox("الفاصل الزمني",list(INTERVAL_PERIODS.keys()),index=5)
    period=st.selectbox("الفترة التاريخية",INTERVAL_PERIODS[interval],index=min(2,len(INTERVAL_PERIODS[interval])-1))
    horizon=st.selectbox("أفق التحقق",[1,3,5,10],index=2,help="عدد الشموع التي نقيس بعدها هل كانت الإشارة صحيحة تاريخياً.")
    run=st.button("🔎 حلّل بـ60 محللاً",type="primary")

if run:
    try:
        raw=load_data(ticker,period,interval)
        if raw.empty: st.error("تعذر جلب البيانات. تحقق من الرمز والفاصل الزمني."); st.stop()
        x=add_indicators(raw)
        if len(x)<80: st.error(f"البيانات غير كافية بعد تجهيز المؤشرات ({len(x)} شمعة). زد الفترة."); st.stop()
        specs=build_agents(x)
        rdf,groups,weighted,agreement=evaluate_agents(specs,x)
        decision=decision_engine(weighted,groups,agreement,float(x.iloc[-1].ATR_PCT))
        val,vg=validation(specs,x,horizon)
        r=x.iloc[-1]; entry=float(r.Close); atrv=float(r.ATR14)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("السعر",f"${entry:.2f}"); c2.metric("قرار المحرك",decision); c3.metric("الدرجة الموزونة",f"{weighted:.1f}/100"); c4.metric("توافق المجموعات",f"{agreement:.0f}%")
        st.caption(f"تم تشغيل {len(rdf)} محللًا داخل 10 مجموعات. القرار لا يعتمد على متوسط 60 رقمًا متشابهًا؛ كل مجموعة لها وزن مستقل.")
        if decision=="LONG BIAS":
            st.success(f"تحيز صاعد | مرجع دخول ${entry:.2f} | وقف تقريبي ${entry-1.5*atrv:.2f} | TP1 ${entry+1.5*atrv:.2f} | TP2 ${entry+3*atrv:.2f}")
        elif decision=="SHORT BIAS":
            st.error(f"تحيز هابط | مرجع دخول ${entry:.2f} | وقف تقريبي ${entry+1.5*atrv:.2f} | TP1 ${entry-1.5*atrv:.2f} | TP2 ${entry-3*atrv:.2f}")
        else: st.warning("المحرك لم يجد أفضلية جماعية كافية للدخول الآن.")

        st.subheader("📊 درجات المجموعات")
        gdf=pd.DataFrame([{"Group":g,"Weight %":GROUP_WEIGHTS[g]*100,"Score":groups.get(g,0),"Weighted contribution":groups.get(g,0)*GROUP_WEIGHTS[g],"Historical hit rate %":vg.get(g,np.nan)} for g in GROUP_WEIGHTS])
        st.dataframe(gdf.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)

        st.subheader("🤖 مراجعة AI الاختيارية")
        group_for_llm=gdf[["Group","Score","Historical hit rate %"]].to_dict("records")
        review=llm_review(ticker,r,pd.DataFrame(group_for_llm),decision)
        if review: st.info(review)
        else: st.caption("مراجع LLM اختياري: أضف OPENAI_API_KEY في Streamlit Secrets. الـ60 محللًا الكميون لا يحتاجون API.")

        st.subheader("🔬 التحقق التاريخي")
        st.caption(f"Hit rate = نسبة المرات التي اتجه فيها العائد بعد {horizon} شموع في نفس اتجاه إشارة المحلل. هذا ليس Backtest كاملًا للأرباح ولا يشمل السبريد والانزلاق.")
        st.dataframe(val.sort_values(["Group","Hit rate %"],ascending=[True,False]),use_container_width=True,hide_index=True)

        st.subheader("🤖 جميع المحللين الـ60")
        st.dataframe(rdf.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)

        st.subheader("السعر والمؤشرات")
        cols=["Close","SMA20","SMA50"]
        if x["SMA200"].notna().any(): cols.append("SMA200")
        st.line_chart(x[cols].tail(240))

        st.subheader("ملخص القرار")
        st.json({"ticker":ticker,"interval":interval,"period":period,"last_bar":str(x.index[-1]),"decision":decision,"agents":len(rdf),"groups":len(groups),"weighted_score":round(weighted,1),"group_agreement_pct":round(agreement,1),"RSI14":round(float(r.RSI14),1),"ATR%":round(float(r.ATR_PCT),2),"volume_ratio":round(float(r.VOL_RATIO),2),"validation_horizon_bars":horizon})
        st.caption("نظام بحثي/تحليلي وليس توصية استثمارية. قوة النظام يجب أن تثبت باختبار Walk-Forward خارج العينة، وليس بكثرة عدد المحللين.")
    except Exception as e:
        st.exception(e)
else:
    st.markdown("""
### النسخة المعاد هندستها
- **60 محللًا كميًا فعليًا** موزعين على **10 عائلات تحليلية** لتقليل تكرار الإشارة.
- القرار النهائي لا يأخذ متوسط الـ60 مباشرة؛ بل يحسب **درجة كل مجموعة ثم يطبق أوزانًا مستقلة**.
- يوجد **توافق على مستوى المجموعات** حتى لا ينتج قرار قوي من مجموعة واحدة.
- أضفت **تحققًا تاريخيًا لكل محلل** على أفق 1/3/5/10 شموع لمعرفة من كان أداؤه أفضل.
- أضفت فواصل زمنية من **1 دقيقة إلى يومي**، مع فترات مناسبة لكل interval بدل استخدام 6 أشهر دائمًا.
- أضفت حاجز **HIGH RISK / NO TRADE** عندما يكون ATR مرتفعًا جدًا.
- مراجع GPT اختياري، وليس 60 استدعاء API.

**مهم:** كلمة AI في الواجهة تعني طبقة تحليل/Ensemble، أما الـ60 نفسهم فهم نماذج كمية Rule-Based وليست 60 نماذج GPT مستقلة.
""")
