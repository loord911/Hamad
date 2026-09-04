import os, json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="US Stock AI Agent Desk", page_icon="📈", layout="wide")
st.title("📈 US Stock AI Agent Desk — Re-Engineered")
st.caption("60 محللًا كميًا متخصصًا + تحليل متعدد الفواصل + مدير قرار. النظام تحليلي ولا يرسل أوامر تداول.")

# 4H is created by resampling 1H because Yahoo Finance does not provide a native 4H interval.
INTERVAL_PERIODS = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["5d", "1mo", "3mo"],
    "15m": ["1mo", "3mo", "6mo"],
    "30m": ["1mo", "3mo", "6mo"],
    "1h": ["1mo", "3mo", "6mo", "1y"],
    "4h": ["3mo", "6mo", "1y"],
    "1d": ["3mo", "6mo", "1y", "2y", "5y"],
    "1wk": ["6mo", "1y", "2y", "5y", "10y"],
}

MTF_FETCH = {
    "1m": ("1m", "5d"),
    "5m": ("5m", "1mo"),
    "15m": ("15m", "3mo"),
    "30m": ("30m", "3mo"),
    "1h": ("1h", "6mo"),
    "4h": ("1h", "1y"),
    "1D": ("1d", "2y"),
    "1W": ("1wk", "5y"),
}

@st.cache_data(ttl=180)
def load_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")

def resample_4h(df):
    if df.empty:
        return df
    x = df.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        return x
    rule = "4h"
    out = x.resample(rule, origin="start_day").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])
    return out

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
    x=x.replace([np.inf,-np.inf],np.nan)
    required=["Close","SMA20","SMA50","RSI14","ATR14","MACD","VOL20","BB_PCT","STO_K","CCI20","OBV","SLOPE20"]
    return x.dropna(subset=required)

def clamp_series(s):
    return s.replace([np.inf,-np.inf],np.nan).fillna(0).clip(-100,100)

def sgn(v): return np.sign(v).fillna(0)

def build_agents(x):
    r=x; specs=[]
    def add(group,name,score,reason): specs.append({"Group":group,"Agent":name,"Score":clamp_series(score),"Reason":reason})
    add("Trend","SMA Structure",35*sgn(r.Close-r.SMA20)+35*sgn(r.SMA20-r.SMA50)+20*sgn(r.SMA50-r.SMA200),"السعر وSMA20/50/200")
    add("Trend","EMA Structure",40*sgn(r.EMA20-r.EMA50)+30*sgn(r.Close-r.EMA20)+20*sgn(r.EMA10-r.EMA20),"EMA10/20/50 + السعر")
    add("Trend","Trend Slope 20",sgn(r.SLOPE20)*(np.abs(r.SLOPE20)/r.Close*10000),"ميل 20 شمعة")
    add("Trend","Trend Slope 50",sgn(r.SLOPE50)*(np.abs(r.SLOPE50)/r.Close*10000),"ميل 50 شمعة")
    add("Trend","Long-Term Position",30*sgn(r.Close-r.SMA200)+30*sgn(r.SMA50-r.SMA200)+20*sgn(r.RET60),"الموقع طويل الأجل")
    add("Trend","EMA Alignment",70*((r.EMA5>r.EMA10)&(r.EMA10>r.EMA20)&(r.EMA20>r.EMA50)).astype(float)-70*((r.EMA5<r.EMA10)&(r.EMA10<r.EMA20)&(r.EMA20<r.EMA50)).astype(float),"اصطفاف EMA")
    add("Momentum","RSI14 Momentum",(r.RSI14-50)*1.5,"RSI14")
    add("Momentum","RSI7 Momentum",(r.RSI7-50)*1.2,"RSI7")
    add("Momentum","ROC10",r.ROC10*7,"عائد 10 شموع")
    add("Momentum","ROC20",r.ROC20*5,"عائد 20 شمعة")
    add("Momentum","Short Return",r.RET5*10,"عائد قصير")
    add("Momentum","Medium Return",r.RET20*5,"عائد متوسط")
    z=(r.Close-r.SMA20)/r.Close.rolling(20).std().replace(0,np.nan)
    add("Mean Reversion","Z-Score Reversion",-z*35,"بعد السعر عن SMA20")
    add("Mean Reversion","Bollinger Reversion",-(r.BB_PCT-.5)*90,"موضع Bollinger")
    add("Mean Reversion","RSI Reversal",70*((r.RSI14.shift(1)<30)&(r.RSI14>r.RSI14.shift(1))).astype(float)-70*((r.RSI14.shift(1)>70)&(r.RSI14<r.RSI14.shift(1))).astype(float),"انعكاس RSI")
    add("Mean Reversion","Stochastic Reversal",75*((r.STO_K.shift(1)<20)&(r.STO_K>r.STO_K.shift(1))).astype(float)-75*((r.STO_K.shift(1)>80)&(r.STO_K<r.STO_K.shift(1))).astype(float),"انعكاس Stochastic")
    add("Mean Reversion","Oversold Bounce",65*((r.RSI14<35)&(r.RET1>0)).astype(float)-65*((r.RSI14>65)&(r.RET1<0)).astype(float),"ارتداد من التشبع")
    add("Mean Reversion","Overextension",-np.clip(r.DIST_SMA20*8,-100,100),"تمدد السعر عن المتوسط")
    add("Oscillators","MACD Direction",65*sgn(r.MACD-r.MACD_SIGNAL)+25*sgn(r.MACD_HIST),"MACD مقابل Signal")
    add("Oscillators","MACD Cross",80*((r.MACD>r.MACD_SIGNAL)&(r.MACD.shift(1)<=r.MACD_SIGNAL.shift(1))).astype(float)-80*((r.MACD<r.MACD_SIGNAL)&(r.MACD.shift(1)>=r.MACD_SIGNAL.shift(1))).astype(float),"تقاطع MACD")
    add("Oscillators","MACD Acceleration",55*sgn(r.MACD_HIST-r.MACD_HIST.shift(1)),"تسارع Histogram")
    add("Oscillators","Stochastic",55*sgn(r.STO_K-r.STO_D),"K مقابل D")
    add("Oscillators","CCI",np.clip(r.CCI20/2,-100,100),"CCI20")
    add("Oscillators","Williams",45*sgn(r.WILLR-r.WILLR.shift(1)),"Williams %R")
    add("Volume","Volume Confirmation",50*sgn(r.RET5)*np.clip(r.VOL_RATIO,0,2)/2,"الحجم مع العائد")
    add("Volume","Volume Surge",70*sgn(r.BODY_PCT)*(r.VOL_Z>2).astype(float),"طفرة حجم")
    add("Volume","OBV Trend",65*sgn(r.OBV-r.OBV_EMA20),"OBV مقابل EMA")
    add("Volume","MFI",np.where((r.MFI14>50)&(r.MFI14<80),45,np.where((r.MFI14<50)&(r.MFI14>20),-45,0)),"MFI14")
    add("Volume","Liquidity Proxy",np.clip((r.VOL_RATIO-1)*60,-60,60),"حجم مقابل متوسط 20")
    add("Volume","Volume Price Agreement",50*sgn(r.RET1)*np.clip(r.VOL_RATIO,0,2)/2,"اتفاق السعر والحجم")
    add("Risk","ATR Regime",np.where(r.ATR_PCT<5,30,np.where(r.ATR_PCT>9,-35,0)),"ATR%")
    add("Risk","ATR Relative",np.clip((r.ATR20_AVG-r.ATR_PCT)*15,-60,60),"ATR مقابل متوسطه")
    add("Risk","Range Expansion",55*sgn(r.BODY_PCT)*(r.RANGE_PCT>r.RANGE_PCT.rolling(20).mean()*1.5).astype(float),"اتساع النطاق")
    add("Risk","Volatility Breakout",60*sgn(r.RET5)*(r.ATR_PCT>r.ATR20_AVG*1.25).astype(float),"ارتفاع التذبذب مع الاتجاه")
    add("Risk","Candle Risk",-45*(r.ATR_PCT>10).astype(float),"مخاطر التذبذب")
    add("Risk","Risk/Trend Balance",np.where((r.Close>r.SMA20)&(r.ATR_PCT<8),40,np.where((r.Close<r.SMA20)&(r.ATR_PCT>8),-40,0)),"اتزان الاتجاه والمخاطر")
    add("Price Action","Candle Body",55*((r.BODY_PCT>1)&(r.CLOSE_LOC>.65)).astype(float)-55*((r.BODY_PCT<-1)&(r.CLOSE_LOC<.35)).astype(float),"جسم الشمعة")
    add("Price Action","Close Location",45*sgn(r.CLOSE_LOC-.5),"موقع الإغلاق داخل الشمعة")
    add("Price Action","Gap",35*sgn(r.GAP_PCT)*(r.GAP_PCT.abs()>1).astype(float),"الفجوة")
    add("Price Action","Higher High",55*((r.High>r.High.shift(1))&(r.Low>=r.Low.shift(1))).astype(float)-55*((r.High<r.High.shift(1))&(r.Low<=r.Low.shift(1))).astype(float),"Higher High/Lower Low")
    add("Price Action","Higher Low",60*((r.Low>r.Low.shift(1))&(r.Close>r.Close.shift(1))).astype(float)-60*((r.Low<r.Low.shift(1))&(r.Close<r.Close.shift(1))).astype(float),"Higher Low/Lower High")
    add("Price Action","Structure vs SMA",60*((r.Close>r.SMA20)&(r.SMA20>r.SMA50)).astype(float)-60*((r.Close<r.SMA20)&(r.SMA20<r.SMA50)).astype(float),"هيكل السعر")
    add("Breakout","20D Breakout",85*((r.Close>r.HIGH20).astype(float))-85*((r.Close<r.LOW20).astype(float)),"اختراق 20 يوم")
    add("Breakout","60D Breakout",90*((r.Close>r.HIGH60).astype(float))-90*((r.Close<r.LOW60).astype(float)),"اختراق 60 يوم")
    add("Breakout","20D Support",60*((r.Close-r.LOW20).abs()/r.Close<.025).astype(float),"قرب دعم 20 يوم")
    add("Breakout","20D Resistance",-60*((r.HIGH20-r.Close).abs()/r.Close<.025).astype(float),"قرب مقاومة 20 يوم")
    add("Breakout","60D Support",55*((r.Close-r.LOW60).abs()/r.Close<.035).astype(float),"قرب دعم 60 يوم")
    add("Breakout","60D Resistance",-55*((r.HIGH60-r.Close).abs()/r.Close<.035).astype(float),"قرب مقاومة 60 يوم")
    add("Confirmation","SMA20 Distance",np.clip(r.DIST_SMA20*10,-70,70),"المسافة عن SMA20")
    add("Confirmation","SMA50 Distance",np.clip(r.DIST_SMA50*7,-60,60),"المسافة عن SMA50")
    add("Confirmation","SMA200 Distance",np.clip(r.DIST_SMA200*3,-50,50),"المسافة عن SMA200")
    add("Confirmation","Trend + MACD",50*(((r.Close>r.SMA20)&(r.MACD>r.MACD_SIGNAL)).astype(float)-((r.Close<r.SMA20)&(r.MACD<r.MACD_SIGNAL)).astype(float)),"اتفاق الاتجاه وMACD")
    add("Confirmation","Trend + RSI",45*(((r.Close>r.SMA20)&(r.RSI14>50)).astype(float)-((r.Close<r.SMA20)&(r.RSI14<50)).astype(float)),"اتفاق الاتجاه وRSI")
    add("Confirmation","Triple Confirmation",70*(((r.Close>r.SMA20)&(r.MACD>r.MACD_SIGNAL)&(r.RSI14>50)).astype(float)-((r.Close<r.SMA20)&(r.MACD<r.MACD_SIGNAL)&(r.RSI14<50)).astype(float)),"سعر + MACD + RSI")
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
    for a in specs:
        s=float(a["Score"].iloc[-1])
        rows.append({"Group":a["Group"],"Agent":a["Agent"],"Score":s,"Reason":a["Reason"]})
    df=pd.DataFrame(rows)
    group=df.groupby("Group")["Score"].mean().to_dict()
    weighted=sum(GROUP_WEIGHTS[g]*group.get(g,0) for g in GROUP_WEIGHTS)
    group_agreement=sum(GROUP_WEIGHTS[g] for g in GROUP_WEIGHTS if abs(group.get(g,0))>=20)*100
    return df,group,weighted,group_agreement

def validation(specs,x,horizon):
    future=x.Close.shift(-horizon)/x.Close-1; out=[]
    for a in specs:
        sig=a["Score"].shift(1); mask=sig.notna() & future.notna() & (sig.abs()>=20)
        if mask.sum()<25: hit=np.nan; n=int(mask.sum())
        else: hit=float((np.sign(sig[mask])==np.sign(future[mask])).mean()*100); n=int(mask.sum())
        out.append({"Group":a["Group"],"Agent":a["Agent"],"Hit rate %":hit,"Samples":n})
    v=pd.DataFrame(out); vg=v.groupby("Group")["Hit rate %"].mean().sort_values(ascending=False)
    return v,vg

def decision_engine(weighted, group, agreement, atr_pct):
    bullish=sum(v>=20 for v in group.values()); bearish=sum(v<=-20 for v in group.values())
    if atr_pct>=15: return "HIGH RISK / NO TRADE"
    if weighted>=28 and bullish>=6 and agreement>=60: return "LONG BIAS"
    if weighted<=-28 and bearish>=6 and agreement>=60: return "SHORT BIAS"
    return "NO TRADE"

def find_levels(x):
    """Find clustered pivot support/resistance plus rolling highs/lows."""
    r=x.iloc[-1]; price=float(r.Close); atrv=float(r.ATR14)
    tol=max(atrv*0.65, price*0.004)
    highs=x["High"].rolling(3,center=True).max()
    lows=x["Low"].rolling(3,center=True).min()
    piv_high=x.loc[x["High"].eq(highs),"High"].dropna().tail(120).tolist()
    piv_low=x.loc[x["Low"].eq(lows),"Low"].dropna().tail(120).tolist()

    def cluster(values):
        levels=[]
        for val in sorted([float(v) for v in values]):
            if not levels or abs(val-levels[-1][-1])>tol:
                levels.append([val])
            else:
                levels[-1].append(val)
        return [float(np.mean(g)) for g in levels if len(g)>=1]

    supports=cluster(piv_low + [float(r.LOW20),float(r.LOW60),float(r.SMA20),float(r.SMA50)])
    resistances=cluster(piv_high + [float(r.HIGH20),float(r.HIGH60),float(r.SMA20),float(r.SMA50)])
    supports=sorted({round(v,2) for v in supports if v < price-tol*0.15}, reverse=True)
    resistances=sorted({round(v,2) for v in resistances if v > price+tol*0.15})
    s1=supports[0] if supports else round(price-1.5*atrv,2)
    s2=supports[1] if len(supports)>1 else round(price-3*atrv,2)
    r1=resistances[0] if resistances else round(price+1.5*atrv,2)
    r2=resistances[1] if len(resistances)>1 else round(price+3*atrv,2)
    return {"support1":s1,"support2":s2,"resistance1":r1,"resistance2":r2,"atr":atrv}

def trade_levels(price, decision, levels):
    s1,s2,r1,r2=levels["support1"],levels["support2"],levels["resistance1"],levels["resistance2"]
    atrv=levels["atr"]
    if decision=="LONG BIAS":
        stop=round(min(s1-0.25*atrv, price-1.25*atrv),2)
        tp1=round(r1,2) if r1>price else round(price+1.5*atrv,2)
        tp2=round(r2,2) if r2>tp1 else round(max(tp1+0.5*atrv,price+3*atrv),2)
    elif decision=="SHORT BIAS":
        stop=round(max(r1+0.25*atrv, price+1.25*atrv),2)
        tp1=round(s1,2) if s1<price else round(price-1.5*atrv,2)
        tp2=round(s2,2) if s2<tp1 else round(min(tp1-0.5*atrv,price-3*atrv),2)
    else:
        stop=tp1=tp2=np.nan
    return stop,tp1,tp2

def plot_price_chart(x, levels, decision, max_bars=180):
    d=x.tail(max_bars).copy(); fig=go.Figure()
    fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="السعر"))
    for col in ["SMA20","SMA50","SMA200"]:
        if col in d and d[col].notna().any():
            fig.add_trace(go.Scatter(x=d.index,y=d[col],mode="lines",name=col))
    lines=[("الدعم 1",levels["support1"]),("الدعم 2",levels["support2"]),("المقاومة 1",levels["resistance1"]),("المقاومة 2",levels["resistance2"])]
    for name,val in lines:
        fig.add_hline(y=val,annotation_text=f"{name}: {val:.2f}",annotation_position="top left",line_dash="dot")
    if decision in ("LONG BIAS","SHORT BIAS"):
        stop,tp1,tp2=trade_levels(float(x.Close.iloc[-1]),decision,levels)
        for name,val in [("وقف",stop),("TP1",tp1),("TP2",tp2)]:
            fig.add_hline(y=val,annotation_text=f"{name}: {val:.2f}",annotation_position="bottom right",line_dash="dash")
    fig.update_layout(height=620,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10),legend=dict(orientation="h"))
    return fig

def direction_from_x(x):
    if len(x)<60: return "بيانات غير كافية",0.0
    specs=build_agents(x); _,groups,weighted,agreement=evaluate_agents(specs,x)
    if weighted>=20 and agreement>=50: bias="صاعد"
    elif weighted<=-20 and agreement>=50: bias="هابط"
    else: bias="محايد"
    return bias,float(weighted)

def multi_timeframe(ticker):
    rows=[]
    for label,(fetch_interval,period) in MTF_FETCH.items():
        try:
            raw=load_data(ticker,period,fetch_interval)
            if label=="4h": raw=resample_4h(raw)
            x=add_indicators(raw)
            bias,score=direction_from_x(x)
            rows.append({"الفاصل":label,"الاتجاه":bias,"الدرجة":round(score,1),"الشموع":len(x)})
        except Exception as e:
            rows.append({"الفاصل":label,"الاتجاه":"خطأ","الدرجة":np.nan,"الشموع":0})
    return pd.DataFrame(rows)

def llm_review(ticker,r,group_df,decision,levels):
    key=None
    try: key=st.secrets.get("OPENAI_API_KEY")
    except Exception: key=os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None: return None
    try:
        client=OpenAI(api_key=key)
        groups=group_df.to_dict("records")
        prompt=(f"راجع سهم {ticker} كمراجع كمي مستقل. السعر {r.Close:.2f}. القرار {decision}. "
                f"الدعم {levels['support1']:.2f}/{levels['support2']:.2f} والمقاومة {levels['resistance1']:.2f}/{levels['resistance2']:.2f}. "
                f"نتائج المجموعات: {json.dumps(groups,ensure_ascii=False)}. "
                "أعطني تحيزاً صاعداً/محايداً/هابطاً، 3 أسباب، خطرين، وهل توجد أفضلية دخول واضحة. لا تعط يقيناً ولا توصية مضمونة.")
        resp=client.responses.create(model="gpt-4.1-mini",input=prompt)
        return resp.output_text
    except Exception as e: return f"تعذر تشغيل مراجع LLM: {e}"

AGENT_GUIDE = {
"Trend":"يقيس الاتجاه عبر SMA/EMA، الميل، والموقع طويل الأجل.",
"Momentum":"يقيس قوة الحركة الحالية عبر RSI وROC والعوائد القصيرة والمتوسطة.",
"Mean Reversion":"يبحث عن تمدد السعر عن المتوسط واحتمالات الارتداد.",
"Oscillators":"يفحص MACD وStochastic وCCI وWilliams %R.",
"Volume":"يفحص تأكيد الحجم للحركة باستخدام Volume Ratio وOBV وMFI.",
"Risk":"يقيس ATR والتذبذب واتساع النطاق وموازنة المخاطر مع الاتجاه.",
"Price Action":"يحلل جسم الشمعة، مكان الإغلاق، الفجوات وبنية القمم والقيعان.",
"Breakout":"يفحص اختراقات 20 و60 فترة وقرب السعر من الدعم والمقاومة.",
"Confirmation":"يبحث عن اتفاق مستقل بين الاتجاه وMACD وRSI والمسافات عن المتوسطات.",
"Regime":"يصنف نظام السوق: اتجاه، زخم، تذبذب، وجودة البيانات واتساق المجموعة."
}

with st.sidebar:
    st.header("إعدادات التحليل")
    ticker=st.text_input("رمز السهم الأمريكي",value="FIGR").strip().upper()
    interval_label=st.selectbox("الفاصل الأساسي",list(INTERVAL_PERIODS.keys()),index=5)
    period=st.selectbox("الفترة التاريخية",INTERVAL_PERIODS[interval_label],index=min(2,len(INTERVAL_PERIODS[interval_label])-1))
    horizon=st.selectbox("أفق التحقق التاريخي",[1,3,5,10],index=2)
    run=st.button("🔎 حلّل السهم",type="primary")

if run:
    try:
        actual_interval=interval_label
        raw=load_data(ticker,period,"1h" if interval_label=="4h" else interval_label)
        if interval_label=="4h": raw=resample_4h(raw)
        if raw.empty: st.error("تعذر جلب البيانات. تحقق من الرمز والفاصل الزمني."); st.stop()
        x=add_indicators(raw)
        if len(x)<80: st.error(f"البيانات غير كافية بعد تجهيز المؤشرات ({len(x)} شمعة). زد الفترة."); st.stop()
        specs=build_agents(x); rdf,groups,weighted,agreement=evaluate_agents(specs,x)
        decision=decision_engine(weighted,groups,agreement,float(x.iloc[-1].ATR_PCT))
        val,vg=validation(specs,x,horizon); r=x.iloc[-1]; price=float(r.Close)
        levels=find_levels(x); stop,tp1,tp2=trade_levels(price,decision,levels)

        st.subheader("🧭 اتجاه السوق متعدد الفواصل")
        mtf=multi_timeframe(ticker)
        st.dataframe(mtf,use_container_width=True,hide_index=True)
        st.caption("4H يتم بناؤه من شموع الساعة. الهدف من MTF هو منع قرار قصير الأجل من تجاهل الاتجاه الأكبر.")

        st.subheader("📌 القرار ومستويات السعر")
        c1,c2,c3=st.columns(3)
        c1.metric("السعر",f"${price:.2f}")
        c2.metric("قرار المحرك",decision)
        c3.metric("درجة المحرك",f"{weighted:.1f}/100")
        if decision=="LONG BIAS":
            st.success(f"سيناريو صاعد | دخول مرجعي ${price:.2f} | وقف ${stop:.2f} | الهدف 1 ${tp1:.2f} | الهدف 2 ${tp2:.2f}")
        elif decision=="SHORT BIAS":
            st.error(f"سيناريو هابط | دخول مرجعي ${price:.2f} | وقف ${stop:.2f} | الهدف 1 ${tp1:.2f} | الهدف 2 ${tp2:.2f}")
        else:
            st.warning("لا توجد أفضلية جماعية كافية للدخول الآن؛ المستويات المعروضة للتخطيط وليست إشارة دخول.")

        l1,l2,l3,l4=st.columns(4)
        l1.metric("الدعم 1",f"${levels['support1']:.2f}")
        l2.metric("الدعم 2",f"${levels['support2']:.2f}")
        l3.metric("المقاومة 1",f"${levels['resistance1']:.2f}")
        l4.metric("المقاومة 2",f"${levels['resistance2']:.2f}")

        st.subheader("📈 الشارت التحليلي")
        st.plotly_chart(plot_price_chart(x,levels,decision),use_container_width=True)
        st.caption("المستويات ناتجة من قمم/قيعان محلية متجمعة مع مستويات 20/60 فترة والمتوسطات، والأهداف تستخدم أقرب مقاومات/دعوم متاحة مع ATR كبديل عند غياب مستوى مناسب.")

        st.subheader("📊 درجات المجموعات")
        gdf=pd.DataFrame([{"Group":g,"Weight %":GROUP_WEIGHTS[g]*100,"Score":groups.get(g,0),"Weighted contribution":groups.get(g,0)*GROUP_WEIGHTS[g],"Historical hit rate %":vg.get(g,np.nan)} for g in GROUP_WEIGHTS])
        st.dataframe(gdf.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)

        st.subheader("🤖 مراجعة AI الاختيارية")
        review=llm_review(ticker,r,gdf[["Group","Score","Historical hit rate %"]],decision,levels)
        if review: st.info(review)
        else: st.caption("مراجع LLM اختياري: أضف OPENAI_API_KEY في Streamlit Secrets. الـ60 محللًا الكميون لا يحتاجون API.")

        st.subheader("🔬 التحقق التاريخي")
        st.caption(f"Hit rate = نسبة المرات التي اتجه فيها العائد بعد {horizon} شموع في نفس اتجاه الإشارة. هذا ليس Backtest كاملًا للأرباح ولا يشمل السبريد والانزلاق.")
        st.dataframe(val.sort_values(["Group","Hit rate %"],ascending=[True,False]),use_container_width=True,hide_index=True)

        st.subheader("🤖 جميع المحللين الـ60")
        st.dataframe(rdf.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)

        with st.expander("📚 شرح كيف يعمل كل وكيل",expanded=False):
            for group,desc in AGENT_GUIDE.items():
                st.markdown(f"**{group} — 6 محللين:** {desc}")
            st.caption("كل مجموعة تحتوي 6 محللين. الدرجات ليست احتمالات ربح؛ هي إشارات معيارية من -100 إلى +100 تستخدم داخل محرك التجميع.")

        st.subheader("📋 بيانات القرار")
        st.json({"ticker":ticker,"interval":interval_label,"period":period,"last_bar":str(x.index[-1]),"decision":decision,"agents":len(rdf),"groups":len(groups),"weighted_score":round(weighted,1),"group_agreement_pct":round(agreement,1),"support1":levels["support1"],"support2":levels["support2"],"resistance1":levels["resistance1"],"resistance2":levels["resistance2"],"RSI14":round(float(r.RSI14),1),"ATR%":round(float(r.ATR_PCT),2),"volume_ratio":round(float(r.VOL_RATIO),2)})
    except Exception as e:
        st.exception(e)
else:
    st.markdown("""
### النسخة الجديدة
- **60 محللًا كميًا فعليًا** موزعين على 10 عائلات تحليلية، 6 محللين لكل عائلة.
- **الشارت أصبح شموعًا** مع SMA20/50/200 ومستويات الدعم والمقاومة.
- النظام يستخرج **دعم 1/2 ومقاومة 1/2** من بنية السعر والقمم والقيعان ومستويات 20/60 فترة.
- عند وجود LONG/SHORT: يعرض **دخولًا مرجعيًا + وقف + TP1 + TP2** بدل رقم ثقة مضلل.
- يوجد **تحليل متعدد الفواصل: 1m، 5m، 15m، 30m، 1h، 4h، 1D، 1W**.
- **4H يُبنى من بيانات الساعة** لأن مصدر Yahoo لا يوفره كفاصل أصلي.
- تبقى أدوات التحقق التاريخي وجدول الـ60 محللًا متاحة للتدقيق.
- يوجد قسم **شرح كيف يعمل كل وكيل**.

**مهم:** الـ60 محللًا هنا نماذج كمية Rule-Based وليست 60 نسخة مستقلة من GPT. كلمة AI تشير إلى منظومة التحليل/التجميع، ومراجع GPT اختياري.
""")

st.divider()
st.subheader("📞 التواصل معي")
st.markdown("إذا عندك ملاحظة، اقتراح، أو تريد الإبلاغ عن مشكلة في التحليل، استخدم وسيلة التواصل التي يضعها صاحب الموقع هنا.")
contact = st.text_input("رابط التواصل (اختياري)", placeholder="https://...")
if contact:
    st.markdown(f"[فتح صفحة التواصل]({contact})")

st.subheader("⚠️ تنبيه")
st.caption("هذا نظام بحثي/تحليلي وليس توصية استثمارية. مستويات الدعم والمقاومة والأهداف تقديرات منهجية وليست ضمانًا للسعر المستقبلي. كثرة المحللين لا تعني دقة أعلى تلقائيًا؛ يجب اختبار النظام خارج العينة واحتساب العمولات والانزلاق.")
