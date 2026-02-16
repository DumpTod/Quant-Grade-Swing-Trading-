# ============================================================
# app.py — World-Class Quant Scanner (Render Deployment)
# ============================================================
# This single file contains the entire scanner + Flask API
# Optimized for Render.com free tier deployment

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import time
import json
import os
import traceback
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from filterpy.kalman import KalmanFilter as KF
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantScanner")

# ============================================================
# STOCK LISTS
# ============================================================
FNO_STOCKS = {
    "AARTIIND","ABB","ABBOTINDIA","ABCAPITAL","ABFRL",
    "ACC","ADANIENT","ADANIPORTS","ALKEM","AMBUJACEM",
    "APOLLOHOSP","APOLLOTYRE","ASHOKLEY","ASIANPAINT",
    "ASTRAL","ATUL","AUBANK","AUROPHARMA","AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BALKRISIND",
    "BALRAMCHIN","BANDHANBNK","BANKBARODA","BATAINDIA",
    "BEL","BERGEPAINT","BHARATFORG","BHARTIARTL","BHEL",
    "BIOCON","BOSCHLTD","BPCL","BRITANNIA","BSOFT",
    "CANBK","CANFINHOME","CHAMBLFERT","CHOLAFIN",
    "CIPLA","COALINDIA","COFORGE","COLPAL","CONCOR",
    "COROMANDEL","CROMPTON","CUB","CUMMINSIND","DABUR",
    "DALBHARAT","DEEPAKNTR","DELTACORP","DIVISLAB",
    "DIXON","DLF","DRREDDY","EICHERMOT","ESCORTS",
    "EXIDEIND","FEDERALBNK","GAIL","GLENMARK",
    "GMRAIRPORT","GNFC","GODREJCP","GODREJPROP",
    "GRANULES","GRASIM","GUJGASLTD","HAL","HAVELLS",
    "HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDCOPPER","HINDPETRO",
    "HINDUNILVR","HONAUT","IBULHSGFIN","ICICIBANK",
    "ICICIGI","ICICIPRULI","IDEA","IDFCFIRSTB",
    "IEX","IGL","INDHOTEL","INDIACEM","INDIAMART",
    "INDIGO","INDUSINDBK","INDUSTOWER","INFY",
    "IOC","IPCALAB","IRCTC","ITC","JINDALSTEL",
    "JKCEMENT","JSWSTEEL","JUBLFOOD","KOTAKBANK",
    "LALPATHLAB","LAURUSLABS","LICHSGFIN","LT",
    "LTIM","LTTS","LUPIN","M&M","M&MFIN",
    "MANAPPURAM","MARICO","MARUTI","UNITDSPR",
    "MCX","METROPOLIS","MFSL","MGL","MOTHERSON",
    "MPHASIS","MRF","MUTHOOTFIN","NATIONALUM",
    "NAUKRI","NAVINFLUOR","NESTLEIND","NMDC",
    "NTPC","OBEROIRLTY","OFSS","ONGC","PAGEIND",
    "PEL","PERSISTENT","PETRONET","PFC","PIDILITIND",
    "PIIND","PNB","POLYCAB","POWERGRID","PVRINOX",
    "RAMCOCEM","RBLBANK","RECLTD","RELIANCE",
    "SAL","SBICARD","SBILIFE","SBIN","SHREECEM",
    "SHRIRAMFIN","SIEMENS","SRF","SUNPHARMA",
    "SUNTV","SYNGENE","TATACHEM","TATACOMM",
    "TATACONSUM","TATAMOTORS","TATAPOWER","TATASTEEL",
    "TCS","TECHM","TITAN","TORNTPHARM","TORNTPOWER",
    "TRENT","TVSMOTOR","UBL","ULTRACEMCO","UPL",
    "VEDL","VOLTAS","WIPRO","ZEEL","ZYDUSLIFE",
    "ADANIENSOL","ADANIGREEN","ADANIPOWER",
    "ATGL","AWL","CGPOWER","DELHIVERY","FACT",
    "GSPL","HUDCO","INDIANB",
    "IRFC","JIOFIN","JSWENERGY","KALYANKJIL",
    "KEI","LICI","LODHA","MAXHEALTH",
    "NHPC","PAYTM","POLICYBZR",
    "POONAWALLA","PRESTIGE","RVNL","SAIL",
    "SJVN","SONACOMS","SUPREMEIND","SYRMA",
    "TATAINVEST","TIINDIA","UNIONBANK","VBL",
    "YESBANK","ZOMATO"
}

ALL_STOCKS = [
    "360ONE","3MINDIA","5PAISA","AARTIDRUGS","AARTIIND","AAVAS",
    "ABB","ABBOTINDIA","ABCAPITAL","ABFRL","ABREL","ACC","ADANIENT",
    "ADANIENSOL","ADANIGREEN","ADANIPORTS","ADANIPOWER","ATGL",
    "ADVENZYMES","AEGISLOG","AFFLE","AIAENG","AJANTPHARM",
    "AKZOINDIA","ALEMBICLTD","ALKEM","ALKYLAMINE","ALLCARGO",
    "ALOKINDS","ARE&M","AMBUJACEM","ANANDRATHI","ANGELONE",
    "ANURAS","APLAPOLLO","APOLLOHOSP","APOLLOTYRE","APTUS",
    "ASAHIINDIA","ASHOKLEY","ASIANPAINT","ASTERDM","ASTRAL",
    "ATUL","AUBANK","AUROPHARMA","AVANTIFEED","AWL","AXISBANK",
    "BAJAJ-AUTO","BAJAJCON","BAJAJFINSV","BAJAJHLDNG","BAJFINANCE",
    "BALKRISIND","BALRAMCHIN","BANDHANBNK","BANKBARODA","BANKINDIA",
    "BASF","BATAINDIA","BAYERCROP","BDL","BEL","BERGEPAINT",
    "BHARATFORG","BHARTIARTL","BHEL","BIOCON","BIRLACORPN",
    "ETERNAL","BLUESTARCO","BOSCHLTD","BPCL","BRIGADE",
    "BRITANNIA","BSE","BSOFT","RBA",
    "CAMS","CANBK","CANFINHOME","CAPLIPOINT","CARBORUNIV",
    "CASTROLIND","CEATLTD","CENTRALBK","ABREL","CENTURYPLY",
    "CERA","CGCL","CGPOWER","CHAMBLFERT","CHEMPLASTS",
    "CHOLAFIN","CIPLA","CLEAN","COALINDIA","COCHINSHIP",
    "COFORGE","COLPAL","CONCOR","CONCORDBIO","COROMANDEL",
    "CRAFTSMAN","CREDITACC","CRISIL","CROMPTON","CSBBANK",
    "CUB","CUMMINSIND","CYIENT","DABUR","DALBHARAT",
    "DATAPATTNS","DCMSHRIRAM","DEEPAKNTR","DEEPAKFERT",
    "DELHIVERY","DELTACORP","DEVYANI","DHANI",
    "DIVISLAB","DIXON","DLF","DMART","DRREDDY",
    "EIDPARRY","EICHERMOT","ELGIEQUIP","EMAMILTD","EMCURE",
    "ENDURANCE","ENGINERSIN","EPL","EQUITASBNK","ERIS",
    "ESCORTS","EXIDEIND","FACT","FEDERALBNK","FINCABLES",
    "FINPIPE","FSL","FLUOROCHEM","FMGOETZE",
    "FORTIS","GAIL","GALAXYSURF","GARFIBRES","GICRE",
    "GILLETTE","GLAXO","GLENMARK","GMRAIRPORT",
    "GNFC","GODFRYPHLP","GODREJCP","GODREJIND","GODREJPROP",
    "GPPL","GRANULES","GRAPHITE","GRASIM","GRINDWELL",
    "GRINFRA","GSPL","GUJALKALI","GUJGASLTD","HAL",
    "HAPPSTMNDS","HATSUN","HAVELLS","HCLTECH","HDFCAMC",
    "HDFCBANK","HDFCLIFE","HEMIPROP","HEROMOTOCO",
    "HFCL","HIKAL","HINDALCO","HINDCOPPER","HINDPETRO",
    "HINDUNILVR","HINDWAREAP","HONAUT","HSCL","HUDCO",
    "IBREALEST","IBULHSGFIN","ICICIBANK","ICICIGI","ICICIPRULI",
    "IDEA","IDFCFIRSTB","IEX","IGL",
    "IIFL","INDHOTEL","INDIACEM","INDIAMART",
    "INDIANB","INDIGO","INDUSINDBK","INDUSTOWER","INFY",
    "INGERRAND","INTELLECT","IOC","IPCALAB","IRCTC",
    "IRFC","ISEC","ITC","ITCHOTELS","ITI",
    "JBCHEPHARM","JBMA","JINDALSTEL","JIOFIN","JKCEMENT",
    "JKLAKSHMI","JKPAPER","JMFINANCIL","JSL","JSWENERGY",
    "JSWINFRA","JSWSTEEL","JTEKTINDIA","JUBLFOOD","JUSTDIAL",
    "JYOTHYLAB","KAJARIACER","KALPATARU","KALYANKJIL",
    "KANSAINER","KAYNES","KEC","KEI","KERNEX",
    "KFINTECH","KIRLOSENG","KNRCON","KPITTECH","KSB",
    "KOTAKBANK","KRBL","KSCL","LAOPALA","LALPATHLAB",
    "LATENTVIEW","LAURUSLABS","LXCHEM","LEMONTREE",
    "LICHSGFIN","LICI","LINDEINDIA","LLOYDSME","LODHA",
    "LT","LTIM","LTTS","LUPIN","LUXIND",
    "M&M","M&MFIN","MAHABANK","CIEINDIA","MAHLIFE",
    "MAHLOG","MANAPPURAM","MAPMYINDIA","MARICO","MARKSANS",
    "MARUTI","MASTEK","MAXHEALTH","MAZDOCK","MCX",
    "UNITDSPR","METROPOLIS","MFSL","MGL","MIDHANI",
    "MINDACORP","MOTHERSON","MPHASIS",
    "MRF","MSUMI","MTARTECH","MUTHOOTFIN",
    "NATCOPHARM","NATIONALUM","NAUKRI","NAVINFLUOR","NAVNETEDUL",
    "NCC","NESTLEIND","NETWORK18","NEWGEN","NHPC",
    "NIACL","NLCINDIA","NMDC","NOCIL","NTPC",
    "NUVOCO","OBEROIRLTY","OFSS","OIL","OLECTRA",
    "ONGC","ORIENTELEC","PAGEIND","PAISALO","PATANJALI",
    "PAYTM","PCBL","PEL","PERSISTENT","PETRONET",
    "PFC","PFIZER","PGHH","PIDILITIND","PIIND",
    "PNB","POLICYBZR","POLYCAB","POONAWALLA","POWERGRID",
    "POWERINDIA","PRESTIGE","PRINCEPIPE","PRSMJOHNSN",
    "PNBHOUSING","PVRINOX","QUESS","RADICO","RAIN",
    "RAJESHEXPO","RALLIS","RAMCOCEM","RKFORGE","RATNAMANI",
    "RAYMOND","RBLBANK","RCF","RECLTD","REDINGTON",
    "RELAXO","RELIANCE","RELIGARE","RENUKA",
    "RITES","ROUTE","RVNL","SAREGAMA","SBICARD",
    "SBILIFE","SBIN","SCHAEFFLER","SCHNEIDER",
    "SHILPAMED","SHREECEM","SHRIRAMFIN","SIEMENS",
    "SJVN","SKFINDIA","SOBHA","SOLARINDS","SONACOMS",
    "SONATSOFTW","SOUTHBANK","SPARC","SRF","STAR",
    "STARHEALTH","STEL","STLTECH","SUDARSCHEM","SUMICHEM",
    "SUNDARMFIN","SUNDRMFAST","SUNPHARMA","SUNTV",
    "SUPREMEIND","SUVEN","SWSOLAR",
    "SYNGENE","SYRMA","TANLA","TARSONS","TATACHEM",
    "TATACOMM","TATACONSUM","TATAELXSI","TATAINVEST",
    "TATAMOTORS","TMCV","TMPV","TATAPOWER","TATASTEEL","TATATECH",
    "TCS","TECHM","THERMAX","THYROCARE","TIINDIA",
    "TIMKEN","TITAN","TORNTPHARM","TORNTPOWER","TRENT",
    "TRIDENT","TRIVENI","TRITURBINE","TTML","TVSMOTOR",
    "UBL","ULTRACEMCO","UNIONBANK","UNITDSPR","UPL",
    "USHAMART","UTIAMC","UJJIVANSFB","VAKRANGEE",
    "VARROC","VBL","VEDL","VENKEYS","VGUARD",
    "VINATIORGA","VIPIND","VOLTAS","VSTIND",
    "WELCORP","WELSPUNLIV","WHIRLPOOL","WIPRO",
    "WOCKPHARMA","YESBANK","ZEEL","ZENSARTECH",
    "ZFCVINDIA","ZODIACLOTH","ZOMATO","ZYDUSLIFE","ZYDUSWELL",
    "SAIL","SUZLON","IREDA","MANKIND","JSWSTEEL",
    "HCLTECH","MAPMYINDIA","AWL","CGPOWER","PPLPHARMA",
    "COCHINSHIP","BSE","CDSL","GRSE","MAZDOCK"
]

ALL_STOCKS = list(dict.fromkeys(ALL_STOCKS))

# ============================================================
# DATA FETCHER
# ============================================================
class DataFetcher:
    def __init__(self):
        self.cache = {}
        self.lock = Lock()
        self.failed_symbols = set()
        self.MIN_BARS = {'1h': 100, '4h': 50, '1d': 200}
        self.SYMBOL_OVERRIDES = {
            "M&M": "M%26M.NS",
            "M&MFIN": "M%26MFIN.NS",
            "ARE&M": "ARE%26M.NS",
            "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
        }

    def _convert(self, symbol):
        if symbol in self.SYMBOL_OVERRIDES:
            return self.SYMBOL_OVERRIDES[symbol]
        if "&" in symbol:
            return f"{symbol.replace('&', '%26')}.NS"
        return f"{symbol}.NS"

    def _resample_4h(self, df):
        if df is None or df.empty: return None
        try:
            r = df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna(subset=['Open','Close'])
            return r if len(r) >= self.MIN_BARS['4h'] else None
        except: return None

    def fetch_one(self, symbol, retries=3):
        yf_sym = self._convert(symbol)
        result = {'symbol': symbol, '1h': None, '4h': None, '1d': None}
        for attempt in range(retries):
            try:
                t = yf.Ticker(yf_sym)
                dd = t.history(period="2y", interval="1d", auto_adjust=True)
                if dd is not None and len(dd) >= self.MIN_BARS['1d']:
                    dd = dd[['Open','High','Low','Close','Volume']].dropna()
                    result['1d'] = dd
                dh = t.history(period="59d", interval="1h", auto_adjust=True)
                if dh is not None and len(dh) >= self.MIN_BARS['1h']:
                    dh = dh[['Open','High','Low','Close','Volume']].dropna()
                    result['1h'] = dh
                    result['4h'] = self._resample_4h(dh)
                if result['1d'] is not None:
                    with self.lock: self.cache[symbol] = result
                    return result
            except:
                if attempt < retries - 1: time.sleep(1*(attempt+1))
        with self.lock: self.failed_symbols.add(symbol)
        return result

    def fetch_all(self, stocks, workers=5):
        data = {}
        total = len(stocks)
        done = 0
        logger.info(f"Fetching {total} stocks...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.fetch_one, s): s for s in stocks}
            for f in as_completed(futs):
                done += 1
                try:
                    r = f.result()
                    if r['1d'] is not None: data[r['symbol']] = r
                except: pass
                if done % 100 == 0 or done == total:
                    logger.info(f"  Progress: {done}/{total} | Loaded: {len(data)}")
        logger.info(f"Fetch complete: {len(data)}/{total}")
        return data

# ============================================================
# FACTOR MODEL
# ============================================================
class FactorModel:
    def __init__(self):
        self.weights = {'momentum':0.30,'quality':0.20,'value':0.15,'volume_trend':0.15,'risk_adj_momentum':0.20}

    def score(self, daily):
        if daily is None or len(daily) < 252: return None
        c = daily['Close']; v = daily['Volume']; r = c.pct_change().dropna()
        factors = {}
        try: factors['momentum'] = (c.iloc[-21]/c.iloc[-252]) - 1
        except: factors['momentum'] = np.nan
        factors['quality'] = -r.iloc[-90:].std() if len(r)>=90 else np.nan
        h52 = c.iloc[-252:].max()
        factors['value'] = c.iloc[-1]/h52 if h52>0 else np.nan
        v5 = v.iloc[-5:].mean(); v50 = v.iloc[-50:].mean()
        factors['volume_trend'] = v5/v50 if v50>0 and len(v)>=50 else np.nan
        r60 = r.iloc[-60:]
        factors['risk_adj_momentum'] = r60.mean()/r60.std() if len(r60)>=60 and r60.std()>0 else np.nan
        valid = sum(1 for x in factors.values() if not np.isnan(x))
        return factors if valid >= 3 else None

    def rank(self, all_data):
        raw = {}
        for sym, d in all_data.items():
            f = self.score(d.get('1d'))
            if f: raw[sym] = f
        if len(raw) < 10: return {}
        df = pd.DataFrame(raw).T
        z = pd.DataFrame(index=df.index)
        for col in df.columns:
            v = df[col].dropna()
            if len(v) > 5 and v.std() > 0: z[col] = (df[col]-v.mean())/v.std()
            else: z[col] = 0.0
        z = z.fillna(0.0)
        comp = sum(z[f]*w for f,w in self.weights.items() if f in z.columns)
        result = {}
        for sym in comp.index:
            result[sym] = {'composite_score': comp[sym], 'rank_percentile': 0, 'factors': {c: z.loc[sym,c] for c in z.columns}, 'raw_factors': raw[sym]}
        sorted_s = sorted(result.keys(), key=lambda s: result[s]['composite_score'], reverse=True)
        n = len(sorted_s)
        for i, s in enumerate(sorted_s): result[s]['rank_percentile'] = (n-i)/n*100
        return result

# ============================================================
# REGIME DETECTOR (HMM)
# ============================================================
class RegimeDetector:
    def __init__(self):
        self.market_regime = None
        self.market_regime_probs = {}
        self.scaler = StandardScaler()

    def _features(self, daily):
        if daily is None or len(daily) < 100: return None
        try:
            c = daily['Close'].values.astype(float)
            v = daily['Volume'].values.astype(float)
            lr = np.diff(np.log(c+1e-10))
            rv = pd.Series(lr).rolling(10).std().values
            vs = pd.Series(v[1:])
            vm = vs.rolling(20).mean(); vstd = vs.rolling(20).std()
            vz = ((vs-vm)/(vstd+1e-10)).values
            s = 20
            feat = np.column_stack([lr[s:], rv[s:], vz[s:]])
            mask = np.all(np.isfinite(feat), axis=1)
            feat = feat[mask]
            return self.scaler.fit_transform(feat) if len(feat) >= 50 else None
        except: return None

    def _label(self, model):
        means = model.means_[:,0]
        si = np.argsort(means)
        if model.n_components == 3: return {si[0]:'bear', si[1]:'sideways', si[2]:'bull'}
        return {si[0]:'bear', si[1]:'bull'}

    def fit_market(self, nifty):
        feat = self._features(nifty)
        if feat is None:
            self.market_regime = 'unknown'
            self.market_regime_probs = {'bull':0.33,'bear':0.33,'sideways':0.34}
            return
        best_m, best_s = None, -np.inf
        for ns in [3, 2]:
            for seed in range(10):
                try:
                    m = GaussianHMM(n_components=ns, covariance_type='diag', n_iter=200, random_state=seed*42, tol=0.001, verbose=False)
                    m.fit(feat)
                    sc = m.score(feat)
                    if sc > best_s: best_s = sc; best_m = m
                except: continue
            if best_m: break
        if not best_m:
            self.market_regime = 'unknown'
            self.market_regime_probs = {'bull':0.33,'bear':0.33,'sideways':0.34}
            return
        states = best_m.predict(feat)
        probs = best_m.predict_proba(feat)
        labels = self._label(best_m)
        self.market_regime = labels[states[-1]]
        self.market_regime_probs = {labels[i]: float(probs[-1,i]) for i in range(best_m.n_components)}
        if 'sideways' not in self.market_regime_probs:
            bp = self.market_regime_probs.get('bull',0.5)
            brp = self.market_regime_probs.get('bear',0.5)
            if abs(bp-brp)<0.3: self.market_regime='sideways'
        logger.info(f"Market Regime: {self.market_regime.upper()} | Probs: {self.market_regime_probs}")

    def filter(self, signal_type):
        if not self.market_regime or self.market_regime=='unknown': return True, 0.5, "Unknown regime"
        m = self.market_regime; p = self.market_regime_probs
        if signal_type=='buy':
            if m=='bull': return True, p.get('bull',0.5), f"Bull ({p.get('bull',0):.0%})"
            elif m=='sideways': return True, p.get('sideways',0.3)*0.7, "Sideways"
            else: return False, 1-p.get('bear',0.5), f"Bear block ({p.get('bear',0):.0%})"
        else:
            if m=='bear': return True, p.get('bear',0.5), f"Bear ({p.get('bear',0):.0%})"
            elif m=='sideways': return True, p.get('sideways',0.3)*0.7, "Sideways"
            else: return False, 1-p.get('bull',0.5), f"Bull block ({p.get('bull',0):.0%})"

# ============================================================
# GARCH MODEL
# ============================================================
class GARCHModel:
    def fit(self, daily, horizon=5):
        if daily is None or len(daily)<100: return None
        try:
            rets = (daily['Close'].pct_change().dropna()*100)
            mr = rets.mean(); sr = rets.std()
            rets = rets[np.abs(rets-mr)<10*sr]
            if len(rets)<100: return None
            mdl = arch_model(rets, mean='Constant', vol='Garch', p=1, q=1, dist='t')
            res = mdl.fit(disp='off', show_warning=False)
            a = res.params.get('alpha[1]',0); b = res.params.get('beta[1]',0)
            cv = res.conditional_volatility.iloc[-1]*np.sqrt(252)/100
            fc = res.forecast(horizon=horizon)
            fv = np.sqrt(np.mean(fc.variance.values[-1,:]))/100
            fva = fv*np.sqrt(252)
            vr = 'low' if cv*100<15 else ('normal' if cv*100<30 else ('high' if cv*100<50 else 'extreme'))
            pm = min(max(0.20/cv if cv>0 else 1, 0.1), 3.0)
            sd = 2*fv*np.sqrt(horizon)
            cp = float(daily['Close'].iloc[-1])
            return {'current_vol_annual':float(cv),'forecast_vol_annual':float(fva),'vol_regime':vr,
                    'persistence':float(a+b),'position_multiplier':float(pm),
                    'stop_distance_pct':float(sd),'current_price':cp,
                    'var_95_1day':float(cp*fv*1.645),'vol_expanding':float(fva>cv)}
        except: return None

    def multi_tf(self, data):
        r = {}
        g = self.fit(data.get('1d'))
        if g: r['daily'] = g
        h = data.get('1h')
        if h is not None and len(h)>50:
            hr = h['Close'].pct_change().dropna()
            r['hourly_realized_vol'] = float(hr.std()*np.sqrt(6.25*252))
        return r if r else None

# ============================================================
# KALMAN FILTER
# ============================================================
class KalmanTrendFilter:
    def run(self, prices, vol=0.02):
        if prices is None or len(prices)<30: return None
        prices = np.array(prices, dtype=float)
        n = len(prices)
        kf = KF(dim_x=2, dim_z=1)
        kf.F = np.array([[1.,1.],[0.,1.]])
        kf.H = np.array([[1.,0.]])
        qs = (prices[0]*vol)**2
        kf.Q = np.array([[qs*0.1,0.],[0.,qs*0.01]])
        kf.R = np.array([[qs*1.0]])
        kf.x = np.array([[prices[0]],[0.]])
        kf.P = np.eye(2)*qs*10
        fp = np.zeros(n); vel = np.zeros(n); unc = np.zeros(n)
        for i in range(n):
            kf.predict(); kf.update(np.array([[prices[i]]]))
            fp[i]=kf.x[0,0]; vel[i]=kf.x[1,0]; unc[i]=np.sqrt(kf.P[0,0])
        return {'filtered_prices':fp,'velocities':vel,'uncertainties':unc}

    def analyze(self, data, garch_r=None):
        result = {}
        ve = 0.02
        if garch_r and 'daily' in garch_r:
            ve = garch_r['daily'].get('current_vol_annual',0.30)/np.sqrt(252)
        tfc = {'1d':{'min':100,'vs':1.0},'4h':{'min':40,'vs':0.6},'1h':{'min':60,'vs':0.4}}
        for tf, cfg in tfc.items():
            df = data.get(tf)
            if df is None or len(df)<cfg['min']: continue
            kr = self.run(df['Close'].values, ve*cfg['vs'])
            if kr is None: continue
            v = kr['velocities']; f = kr['filtered_prices']; u = kr['uncertainties']
            cv = v[-1]; cf = f[-1]; cp = df['Close'].values[-1]
            td = 'up' if cv>0 else ('down' if cv<0 else 'flat')
            ts = abs(cv)/cf if cf>0 else 0
            acc = v[-1]-v[-5] if len(v)>=5 else 0
            lb = min(20,len(v))
            cons = np.sum(v[-lb:]>0)/lb if cv>0 else np.sum(v[-lb:]<0)/lb
            tc = 1.0-min(u[-1]/cp,1.0)
            result[tf] = {'trend_direction':td,'trend_strength':float(ts),'velocity':float(cv),
                          'acceleration':float(acc),'consistency':float(cons),
                          'filtered_price':float(cf),'uncertainty':float(u[-1]),
                          'trend_confidence':float(tc),'accelerating':acc*cv>0}
        if result:
            dirs = [v['trend_direction'] for v in result.values() if isinstance(v,dict) and 'trend_direction' in v]
            bc = dirs.count('up'); brc = dirs.count('down'); tot = len(dirs)
            ac = np.mean([v['trend_confidence'] for v in result.values() if isinstance(v,dict) and 'trend_confidence' in v])
            result['mtf_agreement'] = {'bullish_pct':bc/tot if tot else 0,'bearish_pct':brc/tot if tot else 0,
                                       'all_bullish':bc==tot,'all_bearish':brc==tot,
                                       'conflicting':bc>0 and brc>0,'avg_confidence':float(ac)}
        return result if result else None

# ============================================================
# MONTE CARLO
# ============================================================
class MonteCarloEngine:
    def __init__(self, n_sims=3000):
        self.n_sims = n_sims

    def calibrate(self, daily, garch_r=None):
        if daily is None or len(daily)<60: return None
        rets = daily['Close'].pct_change().dropna().values
        if len(rets)<60: return None
        try: df_t, loc_t, scale_t = stats.t.fit(rets)
        except: df_t, loc_t, scale_t = 5.0, np.mean(rets), np.std(rets)
        if garch_r and 'daily' in garch_r:
            scale_t = garch_r['daily']['current_vol_annual']/np.sqrt(252)
        return {'df':float(df_t),'loc':float(loc_t),'scale':float(scale_t)}

    def simulate(self, entry, sl, tgt, params, days=5, sig_type='buy'):
        if not params: return None
        np.random.seed(42)
        hits_t=0; hits_s=0; pnls=[]
        for _ in range(self.n_sims):
            dr = stats.t.rvs(params['df'],loc=params['loc'],scale=params['scale'],size=days)
            pp = [entry]
            ht=False; hs=False
            for d in range(days):
                np_ = pp[-1]*(1+dr[d]); pp.append(np_)
                if sig_type=='buy':
                    if np_>=tgt: ht=True; break
                    if np_<=sl: hs=True; break
                else:
                    if np_<=tgt: ht=True; break
                    if np_>=sl: hs=True; break
            fp = pp[-1]
            pnl = (fp-entry)/entry if sig_type=='buy' else (entry-fp)/entry
            if ht: hits_t+=1
            elif hs: hits_s+=1
            pnls.append(pnl)
        pa = np.array(pnls); n=self.n_sims
        pp_ = np.sum(pa>0)/n; aw = float(np.mean(pa[pa>0])) if np.sum(pa>0)>0 else 0
        al = float(np.mean(pa[pa<0])) if np.sum(pa<0)>0 else 0
        exp = pp_*aw + (1-pp_)*al
        return {'prob_target_hit':hits_t/n,'prob_stop_hit':hits_s/n,'prob_profit':float(pp_),
                'expected_pnl':float(np.mean(pa)),'avg_win':aw,'avg_loss':al,'expectancy':float(exp),
                'risk_reward_realized':float(abs(aw/al)) if al!=0 else 0}

    def stress(self, entry, sl, tgt, params, sig_type='buy'):
        if not params: return None
        scenarios = {}
        for name, mult in [('normal',1.0),('elevated',1.5),('crisis',2.0),('black_swan',3.0)]:
            sp = params.copy(); sp['scale'] = params['scale']*mult
            r = self.simulate(entry, sl, tgt, sp, 5, sig_type)
            if r: scenarios[name] = r
        robust = scenarios.get('crisis',{}).get('expectancy',0) > 0
        return {'scenarios':scenarios,'robust_under_stress':robust,
                'normal_expectancy':scenarios.get('normal',{}).get('expectancy',0),
                'crisis_expectancy':scenarios.get('crisis',{}).get('expectancy',0)}

# ============================================================
# TRADE STRUCTURER
# ============================================================
class TradeStructurer:
    def calc_levels(self, price, sig_type, garch_r, kalman_r, factor_r):
        if price <= 0: return None
        sp = 0.03
        if garch_r and 'daily' in garch_r:
            g = garch_r['daily']
            sp = g.get('stop_distance_pct',0.03)
            vr = g.get('vol_regime','normal')
            if vr=='low': sp=max(sp,0.015)
            elif vr=='high': sp=min(sp,0.06)
            elif vr=='extreme': sp=min(sp,0.08)
        sp = np.clip(sp, 0.01, 0.08)
        rr = 2.0
        if factor_r:
            cs = factor_r.get('composite_score',0)
            if cs>1.5: rr=3.0
            elif cs>1.0: rr=2.5
        tp = sp*rr
        if kalman_r:
            dk = kalman_r.get('1d',{})
            vel = dk.get('velocity',0)
            if sig_type=='buy' and vel>0:
                proj = (vel*5)/price
                if proj>tp: tp=min(proj, sp*4)
        if sig_type=='buy':
            e=price; sl=price*(1-sp); t=price*(1+tp)
        else:
            e=price; sl=price*(1+sp); t=price*(1-tp)
        risk=abs(e-sl); rew=abs(t-e)
        return {'entry':round(e,2),'stop_loss':round(sl,2),'target':round(t,2),
                'stop_pct':round(sp*100,2),'target_pct':round(tp*100,2),
                'risk_reward':round(rew/risk if risk>0 else 0,2)}

# ============================================================
# BACKTESTER
# ============================================================
class Backtester:
    def test(self, daily, sig_type='buy', lookback=252, sp=0.03, tp=0.06):
        if daily is None or len(daily)<lookback: return None
        c = daily['Close'].values; n=len(c); trades=[]
        for i in range(max(0,n-lookback), n-10, 20):
            ep = c[i]*(1+0.001) if sig_type=='buy' else c[i]*(1-0.001)
            sl = ep*(1-sp) if sig_type=='buy' else ep*(1+sp)
            tg = ep*(1+tp) if sig_type=='buy' else ep*(1-tp)
            xp=None; ht=False; hs=False
            for j in range(1, min(11,n-i)):
                p=c[i+j]
                if sig_type=='buy':
                    if p>=tg: xp=tg; ht=True; break
                    if p<=sl: xp=sl; hs=True; break
                else:
                    if p<=tg: xp=tg; ht=True; break
                    if p>=sl: xp=sl; hs=True; break
            if xp is None: xp=c[min(i+10,n-1)]
            pnl = (xp-ep)/ep if sig_type=='buy' else (ep-xp)/ep
            pnl -= 0.0006
            trades.append({'pnl':pnl,'hit_target':ht,'hit_stop':hs})
        if not trades: return None
        pa = np.array([t['pnl'] for t in trades])
        w=pa[pa>0]; l=pa[pa<=0]
        wr = len(w)/len(pa) if len(pa)>0 else 0
        aw = float(np.mean(w)) if len(w)>0 else 0
        al = float(np.mean(l)) if len(l)>0 else 0
        gp = float(np.sum(w)) if len(w)>0 else 0
        gl = float(abs(np.sum(l))) if len(l)>0 else 0.0001
        exp = wr*aw + (1-wr)*al
        return {'total_trades':len(trades),'win_rate':round(wr,4),'avg_win':round(aw,4),
                'avg_loss':round(al,4),'profit_factor':round(gp/gl,2),
                'expectancy':round(exp,4),
                'target_hit_rate':round(sum(1 for t in trades if t['hit_target'])/len(trades),4),
                'stop_hit_rate':round(sum(1 for t in trades if t['hit_stop'])/len(trades),4),
                'avg_days_held':5.0,'total_pnl':round(float(np.sum(pa)),4)}

# ============================================================
# SIGNAL AGGREGATOR
# ============================================================
class SignalAggregator:
    def __init__(self):
        self.wts = {'factor':0.25,'regime':0.20,'garch':0.15,'kalman':0.25,'monte_carlo':0.15}

    def _fv(self, fr, st):
        if not fr: return 0, 0.0, "No data"
        p = fr.get('rank_percentile',50)
        if st=='buy':
            if p>=80: return 1, min(p/100,0.95), f"Top {100-p:.0f}%"
            elif p>=60: return 1, p/100*0.7, f"Top {100-p:.0f}%"
            elif p<=30: return -1, (100-p)/100, f"Bottom {p:.0f}%"
            return 0, 0.5, f"{p:.0f}th pct"
        else:
            if p<=20: return -1, (100-p)/100*0.9, f"Bottom {p:.0f}%"
            elif p<=40: return -1, (100-p)/100*0.6, f"Bottom {p:.0f}%"
            elif p>=70: return 1, p/100, "Strong — don't short"
            return 0, 0.5, f"{p:.0f}th pct"

    def _rv(self, rd, st):
        ok, conf, reason = rd.filter(st)
        v = (1 if st=='buy' else -1) if ok and conf>0.5 else (-1 if st=='buy' else 1) if not ok else 0
        return v, conf, reason

    def _gv(self, gr, st):
        if not gr or 'daily' not in gr: return 0, 0.5, "No GARCH"
        vr = gr['daily'].get('vol_regime','normal')
        if st=='buy':
            if vr=='low': return 1, 0.7, "Low vol"
            elif vr=='normal': return 1, 0.6, "Normal vol"
            elif vr=='high': return 0, 0.4, "High vol"
            return -1, 0.3, "Extreme vol"
        else:
            if vr in ['high','extreme']: return -1, 0.7, f"{vr.title()} vol"
            elif vr=='normal': return -1, 0.5, "Normal vol"
            return 0, 0.4, "Low vol"

    def _kv(self, kr, st):
        if not kr or 'mtf_agreement' not in kr: return 0, 0.5, "No Kalman"
        m = kr['mtf_agreement']
        if st=='buy':
            if m['all_bullish']: return 1, m['avg_confidence'], "All TF bullish"
            elif m['bullish_pct']>=0.66: return 1, m['avg_confidence']*0.8, f"{m['bullish_pct']:.0%} bullish"
            elif m['all_bearish']: return -1, m['avg_confidence'], "All TF bearish"
            return 0, 0.3 if m['conflicting'] else 0.5, "Mixed"
        else:
            if m['all_bearish']: return -1, m['avg_confidence'], "All TF bearish"
            elif m['bearish_pct']>=0.66: return -1, m['avg_confidence']*0.8, f"{m['bearish_pct']:.0%} bearish"
            elif m['all_bullish']: return 1, m['avg_confidence'], "All TF bullish"
            return 0, 0.5, "Mixed"

    def _mv(self, mr, st):
        if not mr or 'scenarios' not in mr: return 0, 0.5, "No MC"
        n = mr['scenarios'].get('normal',{})
        exp = n.get('expectancy',0); pp = n.get('prob_profit',0.5)
        robust = mr.get('robust_under_stress',False)
        if exp>0.005 and pp>0.55:
            v = 1 if st=='buy' else -1
            r = f"+EV: {exp:.2%}, P(profit): {pp:.0%}"
            if robust: r += " ✓stress"
            return v, min(pp,0.9), r
        elif exp>0 and pp>0.45: return 0, pp*0.6, f"Marginal: {exp:.2%}"
        return -1 if st=='buy' else 1, 1-pp, f"-EV: {exp:.2%}"

    def generate(self, sym, data, fr, rd, gr, kr, tl_b, tl_s, mc_b, mc_s, bt_b, bt_s):
        signals = []
        for st in ['buy','sell']:
            if st=='sell' and sym not in FNO_STOCKS: continue
            fv,fc,fre = self._fv(fr, st)
            rv,rc,rre = self._rv(rd, st)
            gv,gc,gre = self._gv(gr, st)
            kv,kc,kre = self._kv(kr, st)
            mc_r = mc_b if st=='buy' else mc_s
            mv,mc_,mre = self._mv(mc_r, st)
            votes = {'factor':fv,'regime':rv,'garch':gv,'kalman':kv,'monte_carlo':mv}
            confs = {'factor':fc,'regime':rc,'garch':gc,'kalman':kc,'monte_carlo':mc_}
            ag = sum(1 for v in votes.values() if (v>0 if st=='buy' else v<0))
            op = sum(1 for v in votes.values() if (v<0 if st=='buy' else v>0))
            if ag < 3: continue
            if (st=='buy' and rv<0) or (st=='sell' and rv>0): continue
            wc = sum(confs[m]*self.wts[m] for m in self.wts)
            grade = 'A+' if ag==5 else ('A' if ag==4 else 'B+')
            score = wc*100
            tl = tl_b if st=='buy' else tl_s
            bt = bt_b if st=='buy' else bt_s
            if bt:
                hw = bt.get('win_rate',0.5)
                if hw>0.6: score*=1.1
                elif hw<0.35: score*=0.8
            score = np.clip(score, 0, 100)
            hwr = bt.get('win_rate',0)*100 if bt else 0
            signals.append({
                'symbol':sym,'signal_type':st,'is_fno':sym in FNO_STOCKS,
                'grade':grade,'score':round(float(score),1),'confidence':round(float(wc*100),1),
                'agreement':f"{ag}/5",
                'entry':tl['entry'] if tl else 0,'stop_loss':tl['stop_loss'] if tl else 0,
                'target':tl['target'] if tl else 0,'risk_reward':tl['risk_reward'] if tl else 0,
                'stop_pct':tl['stop_pct'] if tl else 0,'target_pct':tl['target_pct'] if tl else 0,
                'historical_win_rate':round(float(hwr),1),
                'model_votes':{
                    'factor':{'vote':fv,'confidence':round(fc,3),'reason':fre,'active':fv!=0},
                    'regime':{'vote':rv,'confidence':round(rc,3),'reason':rre,'active':rv!=0},
                    'garch':{'vote':gv,'confidence':round(gc,3),'reason':gre,'active':gv!=0},
                    'kalman':{'vote':kv,'confidence':round(kc,3),'reason':kre,'active':kv!=0},
                    'monte_carlo':{'vote':mv,'confidence':round(mc_,3),'reason':mre,'active':mv!=0}
                },
                'models_active':ag,'models_opposing':op,
                'monte_carlo_summary':{
                    'prob_profit':mc_r['scenarios']['normal']['prob_profit'] if mc_r and 'scenarios' in mc_r and 'normal' in mc_r['scenarios'] else 0,
                    'expected_pnl':mc_r['scenarios']['normal']['expected_pnl'] if mc_r and 'scenarios' in mc_r and 'normal' in mc_r['scenarios'] else 0,
                    'stress_tested':mc_r.get('robust_under_stress',False) if mc_r else False,
                    'crisis_expectancy':mc_r.get('crisis_expectancy',0) if mc_r else 0
                },
                'backtest':bt if bt else {},
                'garch_summary':{
                    'vol_regime':gr['daily']['vol_regime'] if gr and 'daily' in gr else 'unknown',
                    'current_vol':gr['daily']['current_vol_annual'] if gr and 'daily' in gr else 0,
                    'position_multiplier':gr['daily']['position_multiplier'] if gr and 'daily' in gr else 1
                },
                'kalman_summary':{
                    'trend_direction':kr.get('1d',{}).get('trend_direction','unknown') if kr else 'unknown',
                    'trend_strength':kr.get('1d',{}).get('trend_strength',0) if kr else 0,
                    'mtf_bullish':kr.get('mtf_agreement',{}).get('bullish_pct',0) if kr else 0
                },
                'expectancy':bt.get('expectancy',0) if bt else 0,
                'past_trades':bt.get('total_trades',0) if bt else 0
            })
        return signals

# ============================================================
# MAIN SCANNER
# ============================================================
class QuantScanner:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.factor = FactorModel()
        self.regime = RegimeDetector()
        self.garch = GARCHModel()
        self.kalman = KalmanTrendFilter()
        self.mc = MonteCarloEngine(n_sims=3000)
        self.structurer = TradeStructurer()
        self.bt = Backtester()
        self.agg = SignalAggregator()
        self.last_result = None

    def scan(self, stocks=None, top_n=80, max_sig=50):
        start = time.time()
        stocks = stocks or ALL_STOCKS
        logger.info(f"=== SCAN START: {len(stocks)} stocks ===")

        data = self.fetcher.fetch_all(stocks, workers=5)
        if len(data)<10: return {'buy_signals':[],'sell_signals':[],'error':'Insufficient data'}

        logger.info("Detecting market regime...")
        try:
            nifty = yf.Ticker("^NSEI").history(period="2y", interval="1d", auto_adjust=True)
            nifty = nifty[['Open','High','Low','Close','Volume']]
            self.regime.fit_market(nifty)
        except Exception as e:
            logger.warning(f"Nifty fetch failed: {e}")

        logger.info("Running factor model...")
        factor_results = self.factor.rank(data)
        logger.info(f"Ranked {len(factor_results)} stocks")

        buy_cands = sorted(factor_results.keys(), key=lambda s: factor_results[s]['composite_score'], reverse=True)[:top_n]
        sell_cands = [s for s in sorted(factor_results.keys(), key=lambda s: factor_results[s]['composite_score'])[:top_n] if s in FNO_STOCKS]
        candidates = list(set(buy_cands + sell_cands))
        logger.info(f"Deep analyzing {len(candidates)} stocks...")

        all_signals = []
        for i, sym in enumerate(candidates):
            if (i+1) % 20 == 0: logger.info(f"  Processing {i+1}/{len(candidates)}...")
            sd = data.get(sym)
            if not sd: continue
            daily = sd.get('1d')
            if daily is None or len(daily)<100: continue
            cp = float(daily['Close'].iloc[-1])

            try:
                gr = self.garch.multi_tf(sd)
                kr = self.kalman.analyze(sd, gr)
                fr = factor_results.get(sym)
                tl_b = self.structurer.calc_levels(cp, 'buy', gr, kr, fr)
                tl_s = self.structurer.calc_levels(cp, 'sell', gr, kr, fr) if sym in FNO_STOCKS else None
                dp = self.mc.calibrate(daily, gr)
                mc_b = self.mc.stress(tl_b['entry'],tl_b['stop_loss'],tl_b['target'],dp,'buy') if tl_b and dp else None
                mc_s = self.mc.stress(tl_s['entry'],tl_s['stop_loss'],tl_s['target'],dp,'sell') if tl_s and dp and sym in FNO_STOCKS else None
                bt_b = self.bt.test(daily,'buy',stop_pct=tl_b['stop_pct']/100,tp=tl_b['target_pct']/100) if tl_b else None
                bt_s = self.bt.test(daily,'sell',stop_pct=tl_s['stop_pct']/100,tp=tl_s['target_pct']/100) if tl_s else None
                sigs = self.agg.generate(sym, sd, fr, self.regime, gr, kr, tl_b, tl_s, mc_b, mc_s, bt_b, bt_s)
                all_signals.extend(sigs)
            except Exception as e:
                logger.debug(f"Error {sym}: {e}")

        buy_sigs = sorted([s for s in all_signals if s['signal_type']=='buy'], key=lambda x: x['score'], reverse=True)[:max_sig]
        sell_sigs = sorted([s for s in all_signals if s['signal_type']=='sell'], key=lambda x: x['score'], reverse=True)[:max_sig]
        elapsed = time.time()-start

        result = {
            'buy_signals': buy_sigs,
            'sell_signals': sell_sigs,
            'scan_metadata': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stocks_scanned': len(data),
                'buy_signals_count': len(buy_sigs),
                'sell_signals_count': len(sell_sigs),
                'market_regime': self.regime.market_regime or 'unknown',
                'regime_probs': self.regime.market_regime_probs or {},
                'scan_duration_seconds': round(elapsed, 1),
                'failed_stocks': len(self.fetcher.failed_symbols)
            }
        }
        self.last_result = result
        logger.info(f"=== SCAN COMPLETE: {len(buy_sigs)} buys, {len(sell_sigs)} sells in {elapsed:.0f}s ===")
        return result

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app)

scanner = QuantScanner()
SCAN_RESULTS = {'buy_signals':[],'sell_signals':[],'scan_metadata':{'timestamp':None}}
scan_lock = Lock()

@app.route('/')
def home():
    return jsonify({'name':'Quant Scanner API','version':'2.0',
                    'endpoints':{'/api/scan':'GET scan results','/api/status':'Health check'}})

@app.route('/api/scan', methods=['GET'])
def api_scan():
    global SCAN_RESULTS
    refresh = request.args.get('refresh','false').lower()=='true'
    if refresh or SCAN_RESULTS['scan_metadata'].get('timestamp') is None:
        try:
            result = scanner.scan(ALL_STOCKS, top_n=80, max_sig=50)
            with scan_lock: SCAN_RESULTS = result
        except Exception as e:
            return jsonify({'error':str(e),'buy_signals':[],'sell_signals':[],'scan_metadata':{'error':str(e)}}), 500
    return jsonify(SCAN_RESULTS)

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({'status':'running','last_scan':SCAN_RESULTS['scan_metadata'].get('timestamp','Never'),
                    'buy_signals':SCAN_RESULTS['scan_metadata'].get('buy_signals_count',0),
                    'sell_signals':SCAN_RESULTS['scan_metadata'].get('sell_signals_count',0),
                    'market_regime':SCAN_RESULTS['scan_metadata'].get('market_regime','unknown')})

@app.route('/api/stock/<symbol>', methods=['GET'])
def api_stock(symbol):
    symbol = symbol.upper()
    for s in SCAN_RESULTS.get('buy_signals',[])+SCAN_RESULTS.get('sell_signals',[]):
        if s['symbol']==symbol: return jsonify(s)
    return jsonify({'error':f'{symbol} not in latest scan'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info("Running initial scan on startup...")
    try:
        result = scanner.scan(ALL_STOCKS, top_n=80, max_sig=50)
        SCAN_RESULTS = result
    except Exception as e:
        logger.error(f"Initial scan failed: {e}")
    app.run(host='0.0.0.0', port=port)
