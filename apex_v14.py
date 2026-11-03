"""
APEX PREDATOR v14.0 — HYBRID PARLIAMENT (20 MODELS)
====================================================
15 Rule-based Models + 5 Machine Learning Models.
Ensemble voting (Supermajority 65%) untuk keputusan institusional.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ===== v14.0 ML Models =====
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import RidgeClassifier
    from scipy.signal import butter, filtfilt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

VERSION = "14.0"
NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")

# =========================================================
# CONFIG
# =========================================================
@dataclass
class Config:
    gist_id: str = ""
    gist_token: str = ""
    gist_state_file: str = "apex_state.json"
    local_state_file: str = "apex_state.json"

    symbol: str = "XAUUSD"
    contract_size: float = 100.0
    account_currency: str = "USD"
    price_decimals: int = 3
    mt5_offset: float = 0.0

    account_balance: float = 1000.0
    max_risk_pct_per_trade: float = 2.0
    hard_leverage_cap: float = 50.0
    max_lot: float = 5.0
    min_lot: float = 0.01
    max_daily_loss_pct: float = 4.0
    max_consecutive_losses: int = 3
    dd_pause_hours: int = 24

    atr_sl_mult: float = 1.8
    atr_tp1_mult: float = 1.0
    atr_tp2_mult: float = 2.4
    atr_tp3_mult: float = 4.0
    spread_buffer: float = 0.35

    min_bars_m5: int = 100
    min_bars_m15: int = 100
    min_bars_h1: int = 100
    min_bars_h4: int = 50
    max_stale_mult: float = 3.0
    m5_fetch_size: int = 4500
    price_min: float = 500.0
    price_max: float = 10000.0
    price_max_jump_pct: float = 0.10

    # Parliament rules
    parliament_min_active: int = 8        # Minimal 8 dari 20 model aktif
    parliament_min_ratio: float = 0.65    # 65% supermajority
    parliament_confident_ratio: float = 0.75 # >75% untuk EXECUTE_FULL
    adx_min: float = 22.0

    cooldown_minutes: int = 45
    opposite_cooldown_minutes: int = 120
    max_daily_signals: int = 5

    hurst_trend_min: float = 0.55
    hurst_meanrev_max: float = 0.45
    entropy_chaos_max: float = 0.85

    dry_run: bool = True
    news_blackout_min: int = 30
    news_cache_sec: int = 1800
    news_cache_sec_on_fail: int = 300
    news_max_consecutive_fail: int = 3
    pending_max_age_hours: int = 24
    heartbeat_hour_utc: int = 8
    http_timeout: tuple = (3, 10)
    max_retry: int = 3
    max_journal: int = 500

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            v = os.getenv(f"APEX_{f.upper()}")
            if v is None or not str(v).strip(): continue
            cur = getattr(cfg, f)
            try:
                if isinstance(cur, bool): setattr(cfg, f, str(v).strip().lower() in ("1","true","yes","on"))
                elif isinstance(cur, int): setattr(cfg, f, int(v))
                elif isinstance(cur, float):
                    fv = float(v)
                    if not np.isfinite(fv): raise ValueError
                    setattr(cfg, f, fv)
                else: setattr(cfg, f, str(v).strip())
            except ValueError: logging.warning(f"APEX_{f.upper()}={v!r} invalid")
        cfg.gist_id = os.getenv("GIST_ID", cfg.gist_id).strip()
        cfg.gist_token = os.getenv("GIST_TOKEN", cfg.gist_token).strip()
        mt5_off = os.getenv("APEX_MT5_OFFSET", "").strip()
        if mt5_off:
            try: cfg.mt5_offset = float(mt5_off)
            except ValueError: pass
        cfg.validate()
        return cfg

    def validate(self) -> None:
        errs = []
        if self.account_balance <= 0: errs.append("balance")
        if self.contract_size <= 0: errs.append("contract_size")
        if not (0 < self.atr_tp1_mult < self.atr_tp2_mult < self.atr_tp3_mult): errs.append("TP order")
        if not (0 < self.max_risk_pct_per_trade <= 5): errs.append("max_risk_pct")
        if not (0 < self.hard_leverage_cap <= 200): errs.append("leverage_cap")
        if not (0 < self.max_daily_loss_pct <= 10): errs.append("max_daily_loss_pct")
        if not (0 < self.parliament_min_ratio <= 0.9): errs.append("parliament_min_ratio")
        if self.min_lot <= 0 or self.max_lot < self.min_lot: errs.append("lot range")
        if errs: raise ValueError("CONFIG: " + ", ".join(errs))


# =========================================================
# LOGGING & UTIL
# =========================================================
def setup_logging(verbose=False):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter(f"%(asctime)s [%(levelname)s] [v{VERSION}] %(message)s")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); root.addHandler(ch)

def redact(t):
    s = str(t).replace("\n"," ").replace("\r"," ")
    s = re.sub(r"(token|apikey|api_key)=[^&\s'\"]+", r"\1=***", s, flags=re.I)
    s = re.sub(r"/bot[^/\s'\"]+/", "/bot***/", s)
    s = re.sub(r"Bearer\s+\S+", "Bearer ***", s, flags=re.I)
    return s

def h(s): return html.escape(str(s), quote=False)

class DataFetchError(Exception): pass
class StateLoadError(Exception): pass


# =========================================================
# KVStore
# =========================================================
class KVStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_gist = bool(cfg.gist_id and cfg.gist_token)
        self._cache = None
        if not self.use_gist: logging.warning("GIST not set — local state only.")

    def _headers(self):
        return {"Authorization": f"Bearer {self.cfg.gist_token}", "Accept": "application/vnd.github+json"}

    def _load_gist(self):
        if self._cache is not None: return self._cache
        try:
            r = requests.get(f"https://api.github.com/gists/{self.cfg.gist_id}", headers=self._headers(), timeout=self.cfg.http_timeout)
            if r.status_code == 404: self._cache = {}; return self._cache
            r.raise_for_status()
            files = r.json().get("files", {})
            fn = self.cfg.gist_state_file
            self._cache = json.loads(files[fn]["content"]) if fn in files and files[fn]["content"].strip() else {}
            return self._cache
        except Exception as e:
            logging.error(f"Gist LOAD: {redact(e)}"); raise StateLoadError(str(e)) from e

    def _save_gist(self, data):
        try:
            body = {"files": {self.cfg.gist_state_file: {"content": json.dumps(data, default=str, sort_keys=True)}}}
            r = requests.patch(f"https://api.github.com/gists/{self.cfg.gist_id}", headers=self._headers(), json=body, timeout=self.cfg.http_timeout)
            r.raise_for_status(); self._cache = data; return True
        except Exception as e:
            logging.error(f"Gist SAVE: {redact(e)}"); return False

    def _load_local(self):
        if self._cache is not None: return self._cache
        p = Path(self.cfg.local_state_file)
        self._cache = json.loads(p.read_text()) if p.exists() else {}
        return self._cache

    def _save_local(self, data):
        try:
            Path(self.cfg.local_state_file).write_text(json.dumps(data, default=str, sort_keys=True))
            self._cache = data; return True
        except Exception: return False

    def get(self, k, d=None):
        return (self._load_gist() if self.use_gist else self._load_local()).get(k, d)

    def set(self, k, v):
        data = dict(self._load_gist() if self.use_gist else self._load_local())
        data[k] = v
        return self._save_gist(data) if self.use_gist else self._save_local(data)

    def force_load(self):
        self._load_gist() if self.use_gist else self._load_local()


# =========================================================
# DATA FETCHERS (TwelveData + Yahoo + Stooq)
# =========================================================
def get_dynamic_offset(cfg) -> float:
    oanda_key = os.getenv("OANDA_API_KEY", "").strip()
    if not oanda_key: return cfg.mt5_offset
    try:
        r1 = requests.get("https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/pricing",
                          headers={"Authorization": f"Bearer {oanda_key}"}, timeout=5)
        r1.raise_for_status(); oanda = float(r1.json()["prices"][0]["bids"][0]["price"])
        td_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        r2 = requests.get("https://api.twelvedata.com/price",
                          params={"symbol":"XAU/USD","apikey":td_key}, timeout=5)
        r2.raise_for_status(); td = float(r2.json()["price"])
        return round(oanda - td, 3)
    except Exception as e:
        logging.warning(f"Dynamic offset failed: {redact(e)}"); return cfg.mt5_offset

def _apply_offset(df, off):
    if off != 0.0:
        for c in ("open","high","low","close"): df[c] = df[c] + off
    return df

def validate_ohlcv(df, interval, min_bars, cfg, offset):
    if df is None or len(df) < min_bars: raise DataFetchError(f"Bars {0 if df is None else len(df)} < {min_bars}")
    df = df.dropna(subset=["open","high","low","close"]).copy()
    if (df["high"] < df["low"]).any(): raise DataFetchError("high<low")
    if df["datetime"].duplicated().any(): raise DataFetchError("dupes")
    if not df["datetime"].is_monotonic_increasing: raise DataFetchError("non-monotonic")
    isec = {"5min":300,"15min":900,"1h":3600,"4h":14400}[interval]
    last = df["datetime"].iloc[-1]
    if (datetime.now(timezone.utc) - last).total_seconds() > isec * cfg.max_stale_mult: raise DataFetchError("Stale")
    return _apply_offset(df.reset_index(drop=True), offset)

def resample_ohlc(df, rule):
    d = (df.set_index("datetime").resample(rule, label="left", closed="left")
         .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index())
    return d.reset_index(drop=True)

def fetch_twelvedata(cfg, offset):
    key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not key: raise DataFetchError("No key")
    def _do():
        r = requests.get("https://api.twelvedata.com/time_series",
                         params={"symbol":"XAU/USD","interval":"5min","outputsize":cfg.m5_fetch_size,"apikey":key,"timezone":"UTC"},
                         timeout=cfg.http_timeout)
        r.raise_for_status(); d = r.json()
        if "values" not in d: raise DataFetchError(f"TD: {d.get('message','?')}")
        df = pd.DataFrame(d["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open","high","low","close"): df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = 100.0
        return df.sort_values("datetime").reset_index(drop=True)
    df = _do()
    return {
        "5min": validate_ohlcv(df, "5min", cfg.min_bars_m5, cfg, offset),
        "15min": validate_ohlcv(resample_ohlc(df, "15min"), "15min", cfg.min_bars_m15, cfg, offset),
        "1h": validate_ohlcv(resample_ohlc(df, "1h"), "1h", cfg.min_bars_h1, cfg, offset),
        "4h": validate_ohlcv(resample_ohlc(df, "4h"), "4h", cfg.min_bars_h4, cfg, offset),
    }, "TwelveData"

def fetch_yahoo(cfg, offset):
    headers = {"User-Agent": f"ApexBot/{VERSION}"}
    def _do(symbol, iv, rng):
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                         params={"interval":iv,"range":rng}, headers=headers, timeout=cfg.http_timeout)
        r.raise_for_status()
        res = (r.json().get("chart",{}).get("result") or [None])[0]
        if not res: raise DataFetchError("no result")
        ts = res.get("timestamp",[]); q = res.get("indicators",{}).get("quote",[{}])[0]
        rows = []
        for i,t in enumerate(ts):
            try:
                if any(q.get(k,[None])[i] is None for k in ("open","high","low","close")): continue
                rows.append({"epoch":int(t),"open":float(q["open"][i]),"high":float(q["high"][i]),
                             "low":float(q["low"][i]),"close":float(q["close"][i]),
                             "volume":float((q.get("volume") or [100])[i] or 100)})
            except: continue
        if not rows: raise DataFetchError("0 rows")
        df = pd.DataFrame(rows); df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        return df[["datetime","open","high","low","close","volume"]]
    out = {}
    for iv,cnt,rng,mn in (("5min",200,"5d",cfg.min_bars_m5),("15min",150,"5d",cfg.min_bars_m15),("1h",200,"1mo",cfg.min_bars_h1)):
        out[iv] = validate_ohlcv(_do("XAUUSD=X",{"5min":"5m","15min":"15m","1h":"1h"}[iv],rng), iv, mn, cfg, offset)
    df1h = _do("XAUUSD=X", "1h", "1mo")
    out["4h"] = validate_ohlcv(resample_ohlc(df1h, "4h"), "4h", cfg.min_bars_h4, cfg, offset)
    return out, "Yahoo"

def fetch_stooq(cfg, offset):
    def _do():
        r = requests.get("https://stooq.com/q/d/l/?s=xauusd&i=5", timeout=cfg.http_timeout)
        r.raise_for_status(); df = pd.read_csv(io.StringIO(r.text))
        if "Date" not in df.columns or len(df) < 100: raise DataFetchError("bad CSV")
        df.rename(columns={"Date":"datetime","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.sort_values("datetime").reset_index(drop=True)
    df = _do()
    return {
        "5min": validate_ohlcv(df, "5min", cfg.min_bars_m5, cfg, offset),
        "15min": validate_ohlcv(resample_ohlc(df,"15min"), "15min", cfg.min_bars_m15, cfg, offset),
        "1h": validate_ohlcv(resample_ohlc(df,"1h"), "1h", cfg.min_bars_h1, cfg, offset),
        "4h": validate_ohlcv(resample_ohlc(df,"4h"), "4h", cfg.min_bars_h4, cfg, offset),
    }, "Stooq"

def fetch_all_intervals(cfg):
    offset = get_dynamic_offset(cfg)
    errs = []
    for name, fn in (("TwelveData",fetch_twelvedata),("Yahoo",fetch_yahoo),("Stooq",fetch_stooq)):
        try:
            dfs, src = fn(cfg, offset)
            logging.info(f"Data OK [{src}] (Offset: {offset:+.3f})")
            return dfs, src, offset
        except Exception as e:
            errs.append(f"{name}: {redact(e)}"); logging.warning(f"{name} failed: {redact(e)}")
    raise DataFetchError("ALL FAILED | " + " | ".join(errs))


# =========================================================
# HELPER: INDICATORS
# =========================================================
def atr(df, period=14):
    h_, lo, c_ = df["high"], df["low"], df["close"]
    tr = pd.concat([h_-lo,(h_-c_.shift()).abs(),(lo-c_.shift()).abs()],axis=1).max(axis=1)
    v = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    return 0.0 if pd.isna(v) or v < 1e-6 else float(v)

def adx(df, period=14):
    try:
        up, dn = df["high"].diff(), -df["low"].diff()
        pdm = up.where((up>dn)&(up>0),0.0); mdm = dn.where((dn>up)&(dn>0),0.0)
        tr = pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift()).abs(),(df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
        atr_s = tr.ewm(alpha=1/period, adjust=False).mean().replace(0,np.nan)
        pdi = 100*pdm.ewm(alpha=1/period,adjust=False).mean()/atr_s
        mdi = 100*mdm.ewm(alpha=1/period,adjust=False).mean()/atr_s
        dx = (pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)*100
        v = dx.ewm(alpha=1/period,adjust=False).mean().iloc[-1]
        return 0.0 if pd.isna(v) or v<0 else float(v)
    except: return 0.0

def hurst_exponent(close, max_lag=20):
    try:
        ts = np.log(close.dropna().values)
        if len(ts) < max_lag*2: return 0.5
        lags, tau = [], []
        for lag in range(2, max_lag):
            diff = ts[lag:] - ts[:-lag]; std = float(np.std(diff))
            if std > 1e-12: lags.append(lag); tau.append(std)
        if len(tau) < 4: return 0.5
        slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return float(np.clip(slope, 0.0, 1.0))
    except: return 0.5

def shannon_entropy(close, bins=10):
    try:
        r = close.pct_change().dropna()
        if len(r) < bins*5: return 0.5
        hist, _ = np.histogram(r, bins=bins); p = hist/hist.sum(); p = p[p>0]
        return float(-np.sum(p*np.log(p))/np.log(bins))
    except: return 0.5


# =========================================================
# 15 RULE-BASED MODELS
# =========================================================
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def model_ema_cross(df):
    try:
        e9 = _ema(df["close"], 9); e21 = _ema(df["close"], 21)
        diff = e9 - e21
        if diff.iloc[-1] > 0 and diff.diff().iloc[-1] > 0: return 1, min(1.0, abs(diff.iloc[-1])/(atr(df,14)+1e-6))
        if diff.iloc[-1] < 0 and diff.diff().iloc[-1] < 0: return -1, min(1.0, abs(diff.iloc[-1])/(atr(df,14)+1e-6))
        return 0, 0.0
    except: return 0, 0.0

def model_ema_trend(df):
    try:
        c = df["close"]; e20 = _ema(c, 20); e50 = _ema(c, 50)
        if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]: return 1, 0.9
        if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]: return -1, 0.9
        return 0, 0.0
    except: return 0, 0.0

def model_supertrend(df, period=10, mult=3.0):
    try:
        a = atr(df, period); hl2 = (df["high"]+df["low"])/2
        upper = hl2 + mult*a; lower = hl2 - mult*a
        close = df["close"]; trend = np.ones(len(df))
        for i in range(1, len(df)):
            if close.iloc[i] > upper.iloc[i-1]: trend[i] = 1
            elif close.iloc[i] < lower.iloc[i-1]: trend[i] = -1
            else: trend[i] = trend[i-1]
        if trend[-1] == 1 and trend[-2] == 1: return 1, 0.85
        if trend[-1] == -1 and trend[-2] == -1: return -1, 0.85
        return 0, 0.0
    except: return 0, 0.0

def model_donchian(df, period=20):
    try:
        hh = df["high"].rolling(period).max().iloc[-2]
        ll = df["low"].rolling(period).min().iloc[-2]
        c = df["close"].iloc[-1]
        if c > hh: return 1, 0.8
        if c < ll: return -1, 0.8
        return 0, 0.0
    except: return 0, 0.0

def model_kalman(df):
    try:
        obs = df["close"].values; n = len(obs)
        if n < 10: return 0, 0.0
        x = obs[0]; P = 1.0; Q = 1e-3; R = 1e-2; est = np.empty(n); est[0] = x
        for i in range(1, n):
            P += Q; K = P/(P+R); x = x + K*(obs[i]-x); P *= (1-K); est[i] = x
        slope = est[-1] - est[-5]
        if slope > 0: return 1, min(1.0, abs(slope)/(atr(df,14)+1e-6))
        if slope < 0: return -1, min(1.0, abs(slope)/(atr(df,14)+1e-6))
        return 0, 0.0
    except: return 0, 0.0

def model_macd(df, fast=12, slow=26, signal=9):
    try:
        c = df["close"]; macd = _ema(c, fast) - _ema(c, slow); sig = _ema(macd, signal); hist = macd - sig
        if macd.iloc[-1] > sig.iloc[-1] and hist.diff().iloc[-1] > 0: return 1, 0.9
        if macd.iloc[-1] < sig.iloc[-1] and hist.diff().iloc[-1] < 0: return -1, 0.9
        return 0, 0.0
    except: return 0, 0.0

def model_rsi(df, period=14):
    try:
        delta = df["close"].diff(); gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean(); loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rsi = (100 - 100/(1 + gain/(loss+1e-9))).iloc[-1]; prev = (100 - 100/(1 + gain/(loss+1e-9))).iloc[-2]
        if rsi < 30 and rsi > prev: return 1, 0.85
        if rsi > 70 and rsi < prev: return -1, 0.85
        if 50 < rsi < 65 and rsi > prev: return 1, 0.6
        if 35 < rsi < 50 and rsi < prev: return -1, 0.6
        return 0, 0.0
    except: return 0, 0.0

def model_stochastic(df, k=14, d=3):
    try:
        low = df["low"].rolling(k).min(); high = df["high"].rolling(k).max()
        stoch_k = 100 * (df["close"] - low) / (high - low + 1e-9); stoch_d = stoch_k.rolling(d).mean()
        k_, d_ = stoch_k.iloc[-1], stoch_d.iloc[-1]; kp, dp = stoch_k.iloc[-2], stoch_d.iloc[-2]
        if k_ > d_ and kp <= dp and k_ < 25: return 1, 0.8
        if k_ < d_ and kp >= dp and k_ > 75: return -1, 0.8
        return 0, 0.0
    except: return 0, 0.0

def model_cci(df, period=20):
    try:
        tp = (df["high"]+df["low"]+df["close"])/3; ma = tp.rolling(period).mean(); md = (tp-ma).abs().rolling(period).mean()
        cci = ((tp-ma)/(0.015*md+1e-9)).iloc[-1]; prev = ((tp-ma)/(0.015*md+1e-9)).iloc[-2]
        if prev < -100 and cci > -100: return 1, 0.8
        if prev > 100 and cci < 100: return -1, 0.8
        return 0, 0.0
    except: return 0, 0.0

def model_willr(df, period=14):
    try:
        hh = df["high"].rolling(period).max(); ll = df["low"].rolling(period).min()
        wr = -100 * (hh - df["close"]) / (hh - ll + 1e-9)
        if wr.iloc[-1] < -80 and wr.iloc[-1] > wr.iloc[-2]: return 1, 0.7
        if wr.iloc[-1] > -20 and wr.iloc[-1] < wr.iloc[-2]: return -1, 0.7
        return 0, 0.0
    except: return 0, 0.0

def model_bb(df, period=20, mult=2.0):
    try:
        ma = df["close"].rolling(period).mean(); std = df["close"].rolling(period).std()
        upper = ma + mult*std; lower = ma - mult*std; c = df["close"].iloc[-1]
        if c < lower.iloc[-1]: return 1, 0.75
        if c > upper.iloc[-1]: return -1, 0.75
        return 0, 0.0
    except: return 0, 0.0

def model_keltner(df, period=20, mult=1.5):
    try:
        ema = _ema(df["close"], period); a = atr(df, period)
        upper = ema + mult*a; lower = ema - mult*a; c = df["close"].iloc[-1]
        if c > upper.iloc[-1]: return 1, 0.7
        if c < lower.iloc[-1]: return -1, 0.7
        return 0, 0.0
    except: return 0, 0.0

def model_vwap(df):
    try:
        vol = df["volume"].iloc[-96:]; tp = ((df["high"]+df["low"]+df["close"])/3).iloc[-96:]
        vwap = (tp*vol).sum()/(vol.sum()+1e-9); c = df["close"].iloc[-1]
        if c > vwap * 1.001: return 1, 0.6
        if c < vwap * 0.999: return -1, 0.6
        return 0, 0.0
    except: return 0, 0.0

def model_ichimoku(df):
    try:
        hi, lo, c = df["high"], df["low"], df["close"]
        tenkan = (hi.rolling(9).max()+lo.rolling(9).min())/2; kijun = (hi.rolling(26).max()+lo.rolling(26).min())/2
        senkou_a = ((tenkan+kijun)/2).shift(26); senkou_b = ((hi.rolling(52).max()+lo.rolling(52).min())/2).shift(26)
        if pd.isna(senkou_a.iloc[-1]): return 0, 0.0
        cloud_top = max(senkou_a.iloc[-1], senkou_b.iloc[-1]); cloud_bot = min(senkou_a.iloc[-1], senkou_b.iloc[-1])
        if c.iloc[-1] > cloud_top and tenkan.iloc[-1] > kijun.iloc[-1]: return 1, 0.85
        if c.iloc[-1] < cloud_bot and tenkan.iloc[-1] < kijun.iloc[-1]: return -1, 0.85
        return 0, 0.0
    except: return 0, 0.0

def model_smc(df):
    try:
        d = df.tail(60).reset_index(drop=True); a = atr(df, 14)
        if a <= 0: return 0, 0.0
        bodies = (d["close"]-d["open"]).abs(); cands = bodies.iloc[3:len(d)-1]
        if cands.empty: return 0, 0.0
        imp_idx = int(cands.idxmax()); imp = d.iloc[imp_idx]
        if abs(imp["close"]-imp["open"]) < a*1.2: return 0, 0.0
        is_bull_imp = imp["close"] > imp["open"]; ob = d.iloc[imp_idx-1]; is_bull_ob = ob["close"] > ob["open"]
        if is_bull_imp == is_bull_ob: return 0, 0.0
        lc = d["close"].iloc[-1]; ob_h = max(ob["open"], ob["close"]); ob_l = min(ob["open"], ob["close"])
        if is_bull_imp and lc >= ob_l: return 1, 0.85
        if not is_bull_imp and lc <= ob_h: return -1, 0.85
        return 0, 0.0
    except: return 0, 0.0


# =========================================================
# 5 MACHINE LEARNING MODELS
# =========================================================
def _atr_series(df, period=14):
    h_, lo, c_ = df["high"], df["low"], df["close"]
    tr = pd.concat([h_-lo, (h_-c_.shift()).abs(), (lo-c_.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _rsi_series(close, period=14):
    delta = close.diff(); gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean(); loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100/(1 + gain/(loss+1e-9))

def build_features(df):
    c = df["close"]; vol = df["volume"]; f = pd.DataFrame(index=df.index)
    f["ret_1"] = c.pct_change(1); f["ret_5"] = c.pct_change(5); f["ret_10"] = c.pct_change(10); f["ret_20"] = c.pct_change(20)
    f["atr_norm"] = _atr_series(df, 14) / (c + 1e-9); f["vol_ratio"] = c.pct_change(1).rolling(20).std()
    f["rsi_14"] = _rsi_series(c, 14) / 100.0; f["ema20_dist"] = (c - c.ewm(span=20).mean()) / (c + 1e-9)
    f["ema50_dist"] = (c - c.ewm(span=50).mean()) / (c + 1e-9)
    f["vol_z"] = (vol - vol.rolling(50).mean()) / (vol.rolling(50).std() + 1e-9)
    f["mom_ratio"] = f["ret_5"] / (f["ret_1"].abs().rolling(5).mean() + 1e-9)
    return f.replace([np.inf, -np.inf], np.nan).dropna()

def build_labels(df, forward=10):
    c = df["close"]; future = c.shift(-forward)
    return (future > c).astype(int)

def model_ml_rf(df, forward=10, lookback=800):
    if not HAS_SKLEARN: return 0, 0.0
    try:
        X = build_features(df); y = build_labels(df, forward)
        valid = X.index.intersection(y.dropna().index); X = X.loc[valid]; y = y.loc[valid]
        if len(X) < forward + 150: return 0, 0.0
        train_X = X.iloc[:-forward].iloc[-lookback:]; train_y = y.iloc[:-forward].iloc[-lookback:]
        split = int(len(train_X) * 0.8)
        if split < 100: return 0, 0.0
        model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=1)
        model.fit(train_X.iloc[:split].values, train_y.iloc[:split].values)
        val_acc = model.score(train_X.iloc[split:].values, train_y.iloc[split:].values)
        if val_acc < 0.52: return 0, 0.0
        cur = X.iloc[-1:].values; proba = model.predict_proba(cur)[0]
        up_prob = proba[1] if len(proba) > 1 else 0.5; conf = abs(up_prob - 0.5) * 2
        if up_prob > 0.6: return 1, conf
        if up_prob < 0.4: return -1, conf
        return 0, 0.0
    except Exception as e: return 0, 0.0

def model_ml_gb(df, forward=10, lookback=800):
    if not HAS_SKLEARN: return 0, 0.0
    try:
        X = build_features(df); y = build_labels(df, forward)
        valid = X.index.intersection(y.dropna().index); X = X.loc[valid]; y = y.loc[valid]
        if len(X) < forward + 150: return 0, 0.0
        train_X = X.iloc[:-forward].iloc[-lookback:]; train_y = y.iloc[:-forward].iloc[-lookback:]
        split = int(len(train_X) * 0.8)
        if split < 100: return 0, 0.0
        model = GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(train_X.iloc[:split].values, train_y.iloc[:split].values)
        val_acc = model.score(train_X.iloc[split:].values, train_y.iloc[split:].values)
        if val_acc < 0.52: return 0, 0.0
        cur = X.iloc[-1:].values; proba = model.predict_proba(cur)[0]
        up_prob = proba[1] if len(proba) > 1 else 0.5; conf = abs(up_prob - 0.5) * 2
        if up_prob > 0.6: return 1, conf
        if up_prob < 0.4: return -1, conf
        return 0, 0.0
    except: return 0, 0.0

def model_ml_knn(df, k=20, window=20, forward=10):
    try:
        c = df["close"].values; n = len(c)
        if n < window + forward + 100: return 0, 0.0
        def norm_w(w): return (w - w.mean()) / (w.std() + 1e-9)
        cur_w = norm_w(c[-window:]); dists = []
        for i in range(window, n - forward):
            hist_w = norm_w(c[i-window:i]); dist = float(np.linalg.norm(cur_w - hist_w))
            label = 1 if c[i + forward] > c[i-1] else 0
            dists.append((dist, label))
        if not dists: return 0, 0.0
        dists.sort(key=lambda x: x[0]); top_k = dists[:k]
        up = sum(1 for _, l in top_k if l == 1) / len(top_k); conf = abs(up - 0.5) * 2
        if up > 0.65: return 1, conf
        if up < 0.35: return -1, conf
        return 0, 0.0
    except: return 0, 0.0

def model_ml_ridge(df, forward=10, lookback=800):
    if not HAS_SKLEARN: return 0, 0.0
    try:
        X = build_features(df); y = build_labels(df, forward)
        valid = X.index.intersection(y.dropna().index); X = X.loc[valid]; y = y.loc[valid]
        if len(X) < forward + 150: return 0, 0.0
        train_X = X.iloc[:-forward].iloc[-lookback:]; train_y = y.iloc[:-forward].iloc[-lookback:]
        split = int(len(train_X) * 0.8)
        if split < 100: return 0, 0.0
        model = RidgeClassifier(alpha=1.0)
        model.fit(train_X.iloc[:split].values, train_y.iloc[:split].values)
        val_acc = model.score(train_X.iloc[split:].values, train_y.iloc[split:].values)
        if val_acc < 0.52: return 0, 0.0
        cur = X.iloc[-1:].values; pred = int(model.predict(cur)[0])
        conf = max(0.3, min(1.0, (val_acc - 0.5) * 4))
        return (1 if pred == 1 else -1), conf
    except: return 0, 0.0

def model_wavelet_trend(df, cutoff=0.15):
    if not HAS_SKLEARN: return 0, 0.0
    try:
        c = df["close"].values
        if len(c) < 100: return 0, 0.0
        b, a = butter(2, cutoff); smooth = filtfilt(b, a, c)
        ss = pd.Series(smooth); e_fast = ss.ewm(span=10).mean(); e_slow = ss.ewm(span=30).mean()
        diff = e_fast.iloc[-1] - e_slow.iloc[-1]; slope = e_fast.diff(3).iloc[-1]
        a_val = atr(df, 14)
        if a_val <= 0: return 0, 0.0
        conf = min(1.0, abs(diff) / (a_val + 1e-6))
        if diff > 0 and slope > 0: return 1, conf
        if diff < 0 and slope < 0: return -1, conf
        return 0, 0.0
    except: return 0, 0.0


# =========================================================
# PARLIAMENT (20 MODELS)
# =========================================================
MODELS = [
    ("EMA_Cross",   model_ema_cross,   1.0),
    ("EMA_Trend",   model_ema_trend,   1.0),
    ("Supertrend",  model_supertrend,  1.2),
    ("Donchian",    model_donchian,    0.8),
    ("Kalman",      model_kalman,      1.0),
    ("MACD",        model_macd,        1.2),
    ("RSI",         model_rsi,         1.0),
    ("Stochastic",  model_stochastic,  0.9),
    ("CCI",         model_cci,         0.8),
    ("WilliamsR",   model_willr,       0.7),
    ("BollingerBB", model_bb,          0.9),
    ("Keltner",     model_keltner,     0.8),
    ("VWAP",        model_vwap,        1.0),
    ("Ichimoku",    model_ichimoku,    1.3),
    ("SMC",         model_smc,         1.5),
    ("ML_RandomForest", model_ml_rf,        1.0),
    ("ML_GradientBoost", model_ml_gb,       1.0),
    ("ML_KNN",          model_ml_knn,       0.9),
    ("ML_Ridge",        model_ml_ridge,     0.7),
    ("ML_Wavelet",      model_wavelet_trend, 0.8),
]

@dataclass
class ParliamentResult:
    direction: str = "NEUTRAL"
    vote_score: float = 0.0
    buy_score: float = 0.0
    sell_score: float = 0.0
    active_models: int = 0
    total_models: int = 0
    buy_ratio: float = 0.0
    sell_ratio: float = 0.0
    votes_detail: list = field(default_factory=list)
    skip_reason: str = ""
    regime: dict = field(default_factory=dict)
    adx: float = 0.0
    verdict: str = "CANCEL"

def run_parliament(cfg, m5, m15, h1, h4) -> ParliamentResult:
    res = ParliamentResult()
    res.total_models = len(MODELS)
    adx_val = adx(m5); res.adx = adx_val
    if adx_val < cfg.adx_min:
        res.skip_reason = f"ADX {adx_val:.1f} < {cfg.adx_min}"
        return res

    H = hurst_exponent(h1["close"]); E = shannon_entropy(h1["close"])
    regime = ("TREND" if H > cfg.hurst_trend_min and E < cfg.entropy_chaos_max
              else ("MEANREV" if H < cfg.hurst_meanrev_max else "CHAOS"))
    res.regime = {"hurst": round(H, 3), "entropy": round(E, 3), "regime": regime}
    if regime == "CHAOS":
        res.skip_reason = f"Regime CHAOS (H={H:.2f} E={E:.2f})"
        return res

    def trend_ema(df_, fast=20, slow=50):
        if len(df_) < slow+5: return "NEUTRAL"
        c = df_["close"]; ef = _ema(c, fast); es = _ema(c, slow)
        if c.iloc[-1] > es.iloc[-1] and ef.iloc[-1] > es.iloc[-1]: return "BULLISH"
        if c.iloc[-1] < es.iloc[-1] and ef.iloc[-1] < es.iloc[-1]: return "BEARISH"
        return "NEUTRAL"
    h1_t = trend_ema(h1); h4_t = trend_ema(h4)

    buy_w = sell_w = 0.0; details = []; active = 0
    for name, fn, weight in MODELS:
        vote, conf = fn(m5)
        if vote == 0:
            details.append({"model": name, "vote": "NEUTRAL", "weight": weight, "conf": round(conf,2), "score": 0.0})
            continue
        active += 1; score = weight * conf
        if vote > 0: buy_w += score; v_txt = "BUY"
        else: sell_w += score; v_txt = "SELL"
        details.append({"model": name, "vote": v_txt, "weight": weight, "conf": round(conf,2), "score": round(score,2)})

    res.buy_score = round(buy_w, 2); res.sell_score = round(sell_w, 2)
    res.active_models = active; res.votes_detail = details
    total_w = buy_w + sell_w
    if total_w < 1e-6:
        res.skip_reason = "Semua model NEUTRAL"; return res

    res.buy_ratio = buy_w / total_w; res.sell_ratio = sell_w / total_w; res.vote_score = round(buy_w - sell_w, 2)

    if active < cfg.parliament_min_active:
        res.skip_reason = f"Hanya {active}/{cfg.parliament_min_active} model aktif"; return res

    if res.buy_ratio >= cfg.parliament_min_ratio:
        if h1_t == "BEARISH" or h4_t == "BEARISH":
            res.skip_reason = f"BUY ditolak: HTF {h1_t}/{h4_t} melawan"; return res
        res.direction = "BUY"
    elif res.sell_ratio >= cfg.parliament_min_ratio:
        if h1_t == "BULLISH" or h4_t == "BULLISH":
            res.skip_reason = f"SELL ditolak: HTF {h1_t}/{h4_t} melawan"; return res
        res.direction = "SELL"
    else:
        res.skip_reason = f"Tidak cukup suara (BUY={res.buy_ratio*100:.0f}% SELL={res.sell_ratio*100:.0f}%)"
        return res

    dominant_ratio = max(res.buy_ratio, res.sell_ratio)
    if dominant_ratio >= cfg.parliament_confident_ratio: res.verdict = "EXECUTE_FULL"
    else: res.verdict = "EXECUTE_HALF"
    return res


# =========================================================
# RISK & LEVELS
# =========================================================
def compute_levels(cfg, m5, side):
    a = atr(m5, 14)
    if a <= 0: raise ValueError("ATR=0")
    price = float(m5["close"].iloc[-1]); buf = cfg.spread_buffer*2
    sl_d = max(a*cfg.atr_sl_mult, buf); tp1_d = max(a*cfg.atr_tp1_mult, buf)
    tp2_d = max(a*cfg.atr_tp2_mult, buf*1.5); tp3_d = max(a*cfg.atr_tp3_mult, buf*2)
    p = cfg.price_decimals
    if side == "BUY": return (round(price-sl_d,p), round(price+tp1_d,p), round(price+tp2_d,p), round(price+tp3_d,p))
    return (round(price+sl_d,p), round(price-tp1_d,p), round(price-tp2_d,p), round(price-tp3_d,p))

def compute_lot(cfg, price, sl, conf_pct, journal):
    sl_d = abs(price - sl)
    if sl_d < 1e-6: raise ValueError("SL too tight")
    closed = [j for j in journal if j.get("outcome") in ("WIN_TP3","WIN_BE","LOSS")]
    if len(closed) >= 20:
        wins = sum(1 for j in closed if str(j["outcome"]).startswith("WIN"))
        wr = wins/len(closed); rrs = [abs(float(j.get("r_achieved") or 0)) for j in closed if abs(float(j.get("r_achieved") or 0))>0]
        avg_rr = sum(rrs)/len(rrs) if rrs else 1.5; source = f"journal({len(closed)})"
    else: wr, avg_rr = 0.55, 1.5; source = f"bootstrap({len(closed)})"
    wr = max(0.35, min(0.75, wr)); avg_rr = max(1.0, min(3.0, avg_rr))
    kelly = max(0.0, (wr*avg_rr - (1-wr))/avg_rr); conf_f = max(0.3, min(1.0, conf_pct/100.0))
    risk_pct = min((kelly/2.0)*100*conf_f, cfg.max_risk_pct_per_trade); risk_pct = max(risk_pct, 0.1)
    risk_money = cfg.account_balance * risk_pct / 100.0; loss_per_lot = cfg.contract_size * sl_d
    lot = max(cfg.min_lot, min(cfg.max_lot, round(risk_money/loss_per_lot, 2)))
    notional = lot * cfg.contract_size * price
    if notional / cfg.account_balance > cfg.hard_leverage_cap:
        lot = max(cfg.min_lot, round(cfg.hard_leverage_cap*cfg.account_balance/(cfg.contract_size*price), 2))
    actual_risk = (lot*loss_per_lot/cfg.account_balance)*100
    if actual_risk > cfg.max_risk_pct_per_trade*1.01:
        lot = max(cfg.min_lot, round((cfg.max_risk_pct_per_trade/100*cfg.account_balance)/loss_per_lot, 2))
    note = f"Kelly[{source}] wr={wr*100:.1f}% rr={avg_rr:.2f} risk={risk_pct:.2f}%"
    return lot, risk_pct, note


# =========================================================
# TELEGRAM
# =========================================================
def _tg_hash(msg): return hashlib.sha256(msg.encode()).hexdigest()[:16]

def retry_backoff(fn, attempts=3, base=1.0, max_delay=8.0):
    last = None
    for i in range(attempts):
        try: return fn()
        except requests.HTTPError as e:
            last = e; code = getattr(e.response,"status_code",0)
            if code in (429,500,502,503,504): time.sleep(min(max_delay, base*(2**i))); continue
            raise
        except Exception as e:
            last = e
            if i < attempts-1: time.sleep(min(max_delay, base*(2**i)))
    raise last

def tg_send(cfg, msg, kv):
    if cfg.dry_run: msg = "🧪 <b>DRY RUN</b>\n" + msg
    tok = os.getenv("TELEGRAM_TOKEN","").strip(); chat = os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not tok or not chat:
        if cfg.dry_run: logging.info(f"[DRY TG] {redact(msg)[:200]}"); return True
        return False
    hsh = _tg_hash(msg); dedup = kv.get("tg_sent_hashes", {}); now = datetime.now(timezone.utc)
    dedup = {k:v for k,v in dedup.items() if (now-datetime.fromisoformat(v)).total_seconds()<3600}
    if hsh in dedup: return True
    def _do():
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id":chat,"text":msg,"parse_mode":"HTML"}, timeout=cfg.http_timeout)
        r.raise_for_status(); d = r.json()
        if not d.get("ok"): raise Exception(f"TG: {d.get('description','?')}")
        return True
    try:
        ok = retry_backoff(_do, attempts=cfg.max_retry, base=1.0)
        if ok: dedup[hsh] = now.isoformat(); kv.set("tg_sent_hashes", dedup)
        return ok
    except Exception as e: logging.error(f"TG: {redact(e)}"); return False

def tg_alert(cfg, msg, kv):
    now = datetime.now(timezone.utc); last = kv.get("alert_limit",{}).get("last")
    if last:
        try:
            if (now - datetime.fromisoformat(last)).total_seconds() < 300: return
        except: pass
    if tg_send(cfg, f"<b>⚠️ APEX v{VERSION}</b>\n<code>{h(redact(msg))}</code>", kv):
        kv.set("alert_limit", {"last": now.isoformat()})

# =========================================================
# KILL SWITCH / CIRCUIT / DRAWDOWN / JOURNAL / OUTCOME
# =========================================================
def kill_switch(kv):
    if os.getenv("APEX_KILL","").lower() in ("1","true","yes"): return True, "env"
    ks = kv.get("kill_switch", {})
    return (True, ks.get("reason","?")) if ks.get("active") else (False, "")

def circuit_allow(kv):
    c = kv.get("circuit", {"failures":0,"opened_until":None})
    if c.get("opened_until"):
        try:
            if datetime.now(timezone.utc) < datetime.fromisoformat(c["opened_until"]): return False
        except: pass
    return True

def circuit_fail(kv, cooldown=15):
    c = kv.get("circuit", {"failures":0,"opened_until":None}); f = int(c.get("failures",0))+1; op = c.get("opened_until")
    if f >= 3: op = (datetime.now(timezone.utc)+timedelta(minutes=cooldown)).isoformat(); f=0
    kv.set("circuit", {"failures":f,"opened_until":op})

def circuit_ok(kv): kv.set("circuit", {"failures":0,"opened_until":None})

def drawdown_check(cfg, kv, journal):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tc = [j for j in journal if str(j.get("timestamp","")).startswith(today) and j.get("outcome") in ("WIN_TP3","WIN_BE","LOSS")]
    dr = sum(float(j.get("r_achieved",0) or 0) for j in tc)
    dlp = abs(dr)*cfg.max_risk_pct_per_trade if dr < 0 else 0
    if dlp >= cfg.max_daily_loss_pct: return True, f"Daily loss {dr:.2f}R"
    c = 0
    for j in journal:
        if j.get("outcome") == "LOSS": c+=1
        elif j.get("outcome") in ("WIN_TP3","WIN_BE"): break
    if c >= cfg.max_consecutive_losses: return True, f"{c} consec losses"
    p = kv.get("drawdown_pause", {})
    if p.get("until"):
        try:
            if datetime.now(timezone.utc) < datetime.fromisoformat(p["until"]): return True, "paused"
        except: pass
    return False, ""

def journal_append(kv, entry, max_j=500):
    j = list(kv.get("journal", [])); j = [e for e in j if e.get("id") != entry.get("id")]
    j.append(entry); j.sort(key=lambda x: x.get("timestamp",""), reverse=True)
    return kv.set("journal", j[:max_j])

def journal_get(kv, limit=500): return kv.get("journal", [])[:limit]

def calc_r(entry, sl, tp1, tp2, tp3, h1, h2, h3, be):
    d = abs(entry-sl)
    if d <= 0: return 0.0
    r1,r2,r3 = abs(tp1-entry)/d, abs(tp2-entry)/d, abs(tp3-entry)/d
    if h3: return round((r1+r2+r3)/3.0, 3)
    if h2: return round((r1+r2)/3.0, 3)
    if h1 and be: return round(r1/3.0, 3)
    if be: return 0.0
    return -1.0

def update_outcomes(cfg, kv, m5):
    events = []; now = datetime.now(timezone.utc)
    journal = journal_get(kv, cfg.max_journal); changed_any = False
    for entry in journal:
        if entry.get("outcome") != "PENDING": continue
        anchor = entry.get("signal_bar_dt") or entry["timestamp"]
        try: ets = pd.Timestamp(anchor, tz="UTC")
        except: continue
        if (now - ets.to_pydatetime()) > timedelta(hours=cfg.pending_max_age_hours):
            entry["outcome"] = "STALE"; entry["r_achieved"] = 0.0
            events.append(("STALE", dict(entry))); changed_any = True; continue
        bars = m5[m5["datetime"] > ets].sort_values("datetime")
        if bars.empty: continue
        side = entry["signal"]; ep, sl = float(entry["entry"]), float(entry["sl"])
        tp1,tp2,tp3 = float(entry["tp1"]), float(entry["tp2"]), float(entry["tp3"])
        if ep <= 0 or sl <= 0 or abs(ep-sl) <= 0: continue
        h1_=bool(entry.get("hit_tp1")); h2=bool(entry.get("hit_tp2")); h3=bool(entry.get("hit_tp3"))
        hs=bool(entry.get("hit_sl")); hb=bool(entry.get("hit_be"))
        wo = entry.get("outcome"); wh1 = h1_; ch = False
        for _,b in bars.iterrows():
            if h3 or hs or hb: break
            csl = ep if h1_ else sl
            if side == "BUY":
                if b["low"] <= csl:
                    if h1_: hb = True
                    else: hs = True
                    ch = True; continue
                if not h1_ and b["high"] >= tp1: h1_ = True; ch = True
                if not h2 and b["high"] >= tp2: h2 = True; ch = True
                if not h3 and b["high"] >= tp3: h3 = True; ch = True
            else:
                if b["high"] >= csl:
                    if h1_: hb = True
                    else: hs = True
                    ch = True; continue
                if not h1_ and b["low"] <= tp1: h1_ = True; ch = True
                if not h2 and b["low"] <= tp2: h2 = True; ch = True
                if not h3 and b["low"] <= tp3: h3 = True; ch = True
        entry["hit_tp1"]=int(h1_); entry["hit_tp2"]=int(h2); entry["hit_tp3"]=int(h3)
        entry["hit_sl"]=int(hs); entry["hit_be"]=int(hb)
        if wo == "PENDING":
            if h3: entry["outcome"] = "WIN_TP3"
            elif h2 or hb: entry["outcome"] = "WIN_BE"
            elif hs: entry["outcome"] = "LOSS"
            if entry["outcome"] != "PENDING":
                entry["r_achieved"] = calc_r(ep,sl,tp1,tp2,tp3,h1_,h2,h3,hb); ch = True
        if ch:
            changed_any = True
            if not wh1 and h1_: events.append(("TP1", dict(entry)))
            if wo == "PENDING" and entry["outcome"] != "PENDING": events.append(("CLOSE", dict(entry)))
    if changed_any and not kv.set("journal", journal): return []
    return events

def send_events(cfg, kv, events):
    for kind, e in events:
        try:
            if kind == "TP1": msg = f"<b>🎯 TP1 HIT</b>\n{e['signal']} @ {e['entry']}\nSL→BE"
            elif kind == "CLOSE":
                out = e.get("outcome","?"); r = float(e.get("r_achieved",0) or 0)
                label = "✅ WIN" if out.startswith("WIN") else "❌ LOSS"
                msg = f"<b>{label} {out}</b>\n{e['signal']} @ {e['entry']}\nR: {r:+.2f}R"
            elif kind == "STALE": msg = f"<b>⏸ STALE</b>\n{e['signal']}"
            else: continue
            tg_send(cfg, msg, kv)
        except: pass


# =========================================================
# MARKET / NEWS
# =========================================================
def market_open(now):
    ny = now.astimezone(NY); wd,hr = ny.weekday(), ny.hour
    return not ((wd==4 and hr>=17) or wd==5 or (wd==6 and hr<18))

def in_killzone(now):
    lt = now.astimezone(LONDON); nt = now.astimezone(NY)
    lth = lt.hour+lt.minute/60; nth = nt.hour+nt.minute/60
    return (7.0<=lth<10.0) or (8.0<=nth<11.0) or (13.0<=nth<16.0)

def check_news(cfg, kv):
    now = datetime.now(timezone.utc); cache = kv.get("news_cache", {})
    if cache.get("ts"):
        try:
            age = (now-datetime.fromisoformat(cache["ts"])).total_seconds()
            ttl = cfg.news_cache_sec_on_fail if cache.get("fail") else cfg.news_cache_sec
            if age < ttl: return cache.get("blocked", False), cache.get("reason","")
        except: pass
    def _do():
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=cfg.http_timeout)
        r.raise_for_status(); return r.json() or []
    try:
        events = retry_backoff(_do, attempts=2, base=1.0)
        kv.set("news_fail_streak", 0)
        for e in events:
            if e.get("country")!="USD" or e.get("impact")!="High": continue
            try: dt = datetime.strptime(e["date"], "%Y-%m-%dT%H:%M:%S%z")
            except: continue
            if abs((dt-now).total_seconds()/60) <= cfg.news_blackout_min:
                kv.set("news_cache", {"ts":now.isoformat(),"blocked":True,"reason":e.get("title","?"),"fail":False})
                return True, e.get("title","?")
        kv.set("news_cache", {"ts":now.isoformat(),"blocked":False,"reason":"","fail":False}); return False, ""
    except:
        streak = int(kv.get("news_fail_streak",0))+1; kv.set("news_fail_streak", streak)
        if streak >= cfg.news_max_consecutive_fail:
            kv.set("news_cache", {"ts":now.isoformat(),"blocked":False,"reason":"fail-open","fail":True}); return False, ""
        kv.set("news_cache", {"ts":now.isoformat(),"blocked":True,"reason":"api down","fail":True}); return True, "api down"


# =========================================================
# BUILD MESSAGE
# =========================================================
def build_msg(cfg, run_num, now_str, src, pres, price, sl, tp1, tp2, tp3, lot, note, offset):
    p = cfg.price_decimals
    rr = abs(tp1-price)/abs(price-sl) if abs(price-sl) > 0 else 0

    rule_buy = rule_sell = ml_buy = ml_sell = 0
    for v in pres.votes_detail:
        is_ml = v["model"].startswith("ML_")
        if v["vote"] == "BUY":
            if is_ml: ml_buy += 1
            else: rule_buy += 1
        elif v["vote"] == "SELL":
            if is_ml: ml_sell += 1
            else: rule_sell += 1

    top5 = sorted([v for v in pres.votes_detail if v["vote"] == pres.direction], key=lambda x: x["score"], reverse=True)[:5]
    voters_txt = "\n".join([f"  • {v['model']}: w={v['weight']} c={v['conf']}" for v in top5])

    return (
        f"<b>🏛️ APEX v{VERSION} — HYBRID {pres.direction}</b>\n\n"
        f"<b>{cfg.symbol}</b> @ <code>{price:.{p}f}</code>\n"
        f"SL: <code>{sl:.{p}f}</code> | TP1: <code>{tp1:.{p}f}</code>\n"
        f"TP2: <code>{tp2:.{p}f}</code> | TP3: <code>{tp3:.{p}f}</code>\n"
        f"R:R(TP1): <code>1:{rr:.2f}</code>\n\n"
        f"<b>Lot:</b> <code>{lot}</code>\n<i>{h(note)}</i>\n\n"
        f"<b>Voting (20 models):</b>\n"
        f"BUY: <code>{pres.buy_score}</code> ({pres.buy_ratio*100:.0f}%)\n"
        f"SELL: <code>{pres.sell_score}</code> ({pres.sell_ratio*100:.0f}%)\n"
        f"Active: <code>{pres.active_models}/20</code>\n"
        f"Verdict: <b>{pres.verdict}</b>\n\n"
        f"<b>Breakdown:</b>\n"
        f"Rules: {rule_buy} BUY / {rule_sell} SELL\n"
        f"ML: {ml_buy} BUY / {ml_sell} SELL\n\n"
        f"<b>Top Voters ({pres.direction}):</b>\n<code>{h(voters_txt)}</code>\n\n"
        f"Regime: <code>{pres.regime.get('regime','?')}</code> | ADX: <code>{pres.adx:.1f}</code>\n"
        f"Data: <code>{src}</code> | Offset: <code>{offset:+.3f}</code>\n"
        f"<code>{h(now_str)} UTC</code>\n\n"
        f"<i>Manual execution MT5. BE after TP1.</i>"
    )


# =========================================================
# RUNNER
# =========================================================
def run_once(cfg):
    kv = KVStore(cfg)
    try: kv.force_load()
    except StateLoadError as e:
        logging.error(f"ABORT: {e}"); return
    now = datetime.now(timezone.utc); now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    ks, reason = kill_switch(kv)
    if ks: logging.warning(f"KILL: {reason}"); return
    if now.hour == cfg.heartbeat_hour_utc:
        hb = kv.get("heartbeat", {})
        if hb.get("date") != now.strftime("%Y-%m-%d"):
            if tg_send(cfg, f"<b>APEX v{VERSION} HEARTBEAT</b>", kv):
                kv.set("heartbeat", {"date": now.strftime("%Y-%m-%d")})
    if not market_open(now): logging.info("Market closed"); return
    if not in_killzone(now): logging.info("Outside killzone"); return
    if not circuit_allow(kv): logging.info("Circuit open"); return
    blocked, why = drawdown_check(cfg, kv, journal_get(kv, 200))
    if blocked: logging.info(f"DD: {why}"); return
    if not cfg.dry_run and check_news(cfg, kv)[0]: logging.info("News shield"); return

    st = kv.get("state", {}) or {}; run_num = int(st.get("run_count",0)) + 1

    try:
        dfs, src, offset = fetch_all_intervals(cfg)
        m5, m15, h1, h4 = dfs["5min"], dfs["15min"], dfs["1h"], dfs["4h"]
    except Exception as e:
        circuit_fail(kv); tg_alert(cfg, f"Data: {e}", kv); return
    circuit_ok(kv)

    try:
        ev = update_outcomes(cfg, kv, m5); send_events(cfg, kv, ev)
    except Exception as e: logging.warning(f"Outcome: {e}")

    m5c = m5.iloc[:-1].reset_index(drop=True); m15c = m15.iloc[:-1].reset_index(drop=True)
    h1c = h1.iloc[:-1].reset_index(drop=True); h4c = h4.iloc[:-1].reset_index(drop=True)
    price = float(m5c["close"].iloc[-1]); signal_bar_iso = m5c.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")

    pres = run_parliament(cfg, m5c, m15c, h1c, h4c)
    logging.info(f"Parliament: {pres.direction} | BUY={pres.buy_ratio*100:.0f}% SELL={pres.sell_ratio*100:.0f}% | active={pres.active_models} | verdict={pres.verdict} | skip={pres.skip_reason}")

    should_fire = pres.direction != "NEUTRAL" and pres.verdict != "CANCEL"
    cool = cfg.cooldown_minutes
    if should_fire:
        lt = st.get("last_signal_type", "NEUTRAL"); lts = st.get("last_signal_ts", "")
        if lt != pres.direction: cool = cfg.opposite_cooldown_minutes
        if lts:
            try:
                ldt = datetime.strptime(lts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                if (now - ldt) < timedelta(minutes=cool): should_fire = False
            except: pass

    tg_ok = False; sl = tp1 = tp2 = tp3 = lot = 0.0; note = ""
    if should_fire:
        try:
            sl, tp1, tp2, tp3 = compute_levels(cfg, m5c, pres.direction)
            conf_pct = max(pres.buy_ratio, pres.sell_ratio) * 100
            lot, _, note = compute_lot(cfg, price, sl, conf_pct, journal_get(kv, 500))
            if pres.verdict == "EXECUTE_HALF": lot = max(cfg.min_lot, round(lot/2.0, 2))
        except Exception as e:
            should_fire = False; logging.warning(f"Risk: {e}")

    if should_fire:
        msg = build_msg(cfg, run_num, now_str, src, pres, price, sl, tp1, tp2, tp3, lot, note, offset)
        jid = f"{now_str.replace(' ','T')}_{pres.direction}_{lot}_{signal_bar_iso.replace(' ','T')}"
        entry = {
            "id": jid, "timestamp": now_str, "signal_bar_dt": signal_bar_iso,
            "symbol": cfg.symbol, "signal": pres.direction,
            "entry": round(price, cfg.price_decimals), "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "lot": lot, "buy_score": pres.buy_score, "sell_score": pres.sell_score,
            "active_models": pres.active_models, "verdict": pres.verdict,
            "regime": pres.regime, "outcome": "PENDING", "r_achieved": 0.0,
        }
        if journal_append(kv, entry, cfg.max_journal):
            tg_ok = tg_send(cfg, msg, kv)
            st["last_signal_type"] = pres.direction; st["last_signal_ts"] = now_str

    st["run_count"] = run_num; st["last_price"] = price; st["last_price_ts"] = now_str
    kv.set("state", st)
    logging.info(f"Run #{run_num} | {pres.direction} | fired={tg_ok} | active={pres.active_models}/20")


# =========================================================
# CLI
# =========================================================
def cmd_reports(cfg, limit=20):
    kv = KVStore(cfg); j = journal_get(kv, limit)
    if not j: print("No journal."); return
    for e in j:
        r = float(e.get("r_achieved",0) or 0)
        print(f"[{e['timestamp']}] {e['signal']:4s} @{e['entry']} lot={e['lot']} {e['outcome']:8s} R={r:+.2f} (B={e.get('buy_score',0)} S={e.get('sell_score',0)} active={e.get('active_models',0)})")

def cmd_stats(cfg):
    kv = KVStore(cfg); j = journal_get(kv, 500)
    if not j: print("No journal."); return
    closed = [e for e in j if e.get("outcome") in ("WIN_TP3","WIN_BE","LOSS")]
    b = sum(1 for e in j if e.get("signal")=="BUY"); s = sum(1 for e in j if e.get("signal")=="SELL")
    print(f"Journal: {len(j)} (BUY={b}, SELL={s})")
    if closed:
        w = sum(1 for e in closed if str(e["outcome"]).startswith("WIN"))
        tot = sum(float(e.get("r_achieved",0) or 0) for e in closed)
        print(f"Closed: {len(closed)} | WR: {w/len(closed)*100:.1f}% | R: {tot:+.2f}")

def cmd_smoke(cfg):
    try: dfs, src, offset = fetch_all_intervals(cfg)
    except Exception as e: print(f"Data: {e}"); sys.exit(1)
    m5c = dfs["5min"].iloc[:-1].reset_index(drop=True); m15c = dfs["15min"].iloc[:-1].reset_index(drop=True)
    h1c = dfs["1h"].iloc[:-1].reset_index(drop=True); h4c = dfs["4h"].iloc[:-1].reset_index(drop=True)
    print(f"Source: {src} | Offset: {offset:+.3f}")
    print(f"Last M5: {m5c.iloc[-1]['datetime']} close={m5c.iloc[-1]['close']:.2f}\n")
    pres = run_parliament(cfg, m5c, m15c, h1c, h4c)
    print(f"=== PARLIAMENT (20 models) ===")
    for v in pres.votes_detail:
        print(f"  {v['model']:16s} {v['vote']:8s} w={v['weight']} conf={v['conf']} score={v['score']}")
    print(f"\nBUY score: {pres.buy_score} ({pres.buy_ratio*100:.1f}%)")
    print(f"SELL score: {pres.sell_score} ({pres.sell_ratio*100:.1f}%)")
    print(f"Active: {pres.active_models}/20")
    print(f"Direction: {pres.direction}")
    print(f"Verdict: {pres.verdict}")
    print(f"ADX: {pres.adx:.1f} | Regime: {pres.regime.get('regime','?')}")
    if pres.skip_reason: print(f"Skip: {pres.skip_reason}")

def cmd_test_tg(cfg):
    kv = KVStore(cfg)
    if tg_send(cfg, f"<b>🔔 TEST v{VERSION}</b>\nHybrid Parliament 20 Models ready! 🏛️🤖", kv):
        print("SENT"); sys.exit(0)
    print("FAILED"); sys.exit(1)

def cmd_force_run(cfg):
    global in_killzone
    _o = in_killzone; in_killzone = lambda _: True
    try: run_once(cfg)
    finally: in_killzone = _o

def cmd_selftest(cfg):
    print(f"=== SELFTEST v{VERSION} ==="); kv = KVStore(cfg)
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    failed = []
    print(f"Persistence: {'Gist' if kv.use_gist else 'LOCAL'}")
    if is_ci and not kv.use_gist: failed.append("Gist not configured")
    td = bool(os.getenv("TWELVE_DATA_API_KEY","").strip())
    tg_tok = bool(os.getenv("TELEGRAM_TOKEN","").strip())
    print(f"Secrets: TD={td} TG={tg_tok}")
    if not td or not tg_tok: failed.append("Secrets missing")
    try: kv.force_load(); print("State: OK")
    except Exception as e: failed.append(f"State: {e}")
    try:
        dfs, src, offset = fetch_all_intervals(cfg)
        print(f"Data: {src} M5={len(dfs['5min'])} 4h={len(dfs['4h'])} offset={offset:+.3f}")
    except Exception as e: failed.append(f"Data: {e}")
    if failed:
        print(f"❌ FAILED: {failed}"); sys.exit(1)
    print("✅ PASSED"); sys.exit(0)


if __name__ == "__main__":
    try: cfg = Config.from_env()
    except ValueError as e: print(e); sys.exit(2)
    setup_logging()
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        try:
            if cmd == "run": run_once(cfg)
            elif cmd == "smoke": cmd_smoke(cfg)
            elif cmd == "force-run": cmd_force_run(cfg)
            elif cmd == "test-tg": cmd_test_tg(cfg)
            elif cmd == "reports": cmd_reports(cfg)
            elif cmd == "stats": cmd_stats(cfg)
            elif cmd == "selftest": cmd_selftest(cfg)
            elif cmd == "help": print("cmds: selftest|test-tg|smoke|force-run|run|reports|stats")
        except SystemExit: raise
        except Exception as e: logging.exception(f"FATAL: {e}"); sys.exit(1)
        sys.exit(0)
    run_once(cfg)
