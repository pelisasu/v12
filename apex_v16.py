"""
APEX PREDATOR v16.6 — FINAL BATTLE READY (FORENSIC AUDIT)
===========================================================
Production-hardened XAUUSD signal engine.

v16.6 fixes (FORENSIC FINAL):
  [B1 CRITICAL] TP/SL check per-bar with last_processed_bar_dt tracker
                (prevents pre-fill bars from triggering false TP)
  [B2 HIGH]     Drawdown pause only after max_consecutive_losses;
                clears on WIN
  [B3 MEDIUM]   SL distance enforced [0.5, 2.5] ATR — outside → 1.8 ATR
  [B4 MEDIUM]   Outcomes computed on CLOSED bars only
  [B5 MEDIUM]   Block new signal if ANY pending (filled or unfilled)
  [v16.5]       OB distance guard: > 2 ATR → ATR pullback fallback
  [v16.4]       HTF weighted scoring (H1=2, H4=1)
  [v16.3]       Counter-trend toggle
  [v16.2]       ADX override, CHAOS diagnostics
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import RidgeClassifier
    from scipy.signal import butter, filtfilt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

VERSION = "16.6"
NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")


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
    m5_fetch_size: int = 800
    price_min: float = 500.0
    price_max: float = 10000.0
    price_max_jump_pct: float = 0.10

    parliament_min_active: int = 8
    parliament_min_ratio: float = 0.55
    parliament_half_ratio: float = 0.65
    parliament_full_ratio: float = 0.75
    adx_min: float = 22.0
    adx_trend_override: float = 35.0
    hurst_trend_min: float = 0.53
    hurst_meanrev_max: float = 0.47
    entropy_chaos_max: float = 0.85

    htf_h1_weight: int = 2
    htf_h4_weight: int = 1

    allow_counter_trend: bool = False
    counter_trend_min_adx: float = 40.0
    counter_trend_min_consensus: float = 0.80
    counter_trend_lot_divisor: float = 2.0

    cooldown_minutes: int = 45
    opposite_cooldown_minutes: int = 120
    max_daily_signals: int = 5

    dry_run: bool = True
    news_blackout_min: int = 30
    news_cache_sec: int = 1800
    news_cache_sec_on_fail: int = 300
    news_max_consecutive_fail: int = 3
    pending_max_age_hours: int = 24
    http_timeout: tuple = (3, 10)
    max_retry: int = 3
    max_journal: int = 500

    ml_use_triple_barrier: bool = True
    ml_purge_bars: int = 10
    ml_min_val_acc: float = 0.53

    entry_pullback_atr: float = 0.30
    entry_max_chase_atr: float = 3.0

    ob_max_distance_atr: float = 2.0
    min_sl_atr_mult: float = 0.5
    max_sl_atr_mult: float = 2.5

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in cls.__dataclass_fields__:
            v = os.getenv(f"APEX_{f.upper()}")
            if v is None or not str(v).strip():
                continue
            cur = getattr(cfg, f)
            try:
                if isinstance(cur, bool):
                    setattr(cfg, f, str(v).strip().lower() in ("1", "true", "yes", "on"))
                elif isinstance(cur, int):
                    setattr(cfg, f, int(v))
                elif isinstance(cur, float):
                    fv = float(v)
                    if not np.isfinite(fv):
                        raise ValueError
                    setattr(cfg, f, fv)
                else:
                    setattr(cfg, f, str(v).strip())
            except ValueError:
                pass
        cfg.gist_id = os.getenv("GIST_ID", cfg.gist_id).strip()
        cfg.gist_token = os.getenv("GIST_TOKEN", cfg.gist_token).strip()
        off = os.getenv("APEX_MT5_OFFSET", "").strip()
        if off:
            try:
                cfg.mt5_offset = float(off)
            except ValueError:
                pass
        cfg.validate()
        return cfg

    def validate(self) -> None:
        assert self.parliament_min_ratio < self.parliament_half_ratio < self.parliament_full_ratio
        assert self.max_lot >= self.min_lot > 0
        assert self.m5_fetch_size >= self.min_bars_m5
        assert 0 < self.max_risk_pct_per_trade <= 10
        assert self.account_balance > 0
        assert 0 < self.hurst_meanrev_max < self.hurst_trend_min < 1
        assert self.counter_trend_lot_divisor >= 1.0
        assert self.htf_h1_weight >= 1 and self.htf_h4_weight >= 0
        assert self.ob_max_distance_atr > 0
        assert 0 < self.min_sl_atr_mult < self.max_sl_atr_mult


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(ch)


def redact(t):
    s = str(t).replace("\n", " ").replace("\r", " ")
    return re.sub(r"(token|apikey|api_key)=[^&\s'\"]+", r"\1=***", s, flags=re.I)


class DataFetchError(Exception):
    pass


class StateLoadError(Exception):
    pass


def _request_with_retry(cfg: Config, method: str, url: str, **kw):
    last = None
    for attempt in range(cfg.max_retry):
        try:
            kw.setdefault("timeout", cfg.http_timeout)
            r = requests.request(method, url, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(0.5 * (2 ** attempt))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = str(e)
            time.sleep(0.5 * (2 ** attempt))
    raise DataFetchError(f"retry exhausted: {last}")


class KVStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.use_gist = bool(cfg.gist_id and cfg.gist_token)
        self._cache = None

    def _headers(self):
        return {"Authorization": f"Bearer {self.cfg.gist_token}",
                "Accept": "application/vnd.github+json"}

    def _load_gist(self):
        if self._cache is not None:
            return self._cache
        try:
            r = _request_with_retry(
                self.cfg, "GET", f"https://api.github.com/gists/{self.cfg.gist_id}",
                headers=self._headers())
            files = r.json().get("files", {})
            fn = self.cfg.gist_state_file
            self._cache = json.loads(files[fn]["content"]) if fn in files and files[fn]["content"].strip() else {}
            return self._cache
        except Exception as e:
            raise StateLoadError(str(e))

    def _save_gist(self, data):
        try:
            body = {"files": {self.cfg.gist_state_file: {
                "content": json.dumps(data, default=str, sort_keys=True)}}}
            _request_with_retry(
                self.cfg, "PATCH", f"https://api.github.com/gists/{self.cfg.gist_id}",
                headers=self._headers(), json=body)
            self._cache = data
            return True
        except Exception as e:
            logging.error(f"Gist SAVE: {redact(e)}")
            return False

    def _load_local(self):
        if self._cache is not None:
            return self._cache
        p = Path(self.cfg.local_state_file)
        self._cache = json.loads(p.read_text()) if p.exists() else {}
        return self._cache

    def _save_local(self, data):
        try:
            Path(self.cfg.local_state_file).write_text(json.dumps(data, default=str, sort_keys=True))
            self._cache = data
            return True
        except Exception:
            return False

    def get(self, k, d=None):
        return (self._load_gist() if self.use_gist else self._load_local()).get(k, d)

    def set(self, k, v):
        data = dict(self._load_gist() if self.use_gist else self._load_local())
        data[k] = v
        return self._save_gist(data) if self.use_gist else self._save_local(data)

    def force_load(self):
        self._load_gist() if self.use_gist else self._load_local()


def atr(df, period=14):
    if len(df) < 2:
        return 0.0
    h_, lo, c_ = df["high"], df["low"], df["close"]
    tr = pd.concat([h_ - lo, (h_ - c_.shift()).abs(), (lo - c_.shift()).abs()], axis=1).max(axis=1)
    v = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return 0.0 if pd.isna(v) or v < 1e-6 else float(v)


def adx(df, period=14):
    try:
        if len(df) < period * 2:
            return 0.0
        up, dn = df["high"].diff(), -df["low"].diff()
        pdm = up.where((up > dn) & (up > 0), 0.0)
        mdm = dn.where((dn > up) & (dn > 0), 0.0)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_s = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
        pdi = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
        mdi = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
        dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
        v = dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        return 0.0 if pd.isna(v) or v < 0 else float(v)
    except Exception:
        return 0.0


def hurst_exponent(close, max_lag=20):
    try:
        ts = np.log(close.dropna().values)
        if len(ts) < max_lag * 2:
            return 0.5
        lags, tau = [], []
        for lag in range(2, max_lag):
            diff = ts[lag:] - ts[:-lag]
            std = float(np.std(diff))
            if std > 1e-12:
                lags.append(lag)
                tau.append(std)
        if len(tau) < 4:
            return 0.5
        return float(np.clip(np.polyfit(np.log(lags), np.log(tau), 1)[0], 0.0, 1.0))
    except Exception:
        return 0.5


def shannon_entropy(close, bins=10):
    try:
        r = close.pct_change().dropna()
        if len(r) < bins * 5:
            return 0.5
        hist, _ = np.histogram(r, bins=bins)
        p = hist / hist.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)) / np.log(bins))
    except Exception:
        return 0.5


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _atr_series(df, period=14):
    h_, lo, c_ = df["high"], df["low"], df["close"]
    return (pd.concat([h_ - lo, (h_ - c_.shift()).abs(), (lo - c_.shift()).abs()], axis=1)
            .max(axis=1).ewm(alpha=1 / period, adjust=False).mean())


def _rsi_series(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def model_ema_cross(df):
    if len(df) < 30:
        return 0, 0.0
    diff = _ema(df["close"], 9) - _ema(df["close"], 21)
    if diff.iloc[-1] > 0 and diff.diff().iloc[-1] > 0:
        return 1, min(1.0, abs(diff.iloc[-1]) / (atr(df, 14) + 1e-6))
    if diff.iloc[-1] < 0 and diff.diff().iloc[-1] < 0:
        return -1, min(1.0, abs(diff.iloc[-1]) / (atr(df, 14) + 1e-6))
    return 0, 0.0


def model_ema_trend(df):
    if len(df) < 60:
        return 0, 0.0
    c = df["close"]
    e20, e50 = _ema(c, 20), _ema(c, 50)
    if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]:
        return 1, 0.9
    if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]:
        return -1, 0.9
    return 0, 0.0


def model_supertrend(df, period=10, mult=3.0):
    if len(df) < period + 5:
        return 0, 0.0
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper, lower = hl2 + mult * a, hl2 - mult * a
    close = df["close"]
    trend = np.ones(len(df))
    for i in range(1, len(df)):
        if close.iloc[i] > upper.iloc[i - 1]:
            trend[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    if trend[-1] == 1 and trend[-2] == 1:
        return 1, 0.85
    if trend[-1] == -1 and trend[-2] == -1:
        return -1, 0.85
    return 0, 0.0


def model_donchian(df, period=20):
    if len(df) < period + 2:
        return 0, 0.0
    hh = df["high"].rolling(period).max().iloc[-2]
    ll = df["low"].rolling(period).min().iloc[-2]
    c = df["close"].iloc[-1]
    if c > hh:
        return 1, 0.8
    if c < ll:
        return -1, 0.8
    return 0, 0.0


def model_kalman(df):
    obs = df["close"].values
    if len(obs) < 10:
        return 0, 0.0
    x, P, Q, R = obs[0], 1.0, 1e-3, 1e-2
    est = np.empty(len(obs))
    est[0] = x
    for i in range(1, len(obs)):
        P += Q
        K = P / (P + R)
        x = x + K * (obs[i] - x)
        P *= (1 - K)
        est[i] = x
    slope = est[-1] - est[-5]
    if slope > 0:
        return 1, min(1.0, abs(slope) / (atr(df, 14) + 1e-6))
    if slope < 0:
        return -1, min(1.0, abs(slope) / (atr(df, 14) + 1e-6))
    return 0, 0.0


def model_macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal + 5:
        return 0, 0.0
    c = df["close"]
    macd = _ema(c, fast) - _ema(c, slow)
    sig = _ema(macd, signal)
    hist = macd - sig
    if macd.iloc[-1] > sig.iloc[-1] and hist.diff().iloc[-1] > 0:
        return 1, 0.9
    if macd.iloc[-1] < sig.iloc[-1] and hist.diff().iloc[-1] < 0:
        return -1, 0.9
    return 0, 0.0


def model_rsi(df, period=14):
    if len(df) < period + 2:
        return 0, 0.0
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rsi_s = 100 - 100 / (1 + gain / (loss + 1e-9))
    rsi, prev = rsi_s.iloc[-1], rsi_s.iloc[-2]
    if rsi < 30 and rsi > prev:
        return 1, 0.85
    if rsi > 70 and rsi < prev:
        return -1, 0.85
    if 50 < rsi < 65 and rsi > prev:
        return 1, 0.6
    if 35 < rsi < 50 and rsi < prev:
        return -1, 0.6
    return 0, 0.0


def model_stochastic(df, k=14, d=3):
    if len(df) < k + d + 2:
        return 0, 0.0
    low = df["low"].rolling(k).min()
    high = df["high"].rolling(k).max()
    sk = 100 * (df["close"] - low) / (high - low + 1e-9)
    sd = sk.rolling(d).mean()
    k_, d_ = sk.iloc[-1], sd.iloc[-1]
    kp, dp = sk.iloc[-2], sd.iloc[-2]
    if k_ > d_ and kp <= dp and k_ < 25:
        return 1, 0.8
    if k_ < d_ and kp >= dp and k_ > 75:
        return -1, 0.8
    return 0, 0.0


def model_cci(df, period=20):
    if len(df) < period + 2:
        return 0, 0.0
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    cci = (tp - ma) / (0.015 * md + 1e-9)
    c, p = cci.iloc[-1], cci.iloc[-2]
    if p < -100 and c > -100:
        return 1, 0.8
    if p > 100 and c < 100:
        return -1, 0.8
    return 0, 0.0


def model_willr(df, period=14):
    if len(df) < period + 2:
        return 0, 0.0
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    wr = -100 * (hh - df["close"]) / (hh - ll + 1e-9)
    if wr.iloc[-1] < -80 and wr.iloc[-1] > wr.iloc[-2]:
        return 1, 0.7
    if wr.iloc[-1] > -20 and wr.iloc[-1] < wr.iloc[-2]:
        return -1, 0.7
    return 0, 0.0


def model_bb(df, period=20, mult=2.0):
    if len(df) < period + 2:
        return 0, 0.0
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper, lower = ma + mult * std, ma - mult * std
    c = df["close"].iloc[-1]
    if c < lower.iloc[-1]:
        return 1, 0.75
    if c > upper.iloc[-1]:
        return -1, 0.75
    return 0, 0.0


def model_keltner(df, period=20, mult=1.5):
    if len(df) < period + 2:
        return 0, 0.0
    ema = _ema(df["close"], period)
    a = atr(df, period)
    upper, lower = ema + mult * a, ema - mult * a
    c = df["close"].iloc[-1]
    if c > upper.iloc[-1]:
        return 1, 0.7
    if c < lower.iloc[-1]:
        return -1, 0.7
    return 0, 0.0


def model_vwap(df):
    if len(df) < 100:
        return 0, 0.0
    vol = df["volume"].iloc[-96:]
    tp = ((df["high"] + df["low"] + df["close"]) / 3).iloc[-96:]
    vwap = (tp * vol).sum() / (vol.sum() + 1e-9)
    c = df["close"].iloc[-1]
    if c > vwap * 1.001:
        return 1, 0.6
    if c < vwap * 0.999:
        return -1, 0.6
    return 0, 0.0


def model_ichimoku(df):
    if len(df) < 60:
        return 0, 0.0
    hi, lo, c = df["high"], df["low"], df["close"]
    tenkan = (hi.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    sa = ((tenkan + kijun) / 2).shift(26)
    sb = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    if pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):
        return 0, 0.0
    top = max(sa.iloc[-1], sb.iloc[-1])
    bot = min(sa.iloc[-1], sb.iloc[-1])
    if c.iloc[-1] > top and tenkan.iloc[-1] > kijun.iloc[-1]:
        return 1, 0.85
    if c.iloc[-1] < bot and tenkan.iloc[-1] < kijun.iloc[-1]:
        return -1, 0.85
    return 0, 0.0


def model_smc(df):
    if len(df) < 60:
        return 0, 0.0
    d = df.tail(60).reset_index(drop=True)
    a = atr(df, 14)
    if a <= 0:
        return 0, 0.0
    bodies = (d["close"] - d["open"]).abs()
    cands = bodies.iloc[3:len(d) - 1]
    if cands.empty:
        return 0, 0.0
    imp_idx = int(cands.idxmax())
    if imp_idx < 1:
        return 0, 0.0
    imp = d.iloc[imp_idx]
    if abs(imp["close"] - imp["open"]) < a * 1.2:
        return 0, 0.0
    is_bull_imp = imp["close"] > imp["open"]
    ob = d.iloc[imp_idx - 1]
    is_bull_ob = ob["close"] > ob["open"]
    if is_bull_imp == is_bull_ob:
        return 0, 0.0
    lc = d["close"].iloc[-1]
    ob_h = max(ob["open"], ob["close"])
    ob_l = min(ob["open"], ob["close"])
    if is_bull_imp and lc >= ob_l:
        return 1, 0.85
    if (not is_bull_imp) and lc <= ob_h:
        return -1, 0.85
    return 0, 0.0


def build_features(df):
    c, vol = df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)
    f["ret_1"] = c.pct_change(1)
    f["ret_5"] = c.pct_change(5)
    f["ret_10"] = c.pct_change(10)
    f["ret_20"] = c.pct_change(20)
    f["atr_norm"] = _atr_series(df, 14) / (c + 1e-9)
    f["vol_ratio"] = c.pct_change(1).rolling(20).std()
    f["rsi_14"] = _rsi_series(c, 14) / 100.0
    f["ema20_dist"] = (c - c.ewm(span=20).mean()) / (c + 1e-9)
    f["ema50_dist"] = (c - c.ewm(span=50).mean()) / (c + 1e-9)
    vol_std = vol.rolling(50).std()
    f["vol_z"] = np.where(vol_std > 1e-9, (vol - vol.rolling(50).mean()) / (vol_std + 1e-9), 0.0)
    f["mom_ratio"] = f["ret_5"] / (f["ret_1"].abs().rolling(5).mean() + 1e-9)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def build_labels_triple_barrier(df, horizon=30, atr_tp=1.0, atr_sl=1.8):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    a = _atr_series(df, 14).values
    n = len(c)
    labels = np.zeros(n)
    tp = c + atr_tp * a
    sl = c - atr_sl * a
    for i in range(n - horizon):
        hh = h[i + 1:i + horizon + 1]
        ll = l[i + 1:i + horizon + 1]
        hit_tp = np.where(hh >= tp[i])[0]
        hit_sl = np.where(ll <= sl[i])[0]
        idx_tp = hit_tp[0] if len(hit_tp) else horizon + 1
        idx_sl = hit_sl[0] if len(hit_sl) else horizon + 1
        labels[i] = 0 if idx_tp == idx_sl else (1 if idx_tp < idx_sl else -1)
    return pd.Series(labels[:n - horizon], index=df.index[:n - horizon])


def _ml_predict(model_factory, df, cfg):
    if not HAS_SKLEARN:
        return 0, 0.0
    try:
        X = build_features(df)
        y = build_labels_triple_barrier(df, horizon=cfg.ml_purge_bars * 3)
        y = (y == 1).astype(int)
        valid = X.index.intersection(y.dropna().index)
        X, y = X.loc[valid], y.loc[valid]
        forward = cfg.ml_purge_bars
        if len(X) < forward + 200:
            return 0, 0.0
        trX, trY = X.iloc[:-forward], y.iloc[:-forward]
        if len(trX) < 200:
            return 0, 0.0
        n = len(trX)
        split = int(n * 0.8)
        tr_end = max(100, split - forward)
        X_tr, y_tr = trX.iloc[:tr_end], trY.iloc[:tr_end]
        X_val, y_val = trX.iloc[split:], trY.iloc[split:]
        if len(X_tr) < 100 or len(X_val) < 20:
            return 0, 0.0
        model = model_factory()
        model.fit(X_tr.values, y_tr.values)
        if model.score(X_val.values, y_val.values) < cfg.ml_min_val_acc:
            return 0, 0.0
        proba = model.predict_proba(X.iloc[-1:].values)[0]
        up = proba[1] if len(proba) > 1 else 0.5
        if up > 0.60:
            return 1, abs(up - 0.5) * 2
        if up < 0.40:
            return -1, abs(up - 0.5) * 2
        return 0, 0.0
    except Exception:
        return 0, 0.0


def model_ml_rf(df, cfg):
    return _ml_predict(lambda: RandomForestClassifier(
        n_estimators=80, max_depth=5, min_samples_leaf=8,
        random_state=42, n_jobs=1), df, cfg)


def model_ml_gb(df, cfg):
    return _ml_predict(lambda: GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.08,
        random_state=42), df, cfg)


def model_ml_ridge(df, cfg):
    return _ml_predict(lambda: RidgeClassifier(alpha=1.0), df, cfg)


def model_ml_knn(df, k=20, window=20, forward=10):
    try:
        c = df["close"].values
        n = len(c)
        if n < window + forward + 100:
            return 0, 0.0

        def nw(w):
            return (w - w.mean()) / (w.std() + 1e-9)

        cur = nw(c[-window:])
        dists = []
        for i in range(window, n - forward):
            dist = float(np.linalg.norm(cur - nw(c[i - window:i])))
            dists.append((dist, 1 if c[i + forward] > c[i - 1] else 0))
        if not dists:
            return 0, 0.0
        dists.sort(key=lambda x: x[0])
        up = sum(1 for _, l in dists[:k] if l == 1) / k
        if up > 0.65:
            return 1, abs(up - 0.5) * 2
        if up < 0.35:
            return -1, abs(up - 0.5) * 2
        return 0, 0.0
    except Exception:
        return 0, 0.0


def model_wavelet_trend(df, cutoff=0.15):
    if not HAS_SKLEARN:
        return 0, 0.0
    try:
        c = df["close"].values
        if len(c) < 100:
            return 0, 0.0
        b, a = butter(2, cutoff)
        smooth = pd.Series(filtfilt(b, a, c))
        e_fast = smooth.ewm(span=10).mean()
        e_slow = smooth.ewm(span=30).mean()
        diff = e_fast.iloc[-1] - e_slow.iloc[-1]
        slope = e_fast.diff(3).iloc[-1]
        a_val = atr(df, 14)
        if a_val <= 0:
            return 0, 0.0
        conf = min(1.0, abs(diff) / (a_val + 1e-6))
        if diff > 0 and slope > 0:
            return 1, conf
        if diff < 0 and slope < 0:
            return -1, conf
        return 0, 0.0
    except Exception:
        return 0, 0.0


MODELS = [
    ("EMA_Cross", model_ema_cross, 1.0),
    ("EMA_Trend", model_ema_trend, 1.0),
    ("Supertrend", model_supertrend, 1.2),
    ("Donchian", model_donchian, 0.8),
    ("Kalman", model_kalman, 1.0),
    ("MACD", model_macd, 1.2),
    ("RSI", model_rsi, 1.0),
    ("Stochastic", model_stochastic, 0.9),
    ("CCI", model_cci, 0.8),
    ("WilliamsR", model_willr, 0.7),
    ("BollingerBB", model_bb, 0.9),
    ("Keltner", model_keltner, 0.8),
    ("VWAP", model_vwap, 1.0),
    ("Ichimoku", model_ichimoku, 1.3),
    ("SMC", model_smc, 1.5),
    ("ML_RF", model_ml_rf, 1.0),
    ("ML_GB", model_ml_gb, 1.0),
    ("ML_KNN", model_ml_knn, 0.9),
    ("ML_Ridge", model_ml_ridge, 0.7),
    ("ML_Wavelet", model_wavelet_trend, 0.8),
]


@dataclass
class ParliamentResult:
    direction: str = "NEUTRAL"
    buy_score: float = 0.0
    sell_score: float = 0.0
    active_models: int = 0
    buy_ratio: float = 0.0
    sell_ratio: float = 0.0
    votes_detail: list = field(default_factory=list)
    skip_reason: str = ""
    regime: dict = field(default_factory=dict)
    adx: float = 0.0
    verdict: str = "CANCEL"


class SignalGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run_parliament(self, m5, m15, h1, h4) -> ParliamentResult:
        res = ParliamentResult()
        adx_val = adx(m5)
        res.adx = adx_val
        if adx_val < self.cfg.adx_min:
            res.skip_reason = f"ADX {adx_val:.1f} < {self.cfg.adx_min}"
            return res

        H = hurst_exponent(h1["close"])
        E = shannon_entropy(h1["close"])
        adx_override = adx_val >= self.cfg.adx_trend_override
        if adx_override or (H > self.cfg.hurst_trend_min and E < self.cfg.entropy_chaos_max):
            regime = "TREND"
        elif H < self.cfg.hurst_meanrev_max:
            regime = "MEANREV"
        else:
            regime = "CHAOS"

        res.regime = {"hurst": round(H, 3), "entropy": round(E, 3),
                      "regime": regime, "adx_override": adx_override}

        def trend_ema(df_):
            if len(df_) < 60:
                return "NEUTRAL"
            c = df_["close"]
            ef, es = _ema(c, 20), _ema(c, 50)
            if c.iloc[-1] > es.iloc[-1] and ef.iloc[-1] > es.iloc[-1]:
                return "BULLISH"
            if c.iloc[-1] < es.iloc[-1] and ef.iloc[-1] < es.iloc[-1]:
                return "BEARISH"
            return "NEUTRAL"

        h1_t, h4_t = trend_ema(h1), trend_ema(h4)
        buy_w, sell_w, details, active = 0.0, 0.0, [], 0

        for name, fn, weight in MODELS:
            try:
                if name.startswith("ML_") and name not in ("ML_KNN", "ML_Wavelet"):
                    vote, conf = fn(m5, self.cfg)
                else:
                    vote, conf = fn(m5)
            except Exception as e:
                logging.debug(f"model {name} failed: {e}")
                vote, conf = 0, 0.0

            if vote == 0:
                details.append({"model": name, "vote": "NEUTRAL", "weight": weight,
                                "conf": round(conf, 2), "score": 0.0})
                continue
            active += 1
            score = weight * conf
            if vote > 0:
                buy_w += score
                v_txt = "BUY"
            else:
                sell_w += score
                v_txt = "SELL"
            details.append({"model": name, "vote": v_txt, "weight": weight,
                            "conf": round(conf, 2), "score": round(score, 2)})

        res.buy_score, res.sell_score = round(buy_w, 2), round(sell_w, 2)
        res.active_models, res.votes_detail = active, details
        total_w = buy_w + sell_w
        if total_w < 1e-6:
            res.skip_reason = "Semua model NEUTRAL"
            return res
        res.buy_ratio, res.sell_ratio = buy_w / total_w, sell_w / total_w

        if active < self.cfg.parliament_min_active:
            res.skip_reason = f"Hanya {active}/{self.cfg.parliament_min_active} model aktif"
            return res

        if regime == "CHAOS":
            res.skip_reason = f"Regime CHAOS (H={H:.2f} E={E:.2f})"
            return res

        dom = max(res.buy_ratio, res.sell_ratio)
        if dom >= self.cfg.parliament_full_ratio:
            res.verdict = "EXECUTE_FULL"
        elif dom >= self.cfg.parliament_half_ratio:
            res.verdict = "EXECUTE_HALF"
        elif dom >= self.cfg.parliament_min_ratio:
            res.verdict = "EXECUTE_MICRO"
        else:
            res.skip_reason = f"Consensus {dom * 100:.0f}% < {self.cfg.parliament_min_ratio * 100:.0f}%"
            return res

        def _counter_trend_allowed():
            return (
                self.cfg.allow_counter_trend
                and adx_val >= self.cfg.counter_trend_min_adx
                and dom >= self.cfg.counter_trend_min_consensus
            )

        if res.buy_ratio > res.sell_ratio:
            signal_dir = "BUY"
            h1_a = 1 if h1_t == "BULLISH" else (-1 if h1_t == "BEARISH" else 0)
            h4_a = 1 if h4_t == "BULLISH" else (-1 if h4_t == "BEARISH" else 0)
        else:
            signal_dir = "SELL"
            h1_a = 1 if h1_t == "BEARISH" else (-1 if h1_t == "BULLISH" else 0)
            h4_a = 1 if h4_t == "BEARISH" else (-1 if h4_t == "BULLISH" else 0)

        htf_score = h1_a * self.cfg.htf_h1_weight + h4_a * self.cfg.htf_h4_weight
        max_score = self.cfg.htf_h1_weight + self.cfg.htf_h4_weight
        res.regime["htf_h1"] = h1_t
        res.regime["htf_h4"] = h4_t
        res.regime["htf_score"] = htf_score

        if htf_score >= self.cfg.htf_h1_weight:
            pass
        elif htf_score >= 1:
            if res.verdict == "EXECUTE_FULL":
                res.verdict = "EXECUTE_HALF"
            res.regime["htf_mixed"] = True
        elif htf_score == 0:
            res.verdict = "EXECUTE_HALF"
            res.regime["htf_mixed"] = True
        elif htf_score >= -1:
            if _counter_trend_allowed():
                res.regime["counter_trend"] = True
                res.verdict = "EXECUTE_HALF"
            else:
                res.skip_reason = f"HTF lemah (H1={h1_t} H4={h4_t}, score={htf_score:+d}/{max_score})"
                return res
        else:
            if _counter_trend_allowed():
                res.regime["counter_trend"] = True
                res.verdict = "EXECUTE_MICRO"
            else:
                res.skip_reason = f"HTF both oppose (H1={h1_t} H4={h4_t}, score={htf_score:+d}/{max_score})"
                return res

        res.direction = signal_dir
        return res

    def detect_order_block(self, df, side, lookback=40):
        if len(df) < lookback + 5:
            return None
        d = df.tail(lookback + 5).reset_index(drop=True)
        a = atr(df, 14)
        if a <= 0:
            return None
        for i in range(len(d) - 2, 2, -1):
            body = abs(d["close"].iloc[i] - d["open"].iloc[i])
            if body < 1.2 * a:
                continue
            is_bull = d["close"].iloc[i] > d["open"].iloc[i]
            if side == "BUY" and not is_bull:
                continue
            if side == "SELL" and is_bull:
                continue
            for j in range(i - 1, max(i - 10, 0), -1):
                if side == "BUY" and d["close"].iloc[j] < d["open"].iloc[j]:
                    return {"high": float(d["high"].iloc[j]),
                            "low": float(d["low"].iloc[j]),
                            "bar_index": j, "displacement_idx": i}
                if side == "SELL" and d["close"].iloc[j] > d["open"].iloc[j]:
                    return {"high": float(d["high"].iloc[j]),
                            "low": float(d["low"].iloc[j]),
                            "bar_index": j, "displacement_idx": i}
        return None


class MarketDataHandler:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _apply_offset(self, df, off):
        if off != 0.0:
            for c in ("open", "high", "low", "close"):
                df[c] = df[c] + off
        return df

    def resample_ohlc(self, df, rule):
        return (df.set_index("datetime")
                .resample(rule, label="left", closed="left")
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"})
                .dropna().reset_index())

    def _validate_df(self, df, name, min_bars):
        if df is None or len(df) < min_bars:
            raise DataFetchError(f"{name}: {0 if df is None else len(df)} < {min_bars}")
        last = df["close"].iloc[-1]
        if not (self.cfg.price_min <= last <= self.cfg.price_max):
            raise DataFetchError(f"{name}: price {last} out of range")
        if df["datetime"].isna().any():
            raise DataFetchError(f"{name}: NaN datetime")
        return df

    def _check_staleness(self, df):
        if df.empty:
            return
        age_h = (datetime.now(timezone.utc) - df["datetime"].iloc[-1].to_pydatetime()).total_seconds() / 3600
        if age_h * 60 > self.cfg.max_stale_mult * 5:
            raise DataFetchError(f"Stale data: {age_h:.1f}h old")

    def get_dynamic_offset(self):
        oanda_key = os.getenv("OANDA_API_KEY", "").strip()
        if not oanda_key:
            return self.cfg.mt5_offset
        try:
            r1 = _request_with_retry(
                self.cfg, "GET",
                "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/pricing",
                headers={"Authorization": f"Bearer {oanda_key}"})
            oanda = float(r1.json()["prices"][0]["bids"][0]["price"])
            td_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
            r2 = _request_with_retry(
                self.cfg, "GET", "https://api.twelvedata.com/price",
                params={"symbol": "XAU/USD", "apikey": td_key})
            td = float(r2.json()["price"])
            return round(oanda - td, 3) if td > 0 else self.cfg.mt5_offset
        except Exception:
            return self.cfg.mt5_offset

    def fetch_twelvedata(self, offset):
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("No TD key")
        size = min(self.cfg.m5_fetch_size, 800)

        def _fetch_tf(interval, outputsize):
            r = _request_with_retry(
                self.cfg, "GET", "https://api.twelvedata.com/time_series",
                params={"symbol": "XAU/USD", "interval": interval,
                        "outputsize": outputsize, "apikey": key, "timezone": "UTC"})
            d = r.json()
            if "values" not in d or not d["values"]:
                raise DataFetchError(f"TD {interval}: {d.get('message', 'no values')}")
            df = pd.DataFrame(d["values"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ("open", "high", "low", "close"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["volume"] = 100.0
            df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
            return self._apply_offset(df, offset)

        m5 = _fetch_tf("5min", size)
        h1 = _fetch_tf("1h", 800)
        h4 = _fetch_tf("4h", 800)

        self._validate_df(m5, "TD-M5", self.cfg.min_bars_m5)
        self._validate_df(h1, "TD-H1", self.cfg.min_bars_h1)
        self._validate_df(h4, "TD-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5)

        return {
            "5min": m5,
            "15min": self._apply_offset(self.resample_ohlc(m5, "15min"), offset),
            "1h": h1,
            "4h": h4,
        }, "TwelveData"

    def fetch_yahoo(self, offset):
        headers = {"User-Agent": f"ApexBot/{VERSION}"}

        def _do(symbol, iv, rng):
            r = _request_with_retry(
                self.cfg, "GET",
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval": iv, "range": rng}, headers=headers)
            res = (r.json().get("chart", {}).get("result") or [None])[0]
            if not res:
                raise DataFetchError("Yahoo: no result")
            ts = res.get("timestamp", [])
            q = res.get("indicators", {}).get("quote", [{}])[0]
            if not ts or not q:
                raise DataFetchError("Yahoo: empty quote")
            o, hh, ll, cc = q.get("open", []), q.get("high", []), q.get("low", []), q.get("close", [])
            vv = q.get("volume") or [None] * len(ts)
            n = min(len(ts), len(o), len(hh), len(ll), len(cc))
            rows = []
            for i in range(n):
                if o[i] is None or hh[i] is None or ll[i] is None or cc[i] is None:
                    continue
                vol = vv[i] if i < len(vv) and vv[i] is not None else 100.0
                rows.append({"epoch": int(ts[i]), "open": float(o[i]),
                             "high": float(hh[i]), "low": float(ll[i]),
                             "close": float(cc[i]), "volume": float(vol)})
            if not rows:
                raise DataFetchError("Yahoo: 0 valid rows")
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            return df[["datetime", "open", "high", "low", "close", "volume"]]

        out = {}
        for iv, rng in (("5min", "5d"), ("15min", "5d"), ("1h", "1mo")):
            iv_map = {"5min": "5m", "15min": "15m", "1h": "1h"}
            out[iv] = self._apply_offset(_do("XAUUSD=X", iv_map[iv], rng), offset)
        out["4h"] = self._apply_offset(self.resample_ohlc(_do("XAUUSD=X", "1h", "1mo"), "4h"), offset)

        self._validate_df(out["5min"], "YH-M5", self.cfg.min_bars_m5)
        self._validate_df(out["15min"], "YH-M15", self.cfg.min_bars_m15)
        self._validate_df(out["1h"], "YH-H1", self.cfg.min_bars_h1)
        self._validate_df(out["4h"], "YH-H4", self.cfg.min_bars_h4)
        self._check_staleness(out["5min"])
        return out, "Yahoo"

    def fetch_all_intervals(self):
        offset = self.get_dynamic_offset()
        errs = []
        for name, fn in (("TwelveData", self.fetch_twelvedata), ("Yahoo", self.fetch_yahoo)):
            try:
                dfs, src = fn(offset)
                return dfs, src, offset
            except Exception as e:
                errs.append(f"{name}: {redact(e)}")
        raise DataFetchError(" | ".join(errs))


class NewsSafetyFilter:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def is_news_blackout(self):
        if self.cfg.news_blackout_min <= 0:
            return False, ""
        cached = self.kv.get("news_cache", {}) or {}
        cache_ts = cached.get("ts", 0)
        cache_ok = cached.get("ok", False)
        now_ts = time.time()
        ttl = self.cfg.news_cache_sec if cache_ok else self.cfg.news_cache_sec_on_fail
        if now_ts - cache_ts < ttl:
            return cached.get("blackout", False), cached.get("title", "")

        try:
            r = _request_with_retry(
                self.cfg, "GET",
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
            events = r.json()
            now = datetime.now(timezone.utc)
            blackout, title = False, ""
            for ev in events:
                if ev.get("country") != "USD" or ev.get("impact") != "High":
                    continue
                try:
                    evt = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                except Exception:
                    continue
                if abs((evt - now).total_seconds()) / 60 <= self.cfg.news_blackout_min:
                    blackout, title = True, ev.get("title", "High impact")
                    break
            self.kv.set("news_cache", {"ts": now_ts, "ok": True,
                                       "blackout": blackout, "title": title})
            return blackout, title
        except Exception as e:
            logging.warning(f"News fetch: {redact(e)}")
            self.kv.set("news_cache", {"ts": now_ts, "ok": False,
                                       "blackout": False, "title": ""})
            return False, ""


class RiskManagementEngine:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def compute_levels(self, m5, side, ob):
        """
        v16.6: OB distance guard + SL distance range enforcement.
        """
        a = atr(m5, 14)
        if a <= 0:
            raise ValueError("ATR=0")
        price = float(m5["close"].iloc[-1])
        p = self.cfg.price_decimals
        buf = self.cfg.spread_buffer * 2
        max_ob_dist = a * self.cfg.ob_max_distance_atr
        entry_source = "ATR"

        if side == "BUY":
            use_ob = False
            ob_h = ob_l = price
            if ob:
                ob_h, ob_l = ob["high"], ob["low"]
                if ob_h < price:
                    ob_entry = ob_h - (ob_h - ob_l) * 0.5
                    if (price - ob_entry) <= max_ob_dist:
                        use_ob = True

            if use_ob:
                entry = round(ob_h - (ob_h - ob_l) * 0.5, p)
                sl_raw = round(min(ob_l, entry) - buf, p)
                entry_source = "OB"
            else:
                entry = round(price - a * self.cfg.entry_pullback_atr, p)
                sl_raw = round(entry - a * self.cfg.atr_sl_mult - buf, p)

            entry = min(entry, round(price - a * 0.05, p))

            # v16.6: enforce SL distance range
            sl_dist = entry - sl_raw
            if sl_dist < self.cfg.min_sl_atr_mult * a or sl_dist > self.cfg.max_sl_atr_mult * a:
                sl = round(entry - a * self.cfg.atr_sl_mult - buf, p)
                logging.info(f"SL dist {sl_dist:.2f} outside [{self.cfg.min_sl_atr_mult}*A,{self.cfg.max_sl_atr_mult}*A] → reset to ATR*{self.cfg.atr_sl_mult}")
            else:
                sl = sl_raw

            if entry <= sl:
                raise ValueError(f"invalid BUY entry={entry} sl={sl}")
            return (entry, sl,
                    round(entry + a * self.cfg.atr_tp1_mult, p),
                    round(entry + a * self.cfg.atr_tp2_mult, p),
                    round(entry + a * self.cfg.atr_tp3_mult, p), entry_source)
        else:
            use_ob = False
            ob_h = ob_l = price
            if ob:
                ob_h, ob_l = ob["high"], ob["low"]
                if ob_l > price:
                    ob_entry = ob_l + (ob_h - ob_l) * 0.5
                    if (ob_entry - price) <= max_ob_dist:
                        use_ob = True

            if use_ob:
                entry = round(ob_l + (ob_h - ob_l) * 0.5, p)
                sl_raw = round(max(ob_h, entry) + buf, p)
                entry_source = "OB"
            else:
                entry = round(price + a * self.cfg.entry_pullback_atr, p)
                sl_raw = round(entry + a * self.cfg.atr_sl_mult + buf, p)

            entry = max(entry, round(price + a * 0.05, p))

            # v16.6: enforce SL distance range
            sl_dist = sl_raw - entry
            if sl_dist < self.cfg.min_sl_atr_mult * a or sl_dist > self.cfg.max_sl_atr_mult * a:
                sl = round(entry + a * self.cfg.atr_sl_mult + buf, p)
                logging.info(f"SL dist {sl_dist:.2f} outside [{self.cfg.min_sl_atr_mult}*A,{self.cfg.max_sl_atr_mult}*A] → reset to ATR*{self.cfg.atr_sl_mult}")
            else:
                sl = sl_raw

            if sl <= entry:
                raise ValueError(f"invalid SELL entry={entry} sl={sl}")
            return (entry, sl,
                    round(entry - a * self.cfg.atr_tp1_mult, p),
                    round(entry - a * self.cfg.atr_tp2_mult, p),
                    round(entry - a * self.cfg.atr_tp3_mult, p), entry_source)

    def compute_lot(self, entry_price, sl, conf_pct, journal):
        sl_d = abs(entry_price - sl)
        if sl_d < 1e-6:
            raise ValueError("SL too tight")

        closed = [j for j in journal if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
        if len(closed) >= 20:
            wins = sum(1 for j in closed if str(j["outcome"]).startswith("WIN"))
            wr = wins / len(closed)
            rrs = [abs(float(j.get("r_achieved") or 0)) for j in closed
                   if abs(float(j.get("r_achieved") or 0)) > 0]
            avg_rr = sum(rrs) / len(rrs) if rrs else 1.5
        else:
            wr, avg_rr = 0.50, 1.5

        wr = max(0.35, min(0.75, wr))
        avg_rr = max(1.0, min(3.0, avg_rr))
        kelly = max(0.0, (wr * avg_rr - (1 - wr)) / avg_rr)
        conf_f = max(0.3, min(1.0, conf_pct / 100.0))
        risk_pct = min((kelly / 2.0) * 100 * conf_f, self.cfg.max_risk_pct_per_trade)
        risk_pct = max(risk_pct, 0.1)

        risk_money = self.cfg.account_balance * risk_pct / 100.0
        loss_per_lot = self.cfg.contract_size * sl_d
        lot = max(self.cfg.min_lot, min(self.cfg.max_lot, round(risk_money / loss_per_lot, 2)))

        notional = lot * self.cfg.contract_size * entry_price
        if notional / self.cfg.account_balance > self.cfg.hard_leverage_cap:
            lot = max(self.cfg.min_lot, round(
                self.cfg.hard_leverage_cap * self.cfg.account_balance
                / (self.cfg.contract_size * entry_price), 2))
        return lot, f"wr={wr * 100:.0f}% risk={risk_pct:.1f}%"

    def journal_append(self, entry):
        j = list(self.kv.get("journal", []))
        j = [e for e in j if e.get("id") != entry.get("id")]
        j.append(entry)
        j.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return self.kv.set("journal", j[:self.cfg.max_journal])

    def journal_get(self):
        return (self.kv.get("journal") or [])[:self.cfg.max_journal]

    def update_outcomes(self, m5):
        """
        v16.6 FORENSIC FIX:
          - last_processed_bar_dt tracker → no re-processing (fixes B1, B4)
          - closed bars only
          - consecutive-loss-based pause (fixes B2)
          - clears pause on WIN
        """
        events = []
        now = datetime.now(timezone.utc)
        now_ts = pd.Timestamp.now(tz="UTC")
        journal = self.journal_get()
        changed_any = False
        bar_dur = pd.Timedelta(minutes=5)

        for entry in journal:
            if entry.get("outcome") not in ("PENDING",):
                continue
            anchor = entry.get("signal_bar_dt") or entry["timestamp"]
            try:
                ets = pd.Timestamp(anchor, tz="UTC")
            except Exception:
                continue
            if (now - ets.to_pydatetime()) > timedelta(hours=self.cfg.pending_max_age_hours):
                entry["outcome"] = "STALE"
                entry["r_achieved"] = 0.0
                events.append(("STALE", dict(entry)))
                changed_any = True
                continue

            bars_all = m5[m5["datetime"] > ets].sort_values("datetime")
            if bars_all.empty:
                continue
            bars_closed = bars_all[bars_all["datetime"] + bar_dur <= now_ts]
            if bars_closed.empty:
                continue

            side = entry["signal"]
            ep, sl = float(entry["entry"]), float(entry["sl"])
            tp1, tp2, tp3 = float(entry["tp1"]), float(entry["tp2"]), float(entry["tp3"])
            if ep <= 0 or sl <= 0 or abs(ep - sl) <= 0:
                continue

            filled = bool(entry.get("filled", False))
            h1_ = bool(entry.get("hit_tp1"))
            h2 = bool(entry.get("hit_tp2"))
            h3 = bool(entry.get("hit_tp3"))
            hs = bool(entry.get("hit_sl"))
            hb = bool(entry.get("hit_be"))

            # Determine bars not yet processed
            last_proc = entry.get("last_processed_bar_dt")
            if last_proc:
                try:
                    last_ets = pd.Timestamp(last_proc, tz="UTC")
                    new_bars = bars_closed[bars_closed["datetime"] > last_ets]
                except Exception:
                    new_bars = bars_closed
            else:
                new_bars = bars_closed

            # 1. FILL CHECK
            fill_idx = None
            if not filled:
                for i, (_, b) in enumerate(new_bars.iterrows()):
                    if side == "BUY" and b["low"] <= ep:
                        filled = True
                        fill_idx = i
                        entry["fill_bar_dt"] = b["datetime"].strftime("%Y-%m-%d %H:%M:%S")
                        break
                    if side == "SELL" and b["high"] >= ep:
                        filled = True
                        fill_idx = i
                        entry["fill_bar_dt"] = b["datetime"].strftime("%Y-%m-%d %H:%M:%S")
                        break

                # Chase invalidation
                if not filled and not new_bars.empty:
                    chase = self.cfg.entry_max_chase_atr * abs(ep - sl)
                    if side == "BUY" and float(new_bars["high"].iloc[-1]) > ep + chase:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0
                        entry["r_achieved"] = 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        changed_any = True
                        continue
                    if side == "SELL" and float(new_bars["low"].iloc[-1]) < ep - chase:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0
                        entry["r_achieved"] = 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        changed_any = True
                        continue

                entry["filled"] = int(filled)
                if not filled:
                    if not new_bars.empty:
                        entry["last_processed_bar_dt"] = new_bars.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")
                    changed_any = True
                    continue

            # 2. TP/SL CHECK (from fill bar onwards on new bars)
            bars_to_check = new_bars.iloc[fill_idx:] if fill_idx is not None else new_bars
            if bars_to_check.empty:
                if not new_bars.empty:
                    entry["last_processed_bar_dt"] = new_bars.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")
                    changed_any = True
                continue

            for _, b in bars_to_check.iterrows():
                if h3 or hs or hb:
                    break
                csl = ep if h1_ else sl
                if side == "BUY":
                    if b["low"] <= csl:
                        if h1_:
                            hb = True
                        else:
                            hs = True
                        continue
                    if not h1_ and b["high"] >= tp1:
                        h1_ = True
                    if not h2 and b["high"] >= tp2:
                        h2 = True
                    if not h3 and b["high"] >= tp3:
                        h3 = True
                else:
                    if b["high"] >= csl:
                        if h1_:
                            hb = True
                        else:
                            hs = True
                        continue
                    if not h1_ and b["low"] <= tp1:
                        h1_ = True
                    if not h2 and b["low"] <= tp2:
                        h2 = True
                    if not h3 and b["low"] <= tp3:
                        h3 = True

            entry["hit_tp1"], entry["hit_tp2"], entry["hit_tp3"] = int(h1_), int(h2), int(h3)
            entry["hit_sl"], entry["hit_be"] = int(hs), int(hb)
            entry["last_processed_bar_dt"] = new_bars.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")

            if not (h3 or hs or hb):
                changed_any = True
                continue

            # Finalize outcome
            if h3:
                entry["outcome"] = "WIN_TP3"
            elif h2 or hb:
                entry["outcome"] = "WIN_BE"
            elif hs:
                entry["outcome"] = "LOSS"

            d = abs(ep - sl)
            r1, r2, r3 = abs(tp1 - ep) / d, abs(tp2 - ep) / d, abs(tp3 - ep) / d
            if h3:
                entry["r_achieved"] = round((r1 + r2 + r3) / 3.0, 3)
            elif h2:
                entry["r_achieved"] = round((r1 + r2) / 3.0, 3)
            elif h1_ and hb:
                entry["r_achieved"] = round(r1 / 3.0, 3)
            elif hb:
                entry["r_achieved"] = 0.0
            else:
                entry["r_achieved"] = -1.0

            # v16.6 B2: drawdown pause only on consecutive threshold; clear on WIN
            if entry["outcome"] == "LOSS":
                closed_sorted = sorted(
                    [j for j in journal if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")],
                    key=lambda x: x.get("timestamp", ""), reverse=True
                )
                consec = 0
                for j in closed_sorted:
                    if j["outcome"] == "LOSS":
                        consec += 1
                    else:
                        break
                if consec >= self.cfg.max_consecutive_losses:
                    self.kv.set("last_loss_ts", now.isoformat())
                    logging.warning(f"Consecutive losses = {consec} → drawdown pause {self.cfg.dd_pause_hours}h")
            else:
                # WIN → clear pause
                self.kv.set("last_loss_ts", "")

            events.append(("CLOSE", dict(entry)))
            changed_any = True

        if changed_any:
            self.kv.set("journal", journal)
        return events

    def _today_signals(self, journal):
        today = datetime.now(timezone.utc).date().isoformat()
        return [j for j in journal if j.get("timestamp", "").startswith(today)]

    def _daily_loss_pct(self, journal, balance):
        today = datetime.now(timezone.utc).date().isoformat()
        total = 0.0
        for j in journal:
            if not j.get("timestamp", "").startswith(today):
                continue
            if j.get("outcome") == "LOSS":
                total += (abs(float(j.get("r_achieved", -1.0)))
                          * float(j.get("risk_pct", 2.0)) / 100.0 * balance)
        return (total / balance * 100) if balance > 0 else 0.0

    def _consecutive_losses(self, journal):
        closed = [j for j in journal if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
        closed.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        n = 0
        for j in closed:
            if j["outcome"] == "LOSS":
                n += 1
            else:
                break
        return n

    def _in_drawdown_pause(self):
        last_loss_ts = self.kv.get("last_loss_ts", "")
        if not last_loss_ts:
            return False
        try:
            llt = datetime.fromisoformat(last_loss_ts)
            if llt.tzinfo is None:
                llt = llt.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - llt) < timedelta(hours=self.cfg.dd_pause_hours)
        except Exception:
            return False

    def check_circuit_breakers(self, news_filter: NewsSafetyFilter):
        journal = self.journal_get()
        today = self._today_signals(journal)
        if len(today) >= self.cfg.max_daily_signals:
            return False, f"MAX_DAILY_SIGNALS ({len(today)})"
        dl = self._daily_loss_pct(journal, self.cfg.account_balance)
        if dl >= self.cfg.max_daily_loss_pct:
            return False, f"MAX_DAILY_LOSS ({dl:.1f}%)"
        cl = self._consecutive_losses(journal)
        if cl >= self.cfg.max_consecutive_losses:
            return False, f"MAX_CONSEC_LOSSES ({cl})"
        if self._in_drawdown_pause():
            return False, "DRAWDOWN_PAUSE"
        bo, title = news_filter.is_news_blackout()
        if bo:
            return False, f"NEWS_BLACKOUT ({title})"
        return True, ""


class TelegramNotifier:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def send(self, msg: str) -> bool:
        if self.cfg.dry_run:
            msg = "🧪 <b>DRY RUN</b>\n" + msg
        tok = os.getenv("TELEGRAM_TOKEN", "").strip()
        chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not tok or not chat:
            if self.cfg.dry_run:
                logging.info(f"[DRY TG] {redact(msg)[:200]}")
                return True
            return False

        hsh = hashlib.sha256(msg.encode()).hexdigest()[:16]
        dedup = self.kv.get("tg_sent_hashes", {}) or {}
        now = datetime.now(timezone.utc)
        clean = {}
        for k, v in dedup.items():
            try:
                if (now - datetime.fromisoformat(v)).total_seconds() < 3600:
                    clean[k] = v
            except Exception:
                pass
        dedup = clean
        if hsh in dedup:
            return True

        try:
            r = _request_with_retry(
                self.cfg, "POST",
                f"https://api.telegram.org/bot{tok}/sendMessage",
                json={"chat_id": chat, "text": msg, "parse_mode": "HTML"})
            if r.json().get("ok"):
                dedup[hsh] = now.isoformat()
                self.kv.set("tg_sent_hashes", dedup)
                return True
        except Exception as e:
            logging.error(f"TG: {redact(e)}")
        return False


class TradingBotOrchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.kv = KVStore(cfg)
        self.data = MarketDataHandler(cfg)
        self.signal = SignalGenerator(cfg)
        self.risk = RiskManagementEngine(cfg, self.kv)
        self.news = NewsSafetyFilter(cfg, self.kv)
        self.tg = TelegramNotifier(cfg, self.kv)

    def run_once(self):
        try:
            self.kv.force_load()
        except StateLoadError as e:
            logging.error(f"ABORT state load: {e}")
            return

        now = datetime.now(timezone.utc)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        if os.getenv("APEX_KILL", "").lower() in ("1", "true", "yes"):
            logging.info("APEX_KILL engaged")
            return
        if now.weekday() >= 5:
            logging.info("Weekend")
            return

        ld = now.astimezone(LONDON)
        ny = now.astimezone(NY)
        lh = ld.hour + ld.minute / 60
        nh = ny.hour + ny.minute / 60
        if not (7.0 <= lh < 10.0 or 8.0 <= nh < 11.0 or 13.0 <= nh < 16.0):
            logging.info("Outside killzone")
            return

        ok, reason = self.risk.check_circuit_breakers(self.news)
        if not ok:
            logging.info(f"CB engaged: {reason}")
            return

        st = self.kv.get("state", {}) or {}
        try:
            dfs, src, offset = self.data.fetch_all_intervals()
            m5, m15, h1, h4 = dfs["5min"], dfs["15min"], dfs["1h"], dfs["4h"]
        except Exception as e:
            logging.error(f"Data: {redact(e)}")
            return

        try:
            ev = self.risk.update_outcomes(m5)
            for kind, entry in ev:
                if kind == "CLOSE":
                    msg = (f"<b>📕 Trade Closed</b> {entry['signal']} "
                           f"{entry['symbol']} @ {entry['entry']} → {entry['outcome']} "
                           f"({entry.get('r_achieved', 0):+.2f}R)")
                    self.tg.send(msg)
                elif kind in ("INVALIDATED", "STALE"):
                    self.tg.send(f"<b>⚪ {kind}</b> {entry['signal']} @ {entry['entry']}")
        except Exception as e:
            logging.warning(f"Outcome: {e}")

        m5c = m5.iloc[:-1].reset_index(drop=True)
        m15c = m15.iloc[:-1].reset_index(drop=True)
        h1c = h1.iloc[:-1].reset_index(drop=True)
        h4c = h4.iloc[:-1].reset_index(drop=True)
        if len(m5c) < self.cfg.min_bars_m5 or len(h1c) < self.cfg.min_bars_h1:
            logging.warning("Post-trim insufficient bars")
            return
        price = float(m5c["close"].iloc[-1])
        signal_bar_iso = m5c.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")

        # v16.6 B5: block on ANY pending
        journal = self.risk.journal_get()
        if any(j.get("outcome") == "PENDING" for j in journal):
            logging.info("Pending trade exists — skipping new signal")
            return

        pres = self.signal.run_parliament(m5c, m15c, h1c, h4c)
        logging.info(
            f"Parliament {pres.direction} BUY={pres.buy_ratio * 100:.0f}% "
            f"SELL={pres.sell_ratio * 100:.0f}% verdict={pres.verdict} skip={pres.skip_reason}")

        should_fire = pres.direction != "NEUTRAL" and pres.verdict != "CANCEL"

        if should_fire:
            lt = st.get("last_signal_type", "NEUTRAL")
            lts = st.get("last_signal_ts", "")
            cool = self.cfg.opposite_cooldown_minutes if lt != pres.direction else self.cfg.cooldown_minutes
            if lts:
                try:
                    last = datetime.strptime(lts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    if (now - last) < timedelta(minutes=cool):
                        should_fire = False
                except Exception:
                    pass

        entry = sl = tp1 = tp2 = tp3 = 0.0
        entry_source = ""
        lot = self.cfg.min_lot
        note = ""
        ob = None

        if should_fire:
            ob = self.signal.detect_order_block(m5c, pres.direction, lookback=40)
            try:
                result = self.risk.compute_levels(m5c, pres.direction, ob)
                entry, sl, tp1, tp2, tp3 = result[:5]
                entry_source = result[5] if len(result) > 5 else "?"
            except ValueError as e:
                logging.warning(f"Levels: {e}")
                should_fire = False

        if should_fire:
            conf_pct = max(pres.buy_ratio, pres.sell_ratio) * 100
            try:
                lot, note = self.risk.compute_lot(entry, sl, conf_pct, journal)
            except ValueError as e:
                logging.warning(f"Lot: {e}")
                should_fire = False

        if should_fire:
            if pres.verdict == "EXECUTE_HALF":
                lot = max(self.cfg.min_lot, round(lot / 2.0, 2))
            elif pres.verdict == "EXECUTE_MICRO":
                lot = max(self.cfg.min_lot, round(lot / 4.0, 2))
            if pres.regime.get("counter_trend"):
                lot = max(self.cfg.min_lot, round(lot / self.cfg.counter_trend_lot_divisor, 2))

            mixed_warning = ""
            if pres.regime.get("htf_mixed") and not pres.regime.get("counter_trend"):
                mixed_warning = (
                    f"⚠️ <b>HTF MIXED</b> — H1={pres.regime.get('htf_h1')} "
                    f"H4={pres.regime.get('htf_h4')} "
                    f"(score={pres.regime.get('htf_score', 0):+d})\n\n"
                )
            ct_warning = ""
            if pres.regime.get("counter_trend"):
                ct_warning = (
                    f"⚠️ <b>COUNTER-TREND</b> — H1={pres.regime.get('htf_h1')} "
                    f"H4={pres.regime.get('htf_h4')} "
                    f"(score={pres.regime.get('htf_score', 0):+d})\n"
                    f"<i>Lot ÷{self.cfg.counter_trend_lot_divisor:.0f}×</i>\n\n"
                )

            msg = (
                f"<b>🏛️ APEX v{VERSION} — {pres.direction} (LIMIT)</b>\n\n"
                f"<b>{self.cfg.symbol}</b> limit @ <code>{entry:.3f}</code> [{entry_source}]\n"
                f"SL: <code>{sl:.3f}</code> | TP1: <code>{tp1:.3f}</code>\n"
                f"TP2: <code>{tp2:.3f}</code> | TP3: <code>{tp3:.3f}</code>\n"
                f"Lot: <code>{lot}</code> ({pres.verdict}, {note})\n"
                f"Voting: BUY={pres.buy_ratio * 100:.0f}% SELL={pres.sell_ratio * 100:.0f}%\n"
                f"Regime: {pres.regime.get('regime', '?')} | ADX: {pres.adx:.1f}\n"
                f"HTF: H1={pres.regime.get('htf_h1', '?')} H4={pres.regime.get('htf_h4', '?')} "
                f"(score={pres.regime.get('htf_score', 0):+d})\n"
                f"OB: "
                f"{'✅ ' + str(round(ob['high'], 2)) + '–' + str(round(ob['low'], 2)) if ob else '⚠️ none'}\n\n"
                f"{mixed_warning}"
                f"{ct_warning}"
                f"<i>Pasang limit order di MT5.</i>"
            )
            jid = f"{now_str.replace(' ', 'T')}_{pres.direction}_{lot}"
            entry_data = {
                "id": jid, "timestamp": now_str, "signal_bar_dt": signal_bar_iso,
                "symbol": self.cfg.symbol, "signal": pres.direction,
                "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "lot": lot, "verdict": pres.verdict, "outcome": "PENDING",
                "filled": 0, "r_achieved": 0.0, "last_processed_bar_dt": "",
                "entry_source": entry_source,
                "risk_pct": float(note.split("risk=")[1].rstrip("%")) if "risk=" in note else 2.0,
                "counter_trend": bool(pres.regime.get("counter_trend")),
                "htf_mixed": bool(pres.regime.get("htf_mixed")),
                "htf_score": pres.regime.get("htf_score", 0),
            }
            if self.risk.journal_append(entry_data):
                self.tg.send(msg)
                st["last_signal_type"] = pres.direction
                st["last_signal_ts"] = now_str

        st["run_count"] = int(st.get("run_count", 0)) + 1
        st["last_price"] = price
        st["last_price_ts"] = now_str
        self.kv.set("state", st)

    def smoke(self):
        dfs, src, offset = self.data.fetch_all_intervals()
        m5c = dfs["5min"].iloc[:-1].reset_index(drop=True)
        m15c = dfs["15min"].iloc[:-1].reset_index(drop=True)
        h1c = dfs["1h"].iloc[:-1].reset_index(drop=True)
        h4c = dfs["4h"].iloc[:-1].reset_index(drop=True)

        print(f"Source: {src} | Offset: {offset:+.3f}")
        print(f"Bars: M5={len(m5c)} M15={len(m15c)} H1={len(h1c)} H4={len(h4c)}")
        print(f"Last M5: {m5c.iloc[-1]['datetime']} close={m5c.iloc[-1]['close']:.2f}\n")

        pres = self.signal.run_parliament(m5c, m15c, h1c, h4c)
        print("=== PARLIAMENT ===")
        for v in pres.votes_detail:
            print(f"  {v['model']:16s} {v['vote']:8s} w={v['weight']} "
                  f"conf={v['conf']} score={v['score']}")
        print(f"\nBUY: {pres.buy_score} ({pres.buy_ratio * 100:.1f}%) | "
              f"SELL: {pres.sell_score} ({pres.sell_ratio * 100:.1f}%)")
        print(f"Active: {pres.active_models}/20 | Direction: {pres.direction} | "
              f"Verdict: {pres.verdict}")
        print(f"ADX: {pres.adx:.1f} | Regime: {pres.regime.get('regime', '?')} | "
              f"ADX_override: {pres.regime.get('adx_override', False)}")
        print(f"HTF: H1={pres.regime.get('htf_h1', '?')} | H4={pres.regime.get('htf_h4', '?')} | "
              f"score={pres.regime.get('htf_score', 0):+d} | "
              f"CT: {pres.regime.get('counter_trend', False)} | "
              f"Mixed: {pres.regime.get('htf_mixed', False)}")
        if pres.skip_reason:
            print(f"Skip: {pres.skip_reason}")

        if pres.direction != "NEUTRAL":
            ob = self.signal.detect_order_block(m5c, pres.direction)
            try:
                result = self.risk.compute_levels(m5c, pres.direction, ob)
                e, s, t1, t2, t3 = result[:5]
                entry_src = result[5] if len(result) > 5 else "?"
                cur = m5c.iloc[-1]['close']
                a = atr(m5c, 14)
                dist_atr = abs(cur - e) / a if a > 0 else 0
                sl_dist_atr = abs(e - s) / a if a > 0 else 0
                ob_info = (f"OB found @ {ob['low']:.2f}–{ob['high']:.2f}" if ob else "no OB")
                print(f"\n{ob_info}")
                print(f"Entry source: {entry_src}")
                print(f"Proposed {pres.direction}: entry={e} sl={s} tp1={t1} tp2={t2} tp3={t3}")
                print(f"  Current price: {cur:.3f} | Entry dist: {dist_atr:.2f} ATR | "
                      f"SL dist: {sl_dist_atr:.2f} ATR ({abs(e - s):.3f})")
                print(f"  {'✅ BUY limit below market' if pres.direction == 'BUY' else '✅ SELL limit above market'}")
            except ValueError as ex:
                print(f"  ⚠️ Level error: {ex}")

    def selftest(self):
        errors = []
        if self.cfg.max_lot < self.cfg.min_lot:
            errors.append("max_lot < min_lot")
        if not (self.cfg.parliament_min_ratio < self.cfg.parliament_half_ratio < self.cfg.parliament_full_ratio):
            errors.append("ratio ordering")
        if self.cfg.account_balance <= 0:
            errors.append("account_balance")
        if self.cfg.max_risk_pct_per_trade <= 0 or self.cfg.max_risk_pct_per_trade > 10:
            errors.append("max_risk_pct_per_trade out of safe range")
        if self.cfg.hard_leverage_cap > 200:
            errors.append("hard_leverage_cap > 200")
        if self.cfg.max_daily_loss_pct > 10:
            errors.append("max_daily_loss_pct > 10")
        if not HAS_SKLEARN:
            errors.append("sklearn missing (ML models disabled)")
        if self.cfg.htf_h1_weight < 1:
            errors.append("htf_h1_weight < 1")
        if self.cfg.htf_h4_weight < 0:
            errors.append("htf_h4_weight < 0")
        if self.cfg.ob_max_distance_atr <= 0:
            errors.append("ob_max_distance_atr <= 0")
        if not (0 < self.cfg.min_sl_atr_mult < self.cfg.max_sl_atr_mult):
            errors.append("SL range invalid")
        if errors:
            print("❌ SELFTEST FAILED:", "; ".join(errors))
            sys.exit(1)
        print(f"✅ SELFTEST PASSED (v{VERSION})")

    def test_tg(self):
        ok = self.tg.send(f"<b>🔔 TEST v{VERSION}</b>")
        print("SENT" if ok else "FAILED")
        return ok


if __name__ == "__main__":
    setup_logging()
    try:
        cfg = Config.from_env()
    except Exception as e:
        print(f"CONFIG ERROR: {e}")
        sys.exit(2)

    bot = TradingBotOrchestrator(cfg)
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "run":
            bot.run_once()
        elif cmd == "smoke":
            bot.smoke()
        elif cmd == "test-tg":
            bot.test_tg()
        elif cmd == "selftest":
            bot.selftest()
        elif cmd == "help":
            print("cmds: selftest|test-tg|smoke|run")
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(2)
        sys.exit(0)

    bot.run_once()
