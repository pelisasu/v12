"""
APEX PREDATOR v12.1 — BATTLE-READY
====================================
All P0/P1/P2 patches from v12.0 PLUS v12.1 fixes:
  - Selftest exit code correct (fail-fast in CI)
  - New commands: smoke, backtest, force-run, test-tg
  - Cron-optimized workflow
"""
from __future__ import annotations

import hashlib
import html
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

VERSION = "12.1"
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

    score_threshold: int = 8
    adx_min: float = 22.0
    adx_strong: float = 28.0
    cooldown_minutes: int = 45
    opposite_cooldown_minutes: int = 120
    max_daily_signals: int = 5

    council_execute_min: float = 70.0
    council_half_min: float = 55.0

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
    max_reports: int = 200

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
                logging.warning(f"APEX_{f.upper()}={v!r} invalid")
        cfg.gist_id = os.getenv("GIST_ID", cfg.gist_id).strip()
        cfg.gist_token = os.getenv("GIST_TOKEN", cfg.gist_token).strip()
        cfg.validate()
        return cfg

    def validate(self) -> None:
        errs = []
        if self.account_balance <= 0:
            errs.append("balance")
        if self.contract_size <= 0:
            errs.append("contract_size")
        if self.atr_sl_mult <= 0:
            errs.append("atr_sl_mult")
        if not (0 < self.atr_tp1_mult < self.atr_tp2_mult < self.atr_tp3_mult):
            errs.append("TP order")
        if not (0 < self.score_threshold <= 10):
            errs.append("score_threshold (0,10]")
        if not (0 < self.max_risk_pct_per_trade <= 5):
            errs.append("max_risk_pct (0,5]")
        if not (0 < self.hard_leverage_cap <= 200):
            errs.append("leverage_cap (0,200]")
        if not (0 < self.max_daily_loss_pct <= 10):
            errs.append("max_daily_loss_pct (0,10]")
        if self.min_lot <= 0 or self.max_lot < self.min_lot:
            errs.append("lot range")
        if errs:
            raise ValueError("CONFIG ERROR: " + ", ".join(errs))


# =========================================================
# LOGGING & UTIL
# =========================================================
def setup_logging(verbose: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter(f"%(asctime)s [%(levelname)s] [v{VERSION}] %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)


def redact(t: Any) -> str:
    s = str(t).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"(token|apikey|api_key)=[^&\s'\"]+", r"\1=***", s, flags=re.I)
    s = re.sub(r"/bot[^/\s'\"]+/", "/bot***/", s)
    s = re.sub(r"Bearer\s+\S+", "Bearer ***", s, flags=re.I)
    s = re.sub(r"(Authorization|x-api-key|Private-Token):\s*\S+",
               r"\1: ***", s, flags=re.I)
    return s


def h(s: Any) -> str:
    return html.escape(str(s), quote=False)


def round_half_away(x: float) -> int:
    return int(np.sign(x) * np.floor(np.abs(x) + 0.5))


class DataFetchError(Exception):
    pass


class StateLoadError(Exception):
    pass


# =========================================================
# KVStore — Gist + local fallback, FAIL-CLOSED
# =========================================================
class KVStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.use_gist = bool(cfg.gist_id and cfg.gist_token)
        self._cache: dict | None = None
        if not self.use_gist:
            logging.warning(
                "GIST_ID/GIST_TOKEN tidak diset — state lokal ephemeral. "
                "TIDAK persistent di GitHub Actions."
            )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.cfg.gist_token}",
            "Accept": "application/vnd.github+json",
        }

    def _load_gist(self) -> dict:
        if self._cache is not None:
            return self._cache
        try:
            r = requests.get(
                f"https://api.github.com/gists/{self.cfg.gist_id}",
                headers=self._headers(), timeout=self.cfg.http_timeout,
            )
            if r.status_code == 404:
                self._cache = {}
                return self._cache
            r.raise_for_status()
            files = r.json().get("files", {})
            fname = self.cfg.gist_state_file
            if fname in files:
                content = files[fname].get("content", "{}")
                self._cache = json.loads(content) if content.strip() else {}
            else:
                self._cache = {}
            return self._cache
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", 0)
            logging.error(f"Gist LOAD HTTP {code}: ABORT (fail-closed)")
            raise StateLoadError(f"Gist HTTP {code}") from e
        except Exception as e:
            logging.error(f"Gist LOAD fail: {redact(e)}")
            raise StateLoadError(f"Gist load fail: {e}") from e

    def _save_gist(self, data: dict) -> bool:
        try:
            body = {"files": {self.cfg.gist_state_file: {
                "content": json.dumps(data, default=str, sort_keys=True)
            }}}
            r = requests.patch(
                f"https://api.github.com/gists/{self.cfg.gist_id}",
                headers=self._headers(), json=body, timeout=self.cfg.http_timeout,
            )
            r.raise_for_status()
            self._cache = data
            return True
        except Exception as e:
            logging.error(f"Gist SAVE fail: {redact(e)}")
            return False

    def _load_local(self) -> dict:
        if self._cache is not None:
            return self._cache
        p = Path(self.cfg.local_state_file)
        self._cache = json.loads(p.read_text()) if p.exists() else {}
        return self._cache

    def _save_local(self, data: dict) -> bool:
        try:
            Path(self.cfg.local_state_file).write_text(
                json.dumps(data, default=str, sort_keys=True)
            )
            self._cache = data
            return True
        except Exception as e:
            logging.error(f"Local SAVE fail: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        data = self._load_gist() if self.use_gist else self._load_local()
        return data.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        data = dict(self._load_gist() if self.use_gist else self._load_local())
        data[key] = value
        return self._save_gist(data) if self.use_gist else self._save_local(data)

    def force_load(self) -> None:
        self._load_gist() if self.use_gist else self._load_local()


# =========================================================
# RETRY
# =========================================================
def retry_with_backoff(fn, attempts: int = 3, base: float = 1.0, max_delay: float = 8.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except requests.HTTPError as e:
            last = e
            code = getattr(e.response, "status_code", 0)
            if code in (429, 500, 502, 503, 504):
                delay = min(max_delay, base * (2 ** i))
                logging.warning(f"HTTP {code} → backoff {delay:.1f}s")
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(min(max_delay, base * (2 ** i)))
    raise last


# =========================================================
# DATA
# =========================================================
def validate_ohlcv(df: pd.DataFrame, interval: str, min_bars: int, cfg: Config) -> pd.DataFrame:
    if df is None or len(df) < min_bars:
        raise DataFetchError(f"Bars {0 if df is None else len(df)} < {min_bars}")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if (df["high"] < df["low"]).any():
        raise DataFetchError("high<low")
    if df["datetime"].duplicated().any():
        raise DataFetchError("dupes")
    if not df["datetime"].is_monotonic_increasing:
        raise DataFetchError("non-monotonic")
    isec = {"5min": 300, "15min": 900, "1h": 3600, "4h": 14400}[interval]
    last_ts = df["datetime"].iloc[-1]
    if (datetime.now(timezone.utc) - last_ts).total_seconds() > isec * cfg.max_stale_mult:
        raise DataFetchError("Stale")
    return df.reset_index(drop=True)


def fetch_twelvedata_batch(cfg: Config) -> pd.DataFrame:
    key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not key:
        raise DataFetchError("TwelveData: no key")

    def _do():
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": "XAU/USD", "interval": "5min",
                "outputsize": cfg.m5_fetch_size, "apikey": key,
                "timezone": "UTC",
            },
            timeout=cfg.http_timeout,
        )
        if r.status_code == 429:
            raise requests.HTTPError(response=r)
        r.raise_for_status()
        d = r.json()
        if "values" not in d:
            raise DataFetchError(f"TwelveData: {d.get('message', '?')}")
        df = pd.DataFrame(d["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = 100.0
        df = df.sort_values("datetime").reset_index(drop=True)
        return df[["datetime", "open", "high", "low", "close", "volume"]]

    return retry_with_backoff(_do, attempts=cfg.max_retry, base=1.0)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = (df.set_index("datetime")
           .resample(rule, label="left", closed="left")
           .agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"})
           .dropna().reset_index())
    return d.reset_index(drop=True)


def fetch_yahoo(cfg: Config, interval_str: str, count: int = 200) -> pd.DataFrame:
    imap = {"5min": "5m", "15min": "15m", "1h": "1h"}
    headers = {"User-Agent": f"ApexBot/{VERSION}"}

    def _do(symbol, iv, range_="5d"):
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"interval": iv, "range": range_},
            headers=headers, timeout=cfg.http_timeout,
        )
        r.raise_for_status()
        if "application/json" not in r.headers.get("content-type", ""):
            raise DataFetchError("non-JSON")
        res = (r.json().get("chart", {}).get("result") or [None])[0]
        if not res:
            raise DataFetchError("no result")
        ts = res.get("timestamp", [])
        q = res.get("indicators", {}).get("quote", [{}])[0]
        rows = []
        for i, t in enumerate(ts):
            try:
                if any(q.get(k, [None])[i] is None for k in ("open", "high", "low", "close")):
                    continue
                rows.append({
                    "epoch": int(t), "open": float(q["open"][i]),
                    "high": float(q["high"][i]), "low": float(q["low"][i]),
                    "close": float(q["close"][i]),
                    "volume": float((q.get("volume") or [100])[i] or 100),
                })
            except (IndexError, TypeError):
                continue
        if not rows:
            raise DataFetchError("0 rows")
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        return df[["datetime", "open", "high", "low", "close", "volume"]]

    def _run():
        if interval_str == "4h":
            df1h = _do("XAUUSD=X", "1h", range_="1mo").tail(count * 4 + 20)
            return resample_ohlc(df1h, "4h").tail(count).reset_index(drop=True)
        return _do("XAUUSD=X", imap[interval_str]).tail(count).reset_index(drop=True)

    return retry_with_backoff(_run, attempts=cfg.max_retry)


def fetch_all_intervals(cfg: Config) -> tuple[dict[str, pd.DataFrame], str]:
    errors = []
    try:
        m5 = fetch_twelvedata_batch(cfg)
        m5 = validate_ohlcv(m5, "5min", cfg.min_bars_m5, cfg)
        return {
            "5min": m5,
            "15min": resample_ohlc(m5, "15min"),
            "1h": resample_ohlc(m5, "1h"),
            "4h": resample_ohlc(m5, "4h"),
        }, "TwelveData"
    except Exception as e:
        errors.append(f"TwelveData:{redact(e)}")
        logging.warning(f"TwelveData batch fail: {redact(e)}; fallback Yahoo")

    try:
        out = {}
        for iv, cnt, mn in (
            ("5min", 200, cfg.min_bars_m5),
            ("15min", 150, cfg.min_bars_m15),
            ("1h", 200, cfg.min_bars_h1),
            ("4h", 100, cfg.min_bars_h4),
        ):
            d = fetch_yahoo(cfg, iv, cnt)
            out[iv] = validate_ohlcv(d, iv, mn, cfg)
        return out, "Yahoo"
    except Exception as e:
        errors.append(f"Yahoo:{redact(e)}")
        raise DataFetchError(" | ".join(errors))


# =========================================================
# INDICATORS
# =========================================================
def atr(df: pd.DataFrame, period: int = 14) -> float:
    h_, lo, c_ = df["high"], df["low"], df["close"]
    tr = pd.concat([h_ - lo, (h_ - c_.shift()).abs(),
                    (lo - c_.shift()).abs()], axis=1).max(axis=1)
    v = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return 0.0 if pd.isna(v) or v < 1e-6 else float(v)


def adx(df: pd.DataFrame, period: int = 14) -> float:
    try:
        up, dn = df["high"].diff(), -df["low"].diff()
        pdm = up.where((up > dn) & (up > 0), 0.0)
        mdm = dn.where((dn > up) & (dn > 0), 0.0)
        tr = pd.concat([df["high"] - df["low"],
                        (df["high"] - df["close"].shift()).abs(),
                        (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
        atr_s = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
        pdi = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
        mdi = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
        dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
        v = dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        return 0.0 if pd.isna(v) or v < 0 else float(v)
    except Exception:
        return 0.0


def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    v = (100 - 100 / (1 + gain / (loss + 1e-9))).iloc[-1]
    return 50.0 if pd.isna(v) else float(v)


def hurst_exponent(close: pd.Series, max_lag: int = 20) -> float:
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
        slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


def shannon_entropy(close: pd.Series, bins: int = 10) -> float:
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


def heikin_ashi_trend(df: pd.DataFrame, lookback: int = 5) -> str:
    d = df.tail(max(lookback + 5, 50)).reset_index(drop=True)
    if len(d) < lookback + 2:
        return "NEUTRAL"
    hac = (d["open"] + d["high"] + d["low"] + d["close"]) / 4.0
    hao = np.empty(len(d))
    hao[0] = (d["open"].iloc[0] + d["close"].iloc[0]) / 2.0
    for i in range(1, len(d)):
        hao[i] = (hao[i - 1] + hac.iloc[i - 1]) / 2.0
    tail = (hac - hao).tail(lookback)
    if int((tail > 0).sum()) >= lookback - 1:
        return "BULLISH"
    if int((tail < 0).sum()) >= lookback - 1:
        return "BEARISH"
    return "NEUTRAL"


def trend_ema(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
    if len(df) < slow + 5:
        return "NEUTRAL"
    c = df["close"]
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    slope = ef.diff(3).iloc[-1]
    if c.iloc[-1] > es.iloc[-1] and ef.iloc[-1] > es.iloc[-1] and slope > 0:
        return "BULLISH"
    if c.iloc[-1] < es.iloc[-1] and ef.iloc[-1] < es.iloc[-1] and slope < 0:
        return "BEARISH"
    return "NEUTRAL"


# =========================================================
# SMC
# =========================================================
def find_swing_points(df: pd.DataFrame, left: int = 2, right: int = 2):
    highs, lows = [], []
    for i in range(left, len(df) - right):
        win = df.iloc[i - left:i + right + 1]
        if df["high"].iloc[i] == win["high"].max():
            highs.append(i)
        if df["low"].iloc[i] == win["low"].min():
            lows.append(i)
    return highs, lows


def detect_choch(df: pd.DataFrame) -> dict:
    try:
        if len(df) < 30:
            return {"detected": False, "type": "NONE", "level": 0.0}
        hi, lo = find_swing_points(df)
        if len(hi) < 2 or len(lo) < 2:
            return {"detected": False, "type": "NONE", "level": 0.0}
        lc = df["close"].iloc[-1]
        rh, rl = sorted(hi)[-2:], sorted(lo)[-2:]
        ph, pl = float(df["high"].iloc[rh[0]]), float(df["low"].iloc[rl[0]])
        lsh, lsl = float(df["high"].iloc[rh[-1]]), float(df["low"].iloc[rl[-1]])
        if lsh > ph and lc < lsl:
            return {"detected": True, "type": "BEARISH_CHOCH", "level": lsl}
        if lsl < pl and lc > lsh:
            return {"detected": True, "type": "BULLISH_CHOCH", "level": lsh}
        return {"detected": False, "type": "NONE", "level": 0.0}
    except Exception:
        return {"detected": False, "type": "NONE", "level": 0.0}


def detect_order_block(df: pd.DataFrame, vol_mult: float = 1.3) -> dict:
    EMPTY = {"detected": False, "type": "NONE", "high": 0.0, "low": 0.0, "swept": False}
    try:
        if len(df) < 30:
            return EMPTY
        d = df.tail(60).reset_index(drop=True)
        a = atr(df, 14)
        if a <= 0:
            return EMPTY
        bodies = (d["close"] - d["open"]).abs()
        cands = bodies.iloc[3:len(d) - 1]
        if cands.empty:
            return EMPTY
        imp_idx = int(cands.idxmax())
        imp = d.iloc[imp_idx]
        if abs(imp["close"] - imp["open"]) < a * 1.2:
            return EMPTY
        vol_ma = d["volume"].rolling(20, min_periods=5).mean().iloc[imp_idx]
        if vol_ma > 0 and imp["volume"] < vol_ma * vol_mult:
            return EMPTY
        ob = d.iloc[imp_idx - 1]
        bull_imp = imp["close"] > imp["open"]
        bull_ob = ob["close"] > ob["open"]
        if bull_imp == bull_ob:
            return EMPTY
        ob_h = float(max(ob["open"], ob["close"]))
        ob_l = float(min(ob["open"], ob["close"]))
        pre = d.iloc[max(0, imp_idx - 20):imp_idx - 1]
        swept = False
        if len(pre) >= 5:
            sb = (bull_imp and not bull_ob
                  and ob["low"] < pre["low"].min() and ob["close"] > pre["low"].min()
                  and vol_ma > 0 and ob["volume"] >= vol_ma * 1.1)
            ss = (not bull_imp and bull_ob
                  and ob["high"] > pre["high"].max() and ob["close"] < pre["high"].max()
                  and vol_ma > 0 and ob["volume"] >= vol_ma * 1.1)
            swept = bool(sb or ss)
        lc = float(d["close"].iloc[-1])
        if bull_imp:
            if lc < ob_l:
                return EMPTY
            return {"detected": True, "type": "BULLISH_OB", "high": ob_h, "low": ob_l, "swept": swept}
        if lc > ob_h:
            return EMPTY
        return {"detected": True, "type": "BEARISH_OB", "high": ob_h, "low": ob_l, "swept": swept}
    except Exception:
        return EMPTY


def detect_fvg(df: pd.DataFrame) -> dict:
    try:
        if len(df) < 3:
            return {"detected": False, "type": "NONE", "level": 0.0}
        c1, c3 = df.iloc[-3], df.iloc[-1]
        if c3["low"] > c1["high"]:
            return {"detected": True, "type": "BULLISH_FVG", "level": float(c1["high"])}
        if c3["high"] < c1["low"]:
            return {"detected": True, "type": "BEARISH_FVG", "level": float(c1["low"])}
        return {"detected": False, "type": "NONE", "level": 0.0}
    except Exception:
        return {"detected": False, "type": "NONE", "level": 0.0}


# =========================================================
# SIGNAL
# =========================================================
@dataclass
class Signal:
    type: str = "NEUTRAL"
    score: int = 0
    adx: float = 0.0
    ha: str = "NEUTRAL"
    ob: dict = field(default_factory=lambda: {"detected": False, "type": "NONE", "high": 0, "low": 0, "swept": False})
    choch: dict = field(default_factory=lambda: {"detected": False, "type": "NONE", "level": 0})
    fvg: dict = field(default_factory=lambda: {"detected": False, "type": "NONE", "level": 0})
    conf: dict = field(default_factory=dict)
    regime: dict = field(default_factory=dict)
    skip_reason: str = ""


def compute_confluence(df_m5, trends, ob, choch, fvg, adx_val, ha) -> dict:
    c = df_m5["close"]
    e9 = c.ewm(span=9, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    rsi_v = rsi(c)
    tw = {"m5": 0.5, "m15": 1.0, "h1": 1.5, "h4": 2.0}

    def tv(t):
        return 1 if t == "BULLISH" else (-1 if t == "BEARISH" else 0)

    raw_trend = sum(tw[k] * tv(trends[k]) for k in tw)
    trend_norm = round_half_away(raw_trend * 4 / 5)

    m = 0
    if e9.iloc[-1] > e21.iloc[-1]:
        m += 1
    elif e9.iloc[-1] < e21.iloc[-1]:
        m -= 1
    if rsi_v > 55 and m > 0:
        m += 1
    elif rsi_v < 45 and m < 0:
        m -= 1
    elif rsi_v < 35 and m > 0:
        m += 1
    elif rsi_v > 65 and m < 0:
        m -= 1

    s = 0
    if ob["detected"]:
        base = 2 if ob["type"] == "BULLISH_OB" else -2
        if ob.get("swept"):
            base += 1 if base > 0 else -1
        s += base
    if choch["detected"]:
        s += 1 if choch["type"] == "BULLISH_CHOCH" else -1
    if not ob["detected"] and fvg["detected"]:
        s += 1 if fvg["type"] == "BULLISH_FVG" else -1

    hp = 1 if ha == "BULLISH" else (-1 if ha == "BEARISH" else 0)
    total = trend_norm + m + s + hp
    return {
        "total": total, "trend": trend_norm, "momentum": m,
        "smc": s, "ha": hp, "rsi": round(rsi_v, 1),
        "adx_gate": round(adx_val, 1),
    }


def generate_signal(cfg, m5, m15, h1, h4) -> Signal:
    sig = Signal()
    adx_val = adx(m5)
    sig.adx = adx_val
    if adx_val < cfg.adx_min:
        sig.skip_reason = f"ADX {adx_val:.1f} < {cfg.adx_min}"
        return sig

    ob = detect_order_block(m5)
    choch = detect_choch(m5)
    fvg = detect_fvg(m5)
    sig.ob, sig.choch, sig.fvg = ob, choch, fvg
    if not (ob["detected"] or choch["detected"] or fvg["detected"]):
        sig.skip_reason = "No SMC"
        return sig

    ha = heikin_ashi_trend(m5)
    sig.ha = ha
    trends = {
        "m5": trend_ema(m5), "m15": trend_ema(m15),
        "h1": trend_ema(h1), "h4": trend_ema(h4),
    }
    conf = compute_confluence(m5, trends, ob, choch, fvg, adx_val, ha)
    sig.conf = conf
    sig.score = conf["total"]

    H = hurst_exponent(h1["close"])
    ent = shannon_entropy(h1["close"])
    regime = ("TREND" if H > cfg.hurst_trend_min and ent < cfg.entropy_chaos_max
              else ("MEANREV" if H < cfg.hurst_meanrev_max else "CHAOS"))
    sig.regime = {"hurst": round(H, 3), "entropy": round(ent, 3), "regime": regime}
    if regime == "CHAOS":
        sig.skip_reason = f"Regime CHAOS (H={H:.2f}, E={ent:.2f})"
        return sig

    th = cfg.score_threshold
    if conf["total"] >= th:
        if ha != "BULLISH":
            sig.skip_reason = f"Score{conf['total']:+d} HA={ha}"
            return sig
        if ob["detected"] and ob["type"] != "BULLISH_OB":
            sig.skip_reason = f"OB conflict {ob['type']}"
            return sig
        sig.type = "BUY"
        return sig
    if conf["total"] <= -th:
        if ha != "BEARISH":
            sig.skip_reason = f"Score{conf['total']:+d} HA={ha}"
            return sig
        if ob["detected"] and ob["type"] != "BEARISH_OB":
            sig.skip_reason = f"OB conflict {ob['type']}"
            return sig
        sig.type = "SELL"
        return sig
    sig.skip_reason = f"Score {conf['total']:+d} < ±{th}"
    return sig


# =========================================================
# COUNCIL
# =========================================================
def run_council_vote(cfg, m5, m15, h1, h4, sig) -> tuple[str, float]:
    if sig.type not in ("BUY", "SELL"):
        return "CANCEL", 0.0

    votes, total_w = 0.0, 0.0
    h1t, h4t = trend_ema(h1), trend_ema(h4)
    desired = "BULLISH" if sig.type == "BUY" else "BEARISH"

    total_w += 1.5
    if sig.adx >= cfg.adx_strong and h1t == desired:
        votes += 1.5
    elif sig.adx >= cfg.adx_min and h1t == desired:
        votes += 0.75
    elif sig.adx >= cfg.adx_strong and h1t != desired and h1t != "NEUTRAL":
        votes -= 1.0

    total_w += 2.0
    if h1t == desired and h4t == desired:
        votes += 2.0
    elif h1t == desired or h4t == desired:
        votes += 1.0
    elif (h1t != "NEUTRAL" and h1t != desired) or (h4t != "NEUTRAL" and h4t != desired):
        votes -= 1.0

    total_w += 1.5
    if sig.ob["detected"] and sig.ob.get("swept"):
        votes += 1.5
    elif sig.ob["detected"]:
        votes += 1.0
    desired_choch = "BULLISH_CHOCH" if sig.type == "BUY" else "BEARISH_CHOCH"
    if sig.choch["detected"] and sig.choch["type"] == desired_choch:
        votes += 0.75

    total_w += 1.0
    votes += min(abs(sig.score) / 10.0, 1.0)

    total_w += 1.0
    if sig.ha == desired:
        votes += 1.0

    pct = max(0.0, min(100.0, (votes / total_w) * 100 if total_w else 0.0))
    if pct >= cfg.council_execute_min:
        return "EXECUTE", pct
    if pct >= cfg.council_half_min:
        return "EXECUTE_HALF", pct
    return "CANCEL", pct


# =========================================================
# RISK
# =========================================================
def compute_levels(cfg, m5, side: str) -> tuple[float, float, float, float]:
    a = atr(m5, 14)
    if a <= 0:
        raise ValueError("ATR=0")
    price = float(m5["close"].iloc[-1])
    buf = cfg.spread_buffer * 2
    sl_d = max(a * cfg.atr_sl_mult, buf)
    tp1_d = max(a * cfg.atr_tp1_mult, buf)
    tp2_d = max(a * cfg.atr_tp2_mult, buf * 1.5)
    tp3_d = max(a * cfg.atr_tp3_mult, buf * 2)
    p = cfg.price_decimals
    if side == "BUY":
        return (round(price - sl_d, p), round(price + tp1_d, p),
                round(price + tp2_d, p), round(price + tp3_d, p))
    return (round(price + sl_d, p), round(price - tp1_d, p),
            round(price - tp2_d, p), round(price - tp3_d, p))


def compute_lot(cfg, price: float, sl: float, conf_pct: float, journal: list) -> tuple[float, float, str]:
    sl_d = abs(price - sl)
    if sl_d < 1e-6:
        raise ValueError("SL too tight")

    closed = [j for j in journal if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
    if len(closed) >= 20:
        wins = sum(1 for j in closed if str(j["outcome"]).startswith("WIN"))
        wr = wins / len(closed)
        rrs = [abs(float(j.get("r_achieved") or 0)) for j in closed
               if abs(float(j.get("r_achieved") or 0)) > 0]
        avg_rr = sum(rrs) / len(rrs) if rrs else 1.5
        source = f"journal({len(closed)})"
    else:
        wr, avg_rr = 0.55, 1.5
        source = f"bootstrap({len(closed)})"

    wr = max(0.35, min(0.75, wr))
    avg_rr = max(1.0, min(3.0, avg_rr))
    kelly = max(0.0, (wr * avg_rr - (1 - wr)) / avg_rr)
    conf_f = max(0.3, min(1.0, conf_pct / 100.0))
    risk_pct = min((kelly / 2.0) * 100 * conf_f, cfg.max_risk_pct_per_trade)
    risk_pct = max(risk_pct, 0.1)

    risk_money = cfg.account_balance * risk_pct / 100.0
    loss_per_lot = cfg.contract_size * sl_d
    raw_lot = risk_money / loss_per_lot
    lot = max(cfg.min_lot, min(cfg.max_lot, round(raw_lot, 2)))

    notional = lot * cfg.contract_size * price
    leverage = notional / cfg.account_balance
    if leverage > cfg.hard_leverage_cap:
        lot = round(cfg.hard_leverage_cap * cfg.account_balance / (cfg.contract_size * price), 2)
        lot = max(cfg.min_lot, lot)
        leverage = (lot * cfg.contract_size * price) / cfg.account_balance

    actual_risk = (lot * loss_per_lot / cfg.account_balance) * 100
    if actual_risk > cfg.max_risk_pct_per_trade * 1.01:
        lot = round((cfg.max_risk_pct_per_trade / 100 * cfg.account_balance) / loss_per_lot, 2)
        lot = max(cfg.min_lot, lot)
        actual_risk = (lot * loss_per_lot / cfg.account_balance) * 100
        if actual_risk > cfg.max_risk_pct_per_trade:
            raise ValueError(
                f"min_lot exceeds cap ({actual_risk:.2f}%>{cfg.max_risk_pct_per_trade}%)"
            )
    note = (f"Kelly[{source}] wr={wr*100:.1f}% rr={avg_rr:.2f} "
            f"risk={risk_pct:.2f}% lev={leverage:.1f}x")
    return lot, risk_pct, note


# =========================================================
# TELEGRAM
# =========================================================
def _tg_hash(msg: str) -> str:
    return hashlib.sha256(msg.encode()).hexdigest()[:16]


def tg_send(cfg: Config, msg: str, kv: KVStore) -> bool:
    if cfg.dry_run:
        msg = "🧪 <b>DRY RUN</b>\n" + msg

    tok = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not tok or not chat:
        if cfg.dry_run:
            logging.info(f"[DRY TG] {redact(msg)[:200]}")
            return True
        logging.warning("Telegram creds missing")
        return False

    hsh = _tg_hash(msg)
    dedup = kv.get("tg_sent_hashes", {})
    now = datetime.now(timezone.utc)
    dedup = {k: v for k, v in dedup.items()
             if (now - datetime.fromisoformat(v)).total_seconds() < 3600}
    if hsh in dedup:
        logging.info(f"TG dedup skip: {hsh}")
        return True

    def _do():
        r = requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
            timeout=cfg.http_timeout,
        )
        if r.status_code == 429:
            raise requests.HTTPError(response=r)
        r.raise_for_status()
        d = r.json()
        if not d.get("ok"):
            raise Exception(f"TG: {d.get('description', '?')}")
        return True

    try:
        ok = retry_with_backoff(_do, attempts=cfg.max_retry, base=1.0)
        if ok:
            dedup[hsh] = now.isoformat()
            kv.set("tg_sent_hashes", dedup)
        return ok
    except Exception as e:
        logging.error(f"TG fail: {redact(e)}")
        return False


def tg_alert_error(cfg: Config, msg: str, kv: KVStore) -> None:
    now = datetime.now(timezone.utc)
    last = kv.get("alert_limit", {}).get("last")
    if last:
        try:
            if (now - datetime.fromisoformat(last)).total_seconds() < 300:
                return
        except Exception:
            pass
    if tg_send(cfg, f"<b>⚠️ APEX v{VERSION} ERROR</b>\n<code>{h(redact(msg))}</code>", kv):
        kv.set("alert_limit", {"last": now.isoformat()})


# =========================================================
# KILL SWITCH / CIRCUIT / DRAWDOWN
# =========================================================
def kill_switch_active(kv: KVStore) -> tuple[bool, str]:
    if os.getenv("APEX_KILL", "").lower() in ("1", "true", "yes"):
        return True, "APEX_KILL env"
    ks = kv.get("kill_switch", {})
    if ks.get("active"):
        return True, f"kv: {ks.get('reason', '?')}"
    return False, ""


def circuit_allow(kv: KVStore) -> bool:
    c = kv.get("circuit", {"failures": 0, "opened_until": None})
    if c.get("opened_until"):
        try:
            if datetime.now(timezone.utc) < datetime.fromisoformat(c["opened_until"]):
                return False
        except Exception:
            pass
    return True


def circuit_fail(kv: KVStore, cooldown_min: int = 15) -> None:
    c = kv.get("circuit", {"failures": 0, "opened_until": None})
    f = int(c.get("failures", 0)) + 1
    opened = c.get("opened_until")
    if f >= 3:
        opened = (datetime.now(timezone.utc) + timedelta(minutes=cooldown_min)).isoformat()
        f = 0
        logging.warning(f"Circuit OPEN {cooldown_min}m")
    kv.set("circuit", {"failures": f, "opened_until": opened})


def circuit_ok(kv: KVStore) -> None:
    kv.set("circuit", {"failures": 0, "opened_until": None})


def drawdown_check(cfg: Config, kv: KVStore, journal: list) -> tuple[bool, str]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_closed = [j for j in journal
                    if str(j.get("timestamp", "")).startswith(today)
                    and j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
    day_r = sum(float(j.get("r_achieved", 0) or 0) for j in today_closed)
    day_loss_pct = abs(day_r) * cfg.max_risk_pct_per_trade if day_r < 0 else 0
    if day_loss_pct >= cfg.max_daily_loss_pct:
        return True, f"Daily loss {day_r:.2f}R ≥ cap {cfg.max_daily_loss_pct}%"

    consec = 0
    for j in journal:
        if j.get("outcome") == "LOSS":
            consec += 1
        elif j.get("outcome") in ("WIN_TP3", "WIN_BE"):
            break
    if consec >= cfg.max_consecutive_losses:
        return True, f"{consec} consec losses ≥ {cfg.max_consecutive_losses}"

    p = kv.get("drawdown_pause", {})
    if p.get("until"):
        try:
            if datetime.now(timezone.utc) < datetime.fromisoformat(p["until"]):
                return True, f"pause until {p['until']}"
        except Exception:
            pass
    return False, ""


# =========================================================
# JOURNAL
# =========================================================
def journal_append(kv: KVStore, entry: dict, max_journal: int = 500) -> bool:
    j = list(kv.get("journal", []))
    j = [e for e in j if e.get("id") != entry.get("id")]
    j.append(entry)
    j.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    if not kv.set("journal", j[:max_journal]):
        logging.error(f"Journal SAVE failed id={entry.get('id')}")
        return False
    return True


def journal_get(kv: KVStore, limit: int = 500) -> list:
    return kv.get("journal", [])[:limit]


def calc_r_mixed(entry, sl, tp1, tp2, tp3, h1, h2, h3, be):
    d = abs(entry - sl)
    if d <= 0:
        return 0.0
    r1, r2, r3 = abs(tp1 - entry) / d, abs(tp2 - entry) / d, abs(tp3 - entry) / d
    if h3:
        return round((r1 + r2 + r3) / 3.0, 3)
    if h2:
        return round((r1 + r2) / 3.0, 3)
    if h1 and be:
        return round(r1 / 3.0, 3)
    if be:
        return 0.0
    return -1.0


def update_outcomes(cfg: Config, kv: KVStore, m5: pd.DataFrame) -> list:
    events = []
    now = datetime.now(timezone.utc)
    journal = journal_get(kv, cfg.max_journal)
    changed_any = False

    for entry in journal:
        if entry.get("outcome") != "PENDING":
            continue
        anchor = entry.get("signal_bar_dt") or entry["timestamp"]
        try:
            ets = pd.Timestamp(anchor, tz="UTC")
        except Exception:
            continue

        if (now - ets.to_pydatetime()) > timedelta(hours=cfg.pending_max_age_hours):
            entry["outcome"] = "STALE"
            entry["r_achieved"] = 0.0
            events.append(("STALE", dict(entry)))
            changed_any = True
            continue

        bars = m5[m5["datetime"] > ets].sort_values("datetime")
        if bars.empty:
            continue

        side = entry["signal"]
        ep, sl = float(entry["entry"]), float(entry["sl"])
        tp1, tp2, tp3 = float(entry["tp1"]), float(entry["tp2"]), float(entry["tp3"])
        if ep <= 0 or sl <= 0 or abs(ep - sl) <= 0:
            continue

        hit1 = bool(entry.get("hit_tp1"))
        hit2 = bool(entry.get("hit_tp2"))
        hit3 = bool(entry.get("hit_tp3"))
        hsl = bool(entry.get("hit_sl"))
        hbe = bool(entry.get("hit_be"))
        was_outcome = entry.get("outcome")
        was_hit1 = hit1
        changed = False

        for _, b in bars.iterrows():
            if hit3 or hsl or hbe:
                break
            csl = ep if hit1 else sl
            if side == "BUY":
                if b["low"] <= csl:
                    if hit1:
                        hbe = True
                    else:
                        hsl = True
                    changed = True
                    continue
                if not hit1 and b["high"] >= tp1:
                    hit1 = True
                    changed = True
                if not hit2 and b["high"] >= tp2:
                    hit2 = True
                    changed = True
                if not hit3 and b["high"] >= tp3:
                    hit3 = True
                    changed = True
            else:
                if b["high"] >= csl:
                    if hit1:
                        hbe = True
                    else:
                        hsl = True
                    changed = True
                    continue
                if not hit1 and b["low"] <= tp1:
                    hit1 = True
                    changed = True
                if not hit2 and b["low"] <= tp2:
                    hit2 = True
                    changed = True
                if not hit3 and b["low"] <= tp3:
                    hit3 = True
                    changed = True

        entry["hit_tp1"] = int(hit1)
        entry["hit_tp2"] = int(hit2)
        entry["hit_tp3"] = int(hit3)
        entry["hit_sl"] = int(hsl)
        entry["hit_be"] = int(hbe)

        if was_outcome == "PENDING":
            if hit3:
                entry["outcome"] = "WIN_TP3"
            elif hit2 or hbe:
                entry["outcome"] = "WIN_BE"
            elif hsl:
                entry["outcome"] = "LOSS"
            if entry["outcome"] != "PENDING":
                entry["r_achieved"] = calc_r_mixed(ep, sl, tp1, tp2, tp3, hit1, hit2, hit3, hbe)
                changed = True

        if changed:
            changed_any = True
            if not was_hit1 and hit1:
                events.append(("TP1_HIT", dict(entry)))
            if was_outcome == "PENDING" and entry["outcome"] != "PENDING":
                events.append(("CLOSED", dict(entry)))

    if changed_any:
        if not kv.set("journal", journal):
            logging.error("Journal save FAILED — events suppressed (anti-duplicate)")
            return []
    return events


def send_outcome_events(cfg, kv, events) -> None:
    for kind, e in events:
        try:
            if kind == "TP1_HIT":
                msg = (f"<b>🎯 TP1 HIT (v{VERSION})</b>\n"
                       f"{e.get('signal')} @ <code>{e.get('entry')}</code>\n"
                       f"<b>Action:</b> partial 1/3, SL→BE")
            elif kind == "CLOSED":
                out = e.get("outcome", "?")
                r = float(e.get("r_achieved", 0) or 0)
                if out.startswith("WIN"):
                    msg = (f"<b>✅ WIN {out}</b>\n{e.get('signal')} @ "
                           f"<code>{e.get('entry')}</code>\nR: <code>{r:+.2f}R</code>")
                else:
                    msg = (f"<b>❌ LOSS</b>\n{e.get('signal')} @ "
                           f"<code>{e.get('entry')}</code>\nR: <code>{r:+.2f}R</code>")
            elif kind == "STALE":
                msg = f"<b>⏸ STALE</b>\n{e.get('signal')} @ <code>{e.get('entry')}</code>"
            else:
                continue
            tg_send(cfg, msg, kv)
        except Exception as ex:
            logging.warning(f"Event notify fail: {redact(ex)}")


# =========================================================
# MARKET / NEWS
# =========================================================
def market_is_open(now_utc: datetime) -> bool:
    ny = now_utc.astimezone(NY)
    wd, hr = ny.weekday(), ny.hour
    return not ((wd == 4 and hr >= 17) or wd == 5 or (wd == 6 and hr < 18))


def in_killzone(now_utc: datetime) -> bool:
    lt = now_utc.astimezone(LONDON)
    nt = now_utc.astimezone(NY)
    lth = lt.hour + lt.minute / 60.0
    nth = nt.hour + nt.minute / 60.0
    return (7.0 <= lth < 10.0) or (8.0 <= nth < 11.0) or (13.0 <= nth < 16.0)


def check_news_shield(cfg: Config, kv: KVStore) -> tuple[bool, str]:
    now = datetime.now(timezone.utc)
    cache = kv.get("news_cache", {})
    if cache.get("ts"):
        try:
            age = (now - datetime.fromisoformat(cache["ts"])).total_seconds()
            ttl = cfg.news_cache_sec_on_fail if cache.get("fail") else cfg.news_cache_sec
            if age < ttl:
                return cache.get("blocked", False), cache.get("reason", "")
        except Exception:
            pass

    def _do():
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=cfg.http_timeout,
        )
        r.raise_for_status()
        if "application/json" not in r.headers.get("content-type", ""):
            raise DataFetchError("non-JSON")
        return r.json() or []

    try:
        events = retry_with_backoff(_do, attempts=2, base=1.0)
        kv.set("news_fail_streak", 0)
        if not events:
            kv.set("news_cache", {"ts": now.isoformat(), "blocked": True,
                                  "reason": "empty", "fail": False})
            return True, "empty feed"
        for e in events:
            if e.get("country") != "USD" or e.get("impact") != "High":
                continue
            try:
                ev_dt = datetime.strptime(e["date"], "%Y-%m-%dT%H:%M:%S%z")
            except Exception:
                continue
            if abs((ev_dt - now).total_seconds() / 60) <= cfg.news_blackout_min:
                kv.set("news_cache", {"ts": now.isoformat(), "blocked": True,
                                      "reason": e.get("title", "?"), "fail": False})
                return True, e.get("title", "?")
        kv.set("news_cache", {"ts": now.isoformat(), "blocked": False, "reason": "", "fail": False})
        return False, ""
    except Exception as ex:
        logging.warning(f"News fail: {redact(ex)}")
        streak = int(kv.get("news_fail_streak", 0)) + 1
        kv.set("news_fail_streak", streak)
        if streak >= cfg.news_max_consecutive_fail:
            logging.warning(f"News fail-OPEN after {streak} consec fails")
            kv.set("news_cache", {"ts": now.isoformat(), "blocked": False,
                                  "reason": "fail-open", "fail": True})
            return False, "fail-open"
        kv.set("news_cache", {"ts": now.isoformat(), "blocked": True,
                              "reason": "api down", "fail": True})
        return True, "api down"


# =========================================================
# RUNNER
# =========================================================
def sanity_price(cfg: Config, price: float, st: dict) -> None:
    if not (cfg.price_min < price < cfg.price_max):
        raise DataFetchError(f"Price {price} out of [{cfg.price_min},{cfg.price_max}]")
    if st.get("last_price"):
        ch = abs(price - st["last_price"]) / st["last_price"]
        if ch > cfg.price_max_jump_pct:
            raise DataFetchError(f"Price jump {ch*100:.1f}%")


def _build_msg(cfg, run_num, now_str, src, sig, price, sl, tp1, tp2, tp3, lot, note) -> str:
    p = cfg.price_decimals
    c = sig.conf
    parts = []
    if sig.ob["detected"]:
        parts.append(sig.ob["type"] + ("✓swept" if sig.ob["swept"] else ""))
    if sig.choch["detected"]:
        parts.append(sig.choch["type"])
    if sig.fvg["detected"]:
        parts.append(sig.fvg["type"])
    smc = " | ".join(parts) if parts else "—"
    rr = abs(tp1 - price) / abs(price - sl) if abs(price - sl) > 0 else 0
    warn = f"\n⚠️ <i>{h(note)}</i>" if note else ""
    return (
        f"<b>⚡ APEX v{VERSION} — {h(cfg.symbol)}</b>\n\n"
        f"Signal: <b>{h(sig.type)}</b> @ <code>{price:.{p}f}</code>\n"
        f"SL: <code>{sl:.{p}f}</code> | TP1: <code>{tp1:.{p}f}</code>\n"
        f"TP2: <code>{tp2:.{p}f}</code> | TP3: <code>{tp3:.{p}f}</code>\n"
        f"R:R(TP1): <code>1:{rr:.2f}</code>\n\n"
        f"Lot: <code>{lot}</code> ({cfg.account_currency}){warn}\n\n"
        f"SMC: <code>{h(smc)}</code>\n"
        f"HA: <code>{sig.ha}</code> | ADX: <code>{sig.adx:.1f}</code> | "
        f"RSI: <code>{c.get('rsi', 0)}</code>\n"
        f"Regime: <code>{sig.regime.get('regime', '?')}</code> "
        f"(H={sig.regime.get('hurst')}, E={sig.regime.get('entropy')})\n"
        f"Score: <code>{c.get('total', 0):+d}</code> "
        f"(T{c.get('trend', 0):+d} M{c.get('momentum', 0):+d} "
        f"S{c.get('smc', 0):+d} H{c.get('ha', 0):+d})\n"
        f"Data: <code>{h(src)}</code> | Run #{run_num}\n"
        f"<code>{h(now_str)} UTC</code>\n\n"
        f"<i>Manual execution MT5. BE after TP1.</i>"
    )


def run_once(cfg: Config) -> None:
    kv = KVStore(cfg)

    try:
        kv.force_load()
    except StateLoadError as e:
        logging.error(f"ABORT: state load failed ({e}). Tidak trading tanpa state.")
        try:
            tg_alert_error(cfg, f"State load failed — ABORT. {e}", kv)
        except Exception:
            pass
        return

    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    ks, reason = kill_switch_active(kv)
    if ks:
        logging.warning(f"KILL SWITCH: {reason}")
        return

    if now.hour == cfg.heartbeat_hour_utc:
        hb = kv.get("heartbeat", {})
        if hb.get("date") != now.strftime("%Y-%m-%d"):
            if tg_send(cfg, f"<b>APEX v{VERSION} HEARTBEAT</b>", kv):
                kv.set("heartbeat", {"date": now.strftime("%Y-%m-%d")})

    if not market_is_open(now):
        logging.info("Market closed")
        return
    if not in_killzone(now):
        logging.info("Outside killzone")
        return
    if not circuit_allow(kv):
        logging.info("Circuit open")
        return

    blocked, why = drawdown_check(cfg, kv, journal_get(kv, 200))
    if blocked:
        logging.info(f"DD block: {why}")
        return

    if not cfg.dry_run and check_news_shield(cfg, kv)[0]:
        logging.info("News shield")
        return

    st = kv.get("state", {}) or {}
    run_num = int(st.get("run_count", 0)) + 1

    try:
        dfs, src = fetch_all_intervals(cfg)
        m5 = validate_ohlcv(dfs["5min"], "5min", cfg.min_bars_m5, cfg)
        m15 = validate_ohlcv(dfs["15min"], "15min", cfg.min_bars_m15, cfg)
        h1 = validate_ohlcv(dfs["1h"], "1h", cfg.min_bars_h1, cfg)
        h4 = validate_ohlcv(dfs["4h"], "4h", cfg.min_bars_h4, cfg)
    except Exception as e:
        circuit_fail(kv)
        tg_alert_error(cfg, f"Data: {e}", kv)
        return
    circuit_ok(kv)

    try:
        ev = update_outcomes(cfg, kv, m5)
        send_outcome_events(cfg, kv, ev)
    except Exception as e:
        logging.warning(f"Outcome update: {redact(e)}")

    m5c = m5.iloc[:-1].reset_index(drop=True)
    m15c = m15.iloc[:-1].reset_index(drop=True)
    h1c = h1.iloc[:-1].reset_index(drop=True)
    h4c = h4.iloc[:-1].reset_index(drop=True)

    price = float(m5c["close"].iloc[-1])
    signal_bar_dt = m5c.iloc[-1]["datetime"]
    signal_bar_iso = signal_bar_dt.strftime("%Y-%m-%d %H:%M:%S")

    try:
        sanity_price(cfg, price, st)
    except Exception as e:
        tg_alert_error(cfg, f"Price: {e}", kv)
        return

    sig = generate_signal(cfg, m5c, m15c, h1c, h4c)

    should, cool = False, cfg.cooldown_minutes
    if sig.type != "NEUTRAL":
        lt, lts = st.get("last_signal_type", "NEUTRAL"), st.get("last_signal_ts", "")
        if lt != sig.type:
            cool = cfg.opposite_cooldown_minutes
        if not lts:
            should = True
        else:
            try:
                ldt = datetime.strptime(lts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                if (now - ldt) >= timedelta(minutes=cool):
                    should = True
            except Exception:
                should = True

    action, conf = "CANCEL", 0.0
    sl = tp1 = tp2 = tp3 = lot = 0.0
    lot_note = ""
    blocked_reason = ""

    if should:
        try:
            action, conf = run_council_vote(cfg, m5c, m15c, h1c, h4c, sig)
        except Exception as e:
            logging.exception(f"Council crash: {redact(e)}")
            action, conf = "CANCEL", 0.0
            blocked_reason = f"council-crash: {e}"
        if action == "CANCEL":
            should = False
            blocked_reason = blocked_reason or f"council-cancel({conf:.0f}%)"

    if should:
        try:
            sl, tp1, tp2, tp3 = compute_levels(cfg, m5c, sig.type)
            lot, _, lot_note = compute_lot(cfg, price, sl, conf, journal_get(kv, 500))
            if action == "EXECUTE_HALF":
                lot = max(cfg.min_lot, round(lot / 2.0, 2))
        except (ValueError, TypeError, KeyError) as e:
            should = False
            blocked_reason = f"risk: {e}"

    tg_ok = False
    if should:
        msg = _build_msg(cfg, run_num, now_str, src, sig, price,
                         sl, tp1, tp2, tp3, lot, lot_note)
        jid = f"{now_str.replace(' ', 'T')}_{sig.type}_{lot}_{signal_bar_iso.replace(' ', 'T')}"
        entry = {
            "id": jid,
            "timestamp": now_str,
            "signal_bar_dt": signal_bar_iso,
            "symbol": cfg.symbol,
            "signal": sig.type,
            "entry": round(price, cfg.price_decimals),
            "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "lot": lot,
            "confluence": sig.conf,
            "regime": sig.regime,
            "outcome": "PENDING",
            "r_achieved": 0.0,
            "notify_status": "DRY" if cfg.dry_run else "SENT",
        }
        if not journal_append(kv, entry, cfg.max_journal):
            logging.error("Journal save FAILED — abort signal (anti-duplicate)")
            blocked_reason = "journal-save-fail"
            should = False
        else:
            tg_ok = tg_send(cfg, msg, kv)
            if not tg_ok:
                logging.error("Telegram failed — cooldown still enforced (anti-livelock)")
                blocked_reason = blocked_reason or "tg-fail"
            st["last_signal_type"] = sig.type
            st["last_signal_ts"] = now_str

    st["run_count"] = run_num
    st["last_price"] = price
    st["last_price_ts"] = now_str
    kv.set("state", st)

    logging.info(
        f"Run #{run_num} | {sig.type} | action={action} conf={conf:.0f}% "
        f"| fired={tg_ok} | block={blocked_reason[:60]}"
    )


# =========================================================
# CLI COMMANDS
# =========================================================
def cmd_reports(cfg: Config, limit: int = 20) -> None:
    kv = KVStore(cfg)
    journal = journal_get(kv, limit)
    if not journal:
        print("No journal.")
        return
    for j in journal:
        r = float(j.get("r_achieved") or 0)
        print(f"[{j['timestamp']}] {j['signal']:4s} @{j['entry']} "
              f"lot={j['lot']} outcome={j['outcome']:8s} R={r:+.2f}")


def cmd_stats(cfg: Config) -> None:
    kv = KVStore(cfg)
    j = journal_get(kv, 500)
    if not j:
        print("No journal.")
        return
    closed = [e for e in j if e.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
    buys = sum(1 for e in j if e.get("signal") == "BUY")
    sells = sum(1 for e in j if e.get("signal") == "SELL")
    print(f"Journal: {len(j)} total (BUY={buys}, SELL={sells})")
    if not closed:
        print("No closed trades yet.")
        return
    wins = sum(1 for e in closed if str(e["outcome"]).startswith("WIN"))
    total_r = sum(float(e.get("r_achieved") or 0) for e in closed)
    print(f"Closed: {len(closed)} | WR: {wins/len(closed)*100:.1f}% | "
          f"Total R: {total_r:+.2f} | Avg R: {total_r/len(closed):+.3f}")


def cmd_reset_drawdown(cfg: Config) -> None:
    kv = KVStore(cfg)
    kv.set("drawdown_pause", {"until": None})
    print("Drawdown reset.")


def cmd_reset_price(cfg: Config) -> None:
    kv = KVStore(cfg)
    st = kv.get("state", {}) or {}
    st.pop("last_price", None)
    st.pop("last_price_ts", None)
    kv.set("state", st)
    print("Price sanity reset.")


def cmd_kill(cfg: Config, reason: str = "manual") -> None:
    kv = KVStore(cfg)
    kv.set("kill_switch", {"active": True, "reason": reason})
    print(f"KILL SWITCH ACTIVE: {reason}")


def cmd_unkill(cfg: Config) -> None:
    kv = KVStore(cfg)
    kv.set("kill_switch", {"active": False, "reason": ""})
    print("KILL SWITCH disabled.")


def _test_direction_symmetry() -> None:
    cfg = Config()
    cfg.adx_min = 0
    cfg.score_threshold = 8

    def _make_df(direction, n=100):
        rng = np.random.RandomState(42)
        base = 2000.0
        if direction == "BULLISH":
            closes = base + np.linspace(0, 20, n) + rng.randn(n) * 0.05
        else:
            closes = base - np.linspace(0, 20, n) + rng.randn(n) * 0.05
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) + 0.1
        lows = np.minimum(opens, closes) - 0.1
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": np.full(n, 100.0),
        })

    no_choch = {"detected": False, "type": "NONE", "level": 0}
    no_fvg = {"detected": False, "type": "NONE", "level": 0}
    bull_ob = {"detected": True, "type": "BULLISH_OB", "high": 2010, "low": 2005, "swept": True}
    bear_ob = {"detected": True, "type": "BEARISH_OB", "high": 2010, "low": 2005, "swept": True}
    bull_trends = {"m5": "BULLISH", "m15": "BULLISH", "h1": "BULLISH", "h4": "BULLISH"}
    bear_trends = {"m5": "BEARISH", "m15": "BEARISH", "h1": "BEARISH", "h4": "BEARISH"}

    bull = compute_confluence(_make_df("BULLISH"), bull_trends, bull_ob, no_choch, no_fvg, 30.0, "BULLISH")
    bear = compute_confluence(_make_df("BEARISH"), bear_trends, bear_ob, no_choch, no_fvg, 30.0, "BEARISH")

    assert bull["total"] == -bear["total"], \
        f"ASYMMETRIC: bull={bull['total']:+d}, bear={bear['total']:+d}"
    assert bull["total"] > 0 and bear["total"] < 0, "Sign wrong"

    bull_adx_hi = compute_confluence(_make_df("BULLISH"), bull_trends, bull_ob, no_choch, no_fvg, 40.0, "BULLISH")
    assert bull["total"] == bull_adx_hi["total"], "ADX masih mempengaruhi skor — P0-1 REGRESSED"

    print(f"  ✓ Symmetry: bull={bull['total']:+d}, bear={bear['total']:+d}")
    print(f"  ✓ ADX-independent: ADX 30 vs 40 → identik ({bull['total']:+d})")


def cmd_selftest(cfg: Config) -> None:
    print(f"=== SELFTEST v{VERSION} ===\n")
    kv = KVStore(cfg)
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    failed = []

    print(f"[1/5] Persistence: {'Gist' if kv.use_gist else 'LOCAL'}")
    if is_ci and not kv.use_gist:
        print("  ✗ CRITICAL: Gist tidak dikonfigurasi di CI.")
        print("    Fix: tambah GIST_ID + GIST_TOKEN secrets.")
        failed.append("Gist not configured in CI")
    elif not kv.use_gist:
        print("  ⚠️  LOCAL mode — OK untuk lokal, BUKAN production.")

    print("\n[2/5] Secrets:")
    td = bool(os.getenv("TWELVE_DATA_API_KEY", "").strip())
    tg_tok = bool(os.getenv("TELEGRAM_TOKEN", "").strip())
    tg_chat = bool(os.getenv("TELEGRAM_CHAT_ID", "").strip())
    print(f"  TwelveData key:   {'✓' if td else '✗'}")
    print(f"  Telegram token:   {'✓' if tg_tok else '✗'}")
    print(f"  Telegram chat_id: {'✓' if tg_chat else '✗'}")
    if not td:
        failed.append("TWELVE_DATA_API_KEY missing")
    if not tg_tok or not tg_chat:
        failed.append("Telegram creds incomplete")

    print("\n[3/5] State load:")
    try:
        kv.force_load()
        print("  ✓ State load OK")
    except StateLoadError as e:
        print(f"  ✗ State load FAIL: {e}")
        failed.append(f"State: {e}")

    print("\n[4/5] Direction symmetry (P0-1 regression):")
    try:
        _test_direction_symmetry()
    except AssertionError as e:
        print(f"  ✗ {e}")
        failed.append(f"Symmetry: {e}")

    print("\n[5/5] Data fetch:")
    try:
        dfs, src = fetch_all_intervals(cfg)
        print(f"  ✓ {src}: M5={len(dfs['5min'])} 15m={len(dfs['15min'])} "
              f"1h={len(dfs['1h'])} 4h={len(dfs['4h'])}")
        print(f"    Last M5: {dfs['5min']['close'].iloc[-1]:.2f} "
              f"@ {dfs['5min']['datetime'].iloc[-1]}")
    except DataFetchError as e:
        if "Stale" in str(e):
            print(f"  ⚠️  Stale data (market closed?) — API reachable, skip.")
        else:
            print(f"  ✗ Data FAIL: {e}")
            failed.append(f"Data: {e}")
    except Exception as e:
        print(f"  ✗ Data FAIL: {e}")
        failed.append(f"Data: {e}")

    print()
    if failed:
        print(f"=== ❌ SELFTEST FAILED ({len(failed)} issue(s)) ===")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("=== ✅ SELFTEST PASSED — siap deploy ===")
    sys.exit(0)


def cmd_smoke(cfg: Config) -> None:
    print(f"=== SMOKE @ {datetime.now(timezone.utc).isoformat()} ===\n")
    try:
        dfs, src = fetch_all_intervals(cfg)
    except Exception as e:
        print(f"✗ Data: {e}")
        sys.exit(1)

    m5c = dfs["5min"].iloc[:-1].reset_index(drop=True)
    m15c = dfs["15min"].iloc[:-1].reset_index(drop=True)
    h1c = dfs["1h"].iloc[:-1].reset_index(drop=True)
    h4c = dfs["4h"].iloc[:-1].reset_index(drop=True)

    print(f"Source: {src}")
    print(f"Last M5: {m5c.iloc[-1]['datetime']} close={m5c.iloc[-1]['close']:.2f}")
    for tf, df in (("M5", m5c), ("M15", m15c), ("H1", h1c), ("H4", h4c)):
        print(f"  {tf:3s} trend={trend_ema(df):8s}  HA={heikin_ashi_trend(df)}")
    print(f"\nADX={adx(m5c):.1f}  RSI={rsi(m5c['close']):.1f}")
    ob, ch, fv = detect_order_block(m5c), detect_choch(m5c), detect_fvg(m5c)
    print(f"OB={ob['type']} CHoCH={ch['type']} FVG={fv['type']}")
    H, E = hurst_exponent(h1c["close"]), shannon_entropy(h1c["close"])
    print(f"Regime: H={H:.3f} E={E:.3f}")
    print(f"\n--- SIGNAL ---")
    sig = generate_signal(cfg, m5c, m15c, h1c, h4c)
    if sig.type == "NEUTRAL":
        print(f"NEUTRAL — {sig.skip_reason}")
    else:
        print(f"{sig.type} score={sig.score:+d} conf={sig.conf}")
        action, conf = run_council_vote(cfg, m5c, m15c, h1c, h4c, sig)
        print(f"Council: {action} @ {conf:.1f}%")
        if action != "CANCEL":
            try:
                price = float(m5c["close"].iloc[-1])
                sl, tp1, tp2, tp3 = compute_levels(cfg, m5c, sig.type)
                kv = KVStore(cfg)
                lot, _, note = compute_lot(cfg, price, sl, conf, journal_get(kv, 500))
                print(f"Entry={price:.3f} SL={sl:.3f} TP1={tp1:.3f} TP2={tp2:.3f} TP3={tp3:.3f}")
                print(f"Lot={lot}  {note}")
            except Exception as e:
                print(f"✗ Level: {e}")
    print(f"\n[INFO] Smoke tidak kirim TG, tidak tulis journal.")


def _simulate_trade(future: pd.DataFrame, side: str, entry: float,
                    sl: float, tp1: float, tp2: float, tp3: float) -> float:
    if abs(entry - sl) <= 0:
        return 0.0
    h1 = h2 = h3 = hsl = hbe = False
    for _, b in future.iterrows():
        if h3 or hsl or hbe:
            break
        csl = entry if h1 else sl
        if side == "BUY":
            if b["low"] <= csl:
                if h1: hbe = True
                else: hsl = True
                continue
            if not h1 and b["high"] >= tp1: h1 = True
            if not h2 and b["high"] >= tp2: h2 = True
            if not h3 and b["high"] >= tp3: h3 = True
        else:
            if b["high"] >= csl:
                if h1: hbe = True
                else: hsl = True
                continue
            if not h1 and b["low"] <= tp1: h1 = True
            if not h2 and b["low"] <= tp2: h2 = True
            if not h3 and b["low"] <= tp3: h3 = True
    return calc_r_mixed(entry, sl, tp1, tp2, tp3, h1, h2, h3, hbe)


def cmd_backtest(cfg: Config, max_samples: int = 800) -> None:
    print(f"=== BACKTEST LIGHTWEIGHT ({max_samples} samples) ===\n")
    try:
        dfs, src = fetch_all_intervals(cfg)
    except Exception as e:
        print(f"✗ Data: {e}")
        sys.exit(1)

    m5 = dfs["5min"]
    m15_full, h1_full, h4_full = dfs["15min"], dfs["1h"], dfs["4h"]

    if len(m5) < 300:
        print(f"✗ Not enough M5 bars: {len(m5)}")
        sys.exit(1)

    print(f"Source: {src} | M5 bars: {len(m5)} "
          f"({m5.iloc[0]['datetime']} → {m5.iloc[-1]['datetime']})")

    indices = np.linspace(200, len(m5) - 150,
                          num=min(max_samples, len(m5) - 350)).astype(int)

    buys = sells = neutral = 0
    trades = []

    for idx in indices:
        try:
            m5_w = m5.iloc[:idx + 1]
            ts = m5_w.iloc[-1]["datetime"]
            m15_w = m15_full[m15_full["datetime"] <= ts]
            h1_w = h1_full[h1_full["datetime"] <= ts]
            h4_w = h4_full[h4_full["datetime"] <= ts]
            if (len(m15_w) < cfg.min_bars_m15 or len(h1_w) < cfg.min_bars_h1
                    or len(h4_w) < cfg.min_bars_h4):
                continue

            sig = generate_signal(cfg, m5_w, m15_w, h1_w, h4_w)
            if sig.type == "BUY":
                buys += 1
            elif sig.type == "SELL":
                sells += 1
            else:
                neutral += 1
                continue

            future = m5.iloc[idx + 1: idx + 101]
            if len(future) < 10:
                continue
            entry = float(m5_w.iloc[-1]["close"])
            sl, tp1, tp2, tp3 = compute_levels(cfg, m5_w, sig.type)
            r = _simulate_trade(future, sig.type, entry, sl, tp1, tp2, tp3)
            trades.append((sig.type, r))
        except Exception:
            continue

    print(f"\nSignals: {buys + sells} (BUY={buys}, SELL={sells}, NEUTRAL={neutral})")
    if buys + sells > 0:
        ratio = buys / (buys + sells)
        marker = "✓ OK" if 0.35 <= ratio <= 0.65 else "⚠️  BIAS"
        print(f"BUY/SELL ratio: {ratio*100:.1f}%  {marker}  (target 35–65%)")

    for side in ("BUY", "SELL"):
        rs = [r for s, r in trades if s == side]
        if not rs:
            continue
        wins = sum(1 for r in rs if r > 0)
        avg = sum(rs) / len(rs)
        print(f"  {side}: n={len(rs)}  WR={wins/len(rs)*100:.1f}%  avgR={avg:+.3f}")

    if trades:
        all_r = [r for _, r in trades]
        wins = sum(1 for r in all_r if r > 0)
        total = sum(all_r)
        print(f"\nOverall: n={len(all_r)}  WR={wins/len(all_r)*100:.1f}%  "
              f"totalR={total:+.2f}  avgR={total/len(all_r):+.3f}")
    print(f"\n⚠️  Backtest ≠ forward. Paper trade tetap wajib.")


def cmd_force_run(cfg: Config) -> None:
    global in_killzone
    confirmed = "--confirm" in sys.argv
    if not cfg.dry_run and not confirmed:
        print("✗ force-run di live mode butuh --confirm flag.")
        sys.exit(1)

    print("=" * 60)
    print(f"FORCE RUN — bypass killzone, safety tetap aktif")
    print(f"Mode: {'DRY' if cfg.dry_run else 'LIVE --confirmed'}")
    print("=" * 60)

    _orig_kz = in_killzone

    def _always_kz(_):
        return True

    in_killzone = _always_kz
    try:
        run_once(cfg)
    finally:
        in_killzone = _orig_kz


def cmd_test_tg(cfg: Config) -> None:
    print("=== TEST TELEGRAM INTEGRATION ===")
    kv = KVStore(cfg)
    msg = (f"<b>🔔 TEST MESSAGE — APEX v{VERSION}</b>\n\n"
           f"Jika Anda menerima pesan ini, integrasi Telegram <b>BERHASIL</b>! ✅\n"
           f"Bot siap tempur menunggu sinyal SMC.")
    ok = tg_send(cfg, msg, kv)
    if ok:
        print("✓ Test message SENT. Cek HP Anda.")
    else:
        print("✗ Test message FAILED. Cek TOKEN & CHAT_ID di GitHub Secrets.")
        sys.exit(1)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    try:
        cfg = Config.from_env()
    except ValueError as e:
        print(e)
        sys.exit(2)
    setup_logging()
    if not cfg.dry_run:
        print(f"⚠️ LIVE MODE. Balance=${cfg.account_balance} "
              f"Risk={cfg.max_risk_pct_per_trade}% "
              f"Leverage cap={cfg.hard_leverage_cap}x")

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        try:
            if cmd == "run":
                run_once(cfg)
            elif cmd == "smoke":
                cmd_smoke(cfg)
            elif cmd == "backtest":
                n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 800
                cmd_backtest(cfg, max_samples=n)
            elif cmd == "force-run":
                cmd_force_run(cfg)
            elif cmd == "test-tg":
                cmd_test_tg(cfg)
            elif cmd == "reports":
                cmd_reports(cfg)
            elif cmd == "stats":
                cmd_stats(cfg)
            elif cmd == "reset-drawdown":
                cmd_reset_drawdown(cfg)
            elif cmd == "reset-price":
                cmd_reset_price(cfg)
            elif cmd == "kill":
                cmd_kill(cfg, sys.argv[2] if len(sys.argv) > 2 else "manual")
            elif cmd == "unkill":
                cmd_unkill(cfg)
            elif cmd == "selftest":
                cmd_selftest(cfg)
            elif cmd == "help":
                print("Commands:")
                print("  selftest         — fail-fast validation")
                print("  test-tg          — force send test message to Telegram")
                print("  smoke            — analyse now outside killzone (dry)")
                print("  backtest [N]     — historical symmetry + WR check")
                print("  force-run        — bypass killzone (add --confirm if live)")
                print("  run              — normal production run")
                print("  reports|stats    — audit")
                print("  kill [reason]|unkill|reset-drawdown|reset-price")
            else:
                print(f"Unknown: {cmd}")
        except SystemExit:
            raise
        except Exception as e:
            logging.exception(f"FATAL: {e}")
            sys.exit(1)
        sys.exit(0)

    run_once(cfg)
