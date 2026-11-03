#!/usr/bin/env python3
"""
APEX PREDATOR v18.0 — Production Institutional Hardened
========================================================
OOP architecture with fail-safe layers. See CHANGELOG.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    from sklearn.ensemble import RandomForestClassifier
    from scipy.signal import butter, filtfilt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

VERSION = "18.0"
NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
UTC = timezone.utc


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
@dataclass
class Config:
    gist_id: str = ""
    gist_token: str = ""
    gist_state_file: str = "apex_state_v18.json"
    local_state_file: str = "apex_state_v18.json"
    symbol: str = "XAUUSD"
    contract_size: float = 100.0
    account_currency: str = "USD"
    price_decimals: int = 2
    mt5_offset: float = 0.0
    account_balance: float = 1000.0
    max_risk_pct_per_trade: float = 1.0
    hard_leverage_cap: float = 30.0
    max_lot: float = 5.0
    min_lot: float = 0.01
    max_daily_loss_pct: float = 3.0
    max_consecutive_losses: int = 3
    dd_pause_hours: int = 24
    atr_period: int = 14
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
    enable_ml_models: bool = True
    cooldown_minutes: int = 45
    opposite_cooldown_minutes: int = 120
    max_daily_signals: int = 5
    hurst_trend_min: float = 0.55
    hurst_meanrev_max: float = 0.45
    entropy_chaos_max: float = 0.85
    dry_run: bool = True
    kill_switch: bool = False
    news_blackout_min: int = 30
    news_cache_sec: int = 1800
    news_cache_sec_on_fail: int = 300
    news_max_consecutive_fail: int = 3
    pending_max_age_hours: int = 24
    http_timeout: tuple = (3, 10)
    max_retry: int = 3
    max_journal: int = 500
    ml_purge_bars: int = 10
    ml_min_val_acc: float = 0.56
    entry_pullback_atr: float = 0.30
    entry_max_chase_atr: float = 2.0
    skip_killzone_filter: bool = False
    h1_cache_minutes: int = 55
    h4_cache_minutes: int = 220
    td_min_interval_sec: float = 1.5

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for name in cls.__dataclass_fields__:
            env_key = f"APEX_{name.upper()}"
            raw = os.getenv(env_key)
            if raw is None or not str(raw).strip():
                continue
            cur = getattr(cfg, name)
            try:
                if isinstance(cur, bool):
                    setattr(cfg, name, str(raw).strip().lower() in ("1", "true", "yes", "on"))
                elif isinstance(cur, int):
                    setattr(cfg, name, int(raw))
                elif isinstance(cur, float):
                    v = float(raw)
                    if not np.isfinite(v):
                        raise ValueError("non-finite")
                    setattr(cfg, name, v)
                elif isinstance(cur, str):
                    setattr(cfg, name, str(raw).strip())
            except (ValueError, TypeError):
                logging.warning(f"Config: ignore invalid env {env_key}")
        cfg.gist_id = os.getenv("GIST_ID", cfg.gist_id).strip()
        cfg.gist_token = os.getenv("GIST_TOKEN", cfg.gist_token).strip()
        off = os.getenv("APEX_MT5_OFFSET", "").strip()
        if off:
            try:
                cfg.mt5_offset = float(off)
            except ValueError:
                pass
        if os.getenv("APEX_KILL_SWITCH", "").lower() in ("1", "true", "yes"):
            cfg.kill_switch = True
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not (0 < self.parliament_min_ratio < self.parliament_half_ratio < self.parliament_full_ratio <= 1.0):
            raise ValueError("parliament ratio ordering invalid")
        if not (self.max_lot >= self.min_lot > 0):
            raise ValueError("lot bounds invalid")
        if self.m5_fetch_size < self.min_bars_m5:
            raise ValueError("m5_fetch_size < min_bars_m5")
        if self.account_balance <= 0:
            raise ValueError("account_balance must be > 0")
        if not (0 < self.max_risk_pct_per_trade <= 10):
            raise ValueError("max_risk_pct_per_trade out of range")
        if not (0 < self.max_daily_loss_pct <= 20):
            raise ValueError("max_daily_loss_pct out of range")


# ════════════════════════════════════════════════════════════
# LOGGING & UTIL
# ════════════════════════════════════════════════════════════
def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(h)


def redact(text) -> str:
    s = str(text).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"(token|apikey|api_key|bearer)\s*[=:]\s*[^\s&'\"]+", r"\1=***", s, flags=re.I)
    return s


class DataFetchError(Exception):
    pass


class StateError(Exception):
    pass


# ════════════════════════════════════════════════════════════
# HTTP CLIENT
# ════════════════════════════════════════════════════════════
class HTTPClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.cfg.http_timeout)
        last_err = "unknown"
        for attempt in range(self.cfg.max_retry):
            try:
                r = requests.request(method, url, **kwargs)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}"
                    self._backoff(attempt)
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                last_err = redact(e)
                self._backoff(attempt)
        raise DataFetchError(f"retry exhausted: {last_err}")

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(0.5 * (2 ** attempt) + random.uniform(0, 0.3))


# ════════════════════════════════════════════════════════════
# STATE STORE
# ════════════════════════════════════════════════════════════
class KVStore:
    def __init__(self, cfg: Config, http: HTTPClient):
        self.cfg = cfg
        self.http = http
        self.use_gist = bool(cfg.gist_id and cfg.gist_token)
        self._cache: Optional[dict] = None

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.cfg.gist_token}",
            "Accept": "application/vnd.github+json",
        }

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if self.use_gist:
            try:
                r = self.http.request(
                    "GET", f"https://api.github.com/gists/{self.cfg.gist_id}",
                    headers=self._headers(),
                )
                files = r.json().get("files", {})
                fn = self.cfg.gist_state_file
                content = files.get(fn, {}).get("content", "").strip()
                self._cache = json.loads(content) if content else {}
            except Exception as e:
                raise StateError(f"gist load: {redact(e)}")
        else:
            p = Path(self.cfg.local_state_file)
            try:
                self._cache = json.loads(p.read_text()) if p.exists() else {}
            except Exception as e:
                raise StateError(f"local load: {e}")
        return self._cache

    def _save(self, data: dict) -> bool:
        if self.use_gist:
            try:
                body = {"files": {self.cfg.gist_state_file: {
                    "content": json.dumps(data, default=str, sort_keys=True)
                }}}
                self.http.request(
                    "PATCH", f"https://api.github.com/gists/{self.cfg.gist_id}",
                    headers=self._headers(), json=body,
                )
                self._cache = data
                return True
            except Exception as e:
                logging.error(f"gist save: {redact(e)}")
                return False
        else:
            try:
                Path(self.cfg.local_state_file).write_text(
                    json.dumps(data, default=str, sort_keys=True)
                )
                self._cache = data
                return True
            except Exception as e:
                logging.error(f"local save: {e}")
                return False

    def get(self, key: str, default=None):
        try:
            return self._load().get(key, default)
        except StateError:
            return default

    def set(self, key: str, value) -> bool:
        try:
            data = dict(self._load())
        except StateError:
            data = {}
        data[key] = value
        return self._save(data)

    def force_load(self) -> None:
        self._cache = None
        self._load()


# ════════════════════════════════════════════════════════════
# INDICATORS
# ════════════════════════════════════════════════════════════
class Indicators:
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def atr_last(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < 2:
            return 0.0
        v = Indicators.atr_series(df, period).iloc[-1]
        return 0.0 if pd.isna(v) or v < 1e-6 else float(v)

    @staticmethod
    def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
        d = close.diff()
        gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-9))

    @staticmethod
    def adx_last(df: pd.DataFrame, period: int = 14) -> float:
        try:
            if len(df) < period * 2:
                return 0.0
            up = df["high"].diff()
            dn = -df["low"].diff()
            pdm = up.where((up > dn) & (up > 0), 0.0)
            mdm = dn.where((dn > up) & (dn > 0), 0.0)
            tr = pd.concat(
                [df["high"] - df["low"],
                 (df["high"] - df["close"].shift()).abs(),
                 (df["low"] - df["close"].shift()).abs()],
                axis=1,
            ).max(axis=1)
            atr_s = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
            pdi = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
            mdi = 100 * mdm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
            dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
            v = dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
            return 0.0 if pd.isna(v) or v < 0 else float(v)
        except Exception:
            return 0.0

    @staticmethod
    def hurst(close: pd.Series, max_lag: int = 20) -> float:
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

    @staticmethod
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


# ════════════════════════════════════════════════════════════
# MARKET DATA HANDLER
# ════════════════════════════════════════════════════════════
class MarketDataHandler:
    def __init__(self, cfg: Config, http: HTTPClient, kv: KVStore):
        self.cfg = cfg
        self.http = http
        self.kv = kv
        self._last_td_call = 0.0

    def _validate(self, df: pd.DataFrame, name: str, min_bars: int) -> None:
        if df is None or len(df) < min_bars:
            raise DataFetchError(f"{name}: {0 if df is None else len(df)} < {min_bars} bars")
        if df["datetime"].isna().any():
            raise DataFetchError(f"{name}: NaN datetime")
        if df["datetime"].duplicated().any():
            raise DataFetchError(f"{name}: duplicate datetime")
        last = float(df["close"].iloc[-1])
        if not (self.cfg.price_min <= last <= self.cfg.price_max):
            raise DataFetchError(f"{name}: price {last} out of range")
        if len(df) > 2:
            prev = float(df["close"].iloc[-2])
            jump = abs(last - prev) / max(prev, 1e-9)
            if jump > self.cfg.price_max_jump_pct:
                raise DataFetchError(f"{name}: price jump {jump:.1%} > limit")

    def _check_staleness(self, df: pd.DataFrame, tf_minutes: int, name: str) -> None:
        if df.empty:
            return
        last_ts = df["datetime"].iloc[-1]
        if last_ts.tz is None:
            last_ts = last_ts.tz_localize("UTC")
        age_min = (datetime.now(UTC) - last_ts.to_pydatetime()).total_seconds() / 60
        if age_min > self.cfg.max_stale_mult * tf_minutes:
            raise DataFetchError(f"{name}: stale {age_min:.0f}min old")

    def _compute_offset(self) -> float:
        oanda_key = os.getenv("OANDA_API_KEY", "").strip()
        td_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not (oanda_key and td_key):
            return self.cfg.mt5_offset
        try:
            r1 = self.http.request(
                "GET", "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/pricing",
                headers={"Authorization": f"Bearer {oanda_key}"},
            )
            oanda = float(r1.json()["prices"][0]["bids"][0]["price"])
            r2 = self.http.request(
                "GET", "https://api.twelvedata.com/price",
                params={"symbol": "XAU/USD", "apikey": td_key},
            )
            td = float(r2.json()["price"])
            if td <= 0:
                return self.cfg.mt5_offset
            off = oanda - td
            if abs(off) > 1.0:
                logging.warning(f"offset {off:+.3f} rejected")
                return self.cfg.mt5_offset
            return round(off, 3)
        except Exception as e:
            logging.warning(f"offset compute failed: {redact(e)}")
            return self.cfg.mt5_offset

    @staticmethod
    def _apply_offset(df: pd.DataFrame, off: float) -> pd.DataFrame:
        if off == 0.0:
            return df
        out = df.copy()
        for c in ("open", "high", "low", "close"):
            out[c] = out[c] + off
        return out

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return (
            df.set_index("datetime")
              .resample(rule, label="left", closed="left")
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"})
              .dropna()
              .reset_index()
        )

    def _rate_limit_td(self) -> None:
        elapsed = time.time() - self._last_td_call
        wait = self.cfg.td_min_interval_sec - elapsed
        if wait > 0:
            logging.debug(f"TD rate-limit sleep {wait:.2f}s")
            time.sleep(wait)
        self._last_td_call = time.time()

    def _cached_td_fetch(self, interval: str, outputsize: int,
                         ttl_minutes: int, offset: float) -> pd.DataFrame:
        cache_key = f"td_cache_{interval}"
        cache = self.kv.get(cache_key, {}) or {}
        cache_ts = float(cache.get("ts", 0) or 0)
        cache_data = cache.get("data")

        if cache_data and ttl_minutes > 0 and (time.time() - cache_ts) < ttl_minutes * 60:
            age = (time.time() - cache_ts) / 60.0
            logging.info(f"TD {interval}: cache hit (age {age:.1f}min)")
            df = pd.DataFrame(cache_data)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df.sort_values("datetime").reset_index(drop=True)
            return self._apply_offset(df, offset)

        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("no TD key")

        self._rate_limit_td()
        r = self.http.request(
            "GET", "https://api.twelvedata.com/time_series",
            params={
                "symbol": "XAU/USD", "interval": interval,
                "outputsize": outputsize, "apikey": key, "timezone": "UTC",
            },
        )
        data = r.json()
        if "values" not in data or not data["values"]:
            if cache_data:
                logging.warning(f"TD {interval}: API failed — using stale cache")
                df = pd.DataFrame(cache_data)
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["open", "high", "low", "close"])
                df = df.sort_values("datetime").reset_index(drop=True)
                return self._apply_offset(df, offset)
            raise DataFetchError(
                f"TD {interval}: {data.get('message', 'no values')}"
            )

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = 100.0
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.sort_values("datetime").reset_index(drop=True)

        cache_copy = df.copy()
        cache_copy["datetime"] = cache_copy["datetime"].astype(str)
        self.kv.set(cache_key, {
            "ts": time.time(),
            "data": cache_copy[["datetime", "open", "high", "low", "close", "volume"]].to_dict("records"),
        })

        return self._apply_offset(df, offset)

    def _fetch_td(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
        if not key:
            raise DataFetchError("no TD key")
        size = min(self.cfg.m5_fetch_size, 800)

        m5 = self._cached_td_fetch("5min", size, ttl_minutes=0, offset=offset)
        h1 = self._cached_td_fetch("1h", 800, ttl_minutes=self.cfg.h1_cache_minutes, offset=offset)
        h4 = self._cached_td_fetch("4h", 800, ttl_minutes=self.cfg.h4_cache_minutes, offset=offset)

        self._validate(m5, "TD-M5", self.cfg.min_bars_m5)
        self._validate(h1, "TD-H1", self.cfg.min_bars_h1)
        self._validate(h4, "TD-H4", self.cfg.min_bars_h4)
        self._check_staleness(m5, 5, "TD-M5")
        self._check_staleness(h1, 60, "TD-H1")
        self._check_staleness(h4, 240, "TD-H4")
        m15 = self._apply_offset(self._resample(m5, "15min"), offset)
        self._validate(m15, "TD-M15", self.cfg.min_bars_m15)
        return {"5min": m5, "15min": m15, "1h": h1, "4h": h4}, "TwelveData"

    def _fetch_yahoo(self, offset: float) -> Tuple[Dict[str, pd.DataFrame], str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
        symbol = "GC=F"

        def fetch(iv: str, rng: str) -> pd.DataFrame:
            r = self.http.request(
                "GET", f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval": iv, "range": rng}, headers=headers,
            )
            res = (r.json().get("chart", {}).get("result") or [None])[0]
            if not res:
                raise DataFetchError(f"YH {iv}: no result")
            ts = res.get("timestamp", [])
            q = res.get("indicators", {}).get("quote", [{}])[0]
            if not ts or not q:
                raise DataFetchError(f"YH {iv}: empty quote")
            o = q.get("open", [])
            hh = q.get("high", [])
            ll = q.get("low", [])
            cc = q.get("close", [])
            n = min(len(ts), len(o), len(hh), len(ll), len(cc))
            rows = []
            for i in range(n):
                if None in (o[i], hh[i], ll[i], cc[i]):
                    continue
                rows.append({
                    "epoch": int(ts[i]),
                    "open": float(o[i]), "high": float(hh[i]),
                    "low": float(ll[i]), "close": float(cc[i]),
                    "volume": 100.0,
                })
            if not rows:
                raise DataFetchError(f"YH {iv}: 0 valid rows")
            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            return df[["datetime", "open", "high", "low", "close", "volume"]]

        out = {}
        for iv, rng in (("5min", "5d"), ("15min", "5d"), ("1h", "1mo")):
            iv_map = {"5min": "5m", "15min": "15m", "1h": "1h"}
            out[iv] = self._apply_offset(fetch(iv_map[iv], rng), offset)
        out["4h"] = self._apply_offset(
            self._resample(fetch("1h", "1mo"), "4h"), offset
        )
        for tf, mins, min_bars in (
            ("5min", 5, self.cfg.min_bars_m5),
            ("15min", 15, self.cfg.min_bars_m15),
            ("1h", 60, self.cfg.min_bars_h1),
            ("4h", 240, self.cfg.min_bars_h4),
        ):
            self._validate(out[tf], f"YH-{tf}", min_bars)
            self._check_staleness(out[tf], mins, f"YH-{tf}")
        return out, "Yahoo"

    def fetch_all(self) -> Tuple[Dict[str, pd.DataFrame], str, float]:
        offset = self._compute_offset()
        errors = []
        for name, fn in (("TwelveData", self._fetch_td), ("Yahoo", self._fetch_yahoo)):
            try:
                dfs, src = fn(offset)
                logging.info(f"Data source: {src} | offset={offset:+.3f}")
                return dfs, src, offset
            except Exception as e:
                errors.append(f"{name}: {redact(e)}")
                logging.warning(f"Data {name} failed: {redact(e)}")
        raise DataFetchError(" | ".join(errors))


# ════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════
@dataclass
class SignalResult:
    direction: str = "NEUTRAL"
    buy_score: float = 0.0
    sell_score: float = 0.0
    buy_ratio: float = 0.0
    sell_ratio: float = 0.0
    active_models: int = 0
    verdict: str = "CANCEL"
    skip_reason: str = ""
    votes_detail: list = field(default_factory=list)
    regime: str = "UNKNOWN"
    adx: float = 0.0
    hurst: float = 0.5
    entropy: float = 0.5


class SignalGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model_registry = self._build_registry()

    @staticmethod
    def _ema_cross(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 30:
            return 0, 0.0
        e9 = Indicators.ema(df["close"], 9)
        e21 = Indicators.ema(df["close"], 21)
        diff = e9 - e21
        a = Indicators.atr_last(df) + 1e-9
        if diff.iloc[-1] > 0 and diff.diff().iloc[-1] > 0:
            return 1, min(1.0, abs(diff.iloc[-1]) / a)
        if diff.iloc[-1] < 0 and diff.diff().iloc[-1] < 0:
            return -1, min(1.0, abs(diff.iloc[-1]) / a)
        return 0, 0.0

    @staticmethod
    def _ema_trend(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 60:
            return 0, 0.0
        c = df["close"]
        e20, e50 = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1]:
            return 1, 0.9
        if c.iloc[-1] < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1]:
            return -1, 0.9
        return 0, 0.0

    @staticmethod
    def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Tuple[int, float]:
        if len(df) < period + 5:
            return 0, 0.0
        a = Indicators.atr_series(df, period)
        hl2 = (df["high"] + df["low"]) / 2
        upper = (hl2 + mult * a).values
        lower = (hl2 - mult * a).values
        close = df["close"].values
        trend = np.ones(len(df))
        for i in range(1, len(df)):
            if close[i] > upper[i - 1]:
                trend[i] = 1
            elif close[i] < lower[i - 1]:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]
        if trend[-1] == 1 and trend[-2] == 1:
            return 1, 0.85
        if trend[-1] == -1 and trend[-2] == -1:
            return -1, 0.85
        return 0, 0.0

    @staticmethod
    def _donchian(df: pd.DataFrame, period: int = 20) -> Tuple[int, float]:
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

    @staticmethod
    def _kalman(df: pd.DataFrame) -> Tuple[int, float]:
        obs = df["close"].values
        n = len(obs)
        if n < 10:
            return 0, 0.0
        x = obs[0]
        P, Q, R = 1.0, 1e-3, 1e-2
        est = np.empty(n)
        est[0] = x
        for i in range(1, n):
            P += Q
            K = P / (P + R)
            x = x + K * (obs[i] - x)
            P *= (1 - K)
            est[i] = x
        slope = est[-1] - est[-5]
        a = Indicators.atr_last(df) + 1e-9
        if slope > 0:
            return 1, min(1.0, abs(slope) / a)
        if slope < 0:
            return -1, min(1.0, abs(slope) / a)
        return 0, 0.0

    @staticmethod
    def _macd(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 35:
            return 0, 0.0
        c = df["close"]
        macd = Indicators.ema(c, 12) - Indicators.ema(c, 26)
        sig = Indicators.ema(macd, 9)
        hist = macd - sig
        if macd.iloc[-1] > sig.iloc[-1] and hist.diff().iloc[-1] > 0:
            return 1, 0.9
        if macd.iloc[-1] < sig.iloc[-1] and hist.diff().iloc[-1] < 0:
            return -1, 0.9
        return 0, 0.0

    @staticmethod
    def _rsi(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 16:
            return 0, 0.0
        rsi = Indicators.rsi_series(df["close"], 14)
        cur, prev = rsi.iloc[-1], rsi.iloc[-2]
        if cur < 30 and cur > prev:
            return 1, 0.85
        if cur > 70 and cur < prev:
            return -1, 0.85
        if 50 < cur < 65 and cur > prev:
            return 1, 0.6
        if 35 < cur < 50 and cur < prev:
            return -1, 0.6
        return 0, 0.0

    @staticmethod
    def _stochastic(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 20:
            return 0, 0.0
        low = df["low"].rolling(14).min()
        high = df["high"].rolling(14).max()
        k = 100 * (df["close"] - low) / (high - low + 1e-9)
        d = k.rolling(3).mean()
        k_, d_ = k.iloc[-1], d.iloc[-1]
        kp, dp = k.iloc[-2], d.iloc[-2]
        if k_ > d_ and kp <= dp and k_ < 25:
            return 1, 0.8
        if k_ < d_ and kp >= dp and k_ > 75:
            return -1, 0.8
        return 0, 0.0

    @staticmethod
    def _bb(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> Tuple[int, float]:
        if len(df) < period + 2:
            return 0, 0.0
        ma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = (ma + mult * std).iloc[-1]
        lower = (ma - mult * std).iloc[-1]
        c = df["close"].iloc[-1]
        if c < lower:
            return 1, 0.75
        if c > upper:
            return -1, 0.75
        return 0, 0.0

    @staticmethod
    def _ichimoku(df: pd.DataFrame) -> Tuple[int, float]:
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

    @staticmethod
    def _smc(df: pd.DataFrame) -> Tuple[int, float]:
        if len(df) < 60:
            return 0, 0.0
        d = df.tail(60).reset_index(drop=True)
        a = Indicators.atr_last(df)
        if a <= 0:
            return 0, 0.0
        bodies = (d["close"] - d["open"]).abs()
        cands = bodies.iloc[3:-1]
        if cands.empty:
            return 0, 0.0
        imp_idx = int(cands.idxmax())
        if imp_idx < 1 or abs(d["close"].iloc[imp_idx] - d["open"].iloc[imp_idx]) < a * 1.2:
            return 0, 0.0
        is_bull_imp = d["close"].iloc[imp_idx] > d["open"].iloc[imp_idx]
        ob = d.iloc[imp_idx - 1]
        is_bull_ob = ob["close"] > ob["open"]
        if is_bull_imp == is_bull_ob:
            return 0, 0.0
        last_close = d["close"].iloc[-1]
        ob_h = max(ob["open"], ob["close"])
        ob_l = min(ob["open"], ob["close"])
        buf = a * 0.5
        if is_bull_imp and ob_l - buf <= last_close <= ob_h + buf:
            return 1, 0.85
        if (not is_bull_imp) and ob_l - buf <= last_close <= ob_h + buf:
            return -1, 0.85
        return 0, 0.0

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        c, vol = df["close"], df["volume"]
        f = pd.DataFrame(index=df.index)
        f["ret_1"] = c.pct_change(1)
        f["ret_5"] = c.pct_change(5)
        f["ret_10"] = c.pct_change(10)
        f["atr_norm"] = Indicators.atr_series(df) / (c + 1e-9)
        f["vol_ratio"] = c.pct_change(1).rolling(20).std()
        f["rsi_14"] = Indicators.rsi_series(c, 14) / 100.0
        f["ema20_dist"] = (c - Indicators.ema(c, 20)) / (c + 1e-9)
        f["ema50_dist"] = (c - Indicators.ema(c, 50)) / (c + 1e-9)
        vs = vol.rolling(50).std()
        f["vol_z"] = np.where(vs > 1e-9, (vol - vol.rolling(50).mean()) / (vs + 1e-9), 0.0)
        return f.replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _triple_barrier_labels(
        df: pd.DataFrame, horizon: int = 30, tp_mult: float = 1.0, sl_mult: float = 1.8
    ) -> pd.Series:
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        a = Indicators.atr_series(df).values
        n = len(c)
        labels = np.zeros(n)
        for i in range(n - horizon):
            tp = c[i] + tp_mult * a[i]
            sl = c[i] - sl_mult * a[i]
            hh = h[i + 1:i + horizon + 1]
            ll = l[i + 1:i + horizon + 1]
            hit_tp = np.where(hh >= tp)[0]
            hit_sl = np.where(ll <= sl)[0]
            idx_tp = hit_tp[0] if len(hit_tp) else horizon + 1
            idx_sl = hit_sl[0] if len(hit_sl) else horizon + 1
            if idx_tp < idx_sl:
                labels[i] = 1
            elif idx_sl < idx_tp:
                labels[i] = -1
        return pd.Series(labels[:n - horizon], index=df.index[:n - horizon])

    def _ml_predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        if not (HAS_SKLEARN and self.cfg.enable_ml_models):
            return 0, 0.0
        try:
            horizon = max(self.cfg.ml_purge_bars * 3, 20)
            X = self._build_features(df)
            y_raw = self._triple_barrier_labels(df, horizon=horizon)
            y = (y_raw == 1).astype(int)
            valid = X.index.intersection(y.dropna().index)
            X, y = X.loc[valid], y.loc[valid]
            forward = horizon + self.cfg.ml_purge_bars
            if len(X) < forward + 200:
                return 0, 0.0
            trX = X.iloc[:-forward]
            trY = y.iloc[:-forward]
            if len(trX) < 200:
                return 0, 0.0
            split = int(len(trX) * 0.8)
            purged_end = max(100, split - forward)
            X_tr, y_tr = trX.iloc[:purged_end], trY.iloc[:purged_end]
            X_val, y_val = trX.iloc[split:], trY.iloc[split:]
            if len(X_tr) < 100 or len(X_val) < 20:
                return 0, 0.0
            model = RandomForestClassifier(
                n_estimators=80, max_depth=5, min_samples_leaf=8,
                random_state=42, n_jobs=1,
            )
            model.fit(X_tr.values, y_tr.values)
            val_acc = float(model.score(X_val.values, y_val.values))
            if val_acc < self.cfg.ml_min_val_acc:
                return 0, 0.0
            proba = model.predict_proba(X.iloc[-1:].values)[0]
            up = float(proba[1]) if len(proba) > 1 else 0.5
            if up > 0.62:
                return 1, (up - 0.5) * 2
            if up < 0.38:
                return -1, (0.5 - up) * 2
            return 0, 0.0
        except Exception as e:
            logging.debug(f"ML predict error: {e}")
            return 0, 0.0

    @staticmethod
    def _wavelet(df: pd.DataFrame) -> Tuple[int, float]:
        if not HAS_SKLEARN or len(df) < 100:
            return 0, 0.0
        try:
            c = df["close"].values
            b, a = butter(2, 0.15)
            smooth = filtfilt(b, a, c)
            ss = pd.Series(smooth)
            ef = ss.ewm(span=10).mean()
            es = ss.ewm(span=30).mean()
            diff = ef.iloc[-1] - es.iloc[-1]
            slope = ef.diff(3).iloc[-1]
            a_val = Indicators.atr_last(df)
            if a_val <= 0:
                return 0, 0.0
            conf = min(1.0, abs(diff) / a_val)
            if diff > 0 and slope > 0:
                return 1, conf
            if diff < 0 and slope < 0:
                return -1, conf
            return 0, 0.0
        except Exception:
            return 0, 0.0

    def _build_registry(self) -> List[Tuple[str, Callable, float]]:
        registry: List[Tuple[str, Callable, float]] = [
            ("EMA_Cross", self._ema_cross, 1.0),
            ("EMA_Trend", self._ema_trend, 1.0),
            ("Supertrend", self._supertrend, 1.2),
            ("Donchian", self._donchian, 0.8),
            ("Kalman", self._kalman, 1.0),
            ("MACD", self._macd, 1.2),
            ("RSI", self._rsi, 1.0),
            ("Stochastic", self._stochastic, 0.9),
            ("BollingerBB", self._bb, 0.9),
            ("Ichimoku", self._ichimoku, 1.3),
            ("SMC", self._smc, 1.5),
            ("Wavelet", self._wavelet, 0.8),
        ]
        if self.cfg.enable_ml_models:
            registry.append(("ML_RF", lambda df: self._ml_predict(df), 1.0))
        return registry

    def _regime(self, h1: pd.DataFrame) -> Tuple[str, float, float]:
        H = Indicators.hurst(h1["close"])
        E = Indicators.shannon_entropy(h1["close"])
        if H > self.cfg.hurst_trend_min and E < self.cfg.entropy_chaos_max:
            return "TREND", H, E
        if H < self.cfg.hurst_meanrev_max:
            return "MEANREV", H, E
        return "CHAOS", H, E

    @staticmethod
    def _htf_trend(df: pd.DataFrame) -> str:
        if len(df) < 60:
            return "NEUTRAL"
        c = df["close"]
        ef, es = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if c.iloc[-1] > es.iloc[-1] and ef.iloc[-1] > es.iloc[-1]:
            return "BULLISH"
        if c.iloc[-1] < es.iloc[-1] and ef.iloc[-1] < es.iloc[-1]:
            return "BEARISH"
        return "NEUTRAL"

    def generate(self, m5: pd.DataFrame, m15: pd.DataFrame,
                 h1: pd.DataFrame, h4: pd.DataFrame) -> SignalResult:
        res = SignalResult()
        adx_val = Indicators.adx_last(m5)
        res.adx = adx_val
        if adx_val < self.cfg.adx_min:
            res.skip_reason = f"ADX {adx_val:.1f} < {self.cfg.adx_min}"
            return res

        regime, H, E = self._regime(h1)
        res.regime, res.hurst, res.entropy = regime, H, E
        if regime == "CHAOS":
            res.skip_reason = f"CHAOS regime (H={H:.2f} E={E:.2f})"
            return res

        h4_trend = self._htf_trend(h4)

        buy_w = sell_w = 0.0
        details = []
        active = 0
        for name, fn, weight in self._model_registry:
            try:
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

        res.buy_score = round(buy_w, 2)
        res.sell_score = round(sell_w, 2)
        res.active_models = active
        res.votes_detail = details
        total = buy_w + sell_w
        if total < 1e-6:
            res.skip_reason = "all models neutral"
            return res
        res.buy_ratio = buy_w / total
        res.sell_ratio = sell_w / total

        if active < self.cfg.parliament_min_active:
            res.skip_reason = f"active {active} < {self.cfg.parliament_min_active}"
            return res

        dom = max(res.buy_ratio, res.sell_ratio)
        if dom >= self.cfg.parliament_full_ratio:
            res.verdict = "EXECUTE_FULL"
        elif dom >= self.cfg.parliament_half_ratio:
            res.verdict = "EXECUTE_HALF"
        elif dom >= self.cfg.parliament_min_ratio:
            res.verdict = "EXECUTE_MICRO"
        else:
            res.skip_reason = f"consensus {dom:.0%} too weak"
            return res

        if res.buy_ratio > res.sell_ratio:
            if h4_trend == "BEARISH":
                res.skip_reason = "BUY blocked: H4 bearish"
                return res
            res.direction = "BUY"
        else:
            if h4_trend == "BULLISH":
                res.skip_reason = "SELL blocked: H4 bullish"
                return res
            res.direction = "SELL"
        return res

    @staticmethod
    def detect_order_block(df: pd.DataFrame, side: str, lookback: int = 40) -> Optional[dict]:
        if len(df) < lookback + 5:
            return None
        d = df.tail(lookback + 5).reset_index(drop=True)
        a = Indicators.atr_last(df)
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
                            "low": float(d["low"].iloc[j])}
                if side == "SELL" and d["close"].iloc[j] > d["open"].iloc[j]:
                    return {"high": float(d["high"].iloc[j]),
                            "low": float(d["low"].iloc[j])}
        return None

    def compute_levels(self, m5: pd.DataFrame, side: str,
                       ob: Optional[dict]) -> Tuple[float, float, float, float, float]:
        a = Indicators.atr_last(m5)
        if a <= 0:
            raise ValueError("ATR=0")
        price = float(m5["close"].iloc[-1])
        p = self.cfg.price_decimals
        buf = self.cfg.spread_buffer * 2

        ob_h = ob["high"] if ob else price
        ob_l = ob["low"] if ob else price

        if side == "BUY":
            if ob_h < price:
                entry = ob_h - (ob_h - ob_l) * 0.5
            else:
                entry = price - a * self.cfg.entry_pullback_atr
            entry = min(entry, price - a * 0.05)
            if abs(price - entry) > self.cfg.entry_max_chase_atr * a:
                raise ValueError("entry too far (slippage)")
            sl = min(ob_l, entry) - buf
            if entry <= sl:
                raise ValueError(f"invalid BUY entry={entry} sl={sl}")
            return (
                round(entry, p), round(sl, p),
                round(entry + a * self.cfg.atr_tp1_mult, p),
                round(entry + a * self.cfg.atr_tp2_mult, p),
                round(entry + a * self.cfg.atr_tp3_mult, p),
            )
        else:
            if ob_l > price:
                entry = ob_l + (ob_h - ob_l) * 0.5
            else:
                entry = price + a * self.cfg.entry_pullback_atr
            entry = max(entry, price + a * 0.05)
            if abs(entry - price) > self.cfg.entry_max_chase_atr * a:
                raise ValueError("entry too far (slippage)")
            sl = max(ob_h, entry) + buf
            if sl <= entry:
                raise ValueError(f"invalid SELL entry={entry} sl={sl}")
            return (
                round(entry, p), round(sl, p),
                round(entry - a * self.cfg.atr_tp1_mult, p),
                round(entry - a * self.cfg.atr_tp2_mult, p),
                round(entry - a * self.cfg.atr_tp3_mult, p),
            )


# ════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ════════════════════════════════════════════════════════════
class RiskManagementEngine:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def dynamic_balance(self) -> float:
        base = self.cfg.account_balance
        journal = self.kv.get("journal", []) or []
        closed = [j for j in journal
                  if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")
                  and j.get("r_achieved") is not None]
        closed.sort(key=lambda x: x.get("timestamp", ""))
        for j in closed:
            try:
                r = float(j.get("r_achieved", 0.0))
                rp = float(j.get("risk_pct", 1.0))
            except (ValueError, TypeError):
                continue
            base += r * (base * rp / 100.0)
        return max(10.0, round(base, 2))

    def compute_lot(self, balance: float, entry: float, sl: float,
                    confidence_pct: float) -> Tuple[float, float]:
        sl_dist = abs(entry - sl)
        if sl_dist < 1e-6:
            raise ValueError("SL distance too small")
        conf_f = max(0.4, min(1.0, confidence_pct / 100.0))
        base_pct = self.cfg.max_risk_pct_per_trade
        risk_pct = max(0.1, base_pct * conf_f)
        risk_money = balance * risk_pct / 100.0
        lots_raw = risk_money / (sl_dist * self.cfg.contract_size)
        lots = max(self.cfg.min_lot, min(self.cfg.max_lot, round(lots_raw, 2)))
        notional = lots * self.cfg.contract_size * entry
        if balance > 0 and notional / balance > self.cfg.hard_leverage_cap:
            lots = max(
                self.cfg.min_lot,
                round(self.cfg.hard_leverage_cap * balance /
                      (self.cfg.contract_size * entry), 2),
            )
        actual_risk = (lots * self.cfg.contract_size * sl_dist) / balance * 100.0
        return lots, actual_risk

    def _journal(self) -> list:
        return self.kv.get("journal", []) or []

    def _today_signals_filled(self) -> int:
        today = datetime.now(UTC).date().isoformat()
        return sum(
            1 for j in self._journal()
            if j.get("timestamp", "").startswith(today) and j.get("filled")
        )

    def _daily_loss_pct(self, balance: float) -> float:
        today = datetime.now(UTC).date().isoformat()
        total = 0.0
        for j in self._journal():
            if not j.get("timestamp", "").startswith(today):
                continue
            if j.get("outcome") == "LOSS":
                r = abs(float(j.get("r_achieved", -1.0)))
                rp = float(j.get("risk_pct", 1.0))
                total += r * (rp / 100.0)
        return total * 100.0 if balance > 0 else 0.0

    def _consecutive_losses(self) -> int:
        closed = [j for j in self._journal()
                  if j.get("outcome") in ("WIN_TP3", "WIN_BE", "LOSS")]
        closed.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        n = 0
        for j in closed:
            if j["outcome"] == "LOSS":
                n += 1
            else:
                break
        return n

    def _in_drawdown_pause(self) -> bool:
        last = self.kv.get("last_loss_ts", "")
        if not last:
            return False
        try:
            t = datetime.fromisoformat(last)
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            return (datetime.now(UTC) - t) < timedelta(hours=self.cfg.dd_pause_hours)
        except Exception:
            return False

    def check_circuit_breakers(self, balance: float) -> Tuple[bool, str]:
        n = self._today_signals_filled()
        if n >= self.cfg.max_daily_signals:
            return False, f"MAX_DAILY_SIGNALS({n})"
        dl = self._daily_loss_pct(balance)
        if dl >= self.cfg.max_daily_loss_pct:
            return False, f"MAX_DAILY_LOSS({dl:.1f}%)"
        cl = self._consecutive_losses()
        if cl >= self.cfg.max_consecutive_losses:
            return False, f"MAX_CONSEC_LOSSES({cl})"
        if self._in_drawdown_pause():
            return False, "DRAWDOWN_PAUSE"
        return True, ""


# ════════════════════════════════════════════════════════════
# NEWS FILTER
# ════════════════════════════════════════════════════════════
class NewsSafetyFilter:
    def __init__(self, cfg: Config, kv: KVStore, http: HTTPClient):
        self.cfg = cfg
        self.kv = kv
        self.http = http

    def is_blackout(self) -> Tuple[bool, str]:
        if self.cfg.news_blackout_min <= 0:
            return False, ""

        cached = self.kv.get("news_cache", {}) or {}
        now_ts = time.time()
        ttl = (self.cfg.news_cache_sec if cached.get("ok")
               else self.cfg.news_cache_sec_on_fail)
        if now_ts - cached.get("ts", 0) < ttl:
            return bool(cached.get("blackout")), cached.get("title", "")

        fail_count = int(self.kv.get("news_fail_count", 0))
        try:
            r = self.http.request(
                "GET", "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            )
            events = r.json()
            now = datetime.now(UTC)
            blackout, title = False, ""
            for ev in events:
                if ev.get("country") != "USD" or ev.get("impact") != "High":
                    continue
                try:
                    ds = ev["date"].replace("Z", "+00:00")
                    evt = datetime.fromisoformat(ds)
                    if evt.tzinfo is None:
                        evt = evt.replace(tzinfo=UTC)
                except Exception:
                    continue
                if abs((evt - now).total_seconds()) / 60 <= self.cfg.news_blackout_min:
                    blackout, title = True, ev.get("title", "High impact")
                    break
            self.kv.set("news_cache", {
                "ts": now_ts, "ok": True,
                "blackout": blackout, "title": title,
            })
            self.kv.set("news_fail_count", 0)
            return blackout, title
        except Exception as e:
            fail_count += 1
            self.kv.set("news_fail_count", fail_count)
            self.kv.set("news_cache", {
                "ts": now_ts, "ok": False,
                "blackout": False, "title": "",
            })
            logging.warning(f"news fetch failed ({fail_count}): {redact(e)}")
            if fail_count >= self.cfg.news_max_consecutive_fail:
                return True, "news_feed_failure_guard"
            return False, ""


# ════════════════════════════════════════════════════════════
# TELEGRAM
# ════════════════════════════════════════════════════════════
class TelegramNotifier:
    def __init__(self, cfg: Config, kv: KVStore, http: HTTPClient):
        self.cfg = cfg
        self.kv = kv
        self.http = http
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    def send(self, msg: str) -> bool:
        if self.cfg.dry_run:
            msg = "🧪 <b>DRY RUN (v18)</b>\n" + msg
        if not (self.token and self.chat):
            if self.cfg.dry_run:
                logging.info(f"[DRY-TG] {redact(msg)[:300]}")
                return True
            logging.warning("Telegram not configured")
            return False

        digest = hashlib.sha256(msg.encode()).hexdigest()[:16]
        now = datetime.now(UTC)
        dedup = self.kv.get("tg_hashes", {}) or {}
        dedup = {k: v for k, v in dedup.items() if self._recent(v, now, 3600)}
        if digest in dedup:
            return True
        try:
            r = self.http.request(
                "POST", f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat, "text": msg, "parse_mode": "HTML"},
            )
            if r.json().get("ok"):
                dedup[digest] = now.isoformat()
                self.kv.set("tg_hashes", dedup)
                return True
        except Exception as e:
            logging.error(f"TG send: {redact(e)}")
        return False

    @staticmethod
    def _recent(iso_ts: str, now: datetime, sec: int) -> bool:
        try:
            return (now - datetime.fromisoformat(iso_ts)).total_seconds() < sec
        except Exception:
            return False


# ════════════════════════════════════════════════════════════
# OUTCOME TRACKER
# ════════════════════════════════════════════════════════════
class OutcomeTracker:
    def __init__(self, cfg: Config, kv: KVStore):
        self.cfg = cfg
        self.kv = kv

    def update(self, m5: pd.DataFrame) -> List[Tuple[str, dict]]:
        events = []
        now = datetime.now(UTC)
        journal = self.kv.get("journal", []) or []
        changed = False

        for entry in journal:
            if entry.get("outcome") not in ("PENDING", None):
                continue
            anchor = entry.get("signal_bar_dt") or entry.get("timestamp")
            try:
                ts = pd.Timestamp(anchor, tz="UTC")
            except Exception:
                continue

            if (now - ts.to_pydatetime()) > timedelta(hours=self.cfg.pending_max_age_hours):
                entry["outcome"] = "STALE"
                entry["r_achieved"] = 0.0
                events.append(("STALE", dict(entry)))
                changed = True
                continue

            side = entry["signal"]
            ep = float(entry["entry"])
            sl = float(entry["sl"])
            tp1 = float(entry["tp1"])
            tp2 = float(entry["tp2"])
            tp3 = float(entry["tp3"])
            if ep <= 0 or sl <= 0:
                continue

            bars = m5[m5["datetime"] > ts].sort_values("datetime")
            if bars.empty:
                continue

            filled = bool(entry.get("filled"))
            fill_idx: Optional[int] = None
            if not filled:
                for idx, b in enumerate(bars.itertuples(index=False)):
                    if side == "BUY" and b.low <= ep:
                        filled, fill_idx = True, idx
                        break
                    if side == "SELL" and b.high >= ep:
                        filled, fill_idx = True, idx
                        break
                if not filled:
                    last_high = float(bars["high"].iloc[-1])
                    last_low = float(bars["low"].iloc[-1])
                    limit = self.cfg.entry_max_chase_atr * abs(ep - sl)
                    if side == "BUY" and last_high > ep + limit:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0
                        entry["r_achieved"] = 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        changed = True
                        continue
                    if side == "SELL" and last_low < ep - limit:
                        entry["outcome"] = "INVALIDATED"
                        entry["filled"] = 0
                        entry["r_achieved"] = 0.0
                        events.append(("INVALIDATED", dict(entry)))
                        changed = True
                        continue
                    continue
                entry["filled"] = 1
                changed = True

            walk = bars.iloc[fill_idx:] if fill_idx is not None else bars

            h1_ = bool(entry.get("hit_tp1"))
            h2 = bool(entry.get("hit_tp2"))
            h3 = bool(entry.get("hit_tp3"))
            hs = bool(entry.get("hit_sl"))
            hb = bool(entry.get("hit_be"))
            closed_here = False

            for b in walk.itertuples(index=False):
                if h3 or hs or hb:
                    break
                csl = ep if h1_ else sl
                if side == "BUY":
                    if b.low <= csl:
                        if h1_:
                            hb = True
                        else:
                            hs = True
                        continue
                    if not h1_ and b.high >= tp1:
                        h1_ = True
                    if not h2 and b.high >= tp2:
                        h2 = True
                    if not h3 and b.high >= tp3:
                        h3 = True
                else:
                    if b.high >= csl:
                        if h1_:
                            hb = True
                        else:
                            hs = True
                        continue
                    if not h1_ and b.low <= tp1:
                        h1_ = True
                    if not h2 and b.low <= tp2:
                        h2 = True
                    if not h3 and b.low <= tp3:
                        h3 = True

            entry["hit_tp1"] = int(h1_)
            entry["hit_tp2"] = int(h2)
            entry["hit_tp3"] = int(h3)
            entry["hit_sl"] = int(hs)
            entry["hit_be"] = int(hb)

            if entry.get("outcome") == "PENDING":
                if h3:
                    entry["outcome"] = "WIN_TP3"
                elif h2 or hb:
                    entry["outcome"] = "WIN_BE"
                elif hs:
                    entry["outcome"] = "LOSS"
                if entry["outcome"] != "PENDING":
                    dist = abs(ep - sl)
                    r1 = abs(tp1 - ep) / dist
                    r2 = abs(tp2 - ep) / dist
                    r3 = abs(tp3 - ep) / dist
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
                    events.append(("CLOSE", dict(entry)))
                    changed = True
                    closed_here = True
                    if entry["outcome"] == "LOSS":
                        self.kv.set("last_loss_ts", now.isoformat())

            if not closed_here and (h1_ or h2 or h3 or hs or hb):
                changed = True

        if changed:
            self.kv.set("journal", journal)
        return events


# ════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════
class TradingBotOrchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.http = HTTPClient(cfg)
        self.kv = KVStore(cfg, self.http)
        self.data = MarketDataHandler(cfg, self.http, self.kv)
        self.signal = SignalGenerator(cfg)
        self.risk = RiskManagementEngine(cfg, self.kv)
        self.news = NewsSafetyFilter(cfg, self.kv, self.http)
        self.tg = TelegramNotifier(cfg, self.kv, self.http)
        self.outcome = OutcomeTracker(cfg, self.kv)

    def _in_killzone(self, now: datetime) -> bool:
        if self.cfg.skip_killzone_filter:
            return True
        ld = now.astimezone(LONDON)
        ny = now.astimezone(NY)
        lh = ld.hour + ld.minute / 60.0
        nh = ny.hour + ny.minute / 60.0
        return (7.0 <= lh < 10.0) or (8.0 <= nh < 11.0) or (13.0 <= nh < 16.0)

    def run(self) -> None:
        try:
            self.kv.force_load()
        except StateError as e:
            logging.error(f"state load aborted: {e}")
            return

        now = datetime.now(UTC)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        if self.cfg.kill_switch:
            logging.info("KILL switch engaged")
            return
        if now.weekday() >= 5:
            logging.info("weekend — skip")
            return
        if not self._in_killzone(now):
            logging.info("outside killzone — skip")
            return

        balance = self.risk.dynamic_balance()

        ok, reason = self.risk.check_circuit_breakers(balance)
        if not ok:
            logging.info(f"circuit-breaker: {reason}")
            return

        blackout, title = self.news.is_blackout()
        if blackout:
            logging.info(f"news blackout: {title}")
            return

        try:
            dfs, src, offset = self.data.fetch_all()
        except Exception as e:
            logging.error(f"data fetch: {redact(e)}")
            self.tg.send(f"<b>⚠️ Data unavailable</b>\n<code>{redact(e)[:200]}</code>")
            return

        try:
            events = self.outcome.update(dfs["5min"])
            for kind, ent in events:
                if kind == "CLOSE":
                    self.tg.send(
                        f"<b>📕 Closed</b> {ent['signal']} {ent['symbol']}\n"
                        f"Entry: <code>{ent['entry']}</code> → {ent['outcome']} "
                        f"({ent.get('r_achieved', 0):+.2f}R)"
                    )
                elif kind in ("INVALIDATED", "STALE"):
                    self.tg.send(
                        f"<b>⚪ {kind}</b> {ent['signal']} @ {ent['entry']}"
                    )
        except Exception as e:
            logging.warning(f"outcome update: {e}")

        balance = self.risk.dynamic_balance()

        m5c = dfs["5min"].iloc[:-1].reset_index(drop=True)
        m15c = dfs["15min"].iloc[:-1].reset_index(drop=True)
        h1c = dfs["1h"].iloc[:-1].reset_index(drop=True)
        h4c = dfs["4h"].iloc[:-1].reset_index(drop=True)
        if len(m5c) < self.cfg.min_bars_m5 or len(h1c) < self.cfg.min_bars_h1:
            logging.warning("post-trim insufficient bars")
            return
        price = float(m5c["close"].iloc[-1])
        signal_bar_iso = m5c.iloc[-1]["datetime"].strftime("%Y-%m-%d %H:%M:%S")

        journal = self.kv.get("journal", []) or []
        if any(j.get("outcome") == "PENDING" and j.get("filled") for j in journal):
            logging.info("open filled trade — skip new signal")
            return
        pending_unfilled = sum(
            1 for j in journal
            if j.get("outcome") == "PENDING" and not j.get("filled")
        )
        if pending_unfilled >= 2:
            logging.info(f"too many unfilled pendings ({pending_unfilled})")
            return

        result = self.signal.generate(m5c, m15c, h1c, h4c)
        logging.info(
            f"parliament {result.direction} "
            f"BUY={result.buy_ratio:.0%} SELL={result.sell_ratio:.0%} "
            f"verdict={result.verdict} skip={result.skip_reason}"
        )

        if result.direction == "NEUTRAL" or result.verdict == "CANCEL":
            self._update_state(price, now_str)
            return

        state = self.kv.get("state", {}) or {}
        last_type = state.get("last_signal_type", "NEUTRAL")
        last_ts = state.get("last_signal_ts", "")
        cool = (self.cfg.opposite_cooldown_minutes
                if last_type != result.direction
                else self.cfg.cooldown_minutes)
        if last_ts:
            try:
                last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                if (now - last_dt) < timedelta(minutes=cool):
                    logging.info(f"cooldown {cool}min not elapsed")
                    self._update_state(price, now_str)
                    return
            except Exception:
                pass

        ob = self.signal.detect_order_block(m5c, result.direction)
        try:
            entry, sl, tp1, tp2, tp3 = self.signal.compute_levels(
                m5c, result.direction, ob
            )
        except ValueError as e:
            logging.warning(f"level computation: {e}")
            self._update_state(price, now_str)
            return

        conf_pct = max(result.buy_ratio, result.sell_ratio) * 100.0
        try:
            lot, actual_risk = self.risk.compute_lot(balance, entry, sl, conf_pct)
        except ValueError as e:
            logging.warning(f"lot sizing: {e}")
            self._update_state(price, now_str)
            return

        if result.verdict == "EXECUTE_HALF":
            lot = max(self.cfg.min_lot, round(lot / 2.0, 2))
        elif result.verdict == "EXECUTE_MICRO":
            lot = max(self.cfg.min_lot, round(lot / 4.0, 2))

        msg = (
            f"<b>🏛️ APEX v{VERSION} — {result.direction} (LIMIT)</b>\n\n"
            f"<b>{self.cfg.symbol}</b> limit @ <code>{entry}</code>\n"
            f"SL: <code>{sl}</code> | TP1: <code>{tp1}</code>\n"
            f"TP2: <code>{tp2}</code> | TP3: <code>{tp3}</code>\n"
            f"Lot: <code>{lot}</code> (Bal: ${balance:.2f}, "
            f"risk: {actual_risk:.2f}%)\n"
            f"Verdict: {result.verdict} | "
            f"Vote: BUY={result.buy_ratio:.0%} SELL={result.sell_ratio:.0%}\n"
            f"Regime: {result.regime} | ADX: {result.adx:.1f}\n"
            f"OB: "
            + (f"✅ {ob['high']:.2f}–{ob['low']:.2f}" if ob else "⚠️ none")
            + "\n\n<i>Pasang limit order di MT5 pada harga di atas.</i>"
        )

        jid = f"{now_str.replace(' ', 'T')}_{result.direction}_{lot}"
        entry_data = {
            "id": jid,
            "timestamp": now_str,
            "signal_bar_dt": signal_bar_iso,
            "symbol": self.cfg.symbol,
            "signal": result.direction,
            "entry": entry, "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "lot": lot,
            "verdict": result.verdict,
            "outcome": "PENDING",
            "filled": 0,
            "r_achieved": 0.0,
            "risk_pct": round(actual_risk, 3),
        }

        if self._journal_append(entry_data):
            if self.tg.send(msg):
                state["last_signal_type"] = result.direction
                state["last_signal_ts"] = now_str
                self.kv.set("state", state)
                logging.info(f"signal sent {result.direction} @ {entry} lot={lot}")
            else:
                logging.warning("journal saved but TG failed")

        self._update_state(price, now_str)

    def _update_state(self, price: float, now_str: str) -> None:
        state = self.kv.get("state", {}) or {}
        state["run_count"] = int(state.get("run_count", 0)) + 1
        state["last_price"] = price
        state["last_price_ts"] = now_str
        self.kv.set("state", state)

    def _journal_append(self, entry: dict) -> bool:
        if self.cfg.dry_run:
            logging.info(
                f"[DRY] journal skip: {entry.get('id')} "
                f"{entry.get('signal')} @ {entry.get('entry')} "
                f"lot={entry.get('lot')}"
            )
            return True
        j = list(self.kv.get("journal", []) or [])
        j = [e for e in j if e.get("id") != entry.get("id")]
        j.append(entry)
        j.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return self.kv.set("journal", j[: self.cfg.max_journal])


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════
def cmd_smoke(cfg: Config) -> None:
    http = HTTPClient(cfg)
    kv = KVStore(cfg, http)
    data = MarketDataHandler(cfg, http, kv)
    signal = SignalGenerator(cfg)
    dfs, src, offset = data.fetch_all()
    m5c = dfs["5min"].iloc[:-1].reset_index(drop=True)
    m15c = dfs["15min"].iloc[:-1].reset_index(drop=True)
    h1c = dfs["1h"].iloc[:-1].reset_index(drop=True)
    h4c = dfs["4h"].iloc[:-1].reset_index(drop=True)
    print(f"Source: {src} | Offset: {offset:+.3f}")
    print(f"Bars: M5={len(m5c)} M15={len(m15c)} H1={len(h1c)} H4={len(h4c)}")
    print(f"Last M5: {m5c.iloc[-1]['datetime']} close={m5c.iloc[-1]['close']:.2f}\n")
    res = signal.generate(m5c, m15c, h1c, h4c)
    print("=== PARLIAMENT ===")
    for v in res.votes_detail:
        print(f"  {v['model']:16s} {v['vote']:8s} w={v['weight']} "
              f"conf={v['conf']} score={v['score']}")
    print(f"\nBUY: {res.buy_score} ({res.buy_ratio:.0%}) | "
          f"SELL: {res.sell_score} ({res.sell_ratio:.0%})")
    print(f"Active: {res.active_models} | Dir: {res.direction} | Verdict: {res.verdict}")
    print(f"ADX: {res.adx:.1f} | Regime: {res.regime}")
    if res.skip_reason:
        print(f"Skip: {res.skip_reason}")
    if res.direction != "NEUTRAL":
        ob = signal.detect_order_block(m5c, res.direction)
        try:
            e, s, t1, t2, t3 = signal.compute_levels(m5c, res.direction, ob)
            print(f"\nProposed {res.direction}: entry={e} sl={s} "
                  f"tp1={t1} tp2={t2} tp3={t3}")
        except ValueError as ex:
            print(f"  ⚠️ level error: {ex}")


def cmd_selftest(cfg: Config) -> None:
    errors = []
    if cfg.max_lot < cfg.min_lot:
        errors.append("max_lot < min_lot")
    if not (cfg.parliament_min_ratio < cfg.parliament_half_ratio < cfg.parliament_full_ratio):
        errors.append("ratio ordering")
    if cfg.account_balance <= 0:
        errors.append("account_balance")
    if not HAS_SKLEARN:
        errors.append("sklearn missing (ML disabled)")
    if errors:
        print("❌ SELFTEST FAILED:", "; ".join(errors))
        sys.exit(1)
    print(f"✅ SELFTEST PASSED (v{VERSION} Production)")


def main() -> None:
    setup_logging()
    try:
        cfg = Config.from_env()
    except Exception as e:
        print(f"CONFIG ERROR: {e}")
        sys.exit(2)

    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "run"

    if cmd == "smoke":
        cmd_smoke(cfg)
        return
    if cmd == "selftest":
        cmd_selftest(cfg)
        return
    if cmd == "test-tg":
        http = HTTPClient(cfg)
        kv = KVStore(cfg, http)
        tg = TelegramNotifier(cfg, kv, http)
        ok = tg.send(f"<b>🔔 TEST v{VERSION}</b>")
        print("SENT" if ok else "FAILED")
        return
    if cmd == "help":
        print("commands: selftest|test-tg|smoke|run")
        return

    try:
        TradingBotOrchestrator(cfg).run()
    except Exception as e:
        logging.exception("FATAL")
        try:
            http = HTTPClient(cfg)
            kv = KVStore(cfg, http)
            tg = TelegramNotifier(cfg, kv, http)
            tg.send(f"<b>💥 APEX CRASH</b>\n<code>{redact(e)[:400]}</code>")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
