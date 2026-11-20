#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APEX-QUANT-OMNI — HYPERION PRECISION EDITION (BTCUSD & ETHUSD DUAL-CRYPTO)
Dual-Asset, Dual-Mode (SCALP & SWING), Dual-Tier (PRO & SNIPER) Signal Engine
with Multi-Exchange Failover, 24/7 Crypto Regime & Weekend Liquidity Guard,
Cross-Crypto BTC/ETH Correlation Veto, Walk-Forward OOS + L2 Calibration,
Percentage/ATR-Scaled Risk & Auto-Trailing SL, and 2-Way Telegram Bot.
"""

import os
import sys
import time
import math
import json
import sqlite3
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ==============================================================================
# LOGGING & GLOBAL CONSTANTS
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger("APEX_QUANT_CRYPTO")

DB_PATH = os.getenv("DB_PATH", "trading_signals.db")
CHART_PATH = os.getenv("CHART_PATH", "backtest_performance.png")

MIN_LOT = 0.01
LOT_STEP = 0.01
MAX_LOT = float(os.getenv("MAX_LOT", "25.0"))

DEFAULT_ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000.0"))
DEFAULT_RISK_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
SIGNAL_TIER_MODE = os.getenv("SIGNAL_TIER_MODE", "BOTH").strip().upper()

# Realistic Crypto Spread & Slippage as percentage of spot price (0.03% = ~$27 on $90k BTC, ~$0.90 on $3k ETH)
CRYPTO_SPREAD_PCT = float(os.getenv("CRYPTO_SPREAD_PCT", "0.0003"))
CRYPTO_SLIPPAGE_PCT = float(os.getenv("CRYPTO_SLIPPAGE_PCT", "0.0001"))

RETRY_BACKOFFS = [2, 4, 8]
CIRCUIT_MAX_FAILURES = 5
CIRCUIT_COOLDOWN_MINUTES = 60

GRID_ATR_PERIODS = [10, 14, 20]
GRID_Z_THRESHOLDS = [1.5, 2.0, 2.5]


@dataclass(frozen=True)
class AssetSpec:
    symbol: str
    coinbase_pair: str
    kraken_pair: str
    yf_ticker: str
    contract_size: float
    pip_size: float
    price_decimals: int
    price_offset: float
    min_valid_price: float


BTC_SPEC = AssetSpec(
    symbol="BTCUSD",
    coinbase_pair="BTC-USD",
    kraken_pair="XBTUSD",
    yf_ticker="BTC-USD",
    contract_size=float(os.getenv("BTC_CONTRACT_SIZE", "1.0")),
    pip_size=float(os.getenv("BTC_PIP_SIZE", "1.0")),
    price_decimals=2,
    price_offset=float(os.getenv("BTC_PRICE_OFFSET", "0.0")),
    min_valid_price=1000.0,
)

ETH_SPEC = AssetSpec(
    symbol="ETHUSD",
    coinbase_pair="ETH-USD",
    kraken_pair="ETHUSD",
    yf_ticker="ETH-USD",
    contract_size=float(os.getenv("ETH_CONTRACT_SIZE", "1.0")),
    pip_size=float(os.getenv("ETH_PIP_SIZE", "0.01")),
    price_decimals=2,
    price_offset=float(os.getenv("ETH_PRICE_OFFSET", "0.0")),
    min_valid_price=50.0,
)

ASSETS = [BTC_SPEC, ETH_SPEC]


@dataclass(frozen=True)
class ModeConfig:
    mode_name: str
    exec_tf: str
    mid_tf: str
    htf_tf: str
    sl_atr_mult: float
    tp_r_multiples: Tuple[float, float, float]
    tp1_close_pct: float
    trail_atr_mult: float
    min_score: float
    sniper_score: float
    min_ker: float
    max_chop: float
    min_adx: float
    cooldown_minutes: int
    min_backtest_trades: int
    z_window: int
    vwap_window: int
    swing_lr: int
    stale_max_minutes: int
    min_sl_pct: float
    use_swing_anchor_sl: bool
    bos_lookback: int


SCALP_CONFIG = ModeConfig(
    mode_name="SCALP",
    exec_tf="5m",
    mid_tf="15m",
    htf_tf="1h",
    sl_atr_mult=1.65,
    tp_r_multiples=(1.25, 2.00, 3.00),
    tp1_close_pct=0.55,
    trail_atr_mult=0.95,
    min_score=76.0,
    sniper_score=86.0,
    min_ker=0.33,
    max_chop=56.0,
    min_adx=19.5,
    cooldown_minutes=30,
    min_backtest_trades=10,
    z_window=20,
    vwap_window=48,
    swing_lr=3,
    stale_max_minutes=20,
    min_sl_pct=0.0022,
    use_swing_anchor_sl=False,
    bos_lookback=14,
)

SWING_CONFIG = ModeConfig(
    mode_name="SWING",
    exec_tf="15m",
    mid_tf="1h",
    htf_tf="4h",
    sl_atr_mult=1.90,
    tp_r_multiples=(1.50, 2.50, 4.00),
    tp1_close_pct=0.50,
    trail_atr_mult=0.85,
    min_score=75.0,
    sniper_score=86.0,
    min_ker=0.34,
    max_chop=56.0,
    min_adx=20.0,
    cooldown_minutes=120,
    min_backtest_trades=6,
    z_window=24,
    vwap_window=32,
    swing_lr=4,
    stale_max_minutes=45,
    min_sl_pct=0.0045,
    use_swing_anchor_sl=True,
    bos_lookback=20,
)


@dataclass
class SignalCandidate:
    symbol: str
    mode: str
    direction: str
    tier: str = "NONE"
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    lot_size: float = 0.01
    risk_usd: float = 0.0
    score: float = 0.0
    p_win: float = 0.0
    ev_r: float = 0.0
    ker: float = 0.0
    adx: float = 0.0
    z_score: float = 0.0
    atr: float = 0.0
    regime: str = "UNKNOWN"
    mtf_bias: str = "NEUTRAL"
    cross_ctx: str = "NEUTRAL"
    candle_ts: str = ""
    reason: str = ""
    pillar_details: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# UTILITY FUNCTIONS (24/7 CRYPTO SESSION GUARD & MT5 LOT SIZING)
# ==============================================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def check_crypto_session(now_utc: Optional[datetime] = None) -> Tuple[bool, bool, str]:
    """
    Crypto is 24/7. Returns (is_tradeable, is_weekend_thin, session_tag).
    Blocks 23:56 - 00:05 UTC daily close/funding settlement spike window.
    """
    dt = now_utc or utc_now()
    mins = dt.hour * 60 + dt.minute
    if mins >= (23 * 60 + 56) or mins <= 5:
        return False, False, "DAILY_FUNDING_ROLLOVER"
    is_weekend = dt.weekday() in (5, 6)
    if is_weekend:
        return True, True, "WEEKEND_24_7"
    if 12 <= dt.hour < 21:
        return True, False, "US_PEAK_24_7"
    if 7 <= dt.hour < 12:
        return True, False, "EU_ACTIVE_24_7"
    return True, False, "ASIA_24_7"


def calculate_crypto_lot_size(
    entry_price: float,
    sl_price: float,
    spec: AssetSpec,
    account_balance: float = DEFAULT_ACCOUNT_BALANCE,
    risk_pct: float = DEFAULT_RISK_PCT,
) -> float:
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 1e-6 or account_balance <= 0 or risk_pct <= 0:
        return MIN_LOT
    risk_usd = account_balance * (risk_pct / 100.0)
    sl_pips = sl_distance / spec.pip_size
    tick_val = spec.contract_size * spec.pip_size
    raw_lot = risk_usd / (sl_pips * tick_val + 1e-8)
    rounded_lot = math.floor(raw_lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(MAX_LOT, round(rounded_lot, 2)))


# ==============================================================================
# SQLITE STATE MANAGER (PER-SYMBOL & PER-MODE PERSISTENCE)
# ==============================================================================
class StateManager:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL DEFAULT 'BTCUSD',
                    mode TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry REAL NOT NULL,
                    initial_sl REAL NOT NULL,
                    current_sl REAL NOT NULL,
                    tp1 REAL NOT NULL,
                    tp2 REAL NOT NULL,
                    tp3 REAL NOT NULL,
                    lot_size REAL NOT NULL,
                    score REAL NOT NULL,
                    p_win REAL NOT NULL,
                    ev_r REAL NOT NULL,
                    atr REAL NOT NULL,
                    regime TEXT NOT NULL,
                    status TEXT NOT NULL,
                    candle_ts TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(signals)").fetchall()]
            if "symbol" not in cols:
                conn.execute("ALTER TABLE signals ADD COLUMN symbol TEXT NOT NULL DEFAULT 'BTCUSD'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_sym_mode_ts ON signals(symbol, mode, candle_ts)")
            conn.commit()

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        now_s = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, now_s),
            )
            conn.commit()

    def is_circuit_open(self) -> Tuple[bool, str]:
        open_until = self.get_state("circuit_open_until")
        if not open_until:
            return False, ""
        try:
            if utc_now() < datetime.fromisoformat(open_until):
                return True, open_until
        except ValueError:
            pass
        self.set_state("circuit_open_until", "")
        self.set_state("circuit_failures", "0")
        return False, ""

    def record_failure(self) -> Tuple[int, bool]:
        failures = int(self.get_state("circuit_failures", "0") or "0") + 1
        self.set_state("circuit_failures", str(failures))
        tripped = False
        if failures >= CIRCUIT_MAX_FAILURES:
            self.set_state("circuit_open_until", (utc_now() + timedelta(minutes=CIRCUIT_COOLDOWN_MINUTES)).isoformat())
            tripped = True
        return failures, tripped

    def record_success(self) -> None:
        self.set_state("circuit_failures", "0")
        self.set_state("circuit_open_until", "")
        self.set_state("last_healthy_run", utc_now().isoformat())

    def is_duplicate_or_cooldown(
        self, symbol: str, mode: str, direction: str, candle_ts: str, cooldown_minutes: int, score: float
    ) -> Tuple[bool, str]:
        with self._connect() as conn:
            dup = conn.execute(
                "SELECT id FROM signals WHERE symbol = ? AND mode = ? AND direction = ? AND candle_ts = ?",
                (symbol, mode, direction, candle_ts),
            ).fetchone()
            if dup:
                return True, f"DUPLICATE_CANDLE_{candle_ts}"

            last_sig = conn.execute(
                "SELECT direction, created_at FROM signals WHERE symbol = ? AND mode = ? ORDER BY id DESC LIMIT 1",
                (symbol, mode),
            ).fetchone()
            if last_sig:
                elapsed_min = (utc_now() - datetime.fromisoformat(last_sig["created_at"])).total_seconds() / 60.0
                if elapsed_min < cooldown_minutes:
                    return True, f"COOLDOWN_ACTIVE_{elapsed_min:.1f}m/{cooldown_minutes}m"
                if last_sig["direction"] != direction and elapsed_min < (cooldown_minutes * 2.0) and score < 88.0:
                    return True, f"ANTI_WHIPSAW_FLIP_GUARD_{elapsed_min:.1f}m"
        return False, ""

    def insert_signal(self, sig: SignalCandidate) -> int:
        now_s = utc_now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO signals (
                    symbol, mode, direction, entry, initial_sl, current_sl,
                    tp1, tp2, tp3, lot_size, score, p_win, ev_r,
                    atr, regime, status, candle_ts, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
                """,
                (
                    sig.symbol, sig.mode, sig.direction, sig.entry, sig.sl, sig.sl,
                    sig.tp1, sig.tp2, sig.tp3, sig.lot_size, sig.score,
                    sig.p_win, sig.ev_r, sig.atr, sig.regime, sig.candle_ts,
                    now_s, now_s,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_active_signals(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM signals WHERE status IN ('OPEN', 'BE_LOCKED', 'TP1_TRAIL', 'TP2_TRAIL') ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_signals(self, limit: int = 6) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def update_signal_trail(self, sig_id: int, new_sl: float, new_status: str) -> None:
        now_s = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE signals SET current_sl = ?, status = ?, updated_at = ? WHERE id = ?",
                (round(new_sl, 2), new_status, now_s, sig_id),
            )
            conn.commit()


# ==============================================================================
# NOTIFICATION & INTERACTIVE TELEGRAM POLLING (EXPONENTIAL BACKOFF)
# ==============================================================================
class Notifier:
    def __init__(self, state: StateManager) -> None:
        self.state = state
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        raw_chat_ids = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.authorized_chat_ids = [c.strip() for c in raw_chat_ids.split(",") if c.strip()]
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    def _request_with_backoff(self, method: str, url: str, **kwargs: Any) -> Optional[requests.Response]:
        kwargs.setdefault("timeout", 12)
        for attempt in range(len(RETRY_BACKOFFS) + 1):
            try:
                resp = requests.request(method, url, **kwargs)
                if resp.status_code in (200, 201, 204):
                    return resp
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < len(RETRY_BACKOFFS):
                        wait_s = RETRY_BACKOFFS[attempt]
                        if resp.status_code == 429:
                            try:
                                retry_after = int(resp.json().get("parameters", {}).get("retry_after", wait_s))
                                wait_s = min(max(wait_s, retry_after), 20)
                            except Exception:
                                pass
                        logger.warning(f"HTTP {resp.status_code}; retrying in {wait_s}s")
                        time.sleep(wait_s)
                        continue
                logger.error(f"HTTP {resp.status_code} error: {resp.text[:250]}")
                return resp
            except requests.RequestException as exc:
                if attempt < len(RETRY_BACKOFFS):
                    time.sleep(RETRY_BACKOFFS[attempt])
                else:
                    logger.error(f"Network delivery failed permanently: {exc}")
        return None

    def send_telegram(self, html_text: str, chat_id: Optional[str] = None) -> bool:
        if not self.tg_token or not self.authorized_chat_ids:
            logger.info("Telegram credentials absent; local message:\n" + html_text)
            return False
        targets = [chat_id] if chat_id else self.authorized_chat_ids
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        ok = True
        for cid in targets:
            resp = self._request_with_backoff(
                "POST",
                url,
                json={"chat_id": cid, "text": html_text, "parse_mode": "HTML", "disable_web_page_preview": True},
            )
            if resp is None or resp.status_code != 200:
                ok = False
        return ok

    def send_telegram_photo(self, image_path: str, caption: str = "", chat_id: Optional[str] = None) -> bool:
        if not self.tg_token or not self.authorized_chat_ids or not os.path.exists(image_path):
            return False
        targets = [chat_id] if chat_id else self.authorized_chat_ids
        url = f"https://api.telegram.org/bot{self.tg_token}/sendPhoto"
        ok = True
        for cid in targets:
            with open(image_path, "rb") as img_file:
                resp = self._request_with_backoff(
                    "POST",
                    url,
                    data={"chat_id": cid, "caption": caption[:1024], "parse_mode": "HTML"},
                    files={"photo": img_file},
                )
            if resp is None or resp.status_code != 200:
                ok = False
        return ok

    def send_discord(self, content: str) -> None:
        if not self.discord_webhook:
            return
        clean = content.replace("<b>", "**").replace("</b>", "**").replace("<code>", "`").replace("</code>", "`")
        self._request_with_backoff("POST", self.discord_webhook, json={"content": clean})

    def broadcast(self, html_text: str, target_chat_id: Optional[str] = None) -> None:
        self.send_telegram(html_text, chat_id=target_chat_id)
        if not target_chat_id:
            self.send_discord(html_text)

    def poll_commands(self) -> List[Tuple[str, str]]:
        if not self.tg_token or not self.authorized_chat_ids:
            return []
        offset = int(self.state.get_state("telegram_update_offset", "0") or "0")
        resp = self._request_with_backoff(
            "GET",
            f"https://api.telegram.org/bot{self.tg_token}/getUpdates",
            params={"offset": offset, "timeout": 3, "allowed_updates": ["message"]},
        )
        if resp is None or resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        if not data.get("ok"):
            return []

        commands: List[Tuple[str, str]] = []
        max_uid = offset - 1
        for upd in data.get("result", []):
            uid = int(upd.get("update_id", 0))
            max_uid = max(max_uid, uid)
            msg = upd.get("message") or {}
            chat_id = str((msg.get("chat") or {}).get("id", ""))
            text = (msg.get("text") or "").strip()
            if not text or not chat_id:
                continue
            if chat_id not in self.authorized_chat_ids:
                logger.warning(f"Unauthorized Telegram command from chat_id={chat_id}")
                continue
            cmd_norm = text.lower().split("@")[0].strip()
            if any(cmd_norm.startswith(p) for p in ("/check", "/status", "/chart", "/history", "/sniper", "/help")):
                commands.append((chat_id, text))

        if max_uid >= offset:
            self.state.set_state("telegram_update_offset", str(max_uid + 1))
        return commands


# ==============================================================================
# MULTI-SOURCE CRYPTO DATA ENGINE (COINBASE -> KRAKEN -> YFINANCE -> YAHOO API)
# ==============================================================================
class CryptoDataEngine:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
    }

    @staticmethod
    def _sanitize_ohlcv(df: pd.DataFrame, min_valid_price: float, offset: float = 0.0) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        col_map = {str(c).lower(): c for c in df.columns}
        required = {}
        for std_name in ["open", "high", "low", "close"]:
            if std_name not in col_map:
                return pd.DataFrame()
            required[col_map[std_name]] = std_name.capitalize()
        if "volume" in col_map:
            required[col_map["volume"]] = "Volume"

        df = df.rename(columns=required)
        cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])
        df = df[cols].copy()
        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df["Volume"] = df["Volume"].fillna(0.0).clip(lower=0.0)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df[~df.index.isna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if offset != 0.0:
            for c in ["Open", "High", "Low", "Close"]:
                df[c] = df[c] + offset

        valid_mask = (
            (df["Open"] > min_valid_price)
            & (df["Close"] > min_valid_price)
            & (df["High"] >= df[["Open", "Close"]].max(axis=1))
            & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
            & (df["Low"] > min_valid_price)
        )
        df = df[valid_mask]
        pct_jump = df["Close"].pct_change().abs().fillna(0.0)
        bar_range_pct = ((df["High"] - df["Low"]) / (df["Close"] + 1e-8)).fillna(0.0)
        return df[(pct_jump < 0.08) & (bar_range_pct < 0.10)]

    def fetch_live_ticker(self, spec: AssetSpec) -> Dict[str, Any]:
        try:
            r = requests.get(
                f"https://api.exchange.coinbase.com/products/{spec.coinbase_pair}/ticker",
                headers=self.HEADERS,
                timeout=8,
            )
            if r.status_code == 200:
                q = r.json()
                bid = float(q.get("bid", 0.0) or 0.0) + spec.price_offset
                ask = float(q.get("ask", 0.0) or 0.0) + spec.price_offset
                px = float(q.get("price", 0.0) or 0.0) + spec.price_offset
                mid = ((bid + ask) / 2.0) if (bid > spec.min_valid_price and ask > spec.min_valid_price) else px
                if mid > spec.min_valid_price:
                    ex_sp = abs(ask - bid) if (ask > bid > spec.min_valid_price) else 0.0
                    eff_sp = max(ex_sp, mid * CRYPTO_SPREAD_PCT)
                    return {"mid": round(mid, spec.price_decimals), "spread": round(eff_sp, 2), "source": "coinbase"}
        except Exception:
            pass
        return {"mid": 0.0, "spread": 0.0, "source": "fallback"}

    def _fetch_coinbase_candles(self, spec: AssetSpec, interval: str) -> pd.DataFrame:
        gran_map = {"5m": 300, "15m": 900, "1h": 3600}
        gran = gran_map.get(interval)
        if not gran:
            return pd.DataFrame()
        try:
            resp = requests.get(
                f"https://api.exchange.coinbase.com/products/{spec.coinbase_pair}/candles",
                params={"granularity": gran},
                headers=self.HEADERS,
                timeout=12,
            )
            if resp.status_code != 200:
                return pd.DataFrame()
            rows = resp.json()
            if not isinstance(rows, list) or len(rows) < 5:
                return pd.DataFrame()
            # Coinbase format: [time, low, high, open, close, volume] (newest first)
            df = pd.DataFrame(rows, columns=["Time", "Low", "High", "Open", "Close", "Volume"])
            df.index = pd.to_datetime(df["Time"].astype(float), unit="s", utc=True)
            df = df.sort_index()
            # Drop last currently-forming candle
            df = df.iloc[:-1]
            return self._sanitize_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]], spec.min_valid_price, spec.price_offset)
        except Exception:
            return pd.DataFrame()

    def _fetch_kraken_ohlc(self, spec: AssetSpec, interval: str) -> pd.DataFrame:
        tf_map = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
        try:
            resp = requests.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": spec.kraken_pair, "interval": tf_map.get(interval, 15)},
                headers=self.HEADERS,
                timeout=12,
            )
            if resp.status_code != 200:
                return pd.DataFrame()
            data = resp.json()
            if data.get("error"):
                return pd.DataFrame()
            rows = next((v for k, v in (data.get("result") or {}).items() if k != "last" and isinstance(v, list)), None)
            if not rows or len(rows) < 5:
                return pd.DataFrame()
            df = pd.DataFrame(rows[:-1], columns=["Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"])
            df.index = pd.to_datetime(df["Time"].astype(float), unit="s", utc=True)
            return self._sanitize_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]], spec.min_valid_price, spec.price_offset)
        except Exception:
            return pd.DataFrame()

    def _fetch_yfinance(self, spec: AssetSpec, interval: str, period: str) -> pd.DataFrame:
        try:
            raw = yf.download(
                tickers=spec.yf_ticker, period=period, interval=interval,
                auto_adjust=False, progress=False, threads=False, timeout=15,
            )
            return self._sanitize_ohlcv(raw, spec.min_valid_price, spec.price_offset)
        except Exception:
            return pd.DataFrame()

    def _fetch_yahoo_chart_api(self, spec: AssetSpec, interval: str, range_str: str) -> pd.DataFrame:
        for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
            try:
                resp = requests.get(
                    f"https://{host}/v8/finance/chart/{spec.yf_ticker}",
                    params={"interval": interval, "range": range_str},
                    headers=self.HEADERS,
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                res = (resp.json().get("chart") or {}).get("result") or []
                if not res:
                    continue
                ts = res[0].get("timestamp") or []
                quote = ((res[0].get("indicators") or {}).get("quote") or [{}])[0]
                if not ts or not quote:
                    continue
                df = pd.DataFrame(
                    {
                        "Open": quote.get("open"), "High": quote.get("high"),
                        "Low": quote.get("low"), "Close": quote.get("close"),
                        "Volume": quote.get("volume") or [0.0] * len(ts),
                    },
                    index=pd.to_datetime(ts, unit="s", utc=True),
                )
                clean = self._sanitize_ohlcv(df, spec.min_valid_price, spec.price_offset)
                if not clean.empty:
                    return clean
            except Exception:
                continue
        return pd.DataFrame()

    def fetch_timeframe(self, spec: AssetSpec, interval: str, period: str, min_rows: int = 120) -> Tuple[pd.DataFrame, str]:
        sources = [
            ("COINBASE", lambda: self._fetch_coinbase_candles(spec, interval)),
            ("KRAKEN", lambda: self._fetch_kraken_ohlc(spec, interval)),
            ("YFINANCE", lambda: self._fetch_yfinance(spec, "1h" if interval == "4h" else interval, period)),
            ("YAHOO_V8", lambda: self._fetch_yahoo_chart_api(spec, "1h" if interval == "4h" else interval, period)),
        ]
        for name, loader in sources:
            df = loader()
            if df.empty:
                continue
            if interval == "4h" and name in ("YFINANCE", "YAHOO_V8"):
                df = (
                    df.resample("4h")
                    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                    .dropna(subset=["Open", "High", "Low", "Close"])
                )
            if len(df) >= min_rows:
                logger.info(f"[{spec.symbol} {interval}] loaded from {name} ({len(df)} bars, last={df.index[-1]})")
                return df, f"{name}:{spec.symbol}"
        return pd.DataFrame(), "NONE"

    def load_asset_bundle(self, spec: AssetSpec) -> Dict[str, Any]:
        live_q = self.fetch_live_ticker(spec)
        df_5m, src_5m = self.fetch_timeframe(spec, "5m", "7d", min_rows=150)
        df_15m, src_15m = self.fetch_timeframe(spec, "15m", "14d", min_rows=150)
        df_1h, src_1h = self.fetch_timeframe(spec, "1h", "30d", min_rows=150)
        df_4h, src_4h = self.fetch_timeframe(spec, "4h", "90d", min_rows=100)

        if df_4h.empty and not df_1h.empty:
            df_4h = (
                df_1h.resample("4h")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna(subset=["Open", "High", "Low", "Close"])
            )
            src_4h = f"{src_1h}->4h"

        # Deep Walk-Forward Backtest Series (~30-90d 24/7 crypto history)
        bt_5m = self._fetch_yfinance(spec, "5m", "30d")
        bt_15m = self._fetch_yfinance(spec, "15m", "45d")
        bt_1h = self._fetch_yfinance(spec, "1h", "90d")
        bt_4h = (
            bt_1h.resample("4h")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna(subset=["Open", "High", "Low", "Close"])
            if not bt_1h.empty
            else pd.DataFrame()
        )

        spot_px = live_q["mid"] if live_q["mid"] > spec.min_valid_price else (float(df_5m["Close"].iloc[-1]) if not df_5m.empty else 0.0)
        spread_pts = live_q["spread"] if live_q["spread"] > 0 else round(spot_px * CRYPTO_SPREAD_PCT, 2)

        return {
            "spec": spec,
            "spot": spot_px,
            "spread": spread_pts,
            "sources": {"5m": src_5m, "15m": src_15m, "1h": src_1h, "4h": src_4h},
            "live": {"5m": df_5m, "15m": df_15m, "1h": df_1h, "4h": df_4h},
            "bt": {"5m": bt_5m, "15m": bt_15m, "1h": bt_1h, "4h": bt_4h},
        }

    @staticmethod
    def validate_freshness_and_shock(df: pd.DataFrame, stale_max_minutes: int) -> Tuple[bool, str]:
        if df.empty or len(df) < 80:
            return False, "INSUFFICIENT_DATA"
        age_min = (utc_now() - df.index[-1]).total_seconds() / 60.0
        if age_min > stale_max_minutes:
            return False, f"STALE_DATA_{age_min:.1f}m"

        prev_c = df["Close"].shift(1)
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
        atr_fast, atr_base = tr.rolling(5).mean().iloc[-1], tr.rolling(50).mean().iloc[-1]
        if atr_base > 0 and ((atr_fast / (atr_base + 1e-8)) > 3.0 or (tr.iloc[-1] / (atr_base + 1e-8)) > 4.5):
            return False, f"VOLATILITY_SHOCK_ATR_RATIO={(atr_fast/(atr_base+1e-8)):.2f}"
        return True, "OK"


# ==============================================================================
# 5-PILLAR QUANTITATIVE FEATURE ENGINE + FVG + L2 LOGISTIC CALIBRATOR
# ==============================================================================
class LogisticCalibrator:
    def __init__(self) -> None:
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        if len(y) < 12 or len(np.unique(y)) < 2:
            return False
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0))
        Xs = (X - self.mean) / self.std
        n, d = Xs.shape
        w, b = np.zeros(d, dtype=float), 0.0
        lr, l2 = 0.15, 2.5
        for _ in range(180):
            z = np.clip(Xs @ w + b, -15.0, 15.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y
            w -= lr * ((Xs.T @ err) / n + (l2 / n) * w)
            b -= lr * float(np.mean(err))
        self.weights, self.bias = w, b
        return True

    def predict_proba(self, x: np.ndarray) -> Optional[float]:
        if self.weights is None or self.mean is None or self.std is None:
            return None
        xs = (x - self.mean) / self.std
        z = float(np.clip(xs @ self.weights + self.bias, -10.0, 10.0))
        return float(1.0 / (1.0 + math.exp(-z)))


class QuantFeatureEngine:
    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        prev_c = df["Close"].shift(1)
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        up_m, dn_m = df["High"].diff(), -df["Low"].diff()
        plus_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
        minus_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
        atr = QuantFeatureEngine.compute_atr(df, period) + 1e-8
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        return dx.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def compute_ker(close: pd.Series, window: int = 10) -> pd.Series:
        return (close - close.shift(window)).abs() / (close.diff().abs().rolling(window).sum() + 1e-8)

    @staticmethod
    def compute_choppiness(df: pd.DataFrame, window: int = 14) -> pd.Series:
        prev_c = df["Close"].shift(1)
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
        hl_span = df["High"].rolling(window).max() - df["Low"].rolling(window).min()
        ratio = (tr.rolling(window).sum() / (hl_span + 1e-8)).clip(lower=1e-8)
        return 100.0 * np.log10(ratio) / np.log10(float(window))

    @staticmethod
    def compute_smc_and_fvg_causal(df: pd.DataFrame, lr: int = 3) -> pd.DataFrame:
        n = len(df)
        highs, lows, closes = df["High"].to_numpy(dtype=float), df["Low"].to_numpy(dtype=float), df["Close"].to_numpy(dtype=float)
        last_sh, last_sl = np.full(n, np.nan), np.full(n, np.nan)
        structure = np.zeros(n, dtype=int)
        bos_bull, bos_bear = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        choch_bull, choch_bear = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        sweep_bull, sweep_bear = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        fvg_bull_zone, fvg_bear_zone = np.zeros(n, dtype=int), np.zeros(n, dtype=int)

        cur_sh, cur_sl = np.nan, np.nan
        sh_broken, sl_broken, cur_struct = False, False, 0
        bull_fvg: Optional[Tuple[float, float, int]] = None
        bear_fvg: Optional[Tuple[float, float, int]] = None

        for i in range(max(2, 2 * lr), n):
            if i >= 2 * lr:
                p = i - lr
                win_h, win_l = highs[p - lr : p + lr + 1], lows[p - lr : p + lr + 1]
                if highs[p] == np.max(win_h) and np.sum(win_h == highs[p]) == 1:
                    cur_sh, sh_broken = highs[p], False
                if lows[p] == np.min(win_l) and np.sum(win_l == lows[p]) == 1:
                    cur_sl, sl_broken = lows[p], False

            last_sh[i], last_sl[i] = cur_sh, cur_sl

            if lows[i] > highs[i - 2]:
                bull_fvg = (highs[i - 2], lows[i], i)
            if highs[i] < lows[i - 2]:
                bear_fvg = (highs[i], lows[i - 2], i)

            if bull_fvg is not None:
                fb, ft, fidx = bull_fvg
                if (i - fidx) <= 20 and lows[i] <= ft and closes[i] >= fb:
                    fvg_bull_zone[i] = 1
                elif closes[i] < fb or (i - fidx) > 20:
                    bull_fvg = None

            if bear_fvg is not None:
                fb, ft, fidx = bear_fvg
                if (i - fidx) <= 20 and highs[i] >= fb and closes[i] <= ft:
                    fvg_bear_zone[i] = 1
                elif closes[i] > ft or (i - fidx) > 20:
                    bear_fvg = None

            if not np.isnan(cur_sl) and lows[i] < cur_sl and closes[i] > cur_sl:
                sweep_bull[i] = 1
            if not np.isnan(cur_sh) and highs[i] > cur_sh and closes[i] < cur_sh:
                sweep_bear[i] = 1

            if not np.isnan(cur_sh) and not sh_broken and closes[i] > cur_sh:
                sh_broken = True
                choch_bull[i] = 1 if cur_struct == -1 else 0
                bos_bull[i] = 0 if cur_struct == -1 else 1
                cur_struct = 1
            elif not np.isnan(cur_sl) and not sl_broken and closes[i] < cur_sl:
                sl_broken = True
                choch_bear[i] = 1 if cur_struct == 1 else 0
                bos_bear[i] = 0 if cur_struct == 1 else 1
                cur_struct = -1
            structure[i] = cur_struct

        return pd.DataFrame(
            {
                "last_sh": last_sh, "last_sl": last_sl, "smc_structure": structure,
                "bos_bull": bos_bull, "bos_bear": bos_bear,
                "choch_bull": choch_bull, "choch_bear": choch_bear,
                "sweep_bull": sweep_bull, "sweep_bear": sweep_bear,
                "fvg_bull": fvg_bull_zone, "fvg_bear": fvg_bear_zone,
            },
            index=df.index,
        )

    @staticmethod
    def compute_order_flow_and_vwap(df: pd.DataFrame, vwap_window: int = 32) -> pd.DataFrame:
        candle_range = (df["High"] - df["Low"]) + 1e-8
        body_ratio = (df["Close"] - df["Open"]) / candle_range
        upper_wick_ratio = (df["High"] - df[["Open", "Close"]].max(axis=1)) / candle_range
        lower_wick_ratio = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / candle_range

        if (df["Volume"] > 0).sum() > (len(df) * 0.25):
            vol_base = df["Volume"].replace(0.0, np.nan).ffill().fillna(1.0)
            rel_vol = (vol_base / (vol_base.rolling(20).mean() + 1e-8)).clip(0.25, 3.0)
        else:
            rel_vol = (candle_range / (candle_range.rolling(20).median() + 1e-8)).clip(0.35, 2.5)

        of_score = (body_ratio * rel_vol).rolling(3).mean().fillna(0.0)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        rolling_vwap = (typical_price * rel_vol).rolling(vwap_window).sum() / (rel_vol.rolling(vwap_window).sum() + 1e-8)

        return pd.DataFrame(
            {
                "of_score": of_score, "body_ratio": body_ratio,
                "upper_wick_ratio": upper_wick_ratio, "lower_wick_ratio": lower_wick_ratio,
                "rolling_vwap": rolling_vwap, "rel_vol": rel_vol,
            },
            index=df.index,
        )

    @classmethod
    def compute_htf_bias(cls, df_htf: pd.DataFrame, tf_delta: pd.Timedelta) -> pd.DataFrame:
        if df_htf.empty or len(df_htf) < 55:
            return pd.DataFrame()
        ema_fast = df_htf["Close"].ewm(span=20, adjust=False).mean()
        ema_slow = df_htf["Close"].ewm(span=50, adjust=False).mean()
        smc = cls.compute_smc_and_fvg_causal(df_htf, lr=3)
        adx = cls.compute_adx(df_htf, 14)

        bias = np.where(
            (df_htf["Close"] > ema_fast) & (ema_fast > ema_slow) & (smc["smc_structure"] >= 0),
            1,
            np.where(
                (df_htf["Close"] < ema_fast) & (ema_fast < ema_slow) & (smc["smc_structure"] <= 0),
                -1,
                np.where(ema_fast > ema_slow, 1, -1),
            ),
        )
        return pd.DataFrame(
            {"htf_bias": bias, "htf_adx": adx.to_numpy(), "htf_smc": smc["smc_structure"].to_numpy()},
            index=df_htf.index + tf_delta,
        )

    @classmethod
    def enrich_dataframe(
        cls, df_exec: pd.DataFrame, df_mid: pd.DataFrame, df_htf: pd.DataFrame, cfg: ModeConfig, atr_period: int = 14
    ) -> pd.DataFrame:
        df = df_exec.copy()
        df["atr"] = cls.compute_atr(df, atr_period)
        df["atr_base"] = cls.compute_atr(df, 50)
        df["adx"] = cls.compute_adx(df, 14)
        df["ker"] = cls.compute_ker(df["Close"], 10)
        df["chop"] = cls.compute_choppiness(df, 14)
        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

        z_mean = df["Close"].rolling(cfg.z_window).mean()
        z_std = df["Close"].rolling(cfg.z_window).std(ddof=0)
        df["z_score"] = (df["Close"] - z_mean) / (z_std + 1e-8)
        df["z_prev"] = df["z_score"].shift(1)

        smc = cls.compute_smc_and_fvg_causal(df, lr=cfg.swing_lr)
        of_vwap = cls.compute_order_flow_and_vwap(df, vwap_window=cfg.vwap_window)
        df = pd.concat([df, smc, of_vwap], axis=1)
        df["vwap_dist_atr"] = (df["Close"] - df["rolling_vwap"]) / (df["atr"] + 1e-8)

        mid_delta = pd.Timedelta(minutes=15) if cfg.mid_tf == "15m" else pd.Timedelta(hours=1)
        htf_delta = pd.Timedelta(hours=1) if cfg.htf_tf == "1h" else pd.Timedelta(hours=4)

        df_reset = df.reset_index().rename(columns={df.index.name or "index": "ts"})
        if "ts" not in df_reset.columns:
            df_reset = df_reset.rename(columns={df_reset.columns[0]: "ts"})

        mid_bias_df = cls.compute_htf_bias(df_mid, mid_delta)
        if not mid_bias_df.empty:
            mid_r = mid_bias_df.reset_index().rename(columns={mid_bias_df.index.name or "index": "ts"})
            if "ts" not in mid_r.columns:
                mid_r = mid_r.rename(columns={mid_r.columns[0]: "ts"})
            mid_r = mid_r.rename(columns={"htf_bias": "mid_bias", "htf_adx": "mid_adx", "htf_smc": "mid_smc"})
            df_reset = pd.merge_asof(df_reset.sort_values("ts"), mid_r.sort_values("ts"), on="ts", direction="backward")
        else:
            df_reset["mid_bias"], df_reset["mid_adx"], df_reset["mid_smc"] = 0, 20.0, 0

        htf_bias_df = cls.compute_htf_bias(df_htf, htf_delta)
        if not htf_bias_df.empty:
            htf_r = htf_bias_df.reset_index().rename(columns={htf_bias_df.index.name or "index": "ts"})
            if "ts" not in htf_r.columns:
                htf_r = htf_r.rename(columns={htf_r.columns[0]: "ts"})
            df_reset = pd.merge_asof(df_reset.sort_values("ts"), htf_r.sort_values("ts"), on="ts", direction="backward")
        else:
            df_reset["htf_bias"], df_reset["htf_adx"], df_reset["htf_smc"] = 0, 20.0, 0

        df_out = df_reset.set_index("ts")
        df_out["mid_bias"] = df_out["mid_bias"].fillna(0).astype(int)
        df_out["htf_bias"] = df_out["htf_bias"].fillna(0).astype(int)
        return df_out

    @staticmethod
    def classify_regime(row: pd.Series, cfg: ModeConfig, ts_dt: datetime) -> Tuple[str, bool, float]:
        atr, atr_base = float(row.get("atr", 0.0)), float(row.get("atr_base", 0.0))
        adx, ker, chop = float(row.get("adx", 0.0)), float(row.get("ker", 0.0)), float(row.get("chop", 50.0))
        ema20, ema50 = float(row.get("ema20", 0.0)), float(row.get("ema50", 0.0))
        rel_vol = float(row.get("rel_vol", 1.0))
        bar_range = float(row.get("High", 0.0)) - float(row.get("Low", 0.0))

        if atr > 0 and (bar_range / (atr + 1e-8)) > 2.90:
            return "EXHAUSTION_SPIKE", False, cfg.min_score
        if atr_base > 0 and (atr / (atr_base + 1e-8)) > 2.75:
            return "VOLATILITY_SHOCK", False, cfg.min_score
        if atr_base > 0 and (atr / (atr_base + 1e-8)) < 0.40:
            return "LOW_LIQUIDITY_DEAD", False, cfg.min_score

        _, is_weekend, _ = check_crypto_session(ts_dt)
        eff_ker, eff_adx, eff_score = cfg.min_ker, cfg.min_adx, cfg.min_score
        if cfg.mode_name == "SCALP" and (is_weekend or rel_vol < 0.55):
            eff_ker = max(cfg.min_ker, 0.39)
            eff_adx = max(cfg.min_adx, 22.0)
            eff_score = max(cfg.min_score, 80.0)

        if chop > cfg.max_chop and ker < eff_ker:
            return "CHOPPY_SIDEWAYS", False, eff_score
        if adx < eff_adx and ker < eff_ker:
            return "RANGING_WEAK", False, eff_score
        if ema20 > ema50 and (adx >= eff_adx or ker >= eff_ker):
            return "TREND_BULL", True, eff_score
        if ema20 < ema50 and (adx >= eff_adx or ker >= eff_ker):
            return "TREND_BEAR", True, eff_score
        return "TRANSITIONAL", False, eff_score

    @classmethod
    def evaluate_bar(
        cls, df: pd.DataFrame, idx: int, cfg: ModeConfig, z_thresh: float,
        check_session: bool = True, btc_leader_dir: int = 0, symbol: str = "BTCUSD",
    ) -> Tuple[str, float, str, Dict[str, Any]]:
        if idx < 60 or idx >= len(df):
            return "NO_SIGNAL", 0.0, "WARMUP", {"reason": "WARMUP"}

        row = df.iloc[idx]
        ts_dt = df.index[idx].to_pydatetime()

        if check_session:
            sess_ok, _, sess_reason = check_crypto_session(ts_dt)
            if not sess_ok:
                return "NO_SIGNAL", 0.0, sess_reason, {"reason": sess_reason}

        regime, tradeable, eff_min_score = cls.classify_regime(row, cfg, ts_dt)
        if not tradeable:
            return "NO_SIGNAL", 0.0, regime, {"reason": f"REGIME_BLOCK:{regime}"}

        close, ema20, ema50 = float(row["Close"]), float(row["ema20"]), float(row["ema50"])
        z_val = float(row["z_score"])
        z_prev = float(row["z_prev"]) if not pd.isna(row["z_prev"]) else z_val
        of_score, body_ratio = float(row["of_score"]), float(row["body_ratio"])
        upper_wick, lower_wick = float(row["upper_wick_ratio"]), float(row["lower_wick_ratio"])
        vwap_dist, ker, adx = float(row["vwap_dist_atr"]), float(row["ker"]), float(row["adx"])
        mid_bias, htf_bias, smc_struct = int(row["mid_bias"]), int(row["htf_bias"]), int(row["smc_structure"])

        lb = cfg.bos_lookback
        recent_bos_bull = int(df["bos_bull"].iloc[max(0, idx - lb) : idx + 1].sum() > 0)
        recent_choch_bull = int(df["choch_bull"].iloc[max(0, idx - lb) : idx + 1].sum() > 0)
        recent_bos_bear = int(df["bos_bear"].iloc[max(0, idx - lb) : idx + 1].sum() > 0)
        recent_choch_bear = int(df["choch_bear"].iloc[max(0, idx - lb) : idx + 1].sum() > 0)
        recent_sweep_bull = int(df["sweep_bull"].iloc[max(0, idx - 6) : idx + 1].sum() > 0)
        recent_sweep_bear = int(df["sweep_bear"].iloc[max(0, idx - 6) : idx + 1].sum() > 0)
        recent_fvg_bull = int(df["fvg_bull"].iloc[max(0, idx - 4) : idx + 1].sum() > 0)
        recent_fvg_bear = int(df["fvg_bear"].iloc[max(0, idx - 4) : idx + 1].sum() > 0)

        bull_score, bear_score = 0.0, 0.0

        bull_mtf_ok = (htf_bias >= 0 and mid_bias > 0) or (htf_bias > 0 and mid_bias >= 0)
        bear_mtf_ok = (htf_bias <= 0 and mid_bias < 0) or (htf_bias < 0 and mid_bias <= 0)
        if htf_bias > 0 and mid_bias > 0:
            bull_score += 25.0
        elif bull_mtf_ok:
            bull_score += 17.0
        if htf_bias < 0 and mid_bias < 0:
            bear_score += 25.0
        elif bear_mtf_ok:
            bear_score += 17.0

        if smc_struct > 0:
            bull_score += 13.0
        if recent_bos_bull or recent_choch_bull:
            bull_score += 8.0
        if recent_sweep_bull:
            bull_score += 4.0

        if smc_struct < 0:
            bear_score += 13.0
        if recent_bos_bear or recent_choch_bear:
            bear_score += 8.0
        if recent_sweep_bear:
            bear_score += 4.0

        if of_score > 0.08 and body_ratio > 0.15 and upper_wick < 0.45:
            bull_score += 20.0
        elif body_ratio > 0.05 and upper_wick < 0.50:
            bull_score += 10.0
        if of_score < -0.08 and body_ratio < -0.15 and lower_wick < 0.45:
            bear_score += 20.0
        elif body_ratio < -0.05 and lower_wick < 0.50:
            bear_score += 10.0

        if (-z_thresh <= z_val <= 1.0) and (z_val > z_prev) and close > ema50:
            bull_score += 15.0 if z_prev < -0.30 else 11.0
        elif (1.0 < z_val <= z_thresh) and (z_val >= z_prev) and ker >= 0.45 and close > ema50:
            bull_score += 10.0

        if (-1.0 <= z_val <= z_thresh) and (z_val < z_prev) and close < ema50:
            bear_score += 15.0 if z_prev > 0.30 else 11.0
        elif (-z_thresh <= z_val < -1.0) and (z_val <= z_prev) and ker >= 0.45 and close < ema50:
            bear_score += 10.0

        regime_pts = min(9.0, (ker / 0.60) * 5.0 + (min(adx, 40.0) / 40.0) * 4.0)
        if -1.2 <= vwap_dist <= 1.6 and close >= ema20:
            bull_score += regime_pts + 6.0
        elif vwap_dist <= 2.4:
            bull_score += regime_pts

        if -1.6 <= vwap_dist <= 1.2 and close <= ema20:
            bear_score += regime_pts + 6.0
        elif vwap_dist >= -2.4:
            bear_score += regime_pts

        if recent_fvg_bull:
            bull_score += 5.0
        if recent_fvg_bear:
            bear_score += 5.0

        bull_tech_score = min(100.0, bull_score)
        bear_tech_score = min(100.0, bear_score)

        # Cross-Crypto Leader Bonus/Veto (ETHUSD aligned with BTCUSD macro regime)
        cross_bonus_bull = 5.0 if (symbol == "ETHUSD" and btc_leader_dir > 0) else 0.0
        cross_bonus_bear = 5.0 if (symbol == "ETHUSD" and btc_leader_dir < 0) else 0.0
        bull_final = min(100.0, bull_tech_score + cross_bonus_bull)
        bear_final = min(100.0, bear_tech_score + cross_bonus_bear)

        z_cap = z_thresh + 0.50
        bull_valid = (
            bull_mtf_ok and close > ema20 and ema20 > ema50
            and upper_wick < 0.50 and body_ratio > 0.05
            and z_val < z_cap and bull_final >= eff_min_score
        )
        bear_valid = (
            bear_mtf_ok and close < ema20 and ema20 < ema50
            and lower_wick < 0.50 and body_ratio < -0.05
            and z_val > -z_cap and bear_final >= eff_min_score
        )

        if symbol == "ETHUSD" and bull_valid and btc_leader_dir < 0:
            return "NO_SIGNAL", round(bull_final, 1), regime, {"reason": "CROSS_CRYPTO_BTC_DUMP_VETO", "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}"}
        if symbol == "ETHUSD" and bear_valid and btc_leader_dir > 0:
            return "NO_SIGNAL", round(bear_final, 1), regime, {"reason": "CROSS_CRYPTO_BTC_RALLY_VETO", "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}"}
        if bull_valid and bear_valid:
            return "NO_SIGNAL", max(bull_final, bear_final), regime, {"reason": "CONFLICTING_SIGNALS"}

        tech_score_chosen = bull_tech_score if bull_valid else bear_tech_score
        feat_vec = [
            round(tech_score_chosen / 100.0, 3), round(ker, 3),
            round(min(adx, 50.0) / 50.0, 3), round(min(abs(z_val), 3.0) / 3.0, 3),
            float(recent_fvg_bull if bull_valid else recent_fvg_bear),
            float((recent_bos_bull or recent_choch_bull) if bull_valid else (recent_bos_bear or recent_choch_bear)),
            float(1.0 if (htf_bias == mid_bias and htf_bias != 0) else 0.0),
        ]

        if bull_valid:
            return "BUY", round(bull_final, 1), regime, {"mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}", "fvg": recent_fvg_bull, "feat": feat_vec}
        if bear_valid:
            return "SELL", round(bear_final, 1), regime, {"mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}", "fvg": recent_fvg_bear, "feat": feat_vec}

        top_dir = "BUY" if bull_final >= bear_final else "SELL"
        top_score = round(max(bull_final, bear_final), 1)
        return "NO_SIGNAL", top_score, regime, {
            "reason": f"CONFLUENCE_BELOW_THRESHOLD({top_dir}:{top_score}/{eff_min_score})",
            "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}",
        }

    @staticmethod
    def compute_sl_and_tps(
        direction: str, entry: float, atr: float, last_sl_px: float, last_sh_px: float, cfg: ModeConfig
    ) -> Tuple[float, float, float, float]:
        min_floor = entry * cfg.min_sl_pct
        base_sl_dist = max(atr * cfg.sl_atr_mult, min_floor)
        if direction == "BUY":
            sl = entry - base_sl_dist
            if cfg.use_swing_anchor_sl and not np.isnan(last_sl_px) and last_sl_px < entry:
                swing_dist = entry - (last_sl_px - 0.25 * atr)
                if 0.85 * base_sl_dist <= swing_dist <= 1.45 * base_sl_dist:
                    sl = min(sl, last_sl_px - 0.25 * atr)
            sl_dist = max(entry - sl, min_floor)
            return (
                round(sl, 2), round(entry + sl_dist * cfg.tp_r_multiples[0], 2),
                round(entry + sl_dist * cfg.tp_r_multiples[1], 2), round(entry + sl_dist * cfg.tp_r_multiples[2], 2),
            )
        else:
            sl = entry + base_sl_dist
            if cfg.use_swing_anchor_sl and not np.isnan(last_sh_px) and last_sh_px > entry:
                swing_dist = (last_sh_px + 0.25 * atr) - entry
                if 0.85 * base_sl_dist <= swing_dist <= 1.45 * base_sl_dist:
                    sl = max(sl, last_sh_px + 0.25 * atr)
            sl_dist = max(sl - entry, min_floor)
            return (
                round(sl, 2), round(entry - sl_dist * cfg.tp_r_multiples[0], 2),
                round(entry - sl_dist * cfg.tp_r_multiples[1], 2), round(entry - sl_dist * cfg.tp_r_multiples[2], 2),
            )


# ==============================================================================
# WALK-FORWARD BACKTEST ENGINE & DUAL-TIER OPTIMIZER
# ==============================================================================
class BacktestAndOptimizer:
    @staticmethod
    def run_backtest(
        df: pd.DataFrame, spec: AssetSpec, cfg: ModeConfig, z_thresh: float,
        account_balance: float = DEFAULT_ACCOUNT_BALANCE,
        start_idx: int = 60, end_idx: Optional[int] = None, min_score_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        n = min(len(df), end_idx) if end_idx is not None else len(df)
        start_i = max(60, start_idx)
        equity, peak_equity, max_dd_pct = account_balance, account_balance, 0.0
        equity_curve: List[Tuple[pd.Timestamp, float]] = []
        r_returns, feat_X, feat_y = [], [], []
        wins, losses, tp1_hits = 0, 0, 0
        gross_profit, gross_loss, total_friction_usd = 0.0, 0.0, 0.0
        last_entry_idx = -999
        min_bar_gap = 6 if cfg.mode_name == "SCALP" else 8
        w1 = cfg.tp1_close_pct
        w2 = (1.0 - w1) * 0.50
        w3 = 1.0 - w1 - w2

        i = start_i
        while i < n - 2:
            equity_curve.append((df.index[i], equity))
            if (i - last_entry_idx) < min_bar_gap:
                i += 1
                continue

            direction, score, _, diag = QuantFeatureEngine.evaluate_bar(
                df, i, cfg, z_thresh=z_thresh, check_session=True, symbol=spec.symbol
            )
            if direction not in ("BUY", "SELL") or (min_score_override is not None and score < min_score_override):
                i += 1
                continue

            atr = float(df["atr"].iloc[i])
            if atr <= 0.01 or np.isnan(atr):
                i += 1
                continue

            entry = float(df["Close"].iloc[i])
            last_sl_px = float(df["last_sl"].iloc[i]) if not pd.isna(df["last_sl"].iloc[i]) else np.nan
            last_sh_px = float(df["last_sh"].iloc[i]) if not pd.isna(df["last_sh"].iloc[i]) else np.nan
            sl, tp1, tp2, tp3 = QuantFeatureEngine.compute_sl_and_tps(direction, entry, atr, last_sl_px, last_sh_px, cfg)
            sl_dist = abs(entry - sl)
            be_buf = entry * 0.0004

            risk_usd = equity * (DEFAULT_RISK_PCT / 100.0)
            lot = calculate_crypto_lot_size(entry, sl, spec, equity, DEFAULT_RISK_PCT)
            friction_pts = entry * (CRYPTO_SPREAD_PCT + 2.0 * CRYPTO_SLIPPAGE_PCT)
            friction_rt = (friction_pts / spec.pip_size) * (spec.contract_size * spec.pip_size) * lot
            total_friction_usd += friction_rt

            cur_sl, rem_weight, realized_r = sl, 1.0, 0.0
            tp1_hit, tp2_hit = False, False
            exit_bar = i + 1

            for j in range(i + 1, min(n, i + 180)):
                bar_h, bar_l, bar_c = float(df["High"].iloc[j]), float(df["Low"].iloc[j]), float(df["Close"].iloc[j])
                bar_atr = float(df["atr"].iloc[j]) if not np.isnan(df["atr"].iloc[j]) else atr
                exit_bar = j

                if direction == "BUY":
                    if bar_l <= cur_sl:
                        realized_r += rem_weight * ((cur_sl - entry) / (sl_dist + 1e-8))
                        rem_weight = 0.0
                        break
                    if not tp1_hit and bar_h >= tp1:
                        tp1_hit = True
                        realized_r += w1 * cfg.tp_r_multiples[0]
                        rem_weight -= w1
                        cur_sl = max(cur_sl, entry + be_buf)
                    if tp1_hit and not tp2_hit and bar_h >= tp2:
                        tp2_hit = True
                        realized_r += w2 * cfg.tp_r_multiples[1]
                        rem_weight -= w2
                        cur_sl = max(cur_sl, tp1)
                    if bar_h >= tp3:
                        realized_r += rem_weight * cfg.tp_r_multiples[2]
                        rem_weight = 0.0
                        break
                    if (cfg.mode_name == "SWING" and tp1_hit) or tp2_hit:
                        cur_sl = max(cur_sl, bar_c - (bar_atr * cfg.trail_atr_mult))
                else:
                    if bar_h >= cur_sl:
                        realized_r += rem_weight * ((entry - cur_sl) / (sl_dist + 1e-8))
                        rem_weight = 0.0
                        break
                    if not tp1_hit and bar_l <= tp1:
                        tp1_hit = True
                        realized_r += w1 * cfg.tp_r_multiples[0]
                        rem_weight -= w1
                        cur_sl = min(cur_sl, entry - be_buf)
                    if tp1_hit and not tp2_hit and bar_l <= tp2:
                        tp2_hit = True
                        realized_r += w2 * cfg.tp_r_multiples[1]
                        rem_weight -= w2
                        cur_sl = min(cur_sl, tp1)
                    if bar_l <= tp3:
                        realized_r += rem_weight * cfg.tp_r_multiples[2]
                        rem_weight = 0.0
                        break
                    if (cfg.mode_name == "SWING" and tp1_hit) or tp2_hit:
                        cur_sl = min(cur_sl, bar_c + (bar_atr * cfg.trail_atr_mult))

            if rem_weight > 0:
                last_c = float(df["Close"].iloc[exit_bar])
                realized_r += rem_weight * (((last_c - entry) if direction == "BUY" else (entry - last_c)) / (sl_dist + 1e-8))

            trade_pnl = (realized_r * risk_usd) - friction_rt
            net_r = trade_pnl / (risk_usd + 1e-8)
            equity = max(100.0, equity + trade_pnl)
            r_returns.append(net_r)

            if "feat" in diag:
                feat_X.append(diag["feat"])
                feat_y.append(1 if (tp1_hit or trade_pnl > 0) else 0)

            if tp1_hit:
                tp1_hits += 1
            if trade_pnl > 0:
                wins += 1
                gross_profit += trade_pnl
            else:
                losses += 1
                gross_loss += abs(trade_pnl)

            peak_equity = max(peak_equity, equity)
            max_dd_pct = max(max_dd_pct, ((peak_equity - equity) / (peak_equity + 1e-8)) * 100.0)
            last_entry_idx = i
            i = max(i + 1, exit_bar + 1)

        if n > start_i:
            equity_curve.append((df.index[n - 1], equity))

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        tp1_hit_rate = (tp1_hits / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = (gross_profit / (gross_loss + 1e-8)) if gross_loss > 0 else (3.0 if gross_profit > 0 else 0.0)
        expectancy_r = float(np.mean(r_returns)) if r_returns else 0.0
        std_r = float(np.std(r_returns, ddof=1)) if len(r_returns) >= 3 else 0.0
        sharpe = max(-10.0, min(10.0, (expectancy_r / std_r) * math.sqrt(total_trades))) if (len(r_returns) >= 3 and std_r > 1e-3) else 0.0

        return {
            "symbol": spec.symbol, "mode": cfg.mode_name, "trades": total_trades, "wins": wins, "losses": losses,
            "win_rate": round(win_rate, 2), "tp1_hit_rate": round(tp1_hit_rate, 2),
            "profit_factor": round(min(profit_factor, 9.99), 2), "expectancy_r": round(expectancy_r, 3),
            "sharpe": round(sharpe, 2), "max_dd_pct": round(max_dd_pct, 2),
            "total_return_pct": round(((equity - account_balance) / account_balance) * 100.0, 2),
            "final_equity": round(equity, 2), "friction_usd": round(total_friction_usd, 2),
            "equity_curve": equity_curve, "feat_X": feat_X, "feat_y": feat_y,
        }

    @classmethod
    def grid_search_optimize(
        cls,
        bundle: Dict[str, Any], cfg: ModeConfig,
    ) -> Tuple[int, float, pd.DataFrame, Dict[str, Any], LogisticCalibrator]:
        spec: AssetSpec = bundle["spec"]
        live_dict, bt_dict = bundle["live"], bundle["bt"]

        df_exec_live, df_mid_live, df_htf_live = live_dict[cfg.exec_tf], live_dict[cfg.mid_tf], live_dict[cfg.htf_tf]
        df_exec_bt = bt_dict[cfg.exec_tf] if len(bt_dict[cfg.exec_tf]) > len(df_exec_live) else df_exec_live
        df_mid_bt = bt_dict[cfg.mid_tf] if len(bt_dict[cfg.mid_tf]) > len(df_mid_live) else df_mid_live
        df_htf_bt = bt_dict[cfg.htf_tf] if len(bt_dict[cfg.htf_tf]) > len(df_htf_live) else df_htf_live

        n_bt = len(df_exec_bt)
        split_idx = int(n_bt * 0.70)
        has_oos = (split_idx >= 200) and ((n_bt - split_idx) >= 120)

        cached_bt = {
            atr_p: QuantFeatureEngine.enrich_dataframe(df_exec_bt, df_mid_bt, df_htf_bt, cfg, atr_period=atr_p)
            for atr_p in GRID_ATR_PERIODS
        }

        best_obj, best_atr, best_z, found = -1e9, 14, 2.0, False
        best_is_stats: Optional[Dict[str, Any]] = None
        is_end = split_idx if has_oos else n_bt

        for atr_p in GRID_ATR_PERIODS:
            for z_t in GRID_Z_THRESHOLDS:
                is_stats = cls.run_backtest(
                    cached_bt[atr_p], spec, cfg, z_thresh=z_t, start_idx=60, end_idx=is_end
                )
                if is_stats["trades"] < cfg.min_backtest_trades:
                    continue
                obj = is_stats["expectancy_r"] * math.sqrt(is_stats["trades"]) * min(is_stats["profit_factor"], 3.0) - 0.18 * is_stats["max_dd_pct"]
                if obj > best_obj:
                    best_obj, best_atr, best_z, found, best_is_stats = obj, atr_p, z_t, True, is_stats

        if not found:
            best_atr, best_z = 14, 2.0
            best_is_stats = cls.run_backtest(cached_bt[best_atr], spec, cfg, z_thresh=best_z, start_idx=60, end_idx=is_end)

        win_df = cached_bt[best_atr]
        calibrator = LogisticCalibrator()
        if best_is_stats and len(best_is_stats.get("feat_y", [])) >= 12:
            calibrator.fit(np.array(best_is_stats["feat_X"], dtype=float), np.array(best_is_stats["feat_y"], dtype=float))

        full_stats = cls.run_backtest(win_df, spec, cfg, z_thresh=best_z, start_idx=60)
        sniper_stats = cls.run_backtest(
            win_df, spec, cfg, z_thresh=best_z, start_idx=60, min_score_override=cfg.sniper_score
        )

        if has_oos:
            oos_stats = cls.run_backtest(win_df, spec, cfg, z_thresh=best_z, start_idx=split_idx)
            sniper_oos = cls.run_backtest(
                win_df, spec, cfg, z_thresh=best_z, start_idx=split_idx, min_score_override=cfg.sniper_score
            )
            full_stats["oos_split_ts"] = win_df.index[split_idx]
            full_stats["oos_trades"] = oos_stats["trades"]
            full_stats["oos_win_rate"] = oos_stats["win_rate"]
            full_stats["oos_tp1_hit"] = oos_stats["tp1_hit_rate"]
            full_stats["oos_pf"] = oos_stats["profit_factor"]
            full_stats["oos_ev_r"] = oos_stats["expectancy_r"]
            full_stats["sniper_oos_trades"] = sniper_oos["trades"]
            full_stats["sniper_oos_wr"] = sniper_oos["win_rate"]
        else:
            full_stats["oos_split_ts"] = None
            full_stats["oos_trades"] = full_stats["trades"]
            full_stats["oos_win_rate"] = full_stats["win_rate"]
            full_stats["oos_tp1_hit"] = full_stats["tp1_hit_rate"]
            full_stats["oos_pf"] = full_stats["profit_factor"]
            full_stats["oos_ev_r"] = full_stats["expectancy_r"]
            full_stats["sniper_oos_trades"] = sniper_stats["trades"]
            full_stats["sniper_oos_wr"] = sniper_stats["win_rate"]

        full_stats["opt_atr_period"] = best_atr
        full_stats["opt_z_thresh"] = best_z
        full_stats["bt_bars"] = n_bt
        full_stats["sniper_trades"] = sniper_stats["trades"]
        full_stats["sniper_wr"] = sniper_stats["win_rate"]
        full_stats["sniper_tp1_hit"] = sniper_stats["tp1_hit_rate"]
        full_stats["sniper_pf"] = sniper_stats["profit_factor"]
        full_stats["sniper_dd"] = sniper_stats["max_dd_pct"]
        full_stats["sniper_ret"] = sniper_stats["total_return_pct"]

        df_live_enr = QuantFeatureEngine.enrich_dataframe(df_exec_live, df_mid_live, df_htf_live, cfg, atr_period=best_atr)
        return best_atr, best_z, df_live_enr, full_stats, calibrator

    @staticmethod
    def generate_performance_chart(all_stats: Dict[Tuple[str, str], Dict[str, Any]], out_path: str = CHART_PATH) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor="#0d1117")
        fig.suptitle(
            "APEX-QUANT-OMNI | BTCUSD & ETHUSD Dual-Mode Walk-Forward Backtest (24/7 + Friction + OOS)",
            color="#f0f6fc", fontsize=13, fontweight="bold",
        )
        grid_cfg = [
            (axes[0, 0], ("BTCUSD", "SCALP"), "#f7931a", "BTCUSD SCALP (5M/15M/1H)"),
            (axes[0, 1], ("BTCUSD", "SWING"), "#ffb74d", "BTCUSD SWING (15M/1H/4H)"),
            (axes[1, 0], ("ETHUSD", "SCALP"), "#627eea", "ETHUSD SCALP (5M/15M/1H)"),
            (axes[1, 1], ("ETHUSD", "SWING"), "#29b6f6", "ETHUSD SWING (15M/1H/4H)"),
        ]
        for ax, key, color, label in grid_cfg:
            st = all_stats.get(key) or {}
            ax.set_facecolor("#161b22")
            curve = st.get("equity_curve") or []
            if curve:
                xs, ys = [c[0] for c in curve], [c[1] for c in curve]
                ax.plot(xs, ys, color=color, linewidth=1.7)
                ax.fill_between(xs, DEFAULT_ACCOUNT_BALANCE, ys, color=color, alpha=0.12)
                ax.axhline(DEFAULT_ACCOUNT_BALANCE, color="#8b949e", linestyle="--", linewidth=0.8)
                if st.get("oos_split_ts") is not None:
                    ax.axvline(st["oos_split_ts"], color="#ef5350", linestyle="-.", linewidth=1.1)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            else:
                ax.text(0.5, 0.5, "No Trades Recorded", color="#8b949e", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(
                f"{label}\nPRO: {st.get('trades',0)}T WR {st.get('win_rate',0):.1f}% PF {st.get('profit_factor',0):.2f} Ret {st.get('total_return_pct',0):+.1f}% | "
                f"SNIPER: {st.get('sniper_trades',0)}T WR {st.get('sniper_wr',0):.1f}% PF {st.get('sniper_pf',0):.2f}",
                color="#c9d1d9", fontsize=9, loc="left",
            )
            ax.tick_params(colors="#8b949e")
            ax.grid(color="#30363d", linestyle=":", linewidth=0.6)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)


# ==============================================================================
# SIGNAL GENERATOR & AUTO-TRAILING STOP ENGINE
# ==============================================================================
class SignalEngine:
    @staticmethod
    def build_live_signal(
        df_enriched: pd.DataFrame, spec: AssetSpec, cfg: ModeConfig, z_thresh: float,
        backtest_stats: Dict[str, Any], calibrator: LogisticCalibrator,
        live_spread: float, live_mid: float, btc_leader_dir: int = 0, cross_tag: str = "BTC:NEUT",
    ) -> SignalCandidate:
        if df_enriched.empty or len(df_enriched) < 65:
            return SignalCandidate(symbol=spec.symbol, mode=cfg.mode_name, direction="NO_SIGNAL", reason="EMPTY_DATA")

        idx = len(df_enriched) - 1
        row = df_enriched.iloc[idx]
        candle_ts = df_enriched.index[idx].isoformat()
        close_px, atr = float(row["Close"]), float(row["atr"])
        ker, adx, z_val = float(row["ker"]), float(row["adx"]), float(row["z_score"])
        spot_ref = live_mid if live_mid > spec.min_valid_price else close_px

        direction, score, regime, diag = QuantFeatureEngine.evaluate_bar(
            df_enriched, idx, cfg, z_thresh=z_thresh, check_session=True,
            btc_leader_dir=btc_leader_dir, symbol=spec.symbol,
        )
        mtf_bias = str(diag.get("mtf_bias", "N/A"))

        if direction == "NO_SIGNAL":
            return SignalCandidate(
                symbol=spec.symbol, mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, cross_ctx=cross_tag,
                candle_ts=candle_ts, reason=str(diag.get("reason", "NO_CONFLUENCE")),
            )

        oos_trades = int(backtest_stats.get("oos_trades", 0))
        oos_ev_r = float(backtest_stats.get("oos_ev_r", 0.0))
        if oos_trades >= 6 and oos_ev_r < -0.25 and score < (cfg.min_score + 6.0):
            return SignalCandidate(
                symbol=spec.symbol, mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, cross_ctx=cross_tag,
                candle_ts=candle_ts, reason=f"OOS_DEGRADATION_GUARD(EV={oos_ev_r:+.2f}R)",
            )

        if live_mid > spec.min_valid_price and atr > 0 and abs(live_mid - close_px) > (0.65 * atr):
            return SignalCandidate(
                symbol=spec.symbol, mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(live_mid, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, cross_ctx=cross_tag,
                candle_ts=candle_ts, reason=f"PRICE_DRIFT_GUARD(diff={abs(live_mid-close_px):.2f})",
            )

        half_spread = live_spread / 2.0
        entry = round(spot_ref + half_spread if direction == "BUY" else spot_ref - half_spread, 2)
        last_sl_px = float(row["last_sl"]) if not pd.isna(row["last_sl"]) else np.nan
        last_sh_px = float(row["last_sh"]) if not pd.isna(row["last_sh"]) else np.nan
        sl, tp1, tp2, tp3 = QuantFeatureEngine.compute_sl_and_tps(
            direction, entry, atr, last_sl_px, last_sh_px, cfg
        )
        lot_size = calculate_crypto_lot_size(entry, sl, spec, DEFAULT_ACCOUNT_BALANCE, DEFAULT_RISK_PCT)
        risk_usd = round((abs(entry - sl) / spec.pip_size) * (spec.contract_size * spec.pip_size) * lot_size, 2)

        is_sniper_setup = (score >= cfg.sniper_score and ker >= 0.38)
        s_oos_n = int(backtest_stats.get("sniper_oos_trades", 0))
        if is_sniper_setup and s_oos_n >= 4:
            ref_wr = float(backtest_stats.get("sniper_oos_wr", backtest_stats.get("sniper_wr", 55.0))) / 100.0
            ref_n = s_oos_n
        else:
            ref_wr = float(backtest_stats.get("oos_win_rate", backtest_stats.get("win_rate", 55.0))) / 100.0
            ref_n = oos_trades

        prior_wr = 0.52 + ((score - cfg.min_score) / 100.0) * 0.24
        weight_oos = min(0.50, ref_n / 20.0)
        bayesian_p = weight_oos * ref_wr + (1.0 - weight_oos) * prior_wr

        lr_p = calibrator.predict_proba(np.array(diag["feat"], dtype=float)) if "feat" in diag else None
        blended_p = (0.55 * bayesian_p + 0.45 * lr_p) if lr_p is not None else bayesian_p
        p_win = max(45.0, min(88.0, round(blended_p * 100.0, 1)))

        p_dec = p_win / 100.0
        w1 = cfg.tp1_close_pct
        w2 = (1.0 - w1) * 0.50
        w3 = 1.0 - w1 - w2
        blended_r = (w1 * cfg.tp_r_multiples[0]) + (w2 * cfg.tp_r_multiples[1]) + (w3 * cfg.tp_r_multiples[2])
        ev_r = round((p_dec * blended_r) - ((1.0 - p_dec) * 1.0), 2)

        if ev_r <= 0.15:
            return SignalCandidate(
                symbol=spec.symbol, mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, cross_ctx=cross_tag,
                candle_ts=candle_ts, reason=f"LOW_EXPECTANCY_EV={ev_r:+.2f}R",
            )

        tier = "SNIPER" if (is_sniper_setup and p_win >= 60.0) else "PRO"
        return SignalCandidate(
            symbol=spec.symbol, mode=cfg.mode_name, direction=direction, tier=tier, entry=entry, sl=sl,
            tp1=tp1, tp2=tp2, tp3=tp3, lot_size=lot_size, risk_usd=risk_usd,
            score=score, p_win=p_win, ev_r=ev_r, ker=round(ker, 2), adx=round(adx, 1),
            z_score=round(z_val, 2), atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias,
            cross_ctx=cross_tag, candle_ts=candle_ts, reason=f"SIGNAL_{tier}_CONFIRMED", pillar_details=diag,
        )

    @staticmethod
    def format_telegram_signal(sig: SignalCandidate) -> str:
        prefix = "💎" if sig.tier == "SNIPER" else "⚡"
        return (
            f"{prefix} <b>{sig.symbol} | {sig.mode} | {sig.direction}</b>\n"
            f"🎯 Entry: <code>{sig.entry:.2f}</code> | 🛑 SL: <code>{sig.sl:.2f}</code>\n"
            f"✅ TP1: <code>{sig.tp1:.2f}</code> | TP2: <code>{sig.tp2:.2f}</code> | TP3: <code>{sig.tp3:.2f}</code>\n"
            f"📐 Lot: <code>{sig.lot_size:.2f}</code> | Skor: <code>{sig.score:.0f}/100</code> | Regime: <code>{sig.regime}</code>\n"
            f"🔄 Trail: Aktif @ TP1 (BE) | 🕒 <code>{utc_now().strftime('%H:%M UTC')}</code>\n"
            f"🧠 Edge: P(Win) <code>{sig.p_win:.0f}%</code> | EV <code>{sig.ev_r:+.2f}R</code> | KER <code>{sig.ker:.2f}</code> | <code>{sig.cross_ctx}</code>"
        )


class TrailingStopManager:
    def __init__(self, state: StateManager, notifier: Notifier) -> None:
        self.state = state
        self.notifier = notifier

    def update_active_trades(self, bundles: Dict[str, Dict[str, Any]]) -> List[str]:
        active = self.state.get_active_signals()
        if not active:
            return []
        events: List[str] = []

        for tr in active:
            sig_id, symbol, mode = int(tr["id"]), tr["symbol"], tr["mode"]
            direction, status = tr["direction"], tr["status"]
            bundle = bundles.get(symbol)
            if not bundle:
                continue

            cfg = SCALP_CONFIG if mode == "SCALP" else SWING_CONFIG
            df_ref = bundle["live"]["5m"] if mode == "SCALP" else bundle["live"]["15m"]
            if df_ref.empty:
                df_ref = bundle["live"]["5m"]
            if df_ref.empty:
                continue

            latest_price = bundle["spot"] if bundle["spot"] > 0 else float(df_ref["Close"].iloc[-1])
            since_df = df_ref[df_ref.index >= datetime.fromisoformat(tr["created_at"])]
            high_since = max(float(since_df["High"].max()), latest_price) if not since_df.empty else latest_price
            low_since = min(float(since_df["Low"].min()), latest_price) if not since_df.empty else latest_price

            cur_sl, entry = float(tr["current_sl"]), float(tr["entry"])
            tp1, tp2, tp3, atr = float(tr["tp1"]), float(tr["tp2"]), float(tr["tp3"]), float(tr["atr"])
            be_buf = entry * 0.0004
            trail_step_tol = atr * 0.15

            if direction == "BUY":
                if low_since <= cur_sl:
                    cs = "CLOSED_BE" if cur_sl >= entry else "CLOSED_SL"
                    self.state.update_signal_trail(sig_id, cur_sl, cs)
                    msg = f"🛑 <b>{symbol} | {mode} BUY #{sig_id} {cs}</b> @ <code>{cur_sl:.2f}</code> (Spot: <code>{latest_price:.2f}</code>)"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue
                if high_since >= tp3:
                    self.state.update_signal_trail(sig_id, tp2, "CLOSED_TP3")
                    msg = f"🏆 <b>{symbol} | {mode} BUY #{sig_id} FULL TP3 HIT</b> @ <code>{tp3:.2f}</code>!"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                new_sl, new_status = cur_sl, status
                if high_since >= tp2 and status != "TP2_TRAIL":
                    new_status = "TP2_TRAIL"
                    new_sl = max(cur_sl, tp1, latest_price - atr * cfg.trail_atr_mult)
                elif high_since >= tp1 and status in ("OPEN", "BE_LOCKED"):
                    new_status = "TP1_TRAIL"
                    new_sl = max(cur_sl, entry + be_buf)
                elif status == "TP2_TRAIL" or (mode == "SWING" and status == "TP1_TRAIL"):
                    new_sl = max(cur_sl, latest_price - atr * cfg.trail_atr_mult)

                if new_status != status or new_sl > cur_sl + trail_step_tol:
                    self.state.update_signal_trail(sig_id, new_sl, new_status)
                    msg = f"🔄 <b>TRAIL | {symbol} {mode} BUY #{sig_id}</b> Status: <code>{new_status}</code> | SL: <code>{new_sl:.2f}</code> | Spot: <code>{latest_price:.2f}</code>"
                    events.append(msg)
                    self.notifier.broadcast(msg)
            else:
                if high_since >= cur_sl:
                    cs = "CLOSED_BE" if cur_sl <= entry else "CLOSED_SL"
                    self.state.update_signal_trail(sig_id, cur_sl, cs)
                    msg = f"🛑 <b>{symbol} | {mode} SELL #{sig_id} {cs}</b> @ <code>{cur_sl:.2f}</code> (Spot: <code>{latest_price:.2f}</code>)"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue
                if low_since <= tp3:
                    self.state.update_signal_trail(sig_id, tp2, "CLOSED_TP3")
                    msg = f"🏆 <b>{symbol} | {mode} SELL #{sig_id} FULL TP3 HIT</b> @ <code>{tp3:.2f}</code>!"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                new_sl, new_status = cur_sl, status
                if low_since <= tp2 and status != "TP2_TRAIL":
                    new_status = "TP2_TRAIL"
                    new_sl = min(cur_sl, tp1, latest_price + atr * cfg.trail_atr_mult)
                elif low_since <= tp1 and status in ("OPEN", "BE_LOCKED"):
                    new_status = "TP1_TRAIL"
                    new_sl = min(cur_sl, entry - be_buf)
                elif status == "TP2_TRAIL" or (mode == "SWING" and status == "TP1_TRAIL"):
                    new_sl = min(cur_sl, latest_price + atr * cfg.trail_atr_mult)

                if new_status != status or new_sl < cur_sl - trail_step_tol:
                    self.state.update_signal_trail(sig_id, new_sl, new_status)
                    msg = f"🔄 <b>TRAIL | {symbol} {mode} SELL #{sig_id}</b> Status: <code>{new_status}</code> | SL: <code>{new_sl:.2f}</code> | Spot: <code>{latest_price:.2f}</code>"
                    events.append(msg)
                    self.notifier.broadcast(msg)
        return events


# ==============================================================================
# FORENSIC AUDIT LOGGER & MAIN ORCHESTRATOR
# ==============================================================================
def forensic_audit_log(
    bundles: Dict[str, Dict[str, Any]],
    all_stats: Dict[Tuple[str, str], Dict[str, Any]],
    all_sigs: Dict[Tuple[str, str], SignalCandidate],
    cross_tag: str,
    active_trades: List[Dict[str, Any]],
) -> None:
    _, _, sess_tag = check_crypto_session()
    lines = [
        "=" * 90,
        "APEX-QUANT-OMNI — HYPERION DUAL-CRYPTO FORENSIC AUDIT LOG (BTCUSD & ETHUSD)",
        "=" * 90,
        f"Timestamp UTC     : {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Session={sess_tag} | Tier={SIGNAL_TIER_MODE} | {cross_tag}",
    ]
    for sym in ["BTCUSD", "ETHUSD"]:
        b = bundles[sym]
        lines.append(f"[{sym} FEEDS]    : Spot=${b['spot']:.2f} | Spread=${b['spread']:.2f} | {json.dumps(b['sources'])}")
    lines.append("-" * 90)
    for sym in ["BTCUSD", "ETHUSD"]:
        for m in ["SCALP", "SWING"]:
            st = all_stats[(sym, m)]
            lines.append(
                f"[{sym} {m} PRO]   : Bars={st.get('bt_bars')} | ATR={st.get('opt_atr_period')} | Z={st.get('opt_z_thresh')} | "
                f"T={st.get('trades')} | WR={st.get('win_rate')}% | TP1Hit={st.get('tp1_hit_rate')}% | PF={st.get('profit_factor')} | "
                f"OOS={st.get('oos_win_rate')}%/{st.get('oos_trades')}T (EV={st.get('oos_ev_r'):+.2f}R) | DD={st.get('max_dd_pct')}% | Ret={st.get('total_return_pct'):+.2f}%"
            )
            lines.append(
                f"[{sym} {m} SNIPER]: T={st.get('sniper_trades')} | WR={st.get('sniper_wr')}% | PF={st.get('sniper_pf')} | "
                f"OOS={st.get('sniper_oos_wr')}%/{st.get('sniper_oos_trades')}T | DD={st.get('sniper_dd')}% | Ret={st.get('sniper_ret'):+.2f}%"
            )
    lines.append("-" * 90)
    for sym in ["BTCUSD", "ETHUSD"]:
        for m in ["SCALP", "SWING"]:
            sg = all_sigs[(sym, m)]
            lines.append(
                f"[LIVE {sym} {m}]: {sg.direction} ({sg.tier}) | Px={sg.entry} | Score={sg.score} | Regime={sg.regime} | KER={sg.ker} | ADX={sg.adx} | Reason={sg.reason}"
            )
    lines.append(f"[ACTIVE TRADES]   : {len(active_trades)} active position(s)")
    for t in active_trades:
        lines.append(f"   -> #{t['id']} {t['symbol']} {t['mode']} {t['direction']} Entry={t['entry']} CurSL={t['current_sl']} Status={t['status']}")
    lines.append("=" * 90)
    print("\n".join(lines), flush=True)


def format_on_demand_check(sig: SignalCandidate, bt_stats: Dict[str, Any]) -> str:
    if sig.direction in ("BUY", "SELL"):
        return SignalEngine.format_telegram_signal(sig)
    return (
        f"⏸ <b>{sig.symbol} | {sig.mode} | NO_SIGNAL</b>\n"
        f"💵 Spot: <code>{sig.entry:.2f}</code> | Skor: <code>{sig.score:.0f}</code> | Regime: <code>{sig.regime}</code>\n"
        f"🛡 Filter: <code>{sig.reason}</code> | TP1Hit: <code>{bt_stats.get('tp1_hit_rate', 0):.1f}%</code>"
    )


def format_status_message(
    state: StateManager, bundles: Dict[str, Dict[str, Any]],
    all_stats: Dict[Tuple[str, str], Dict[str, Any]], cross_tag: str,
) -> str:
    active = state.get_active_signals()
    _, _, sess_tag = check_crypto_session()
    btc_px, eth_px = bundles["BTCUSD"]["spot"], bundles["ETHUSD"]["spot"]
    lines = [
        f"📊 <b>APEX-QUANT CRYPTO STATUS</b> | 🕒 <code>{utc_now().strftime('%H:%M UTC')}</code>",
        f"🪙 BTC: <code>{btc_px:.2f}</code> | ETH: <code>{eth_px:.2f}</code> | <code>{sess_tag}</code> | <code>{cross_tag}</code>",
        f"⚡ BTC OOS: Scalp <code>{all_stats[('BTCUSD','SCALP')].get('oos_win_rate',0):.1f}%</code> | Swing <code>{all_stats[('BTCUSD','SWING')].get('oos_win_rate',0):.1f}%</code>",
        f"⚡ ETH OOS: Scalp <code>{all_stats[('ETHUSD','SCALP')].get('oos_win_rate',0):.1f}%</code> | Swing <code>{all_stats[('ETHUSD','SWING')].get('oos_win_rate',0):.1f}%</code>",
        f"📌 Posisi Aktif: <code>{len(active)}</code>",
    ]
    for tr in active[:4]:
        px = bundles.get(tr["symbol"], {}).get("spot", float(tr["entry"]))
        sl_dist = abs(float(tr["entry"]) - float(tr["initial_sl"])) + 1e-8
        float_r = ((px - float(tr["entry"])) if tr["direction"] == "BUY" else (float(tr["entry"]) - px)) / sl_dist
        lines.append(
            f"• #{tr['id']} <b>{tr['symbol']} {tr['mode']} {tr['direction']}</b> @ <code>{tr['entry']:.2f}</code> | "
            f"SL: <code>{tr['current_sl']:.2f}</code> | PnL: <code>{float_r:+.2f}R</code> | <code>{tr['status']}</code>"
        )
    return "\n".join(lines)


def format_history_message(state: StateManager) -> str:
    recent = state.get_recent_signals(limit=6)
    if not recent:
        return "📜 <b>CRYPTO HISTORY</b>\nBelum ada sinyal terekam di database."
    lines = ["📜 <b>CRYPTO LAST 6 SIGNALS</b>"]
    for r in recent:
        lines.append(
            f"• #{r['id']} <b>{r['symbol']} {r['mode']} {r['direction']}</b> @ <code>{r['entry']:.2f}</code> | SL <code>{r['current_sl']:.2f}</code> | <code>{r['status']}</code>"
        )
    return "\n".join(lines)


def ensure_fallback_chart_if_missing(out_path: str = CHART_PATH) -> None:
    if not os.path.exists(out_path):
        BacktestAndOptimizer.generate_performance_chart({}, out_path)


def should_emit_tier(sig: SignalCandidate, tier_mode: str) -> bool:
    if sig.direction not in ("BUY", "SELL"):
        return False
    if tier_mode == "SNIPER_ONLY":
        return sig.tier == "SNIPER"
    if tier_mode == "PRO_ONLY":
        return sig.tier in ("PRO", "SNIPER")
    return True


def main() -> int:
    state = StateManager(DB_PATH)
    notifier = Notifier(state)

    try:
        cb_open, cb_until = state.is_circuit_open()
        pending_cmds = notifier.poll_commands()
        manual_cmd = os.getenv("BOT_COMMAND", "auto").strip().lower()

        if cb_open and manual_cmd == "auto" and not pending_cmds:
            logger.warning(f"Circuit Breaker OPEN until {cb_until}; skipping.")
            ensure_fallback_chart_if_missing()
            return 0

        data_engine = CryptoDataEngine()
        bundles: Dict[str, Dict[str, Any]] = {
            spec.symbol: data_engine.load_asset_bundle(spec) for spec in ASSETS
        }

        for sym, b in bundles.items():
            if b["live"]["5m"].empty or b["live"]["15m"].empty or b["live"]["1h"].empty or b["live"]["4h"].empty:
                failures, tripped = state.record_failure()
                err_msg = f"⚠️ <b>APEX {sym} DATA FAILOVER EXHAUSTED</b> | Sources: <code>{b['sources']}</code> | Fails: <code>{failures}</code>"
                logger.error(err_msg)
                if tripped or pending_cmds:
                    notifier.broadcast(err_msg)
                ensure_fallback_chart_if_missing()
                return 1

        # Update active trailing stops across BTCUSD & ETHUSD
        trail_mgr = TrailingStopManager(state, notifier)
        trail_mgr.update_active_trades(bundles)

        # Optimize & Enrich BTCUSD first so BTC Macro Leader signal feeds ETHUSD Cross-Crypto Veto
        all_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
        all_sigs: Dict[Tuple[str, str], SignalCandidate] = {}

        # 1. BTCUSD Optimization & Live Evaluation
        btc_b = bundles["BTCUSD"]
        _, btc_z_sc, btc_enr_sc, btc_st_sc, btc_cal_sc = BacktestAndOptimizer.grid_search_optimize(btc_b, SCALP_CONFIG)
        _, btc_z_sw, btc_enr_sw, btc_st_sw, btc_cal_sw = BacktestAndOptimizer.grid_search_optimize(btc_b, SWING_CONFIG)
        all_stats[("BTCUSD", "SCALP")] = btc_st_sc
        all_stats[("BTCUSD", "SWING")] = btc_st_sw

        # Determine BTC macro leader direction from BTC 1H/15M MTF + Z-Score
        btc_last_row = btc_enr_sw.iloc[-1]
        btc_htf_bias = int(btc_last_row.get("htf_bias", 0))
        btc_mid_bias = int(btc_last_row.get("mid_bias", 0))
        btc_z = float(btc_last_row.get("z_score", 0.0))
        if btc_htf_bias > 0 and btc_mid_bias > 0 and btc_z > -0.5:
            btc_leader_dir = 1
        elif btc_htf_bias < 0 and btc_mid_bias < 0 and btc_z < 0.5:
            btc_leader_dir = -1
        else:
            btc_leader_dir = 0
        cross_tag = f"BTC:{'BULL' if btc_leader_dir > 0 else ('BEAR' if btc_leader_dir < 0 else 'NEUT')}"

        for cfg, enr_df, opt_z, st, cal in [
            (SCALP_CONFIG, btc_enr_sc, btc_z_sc, btc_st_sc, btc_cal_sc),
            (SWING_CONFIG, btc_enr_sw, btc_z_sw, btc_st_sw, btc_cal_sw),
        ]:
            ok_exec, msg_exec = CryptoDataEngine.validate_freshness_and_shock(btc_b["live"][cfg.exec_tf], cfg.stale_max_minutes)
            ok_mid, msg_mid = CryptoDataEngine.validate_freshness_and_shock(btc_b["live"][cfg.mid_tf], 120)
            if ok_exec and ok_mid:
                all_sigs[("BTCUSD", cfg.mode_name)] = SignalEngine.build_live_signal(
                    enr_df, BTC_SPEC, cfg, opt_z, st, cal, btc_b["spread"], btc_b["spot"],
                    btc_leader_dir=0, cross_tag=cross_tag,
                )
            else:
                all_sigs[("BTCUSD", cfg.mode_name)] = SignalCandidate(
                    symbol="BTCUSD", mode=cfg.mode_name, direction="NO_SIGNAL",
                    entry=round(btc_b["spot"], 2), regime="DATA_GUARD", reason=f"{cfg.exec_tf}:{msg_exec}|{cfg.mid_tf}:{msg_mid}",
                )

        # 2. ETHUSD Optimization & Live Evaluation (with BTC Leader Cross-Crypto Alignment)
        eth_b = bundles["ETHUSD"]
        _, eth_z_sc, eth_enr_sc, eth_st_sc, eth_cal_sc = BacktestAndOptimizer.grid_search_optimize(eth_b, SCALP_CONFIG)
        _, eth_z_sw, eth_enr_sw, eth_st_sw, eth_cal_sw = BacktestAndOptimizer.grid_search_optimize(eth_b, SWING_CONFIG)
        all_stats[("ETHUSD", "SCALP")] = eth_st_sc
        all_stats[("ETHUSD", "SWING")] = eth_st_sw

        for cfg, enr_df, opt_z, st, cal in [
            (SCALP_CONFIG, eth_enr_sc, eth_z_sc, eth_st_sc, eth_cal_sc),
            (SWING_CONFIG, eth_enr_sw, eth_z_sw, eth_st_sw, eth_cal_sw),
        ]:
            ok_exec, msg_exec = CryptoDataEngine.validate_freshness_and_shock(eth_b["live"][cfg.exec_tf], cfg.stale_max_minutes)
            ok_mid, msg_mid = CryptoDataEngine.validate_freshness_and_shock(eth_b["live"][cfg.mid_tf], 120)
            if ok_exec and ok_mid:
                all_sigs[("ETHUSD", cfg.mode_name)] = SignalEngine.build_live_signal(
                    enr_df, ETH_SPEC, cfg, opt_z, st, cal, eth_b["spread"], eth_b["spot"],
                    btc_leader_dir=btc_leader_dir, cross_tag=cross_tag,
                )
            else:
                all_sigs[("ETHUSD", cfg.mode_name)] = SignalCandidate(
                    symbol="ETHUSD", mode=cfg.mode_name, direction="NO_SIGNAL",
                    entry=round(eth_b["spot"], 2), regime="DATA_GUARD", reason=f"{cfg.exec_tf}:{msg_exec}|{cfg.mid_tf}:{msg_mid}",
                )

        BacktestAndOptimizer.generate_performance_chart(all_stats, CHART_PATH)

        # 3. Interactive Commands
        if manual_cmd in ("check", "btc", "eth", "scalp", "swing", "status", "chart", "history", "sniper"):
            target = notifier.authorized_chat_ids[0] if notifier.authorized_chat_ids else None
            pending_cmds.append((target or "", f"/{manual_cmd}"))

        for chat_id, raw_cmd in pending_cmds:
            cmd = raw_cmd.lower().strip()
            target_cid = chat_id if chat_id else None
            if cmd.startswith("/status"):
                notifier.broadcast(format_status_message(state, bundles, all_stats, cross_tag), target_chat_id=target_cid)
            elif cmd.startswith("/chart"):
                cap = (
                    f"📈 <b>BTCUSD & ETHUSD Walk-Forward</b>\n"
                    f"BTC: Scalp <code>{btc_st_sc.get('total_return_pct',0):+.1f}%</code> | Swing <code>{btc_st_sw.get('total_return_pct',0):+.1f}%</code>\n"
                    f"ETH: Scalp <code>{eth_st_sc.get('total_return_pct',0):+.1f}%</code> | Swing <code>{eth_st_sw.get('total_return_pct',0):+.1f}%</code>"
                )
                notifier.send_telegram_photo(CHART_PATH, caption=cap, chat_id=target_cid)
            elif cmd.startswith("/history"):
                notifier.broadcast(format_history_message(state), target_chat_id=target_cid)
            elif cmd.startswith("/sniper"):
                lines = [f"💎 <b>CRYPTO SNIPER SNAPSHOT</b> (<code>{cross_tag}</code>)"]
                for sym in ["BTCUSD", "ETHUSD"]:
                    for m in ["SCALP", "SWING"]:
                        sg = all_sigs[(sym, m)]
                        lines.append(f"• {sym} {m}: <code>{sg.direction} ({sg.tier})</code> {sg.score:.0f}/86 | {sg.reason}")
                notifier.broadcast("\n".join(lines), target_chat_id=target_cid)
            elif cmd.startswith("/help"):
                notifier.broadcast(
                    "🤖 <b>APEX-QUANT CRYPTO COMMANDS</b>\n"
                    "<code>/check</code> — Analisa BTC & ETH (Scalp & Swing)\n"
                    "<code>/check btc</code> | <code>/check eth</code>\n"
                    "<code>/check scalp</code> | <code>/check swing</code>\n"
                    "<code>/sniper</code> — Cek Peluang Sniper A+\n"
                    "<code>/status</code> — Spot, Posisi & OOS WR\n"
                    "<code>/chart</code> — Grafik 4-Panel Equity\n"
                    "<code>/history</code> — 6 Sinyal Terakhir",
                    target_chat_id=target_cid,
                )
            elif cmd.startswith("/check") or cmd in ("/btc", "/eth", "/scalp", "/swing"):
                syms = ["BTCUSD"] if "btc" in cmd else (["ETHUSD"] if "eth" in cmd else ["BTCUSD", "ETHUSD"])
                modes = ["SCALP"] if "scalp" in cmd else (["SWING"] if "swing" in cmd else ["SCALP", "SWING"])
                for s in syms:
                    for m in modes:
                        notifier.broadcast(format_on_demand_check(all_sigs[(s, m)], all_stats[(s, m)]), target_chat_id=target_cid)

        # 4. Auto-Emit Signals (Deduplicated & Cooldown-Protected per Symbol & Mode)
        for sym in ["BTCUSD", "ETHUSD"]:
            for cfg in [SCALP_CONFIG, SWING_CONFIG]:
                sig = all_sigs[(sym, cfg.mode_name)]
                if should_emit_tier(sig, SIGNAL_TIER_MODE):
                    blocked, block_reason = state.is_duplicate_or_cooldown(
                        sig.symbol, sig.mode, sig.direction, sig.candle_ts, cfg.cooldown_minutes, sig.score
                    )
                    if blocked:
                        logger.info(f"[{sig.symbol} {sig.mode}] Signal suppressed: {block_reason}")
                    else:
                        sig_id = state.insert_signal(sig)
                        logger.info(f"[{sig.symbol} {sig.mode}] Emitted #{sig_id} [{sig.tier}] {sig.direction} @ {sig.entry}")
                        notifier.broadcast(SignalEngine.format_telegram_signal(sig))
                elif sig.direction in ("BUY", "SELL"):
                    logger.info(f"[{sig.symbol} {sig.mode}] Tier {sig.tier} skipped by SIGNAL_TIER_MODE={SIGNAL_TIER_MODE}")

        state.record_success()
        forensic_audit_log(bundles, all_stats, all_sigs, cross_tag, state.get_active_signals())
        return 0

    except Exception as exc:
        failures, _ = state.record_failure()
        tb_short = traceback.format_exc()[-500:]
        logger.exception("Fatal error in APEX-QUANT-CRYPTO main loop")
        notifier.broadcast(
            f"🚨 <b>APEX-QUANT CRYPTO SYSTEM ERROR</b> (Fails: <code>{failures}</code>)\n"
            f"<code>{str(exc)[:180]}</code>\n<code>{tb_short[:300]}</code>"
        )
        ensure_fallback_chart_if_missing()
        return 1


if __name__ == "__main__":
    sys.exit(main())
