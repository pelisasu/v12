#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APEX-QUANT-OMNI — HYPERION APEX EDITION (NASDAQ 100 / US100 / NDX)
Dual-Mode (SCALP & SWING) + Dual-Tier (PRO & SNIPER) Signal, Walk-Forward OOS
Backtest, Intermarket US10Y/VIX Veto, USD News Guard & Auto-Trailing Engine.
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

try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    NY_TZ = timezone(timedelta(hours=-5))

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
logger = logging.getLogger("APEX_QUANT_NAS100")

SYMBOL_LABEL = os.getenv("SYMBOL_LABEL", "NAS100").strip().upper()
DB_PATH = os.getenv("DB_PATH", "trading_signals.db")
CHART_PATH = os.getenv("CHART_PATH", "backtest_performance.png")

CONTRACT_SIZE = float(os.getenv("CONTRACT_SIZE", "1.0"))
PIP_SIZE = float(os.getenv("PIP_SIZE", "0.10"))
TICK_VALUE = float(os.getenv("TICK_VALUE", str(CONTRACT_SIZE * PIP_SIZE)))
MIN_LOT = 0.01
LOT_STEP = 0.01
MAX_LOT = float(os.getenv("MAX_LOT", "20.0"))

BROKER_PRICE_SCALE = float(os.getenv("BROKER_PRICE_SCALE", "1.0"))
BROKER_PRICE_OFFSET = float(os.getenv("BROKER_PRICE_OFFSET", "0.0"))

SPREAD_POINTS = float(os.getenv("SPREAD_POINTS", "1.20"))
SLIPPAGE_POINTS = float(os.getenv("SLIPPAGE_POINTS", "0.40"))

DEFAULT_ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000.0"))
DEFAULT_RISK_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
SIGNAL_TIER_MODE = os.getenv("SIGNAL_TIER_MODE", "BOTH").strip().upper()

RETRY_BACKOFFS = [2, 4, 8]
CIRCUIT_MAX_FAILURES = 5
CIRCUIT_COOLDOWN_MINUTES = 60
GRID_ATR_PERIODS = [10, 14, 20]
GRID_Z_THRESHOLDS = [1.5, 2.0, 2.5]


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
    be_buffer_pts: float
    min_sl_pts: float
    use_swing_anchor_sl: bool
    bos_lookback: int


SCALP_CONFIG = ModeConfig(
    mode_name="SCALP",
    exec_tf="5m",
    mid_tf="15m",
    htf_tf="1h",
    sl_atr_mult=1.60,
    tp_r_multiples=(1.20, 2.00, 3.00),
    tp1_close_pct=0.55,
    trail_atr_mult=0.95,
    min_score=76.0,
    sniper_score=86.0,
    min_ker=0.33,
    max_chop=56.0,
    min_adx=19.0,
    cooldown_minutes=30,
    min_backtest_trades=8,
    z_window=20,
    vwap_window=48,
    swing_lr=3,
    stale_max_minutes=45,
    be_buffer_pts=1.50,
    min_sl_pts=18.0,
    use_swing_anchor_sl=False,
    bos_lookback=14,
)

SWING_CONFIG = ModeConfig(
    mode_name="SWING",
    exec_tf="15m",
    mid_tf="1h",
    htf_tf="4h",
    sl_atr_mult=1.85,
    tp_r_multiples=(1.50, 2.50, 4.00),
    tp1_close_pct=0.50,
    trail_atr_mult=0.85,
    min_score=75.0,
    sniper_score=86.0,
    min_ker=0.34,
    max_chop=56.0,
    min_adx=20.0,
    cooldown_minutes=120,
    min_backtest_trades=5,
    z_window=24,
    vwap_window=32,
    swing_lr=4,
    stale_max_minutes=75,
    be_buffer_pts=2.50,
    min_sl_pts=30.0,
    use_swing_anchor_sl=True,
    bos_lookback=20,
)


@dataclass
class SignalCandidate:
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
    confluence_pct: float = 0.0
    p_win: float = 0.0
    ev_r: float = 0.0
    ker: float = 0.0
    adx: float = 0.0
    z_score: float = 0.0
    atr: float = 0.0
    regime: str = "UNKNOWN"
    mtf_bias: str = "NEUTRAL"
    macro_ctx: str = "NEUTRAL"
    candle_ts: str = ""
    reason: str = ""
    pillar_details: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_market_open(now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
    dt_utc = now_utc or utc_now()
    ny = dt_utc.astimezone(NY_TZ)
    wd, mins = ny.weekday(), ny.hour * 60 + ny.minute

    if wd == 4 and mins >= (16 * 60 + 50):
        return False, "WEEKEND_CLOSED"
    if wd == 5:
        return False, "WEEKEND_CLOSED"
    if wd == 6 and mins < (18 * 60 + 5):
        return False, "WEEKEND_CLOSED"
    if (16 * 60 + 55) <= mins <= (18 * 60 + 5):
        return False, "DAILY_MAINT_ROLLOVER"
    if wd == 4 and mins >= (16 * 60 + 30):
        return False, "FRIDAY_LATE_CLOSE"
    return True, "ACTIVE_SESSION"


def is_us_cash_or_premarket_peak(dt_utc: datetime) -> Tuple[bool, bool]:
    ny = dt_utc.astimezone(NY_TZ)
    mins = ny.hour * 60 + ny.minute
    is_peak = (8 * 60 + 30) <= mins <= (16 * 60)
    is_open_trap = (9 * 60 + 28) <= mins <= (9 * 60 + 36)
    return is_peak, is_open_trap


def calculate_nas100_lot_size(
    entry_price: float,
    sl_price: float,
    account_balance: float = DEFAULT_ACCOUNT_BALANCE,
    risk_pct: float = DEFAULT_RISK_PCT,
) -> float:
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 1e-6 or account_balance <= 0 or risk_pct <= 0:
        return MIN_LOT
    risk_usd = account_balance * (risk_pct / 100.0)
    sl_pips = sl_distance / PIP_SIZE
    raw_lot = risk_usd / (sl_pips * TICK_VALUE + 1e-8)
    rounded_lot = math.floor(raw_lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(MAX_LOT, round(rounded_lot, 2)))


# ==============================================================================
# SQLITE STATE MANAGER
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_mode_ts ON signals(mode, candle_ts)")
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
        self, mode: str, direction: str, candle_ts: str, cooldown_minutes: int, score: float
    ) -> Tuple[bool, str]:
        with self._connect() as conn:
            dup = conn.execute(
                "SELECT id FROM signals WHERE mode = ? AND direction = ? AND candle_ts = ?",
                (mode, direction, candle_ts),
            ).fetchone()
            if dup:
                return True, f"DUPLICATE_CANDLE_{candle_ts}"

            last_sig = conn.execute(
                "SELECT direction, created_at FROM signals WHERE mode = ? ORDER BY id DESC LIMIT 1",
                (mode,),
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
                    mode, direction, entry, initial_sl, current_sl,
                    tp1, tp2, tp3, lot_size, score, p_win, ev_r,
                    atr, regime, status, candle_ts, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
                """,
                (
                    sig.mode, sig.direction, sig.entry, sig.sl, sig.sl,
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

    def get_recent_signals(self, limit: int = 5) -> List[Dict[str, Any]]:
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
# NOTIFICATION & INTERACTIVE TELEGRAM POLLING
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
# MACRO NEWS BLACKOUT & INTERMARKET VETO ENGINE (US10Y YIELD + VIX + US NEWS)
# ==============================================================================
class MacroAndNewsGuard:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
    }

    @classmethod
    def check_usd_news_blackout(cls, now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
        dt = now_utc or utc_now()
        try:
            r_biq = requests.get(
                "https://biquote.io/api/calendar",
                params={"countries": "US", "importance": "high"},
                headers=cls.HEADERS,
                timeout=6,
            )
            if r_biq.status_code == 200:
                payload = r_biq.json()
                events = payload if isinstance(payload, list) else (payload.get("events") or payload.get("data") or [])
                for ev in events:
                    raw_dt = ev.get("date") or ev.get("time") or ev.get("timestamp")
                    if not raw_dt:
                        continue
                    ev_dt = datetime.fromisoformat(str(raw_dt).replace("Z", "+00:00")).astimezone(timezone.utc)
                    if abs((dt - ev_dt).total_seconds()) / 60.0 <= 30.0:
                        title = str(ev.get("title") or ev.get("event") or "US_HIGH_NEWS")[:24]
                        return True, f"NEWS_BLACKOUT({title})"
                return False, "NEWS_CLEAR(BIQ)"
        except Exception:
            pass

        try:
            resp = requests.get(
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                headers=cls.HEADERS,
                timeout=6,
            )
            if resp.status_code == 200:
                for ev in resp.json():
                    if str(ev.get("country", "")).upper() != "USD" or str(ev.get("impact", "")).lower() != "high":
                        continue
                    raw_date = ev.get("date")
                    if not raw_date:
                        continue
                    ev_dt = datetime.fromisoformat(raw_date).astimezone(timezone.utc)
                    if abs((dt - ev_dt).total_seconds()) / 60.0 <= 30.0:
                        return True, f"NEWS_BLACKOUT({str(ev.get('title', 'USD_NEWS'))[:24]})"
                return False, "NEWS_CLEAR(FF)"
        except Exception:
            pass

        ny_dt = dt.astimezone(NY_TZ)
        if ny_dt.weekday() == 3 and ny_dt.hour == 8 and 20 <= ny_dt.minute <= 45:
            return True, "NEWS_BLACKOUT(US_0830ET)"
        if ny_dt.weekday() == 4 and ny_dt.day <= 7 and ny_dt.hour == 8 and ny_dt.minute >= 15:
            return True, "NEWS_BLACKOUT(NFP_WINDOW)"
        return False, "NEWS_CLEAR(SCHED)"

    @classmethod
    def fetch_intermarket_bias(cls) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {"tnx_z": 0.0, "tnx_dir": 0, "vix": 0.0, "vix_panic": False, "tag": "TNX:NEUT|VIX:OK"}
        try:
            tnx = yf.download("^TNX", period="5d", interval="15m", progress=False, threads=False, timeout=10)
            if isinstance(tnx.columns, pd.MultiIndex):
                tnx.columns = tnx.columns.get_level_values(0)
            c = pd.to_numeric(tnx["Close"], errors="coerce").dropna() if not tnx.empty else pd.Series(dtype=float)
            if len(c) >= 25:
                z = float((c.iloc[-1] - c.rolling(20).mean().iloc[-1]) / (c.rolling(20).std(ddof=0).iloc[-1] + 1e-8))
                ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
                ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
                slope = (ema9 - ema21) / (float(c.diff().abs().rolling(14).mean().iloc[-1]) + 1e-8)
                t_dir = 1 if (z > 1.25 and slope > 0.25) else (-1 if (z < -1.25 and slope < -0.25) else 0)
                ctx["tnx_z"] = round(z, 2)
                ctx["tnx_dir"] = t_dir
                tnx_lbl = "SURGE" if t_dir > 0 else ("DROP" if t_dir < 0 else "NEUT")
                ctx["tag"] = f"TNX:{tnx_lbl}({z:+.1f})"
        except Exception:
            pass

        try:
            vix = yf.download("^VIX", period="5d", interval="15m", progress=False, threads=False, timeout=8)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            cv = pd.to_numeric(vix["Close"], errors="coerce").dropna() if not vix.empty else pd.Series(dtype=float)
            if len(cv) >= 20:
                v_last = float(cv.iloc[-1])
                vz = float((v_last - cv.rolling(20).mean().iloc[-1]) / (cv.rolling(20).std(ddof=0).iloc[-1] + 1e-8))
                ctx["vix"] = round(v_last, 2)
                ctx["vix_panic"] = (v_last > 35.0) or (vz > 2.5)
                ctx["tag"] += f"|VIX:{v_last:.1f}"
        except Exception:
            pass
        return ctx


# ==============================================================================
# MULTI-SOURCE DATA ENGINE (USTEC / ^NDX / QQQ) + SHOCK GUARD
# ==============================================================================
class MarketDataEngine:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
    }

    @staticmethod
    def _apply_broker_calibration(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or (BROKER_PRICE_SCALE == 1.0 and BROKER_PRICE_OFFSET == 0.0):
            return df
        out = df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            out[col] = out[col] * BROKER_PRICE_SCALE + BROKER_PRICE_OFFSET
        return out

    @staticmethod
    def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
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

        valid_mask = (
            (df["Open"] > 10.0)
            & (df["Close"] > 10.0)
            & (df["High"] >= df[["Open", "Close"]].max(axis=1))
            & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
            & (df["Low"] > 10.0)
        )
        df = df[valid_mask]
        pct_jump = df["Close"].pct_change().abs().fillna(0.0)
        bar_range_pct = ((df["High"] - df["Low"]) / (df["Close"] + 1e-8)).fillna(0.0)
        return df[(pct_jump < 0.05) & (bar_range_pct < 0.06)]

    def fetch_live_spot_quote(self) -> Dict[str, Any]:
        try:
            r = requests.get("https://biquote.io/api/USTEC", headers=self.HEADERS, timeout=8)
            if r.status_code == 200:
                q = r.json()
                bid, ask, mid = float(q.get("bid", 0) or 0), float(q.get("ask", 0) or 0), float(q.get("mid", 0) or 0)
                if mid > 1000.0:
                    mid_cal = mid * BROKER_PRICE_SCALE + BROKER_PRICE_OFFSET
                    raw_sp = abs(ask - bid) if (ask > bid > 1000.0) else float(q.get("spread", SPREAD_POINTS) or SPREAD_POINTS)
                    sp_cal = max(0.50, min(6.00, raw_sp))
                    return {"mid": round(mid_cal, 2), "spread": round(sp_cal, 2), "sources": ["biquote:USTEC"]}
        except Exception:
            pass
        # Fallback reference spot from ^NDX if USTEC quote endpoint is offline
        ndx_ref = self._fetch_yfinance("^NDX", "1d", "5d")
        if not ndx_ref.empty:
            mid_ndx = float(ndx_ref["Close"].iloc[-1]) * BROKER_PRICE_SCALE + BROKER_PRICE_OFFSET
            return {"mid": round(mid_ndx, 2), "spread": SPREAD_POINTS, "sources": ["yf:^NDX"]}
        return {"mid": 0.0, "spread": SPREAD_POINTS, "sources": []}

    def _fetch_biquote_ohlc(self, interval: str, limit: int = 500) -> pd.DataFrame:
        try:
            resp = requests.get(
                "https://biquote.io/api/USTEC/ohlc",
                params={"interval": interval, "limit": limit},
                headers=self.HEADERS,
                timeout=12,
            )
            if resp.status_code != 200:
                return pd.DataFrame()
            bars = [b for b in (resp.json().get("bars") or []) if not b.get("isOpen", False)]
            if not bars:
                return pd.DataFrame()
            records = [
                {
                    "Time": b.get("openTime"),
                    "Open": b.get("open"), "High": b.get("high"),
                    "Low": b.get("low"), "Close": b.get("close"),
                    "Volume": float(b.get("volume") or b.get("tickVolume") or 0.0),
                }
                for b in bars
            ]
            df = pd.DataFrame(records)
            df.index = pd.to_datetime(df["Time"], utc=True, errors="coerce")
            return self._apply_broker_calibration(self._sanitize_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]]))
        except Exception:
            return pd.DataFrame()

    def _fetch_yfinance(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        try:
            raw = yf.download(
                tickers=symbol, period=period, interval=interval,
                auto_adjust=False, progress=False, threads=False, timeout=15,
            )
            return self._sanitize_ohlcv(raw)
        except Exception:
            return pd.DataFrame()

    def _fetch_yahoo_chart_api(self, symbol: str, interval: str, range_str: str) -> pd.DataFrame:
        for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
            try:
                resp = requests.get(
                    f"https://{host}/v8/finance/chart/{symbol}",
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
                clean = self._sanitize_ohlcv(df)
                if not clean.empty:
                    return clean
            except Exception:
                continue
        return pd.DataFrame()

    def fetch_timeframe(self, interval: str, period: str, spot_ref: float, min_rows: int = 120) -> Tuple[pd.DataFrame, str]:
        sources = [
            ("BIQUOTE:USTEC", lambda: self._fetch_biquote_ohlc(interval, limit=500)),
            ("YF:^NDX", lambda: self._apply_broker_calibration(self._fetch_yfinance("^NDX", "1h" if interval == "4h" else interval, period))),
            ("YAPI:^NDX", lambda: self._apply_broker_calibration(self._fetch_yahoo_chart_api("^NDX", "1h" if interval == "4h" else interval, period))),
            ("YF:QQQ(SCALED)", lambda: self._fetch_yfinance("QQQ", "1h" if interval == "4h" else interval, period)),
        ]
        for name, loader in sources:
            df = loader()
            if df.empty:
                continue
            if "QQQ" in name:
                scale = (spot_ref / float(df["Close"].iloc[-1])) if spot_ref > 1000.0 else 41.2
                if 20.0 <= scale <= 80.0:
                    for col in ["Open", "High", "Low", "Close"]:
                        df[col] = df[col] * scale
            if interval == "4h" and name != "BIQUOTE:USTEC":
                df = (
                    df.resample("4h")
                    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                    .dropna(subset=["Open", "High", "Low", "Close"])
                )
            if len(df) >= min_rows:
                logger.info(f"Data [{interval}] loaded from {name} ({len(df)} bars, last={df.index[-1]})")
                return df, name
        return pd.DataFrame(), "NONE"

    def load_deep_calibration_history(self, spot_reference: float) -> Dict[str, Tuple[pd.DataFrame, str]]:
        out: Dict[str, Tuple[pd.DataFrame, str]] = {}
        for label, per in [("5m", "55d"), ("15m", "55d"), ("1h", "90d")]:
            df = self._fetch_yfinance("^NDX", label, per)
            src = "YF:^NDX"
            if df.empty or len(df) < 350:
                df = self._fetch_yfinance("QQQ", label, per)
                src = "YF:QQQ(SCALED)"
            if df.empty or len(df) < 250:
                out[label] = (pd.DataFrame(), "NONE")
                continue
            df = df[df["High"] > df["Low"]].copy()
            if spot_reference > 1000.0 and len(df) > 0:
                ratio = spot_reference / float(df["Close"].iloc[-1])
                if 0.90 <= ratio <= 1.10 or "QQQ" in src:
                    for col in ["Open", "High", "Low", "Close"]:
                        df[col] = df[col] * ratio
            elif "QQQ" in src:
                for col in ["Open", "High", "Low", "Close"]:
                    df[col] = df[col] * 41.2
            out[label] = (df, src)

        df_1h, src_1h = out.get("1h", (pd.DataFrame(), "NONE"))
        if not df_1h.empty:
            df_4h = (
                df_1h.resample("4h")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna(subset=["Open", "High", "Low", "Close"])
            )
            out["4h"] = (df_4h, f"{src_1h}->4h")
        else:
            out["4h"] = (pd.DataFrame(), "NONE")
        return out

    @staticmethod
    def validate_freshness_and_shock(
        df: pd.DataFrame, stale_max_minutes: int, market_open: bool
    ) -> Tuple[bool, str]:
        if df.empty or len(df) < 80:
            return False, "INSUFFICIENT_DATA"
        age_min = (utc_now() - df.index[-1]).total_seconds() / 60.0
        if market_open and age_min > stale_max_minutes:
            return False, f"STALE_DATA_{age_min:.1f}m"

        prev_c = df["Close"].shift(1)
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
        atr_fast, atr_base = tr.rolling(5).mean().iloc[-1], tr.rolling(50).mean().iloc[-1]
        if atr_base > 0 and ((atr_fast / (atr_base + 1e-8)) > 2.85 or (tr.iloc[-1] / (atr_base + 1e-8)) > 4.2):
            return False, f"VOLATILITY_SHOCK_ATR_RATIO={(atr_fast/(atr_base+1e-8)):.2f}"
        return True, "OK"

    def load_all_timeframes(self, spot_ref: float) -> Dict[str, Tuple[pd.DataFrame, str]]:
        df_5m, src_5m = self.fetch_timeframe("5m", "15d", spot_ref, min_rows=120)
        df_15m, src_15m = self.fetch_timeframe("15m", "30d", spot_ref, min_rows=120)
        df_1h, src_1h = self.fetch_timeframe("1h", "90d", spot_ref, min_rows=120)
        df_4h, src_4h = self.fetch_timeframe("4h", "180d", spot_ref, min_rows=80)
        if df_4h.empty and not df_1h.empty:
            df_4h = (
                df_1h.resample("4h")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna(subset=["Open", "High", "Low", "Close"])
            )
            src_4h = f"{src_1h}->4h"
        return {"5m": (df_5m, src_5m), "15m": (df_15m, src_15m), "1h": (df_1h, src_1h), "4h": (df_4h, src_4h)}


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
                "rolling_vwap": rolling_vwap,
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
        bar_range = float(row.get("High", 0.0)) - float(row.get("Low", 0.0))

        if atr > 0 and (bar_range / (atr + 1e-8)) > 2.80:
            return "EXHAUSTION_SPIKE", False, cfg.min_score
        if atr_base > 0 and (atr / (atr_base + 1e-8)) > 2.6:
            return "VOLATILITY_SHOCK", False, cfg.min_score
        if atr_base > 0 and (atr / (atr_base + 1e-8)) < 0.42:
            return "LOW_LIQUIDITY_DEAD", False, cfg.min_score

        is_peak_ny, is_open_trap = is_us_cash_or_premarket_peak(ts_dt)
        if cfg.mode_name == "SCALP" and is_open_trap:
            return "US_OPEN_WHIPSAW_GUARD", False, cfg.min_score

        eff_ker, eff_adx, eff_score = cfg.min_ker, cfg.min_adx, cfg.min_score
        if cfg.mode_name == "SCALP" and not is_peak_ny:
            eff_ker = max(cfg.min_ker, 0.42)
            eff_adx = max(cfg.min_adx, 23.0)
            eff_score = max(cfg.min_score, 82.0)

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
        check_session: bool = True, tnx_dir: int = 0,
    ) -> Tuple[str, float, str, Dict[str, Any]]:
        if idx < 60 or idx >= len(df):
            return "NO_SIGNAL", 0.0, "WARMUP", {"reason": "WARMUP"}

        row = df.iloc[idx]
        ts_dt = df.index[idx].to_pydatetime()

        if check_session:
            sess_ok, sess_reason = is_market_open(ts_dt)
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

        bull_final = min(100.0, bull_tech_score + (5.0 if tnx_dir < 0 else 0.0))
        bear_final = min(100.0, bear_tech_score + (5.0 if tnx_dir > 0 else 0.0))

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

        if bull_valid and tnx_dir > 0:
            return "NO_SIGNAL", round(bull_final, 1), regime, {"reason": "INTERMARKET_TNX_SURGE_VETO", "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}"}
        if bear_valid and tnx_dir < 0:
            return "NO_SIGNAL", round(bear_final, 1), regime, {"reason": "INTERMARKET_TNX_DROP_VETO", "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}"}
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
        base_sl_dist = max(atr * cfg.sl_atr_mult, cfg.min_sl_pts)
        if direction == "BUY":
            sl = entry - base_sl_dist
            if cfg.use_swing_anchor_sl and not np.isnan(last_sl_px) and last_sl_px < entry:
                swing_dist = entry - (last_sl_px - 0.25 * atr)
                if 0.85 * base_sl_dist <= swing_dist <= 1.45 * base_sl_dist:
                    sl = min(sl, last_sl_px - 0.25 * atr)
            sl_dist = max(entry - sl, cfg.min_sl_pts)
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
            sl_dist = max(sl - entry, cfg.min_sl_pts)
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
        df: pd.DataFrame, cfg: ModeConfig, z_thresh: float,
        spread_pts: float = SPREAD_POINTS, account_balance: float = DEFAULT_ACCOUNT_BALANCE,
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

            direction, score, _, diag = QuantFeatureEngine.evaluate_bar(df, i, cfg, z_thresh=z_thresh, check_session=True)
            if direction not in ("BUY", "SELL") or (min_score_override is not None and score < min_score_override):
                i += 1
                continue

            atr = float(df["atr"].iloc[i])
            if atr <= 1.0 or np.isnan(atr):
                i += 1
                continue

            entry = float(df["Close"].iloc[i])
            last_sl_px = float(df["last_sl"].iloc[i]) if not pd.isna(df["last_sl"].iloc[i]) else np.nan
            last_sh_px = float(df["last_sh"].iloc[i]) if not pd.isna(df["last_sh"].iloc[i]) else np.nan
            sl, tp1, tp2, tp3 = QuantFeatureEngine.compute_sl_and_tps(direction, entry, atr, last_sl_px, last_sh_px, cfg)
            sl_dist = abs(entry - sl)

            risk_usd = equity * (DEFAULT_RISK_PCT / 100.0)
            lot = calculate_nas100_lot_size(entry, sl, equity, DEFAULT_RISK_PCT)
            friction_rt = ((spread_pts + 2.0 * SLIPPAGE_POINTS) / PIP_SIZE) * TICK_VALUE * lot
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
                        cur_sl = max(cur_sl, entry + cfg.be_buffer_pts)
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
                        cur_sl = min(cur_sl, entry - cfg.be_buffer_pts)
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
            "mode": cfg.mode_name, "trades": total_trades, "wins": wins, "losses": losses,
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
        df_exec_bt: pd.DataFrame, df_mid_bt: pd.DataFrame, df_htf_bt: pd.DataFrame,
        df_exec_live: pd.DataFrame, df_mid_live: pd.DataFrame, df_htf_live: pd.DataFrame,
        cfg: ModeConfig, spread_pts: float = SPREAD_POINTS,
    ) -> Tuple[int, float, pd.DataFrame, Dict[str, Any], LogisticCalibrator]:
        use_bt_exec = df_exec_bt if len(df_exec_bt) > len(df_exec_live) else df_exec_live
        use_bt_mid = df_mid_bt if len(df_mid_bt) > len(df_mid_live) else df_mid_live
        use_bt_htf = df_htf_bt if len(df_htf_bt) > len(df_htf_live) else df_htf_live

        n_bt = len(use_bt_exec)
        split_idx = int(n_bt * 0.70)
        has_oos = (split_idx >= 180) and ((n_bt - split_idx) >= 120)

        cached_bt = {
            atr_p: QuantFeatureEngine.enrich_dataframe(use_bt_exec, use_bt_mid, use_bt_htf, cfg, atr_period=atr_p)
            for atr_p in GRID_ATR_PERIODS
        }

        best_obj, best_atr, best_z, found = -1e9, 14, 2.0, False
        best_is_stats: Optional[Dict[str, Any]] = None
        is_end = split_idx if has_oos else n_bt

        for atr_p in GRID_ATR_PERIODS:
            for z_t in GRID_Z_THRESHOLDS:
                is_stats = cls.run_backtest(
                    cached_bt[atr_p], cfg, z_thresh=z_t, spread_pts=spread_pts, start_idx=60, end_idx=is_end
                )
                if is_stats["trades"] < cfg.min_backtest_trades:
                    continue
                obj = is_stats["expectancy_r"] * math.sqrt(is_stats["trades"]) * min(is_stats["profit_factor"], 3.0) - 0.18 * is_stats["max_dd_pct"]
                if obj > best_obj:
                    best_obj, best_atr, best_z, found, best_is_stats = obj, atr_p, z_t, True, is_stats

        if not found:
            best_atr, best_z = 14, 2.0
            best_is_stats = cls.run_backtest(cached_bt[best_atr], cfg, z_thresh=best_z, spread_pts=spread_pts, start_idx=60, end_idx=is_end)

        win_df = cached_bt[best_atr]
        calibrator = LogisticCalibrator()
        if best_is_stats and len(best_is_stats.get("feat_y", [])) >= 12:
            calibrator.fit(np.array(best_is_stats["feat_X"], dtype=float), np.array(best_is_stats["feat_y"], dtype=float))

        full_stats = cls.run_backtest(win_df, cfg, z_thresh=best_z, spread_pts=spread_pts, start_idx=60)
        sniper_stats = cls.run_backtest(
            win_df, cfg, z_thresh=best_z, spread_pts=spread_pts, start_idx=60, min_score_override=cfg.sniper_score
        )

        if has_oos:
            oos_stats = cls.run_backtest(win_df, cfg, z_thresh=best_z, spread_pts=spread_pts, start_idx=split_idx)
            sniper_oos = cls.run_backtest(
                win_df, cfg, z_thresh=best_z, spread_pts=spread_pts, start_idx=split_idx, min_score_override=cfg.sniper_score
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
    def generate_performance_chart(scalp_stats: Dict[str, Any], swing_stats: Dict[str, Any], out_path: str = CHART_PATH) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor="#0d1117")
        fig.suptitle(
            f"APEX-QUANT-OMNI | {SYMBOL_LABEL} Dual-Mode Walk-Forward Backtest (Spread + Slippage + OOS)",
            color="#f0f6fc", fontsize=12.5, fontweight="bold",
        )
        for ax, st, color, label in [
            (axes[0], scalp_stats, "#00e676", "SCALP (5M/15M/1H)"),
            (axes[1], swing_stats, "#29b6f6", "SWING (15M/1H/4H)"),
        ]:
            ax.set_facecolor("#161b22")
            curve = st.get("equity_curve") or []
            if curve:
                xs, ys = [c[0] for c in curve], [c[1] for c in curve]
                ax.plot(xs, ys, color=color, linewidth=1.8)
                ax.fill_between(xs, DEFAULT_ACCOUNT_BALANCE, ys, color=color, alpha=0.12)
                ax.axhline(DEFAULT_ACCOUNT_BALANCE, color="#8b949e", linestyle="--", linewidth=0.9)
                if st.get("oos_split_ts") is not None:
                    ax.axvline(st["oos_split_ts"], color="#ffa726", linestyle="-.", linewidth=1.2)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            else:
                ax.text(0.5, 0.5, "No Trades Recorded", color="#8b949e", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(
                f"{label} | PRO: {st.get('trades',0)}T WR {st.get('win_rate',0):.1f}% PF {st.get('profit_factor',0):.2f} Ret {st.get('total_return_pct',0):+.1f}% | "
                f"SNIPER: {st.get('sniper_trades',0)}T WR {st.get('sniper_wr',0):.1f}% PF {st.get('sniper_pf',0):.2f} Ret {st.get('sniper_ret',0):+.1f}%",
                color="#c9d1d9", fontsize=9.5, loc="left",
            )
            ax.tick_params(colors="#8b949e")
            ax.grid(color="#30363d", linestyle=":", linewidth=0.6)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)


# ==============================================================================
# SIGNAL GENERATOR & AUTO-TRAILING STOP ENGINE
# ==============================================================================
class SignalEngine:
    @staticmethod
    def build_live_signal(
        df_enriched: pd.DataFrame, cfg: ModeConfig, z_thresh: float,
        backtest_stats: Dict[str, Any], calibrator: LogisticCalibrator,
        intermarket: Dict[str, Any], news_blackout: Tuple[bool, str],
        live_spread_pts: float = SPREAD_POINTS, live_mid: float = 0.0,
    ) -> SignalCandidate:
        if df_enriched.empty or len(df_enriched) < 65:
            return SignalCandidate(mode=cfg.mode_name, direction="NO_SIGNAL", reason="EMPTY_DATA")

        idx = len(df_enriched) - 1
        row = df_enriched.iloc[idx]
        candle_ts = df_enriched.index[idx].isoformat()
        close_px, atr = float(row["Close"]), float(row["atr"])
        ker, adx, z_val = float(row["ker"]), float(row["adx"]), float(row["z_score"])
        spot_ref = live_mid if live_mid > 1000.0 else close_px
        macro_tag = f"{intermarket.get('tag', 'TNX:N/A')}|{news_blackout[1]}"

        if news_blackout[0]:
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2), atr=round(atr, 2),
                regime="NEWS_GUARD", macro_ctx=macro_tag, candle_ts=candle_ts, reason=news_blackout[1],
            )

        if intermarket.get("vix_panic") and cfg.mode_name == "SCALP":
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2), atr=round(atr, 2),
                regime="VIX_PANIC_GUARD", macro_ctx=macro_tag, candle_ts=candle_ts,
                reason=f"VIX_PANIC_GUARD(VIX={intermarket.get('vix')})",
            )

        direction, score, regime, diag = QuantFeatureEngine.evaluate_bar(
            df_enriched, idx, cfg, z_thresh=z_thresh, check_session=True, tnx_dir=int(intermarket.get("tnx_dir", 0))
        )
        mtf_bias = str(diag.get("mtf_bias", "N/A"))

        if direction == "NO_SIGNAL":
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                score=score, confluence_pct=round(score, 1), ker=round(ker, 2), adx=round(adx, 1),
                z_score=round(z_val, 2), atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias,
                macro_ctx=macro_tag, candle_ts=candle_ts, reason=str(diag.get("reason", "NO_CONFLUENCE")),
            )

        oos_trades = int(backtest_stats.get("oos_trades", 0))
        oos_ev_r = float(backtest_stats.get("oos_ev_r", 0.0))
        if oos_trades >= 6 and oos_ev_r < -0.25 and score < (cfg.min_score + 6.0):
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, macro_ctx=macro_tag,
                candle_ts=candle_ts, reason=f"OOS_DEGRADATION_GUARD(EV={oos_ev_r:+.2f}R)",
            )

        if live_mid > 1000.0 and atr > 0 and abs(live_mid - close_px) > (0.65 * atr):
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(live_mid, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, macro_ctx=macro_tag,
                candle_ts=candle_ts, reason=f"PRICE_DRIFT_GUARD(diff={abs(live_mid-close_px):.2f})",
            )

        half_spread = live_spread_pts / 2.0
        entry = round(spot_ref + half_spread if direction == "BUY" else spot_ref - half_spread, 2)
        last_sl_px = float(row["last_sl"]) if not pd.isna(row["last_sl"]) else np.nan
        last_sh_px = float(row["last_sh"]) if not pd.isna(row["last_sh"]) else np.nan
        sl, tp1, tp2, tp3 = QuantFeatureEngine.compute_sl_and_tps(
            direction, entry, atr, last_sl_px, last_sh_px, cfg
        )
        lot_size = calculate_nas100_lot_size(entry, sl, DEFAULT_ACCOUNT_BALANCE, DEFAULT_RISK_PCT)
        risk_usd = round((abs(entry - sl) / PIP_SIZE) * TICK_VALUE * lot_size, 2)

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
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, macro_ctx=macro_tag,
                candle_ts=candle_ts, reason=f"LOW_EXPECTANCY_EV={ev_r:+.2f}R",
            )

        tier = "SNIPER" if (is_sniper_setup and p_win >= 60.0) else "PRO"
        return SignalCandidate(
            mode=cfg.mode_name, direction=direction, tier=tier, entry=entry, sl=sl,
            tp1=tp1, tp2=tp2, tp3=tp3, lot_size=lot_size, risk_usd=risk_usd,
            score=score, confluence_pct=round(score, 1), p_win=p_win, ev_r=ev_r,
            ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2), atr=round(atr, 2),
            regime=regime, mtf_bias=mtf_bias, macro_ctx=macro_tag, candle_ts=candle_ts,
            reason=f"SIGNAL_{tier}_CONFIRMED", pillar_details=diag,
        )

    @staticmethod
    def format_telegram_signal(sig: SignalCandidate) -> str:
        prefix = "💎" if sig.tier == "SNIPER" else "⚡"
        return (
            f"{prefix} <b>{SYMBOL_LABEL} | {sig.mode} | {sig.direction}</b>\n"
            f"🎯 Entry: <code>{sig.entry:.2f}</code> | 🛑 SL: <code>{sig.sl:.2f}</code>\n"
            f"✅ TP1: <code>{sig.tp1:.2f}</code> | TP2: <code>{sig.tp2:.2f}</code> | TP3: <code>{sig.tp3:.2f}</code>\n"
            f"📐 Lot: <code>{sig.lot_size:.2f}</code> | Skor: <code>{sig.score:.0f}/100</code> | Regime: <code>{sig.regime}</code>\n"
            f"🔄 Trail: Aktif @ TP1 (BE) | 🕒 <code>{utc_now().strftime('%H:%M UTC')}</code>\n"
            f"🧠 Edge: P(Win) <code>{sig.p_win:.0f}%</code> | EV <code>{sig.ev_r:+.2f}R</code> | KER <code>{sig.ker:.2f}</code> | <code>{sig.macro_ctx}</code>"
        )


class TrailingStopManager:
    def __init__(self, state: StateManager, notifier: Notifier) -> None:
        self.state = state
        self.notifier = notifier

    def update_active_trades(
        self, df_5m: pd.DataFrame, df_15m: pd.DataFrame, live_mid: float = 0.0, news_active: bool = False
    ) -> List[str]:
        active = self.state.get_active_signals()
        if not active or df_5m.empty:
            return []
        events: List[str] = []
        latest_price = live_mid if live_mid > 1000.0 else float(df_5m["Close"].iloc[-1])

        for tr in active:
            sig_id, mode, direction, status = int(tr["id"]), tr["mode"], tr["direction"], tr["status"]
            cfg = SCALP_CONFIG if mode == "SCALP" else SWING_CONFIG
            df_ref = df_5m if mode == "SCALP" else df_15m
            if df_ref.empty:
                df_ref = df_5m

            since_df = df_ref[df_ref.index >= datetime.fromisoformat(tr["created_at"])]
            high_since = max(float(since_df["High"].max()), latest_price) if not since_df.empty else latest_price
            low_since = min(float(since_df["Low"].min()), latest_price) if not since_df.empty else latest_price

            cur_sl, entry, init_sl = float(tr["current_sl"]), float(tr["entry"]), float(tr["initial_sl"])
            tp1, tp2, tp3, atr = float(tr["tp1"]), float(tr["tp2"]), float(tr["tp3"]), float(tr["atr"])
            sl_dist = abs(entry - init_sl)

            if direction == "BUY":
                if low_since <= cur_sl:
                    cs = "CLOSED_BE" if cur_sl >= entry else "CLOSED_SL"
                    self.state.update_signal_trail(sig_id, cur_sl, cs)
                    msg = f"🛑 <b>{SYMBOL_LABEL} | {mode} BUY #{sig_id} {cs}</b> @ <code>{cur_sl:.2f}</code> (Spot: <code>{latest_price:.2f}</code>)"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue
                if high_since >= tp3:
                    self.state.update_signal_trail(sig_id, tp2, "CLOSED_TP3")
                    msg = f"🏆 <b>{SYMBOL_LABEL} | {mode} BUY #{sig_id} FULL TP3 HIT</b> @ <code>{tp3:.2f}</code>!"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                new_sl, new_status = cur_sl, status
                if high_since >= tp2 and status != "TP2_TRAIL":
                    new_status = "TP2_TRAIL"
                    new_sl = max(cur_sl, tp1, latest_price - atr * cfg.trail_atr_mult)
                elif high_since >= tp1 and status in ("OPEN", "BE_LOCKED"):
                    new_status = "TP1_TRAIL"
                    new_sl = max(cur_sl, entry + cfg.be_buffer_pts)
                elif news_active and latest_price >= (entry + 0.50 * sl_dist) and status == "OPEN":
                    new_status = "BE_LOCKED"
                    new_sl = max(cur_sl, entry + cfg.be_buffer_pts)
                elif status == "TP2_TRAIL" or (mode == "SWING" and status == "TP1_TRAIL"):
                    new_sl = max(cur_sl, latest_price - atr * cfg.trail_atr_mult)

                if new_status != status or new_sl > cur_sl + 1.50:
                    self.state.update_signal_trail(sig_id, new_sl, new_status)
                    msg = f"🔄 <b>TRAIL | {mode} BUY #{sig_id}</b> Status: <code>{new_status}</code> | SL: <code>{new_sl:.2f}</code> | Spot: <code>{latest_price:.2f}</code>"
                    events.append(msg)
                    self.notifier.broadcast(msg)
            else:
                if high_since >= cur_sl:
                    cs = "CLOSED_BE" if cur_sl <= entry else "CLOSED_SL"
                    self.state.update_signal_trail(sig_id, cur_sl, cs)
                    msg = f"🛑 <b>{SYMBOL_LABEL} | {mode} SELL #{sig_id} {cs}</b> @ <code>{cur_sl:.2f}</code> (Spot: <code>{latest_price:.2f}</code>)"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue
                if low_since <= tp3:
                    self.state.update_signal_trail(sig_id, tp2, "CLOSED_TP3")
                    msg = f"🏆 <b>{SYMBOL_LABEL} | {mode} SELL #{sig_id} FULL TP3 HIT</b> @ <code>{tp3:.2f}</code>!"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                new_sl, new_status = cur_sl, status
                if low_since <= tp2 and status != "TP2_TRAIL":
                    new_status = "TP2_TRAIL"
                    new_sl = min(cur_sl, tp1, latest_price + atr * cfg.trail_atr_mult)
                elif low_since <= tp1 and status in ("OPEN", "BE_LOCKED"):
                    new_status = "TP1_TRAIL"
                    new_sl = min(cur_sl, entry - cfg.be_buffer_pts)
                elif news_active and latest_price <= (entry - 0.50 * sl_dist) and status == "OPEN":
                    new_status = "BE_LOCKED"
                    new_sl = min(cur_sl, entry - cfg.be_buffer_pts)
                elif status == "TP2_TRAIL" or (mode == "SWING" and status == "TP1_TRAIL"):
                    new_sl = min(cur_sl, latest_price + atr * cfg.trail_atr_mult)

                if new_status != status or new_sl < cur_sl - 1.50:
                    self.state.update_signal_trail(sig_id, new_sl, new_status)
                    msg = f"🔄 <b>TRAIL | {mode} SELL #{sig_id}</b> Status: <code>{new_status}</code> | SL: <code>{new_sl:.2f}</code> | Spot: <code>{latest_price:.2f}</code>"
                    events.append(msg)
                    self.notifier.broadcast(msg)
        return events


# ==============================================================================
# FORENSIC AUDIT LOGGER & MAIN ORCHESTRATOR
# ==============================================================================
def forensic_audit_log(
    sources: Dict[str, str], live_quote: Dict[str, Any], intermarket: Dict[str, Any], news_info: Tuple[bool, str],
    scalp_stats: Dict[str, Any], swing_stats: Dict[str, Any],
    scalp_sig: SignalCandidate, swing_sig: SignalCandidate, active_trades: List[Dict[str, Any]],
) -> None:
    lines = [
        "=" * 88,
        f"APEX-QUANT-OMNI — HYPERION APEX FORENSIC AUDIT LOG ({SYMBOL_LABEL})",
        "=" * 88,
        f"Timestamp UTC     : {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')} | TierFilter={SIGNAL_TIER_MODE}",
        f"Data Feeds        : {json.dumps(sources)}",
        f"Live Spot & Spec  : {json.dumps(live_quote)} | Scale={BROKER_PRICE_SCALE} Offset={BROKER_PRICE_OFFSET} | Contract={CONTRACT_SIZE} TickVal=${TICK_VALUE}",
        f"Macro & News      : {intermarket.get('tag')} | {news_info[1]}",
        "-" * 88,
        (
            f"[SCALP PRO]       : Bars={scalp_stats.get('bt_bars')} | ATR={scalp_stats.get('opt_atr_period')} | Z={scalp_stats.get('opt_z_thresh')} | "
            f"Trades={scalp_stats.get('trades')} | WR={scalp_stats.get('win_rate')}% | TP1Hit={scalp_stats.get('tp1_hit_rate')}% | "
            f"PF={scalp_stats.get('profit_factor')} | OOS={scalp_stats.get('oos_win_rate')}%/{scalp_stats.get('oos_trades')}T (EV={scalp_stats.get('oos_ev_r'):+.2f}R) | "
            f"MaxDD={scalp_stats.get('max_dd_pct')}% | Ret={scalp_stats.get('total_return_pct'):+.2f}%"
        ),
        (
            f"[SCALP SNIPER 86+]: Trades={scalp_stats.get('sniper_trades')} | WR={scalp_stats.get('sniper_wr')}% | "
            f"TP1Hit={scalp_stats.get('sniper_tp1_hit')}% | PF={scalp_stats.get('sniper_pf')} | "
            f"OOS={scalp_stats.get('sniper_oos_wr')}%/{scalp_stats.get('sniper_oos_trades')}T | MaxDD={scalp_stats.get('sniper_dd')}% | Ret={scalp_stats.get('sniper_ret'):+.2f}%"
        ),
        (
            f"[SWING PRO]       : Bars={swing_stats.get('bt_bars')} | ATR={swing_stats.get('opt_atr_period')} | Z={swing_stats.get('opt_z_thresh')} | "
            f"Trades={swing_stats.get('trades')} | WR={swing_stats.get('win_rate')}% | TP1Hit={swing_stats.get('tp1_hit_rate')}% | "
            f"PF={swing_stats.get('profit_factor')} | OOS={swing_stats.get('oos_win_rate')}%/{swing_stats.get('oos_trades')}T (EV={swing_stats.get('oos_ev_r'):+.2f}R) | "
            f"MaxDD={swing_stats.get('max_dd_pct')}% | Ret={swing_stats.get('total_return_pct'):+.2f}%"
        ),
        (
            f"[SWING SNIPER 86+]: Trades={swing_stats.get('sniper_trades')} | WR={swing_stats.get('sniper_wr')}% | "
            f"TP1Hit={swing_stats.get('sniper_tp1_hit')}% | PF={swing_stats.get('sniper_pf')} | "
            f"OOS={swing_stats.get('sniper_oos_wr')}%/{swing_stats.get('sniper_oos_trades')}T | MaxDD={swing_stats.get('sniper_dd')}% | Ret={swing_stats.get('sniper_ret'):+.2f}%"
        ),
        "-" * 88,
        f"[LIVE SCALP]      : {scalp_sig.direction} ({scalp_sig.tier}) | Px={scalp_sig.entry} | Score={scalp_sig.score} | Regime={scalp_sig.regime} | KER={scalp_sig.ker} | ADX={scalp_sig.adx} | Reason={scalp_sig.reason}",
        f"[LIVE SWING]      : {swing_sig.direction} ({swing_sig.tier}) | Px={swing_sig.entry} | Score={swing_sig.score} | Regime={swing_sig.regime} | KER={swing_sig.ker} | ADX={swing_sig.adx} | Reason={swing_sig.reason}",
        f"[ACTIVE TRADES]   : {len(active_trades)} active position(s)",
    ]
    for t in active_trades:
        lines.append(f"   -> #{t['id']} {t['mode']} {t['direction']} Entry={t['entry']} CurSL={t['current_sl']} Status={t['status']}")
    lines.append("=" * 88)
    print("\n".join(lines), flush=True)


def format_on_demand_check(sig: SignalCandidate, bt_stats: Dict[str, Any]) -> str:
    if sig.direction in ("BUY", "SELL"):
        return SignalEngine.format_telegram_signal(sig)
    return (
        f"⏸ <b>{SYMBOL_LABEL} | {sig.mode} | NO_SIGNAL</b>\n"
        f"💵 Spot: <code>{sig.entry:.2f}</code> | Skor: <code>{sig.score:.0f}</code> | Regime: <code>{sig.regime}</code>\n"
        f"🛡 Filter: <code>{sig.reason}</code> | TP1Hit: <code>{bt_stats.get('tp1_hit_rate', 0):.1f}%</code>"
    )


def format_status_message(
    state: StateManager, spot_px: float, scalp_stats: Dict[str, Any], swing_stats: Dict[str, Any],
    intermarket: Dict[str, Any], news_info: Tuple[bool, str],
) -> str:
    active = state.get_active_signals()
    _, sess_tag = is_market_open()
    is_peak_ny, _ = is_us_cash_or_premarket_peak(utc_now())
    peak_tag = "NY-PEAK" if is_peak_ny else "OFF-PEAK"
    lines = [
        f"📊 <b>APEX-QUANT {SYMBOL_LABEL} STATUS</b> | 🕒 <code>{utc_now().strftime('%H:%M UTC')}</code>",
        f"💵 Spot: <code>{spot_px:.2f}</code> | <code>{sess_tag} ({peak_tag})</code> | <code>{intermarket.get('tag')}</code>",
        f"📰 News: <code>{news_info[1]}</code> | Tier: <code>{SIGNAL_TIER_MODE}</code>",
        f"⚡ Scalp: OOS WR <code>{scalp_stats.get('oos_win_rate', 0):.1f}%</code> | Sniper PF <code>{scalp_stats.get('sniper_pf', 0):.2f}</code>",
        f"🌊 Swing: OOS WR <code>{swing_stats.get('oos_win_rate', 0):.1f}%</code> | Sniper PF <code>{swing_stats.get('sniper_pf', 0):.2f}</code>",
        f"📌 Posisi Aktif: <code>{len(active)}</code>",
    ]
    for tr in active[:4]:
        sl_dist = abs(float(tr["entry"]) - float(tr["initial_sl"])) + 1e-8
        float_r = ((spot_px - float(tr["entry"])) if tr["direction"] == "BUY" else (float(tr["entry"]) - spot_px)) / sl_dist
        lines.append(
            f"• #{tr['id']} <b>{tr['mode']} {tr['direction']}</b> @ <code>{tr['entry']:.2f}</code> | "
            f"SL: <code>{tr['current_sl']:.2f}</code> | PnL: <code>{float_r:+.2f}R</code> | <code>{tr['status']}</code>"
        )
    return "\n".join(lines)


def format_history_message(state: StateManager) -> str:
    recent = state.get_recent_signals(limit=5)
    if not recent:
        return "📜 <b>APEX HISTORY</b>\nBelum ada sinyal terekam di database."
    lines = [f"📜 <b>{SYMBOL_LABEL} LAST 5 SIGNALS</b>"]
    for r in recent:
        lines.append(
            f"• #{r['id']} <b>{r['mode']} {r['direction']}</b> @ <code>{r['entry']:.2f}</code> | SL <code>{r['current_sl']:.2f}</code> | <code>{r['status']}</code>"
        )
    return "\n".join(lines)


def ensure_fallback_chart_if_missing(out_path: str = CHART_PATH) -> None:
    if os.path.exists(out_path):
        return
    empty_st = {
        "trades": 0, "win_rate": 0.0, "tp1_hit_rate": 0.0, "profit_factor": 0.0,
        "oos_win_rate": 0.0, "oos_tp1_hit": 0.0, "oos_trades": 0,
        "sniper_trades": 0, "sniper_wr": 0.0, "sniper_tp1_hit": 0.0, "sniper_pf": 0.0,
        "sniper_ret": 0.0, "total_return_pct": 0.0, "equity_curve": [],
    }
    BacktestAndOptimizer.generate_performance_chart(empty_st, empty_st, out_path)


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

        market_open, _ = is_market_open()
        news_info = MacroAndNewsGuard.check_usd_news_blackout()
        intermarket = MacroAndNewsGuard.fetch_intermarket_bias()

        data_engine = MarketDataEngine()
        live_quote = data_engine.fetch_live_spot_quote()
        live_spread = float(live_quote.get("spread", SPREAD_POINTS) or SPREAD_POINTS)
        live_mid = float(live_quote.get("mid", 0.0) or 0.0)

        tf_bundle = data_engine.load_all_timeframes(spot_ref=live_mid)
        df_5m, src_5m = tf_bundle["5m"]
        df_15m, src_15m = tf_bundle["15m"]
        df_1h, src_1h = tf_bundle["1h"]
        df_4h, src_4h = tf_bundle["4h"]
        sources = {"5m": src_5m, "15m": src_15m, "1h": src_1h, "4h": src_4h}

        ok_5m, msg_5m = MarketDataEngine.validate_freshness_and_shock(df_5m, SCALP_CONFIG.stale_max_minutes, market_open)
        ok_15m, msg_15m = MarketDataEngine.validate_freshness_and_shock(df_15m, SWING_CONFIG.stale_max_minutes, market_open)
        ok_1h, msg_1h = MarketDataEngine.validate_freshness_and_shock(df_1h, 240, market_open)

        if df_5m.empty or df_15m.empty or df_1h.empty or df_4h.empty:
            failures, tripped = state.record_failure()
            err_msg = f"⚠️ <b>APEX {SYMBOL_LABEL} DATA FAILOVER EXHAUSTED</b> | Sources: <code>{sources}</code> | Fails: <code>{failures}</code>"
            logger.error(err_msg)
            if tripped or pending_cmds:
                notifier.broadcast(err_msg)
            ensure_fallback_chart_if_missing()
            return 1

        spot_px = live_mid if live_mid > 1000.0 else float(df_5m["Close"].iloc[-1])

        trail_mgr = TrailingStopManager(state, notifier)
        if market_open:
            trail_mgr.update_active_trades(df_5m, df_15m, live_mid=live_mid, news_active=news_info[0])

        deep_hist = data_engine.load_deep_calibration_history(spot_reference=spot_px)
        bt_5m, _ = deep_hist.get("5m", (pd.DataFrame(), "NONE"))
        bt_15m, _ = deep_hist.get("15m", (pd.DataFrame(), "NONE"))
        bt_1h, _ = deep_hist.get("1h", (pd.DataFrame(), "NONE"))
        bt_4h, _ = deep_hist.get("4h", (pd.DataFrame(), "NONE"))

        _, opt_z_scalp, df_scalp_enr, scalp_stats, scalp_cal = BacktestAndOptimizer.grid_search_optimize(
            bt_5m, bt_15m, bt_1h, df_5m, df_15m, df_1h, SCALP_CONFIG, spread_pts=live_spread
        )
        _, opt_z_swing, df_swing_enr, swing_stats, swing_cal = BacktestAndOptimizer.grid_search_optimize(
            bt_15m, bt_1h, bt_4h, df_15m, df_1h, df_4h, SWING_CONFIG, spread_pts=live_spread
        )

        BacktestAndOptimizer.generate_performance_chart(scalp_stats, swing_stats, CHART_PATH)

        if ok_5m and ok_15m:
            scalp_sig = SignalEngine.build_live_signal(
                df_scalp_enr, SCALP_CONFIG, opt_z_scalp, scalp_stats, scalp_cal,
                intermarket, news_info, live_spread_pts=live_spread, live_mid=live_mid,
            )
        else:
            scalp_sig = SignalCandidate(
                mode="SCALP", direction="NO_SIGNAL", entry=round(spot_px, 2),
                regime="DATA_GUARD", reason=f"5M:{msg_5m}|15M:{msg_15m}",
            )

        if ok_15m and ok_1h:
            swing_sig = SignalEngine.build_live_signal(
                df_swing_enr, SWING_CONFIG, opt_z_swing, swing_stats, swing_cal,
                intermarket, news_info, live_spread_pts=live_spread, live_mid=live_mid,
            )
        else:
            swing_sig = SignalCandidate(
                mode="SWING", direction="NO_SIGNAL", entry=round(spot_px, 2),
                regime="DATA_GUARD", reason=f"15M:{msg_15m}|1H:{msg_1h}",
            )

        if manual_cmd in ("check", "scalp", "swing", "status", "chart", "history", "sniper"):
            target = notifier.authorized_chat_ids[0] if notifier.authorized_chat_ids else None
            pending_cmds.append((target or "", f"/{manual_cmd}"))

        for chat_id, raw_cmd in pending_cmds:
            cmd = raw_cmd.lower().strip()
            target_cid = chat_id if chat_id else None
            if cmd.startswith("/status"):
                notifier.broadcast(
                    format_status_message(state, spot_px, scalp_stats, swing_stats, intermarket, news_info),
                    target_chat_id=target_cid,
                )
            elif cmd.startswith("/chart"):
                caption = (
                    f"📈 <b>{SYMBOL_LABEL} Dual-Tier Walk-Forward</b>\n"
                    f"SCALP: Pro <code>{scalp_stats.get('total_return_pct',0):+.1f}%</code> | Sniper <code>{scalp_stats.get('sniper_ret',0):+.1f}%</code>\n"
                    f"SWING: Pro <code>{swing_stats.get('total_return_pct',0):+.1f}%</code> | Sniper <code>{swing_stats.get('sniper_ret',0):+.1f}%</code>"
                )
                notifier.send_telegram_photo(CHART_PATH, caption=caption, chat_id=target_cid)
            elif cmd.startswith("/history"):
                notifier.broadcast(format_history_message(state), target_chat_id=target_cid)
            elif cmd.startswith("/sniper"):
                s_msg = (
                    f"💎 <b>{SYMBOL_LABEL} SNIPER SNAPSHOT</b> (<code>{intermarket.get('tag')}</code>)\n"
                    f"• SCALP: <code>{scalp_sig.direction} ({scalp_sig.tier})</code> Skor <code>{scalp_sig.score:.0f}/86</code> | {scalp_sig.reason}\n"
                    f"• SWING: <code>{swing_sig.direction} ({swing_sig.tier})</code> Skor <code>{swing_sig.score:.0f}/86</code> | {swing_sig.reason}"
                )
                notifier.broadcast(s_msg, target_chat_id=target_cid)
            elif cmd.startswith("/help"):
                notifier.broadcast(
                    f"🤖 <b>APEX-QUANT {SYMBOL_LABEL} COMMANDS</b>\n"
                    "<code>/check</code> — Analisa SCALP & SWING\n"
                    "<code>/sniper</code> — Cek Peluang Sniper A+\n"
                    "<code>/status</code> — Status Posisi, TNX & VIX\n"
                    "<code>/chart</code> — Grafik Backtest OOS\n"
                    "<code>/history</code> — 5 Sinyal Terakhir",
                    target_chat_id=target_cid,
                )
            elif cmd.startswith("/check") or cmd in ("/scalp", "/swing"):
                if "scalp" in cmd:
                    notifier.broadcast(format_on_demand_check(scalp_sig, scalp_stats), target_chat_id=target_cid)
                elif "swing" in cmd:
                    notifier.broadcast(format_on_demand_check(swing_sig, swing_stats), target_chat_id=target_cid)
                else:
                    notifier.broadcast(format_on_demand_check(scalp_sig, scalp_stats), target_chat_id=target_cid)
                    notifier.broadcast(format_on_demand_check(swing_sig, swing_stats), target_chat_id=target_cid)

        for sig, cfg in [(scalp_sig, SCALP_CONFIG), (swing_sig, SWING_CONFIG)]:
            if should_emit_tier(sig, SIGNAL_TIER_MODE):
                blocked, block_reason = state.is_duplicate_or_cooldown(
                    sig.mode, sig.direction, sig.candle_ts, cfg.cooldown_minutes, sig.score
                )
                if blocked:
                    logger.info(f"[{sig.mode}] Signal suppressed: {block_reason}")
                else:
                    sig_id = state.insert_signal(sig)
                    logger.info(f"[{sig.mode}] Emitted #{sig_id} [{sig.tier}] {sig.direction} @ {sig.entry}")
                    notifier.broadcast(SignalEngine.format_telegram_signal(sig))
            elif sig.direction in ("BUY", "SELL"):
                logger.info(f"[{sig.mode}] Tier {sig.tier} skipped by SIGNAL_TIER_MODE={SIGNAL_TIER_MODE}")

        state.record_success()
        forensic_audit_log(
            sources, live_quote, intermarket, news_info,
            scalp_stats, swing_stats, scalp_sig, swing_sig, state.get_active_signals()
        )
        return 0

    except Exception as exc:
        failures, _ = state.record_failure()
        tb_short = traceback.format_exc()[-500:]
        logger.exception("Fatal error in APEX-QUANT-OMNI main loop")
        notifier.broadcast(
            f"🚨 <b>APEX-QUANT {SYMBOL_LABEL} SYSTEM ERROR</b> (Fails: <code>{failures}</code>)\n"
            f"<code>{str(exc)[:180]}</code>\n<code>{tb_short[:300]}</code>"
        )
        ensure_fallback_chart_if_missing()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```
