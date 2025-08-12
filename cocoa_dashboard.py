# cocoa_dashboard.py — Multiple Strategies (Donchian weekly long-only + Keltner + costs)
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

TICKER = "CC=F"
DEFAULT_YEARS = 5
st.set_page_config(page_title="Cocoa Dashboard", layout="wide", initial_sidebar_state="collapsed")
# Auto-refresh toggleable in sidebar
st_autorefresh(interval=60000, key="data_refresh")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_prices(symbol: str, start, end) -> pd.DataFrame:
    import time
    import datetime as _dt

    # allow date, datetime, or str
    if isinstance(start, (_dt.date, _dt.datetime)):
        start = _dt.datetime.combine(start, _dt.time.min).strftime("%Y-%m-%d")
    if isinstance(end, (_dt.date, _dt.datetime)):
        end = _dt.datetime.combine(end, _dt.time.min).strftime("%Y-%m-%d")

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.get_level_values(0):
                df = df.xs(symbol, axis=1, level=0)
            else:
                df = df.droplevel(0, axis=1)
        df = df.rename(columns=lambda c: c.title())
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # 1) Try download() a few times (Yahoo can be moody)
    for _ in range(3):
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            interval="1d",
            progress=False,
            threads=False,
        )
        df = _normalize(df)
        if not df.empty:
            return df
        time.sleep(1.2)

    # 2) Try history() with end+1 day (fixes end-exclusive empties)
    try:
        end_plus = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.Ticker(symbol).history(start=start, end=end_plus, auto_adjust=True, interval="1d")
        df = _normalize(df)
        if not df.empty:
            return df
    except Exception:
        pass

    # 3) Last resort: period-based fetch then slice
    try:
        days = max(1, (pd.to_datetime(end) - pd.to_datetime(start)).days)
        years = max(1, days // 365)
        df = yf.download(symbol, period=f"{years}y", auto_adjust=True, interval="1d", progress=False, threads=False)
        df = _normalize(df)
        if not df.empty:
            df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
            if not df.empty:
                return df
    except Exception:
        pass

    raise RuntimeError(
        f"Yahoo returned no daily data for {symbol} in {start} → {end}. "
        "Try a shorter range, click Refresh again, or temporarily switch network/VPN."
    )


def today_utc():
    return dt.datetime.utcnow().date()

# ---------- Helpers ----------

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "D":
        return df.copy()
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    v = df["Volume"].resample(rule).sum() if "Volume" in df else None
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})
    if v is not None:
        out["Volume"] = v
    return out.dropna()

# Costs: simple approximation in bps (basis points)
def apply_costs(pnl_pct: float, commission_bps: float, slippage_bps: float) -> float:
    # Commission charged on entry+exit, slippage once
    total_bps = (2 * commission_bps) + slippage_bps
    return pnl_pct - (total_bps / 100.0)  # convert bps to percent

# ---------- Indicators ----------

def compute_atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

# ---------- Trade struct ----------
@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: str  # "long" or "short"
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None

# ---------- Keltner Breakout (with ADX + Chandelier) ----------

def keltner_bands(df: pd.DataFrame, ema_len: int, atr_len: int, k: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(df["Close"], ema_len)
    atr = compute_atr_wilder(df, atr_len)
    upper = mid + k * atr
    lower = mid - k * atr
    return upper, mid, lower


def chandelier_stops(df: pd.DataFrame, atr: pd.Series, n: int = 22, m: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    long_stop = df["High"].rolling(n, min_periods=n).max() - m * atr
    short_stop = df["Low"].rolling(n, min_periods=n).min() + m * atr
    return long_stop, short_stop


def backtest_keltner(df: pd.DataFrame,
                     ema_len: int = 20,
                     atr_len: int = 20,
                     k: float = 1.25,
                     chand_n: int = 22,
                     chand_mult: float = 3.0,
                     adx_period: int = 14,
                     adx_min: float = 20.0,
                     long_only: bool = False,
                     commission_bps: float = 1.5,
                     slippage_bps: float = 2.0) -> Tuple[List[Trade], pd.Series, dict]:
    close, high, low = df["Close"].astype(float), df["High"].astype(float), df["Low"].astype(float)

    upper, mid, lower = keltner_bands(df, ema_len, atr_len, k)
    atr = compute_atr_wilder(df, atr_len)
    adx = compute_adx(df, adx_period)
    chand_long, chand_short = chandelier_stops(df, atr, chand_n, chand_mult)

    trades: List[Trade] = []
    equity, equity_time = [1.0], [df.index[0]]
    position: Optional[Trade] = None

    for i in range(1, len(df)):
        t = df.index[i]
        px_c, px_h, px_l = float(close.iloc[i]), float(high.iloc[i]), float(low.iloc[i])

        up_prev = float(upper.iloc[i-1]) if np.isfinite(upper.iloc[i-1]) else np.nan
        lo_prev = float(lower.iloc[i-1]) if np.isfinite(lower.iloc[i-1]) else np.nan
        mid_prev = float(mid.iloc[i-1]) if np.isfinite(mid.iloc[i-1]) else np.nan
        adx_ok = float(adx.iloc[i-1]) >= adx_min if np.isfinite(adx.iloc[i-1]) else False
        chand_l = float(chand_long.iloc[i-1]) if np.isfinite(chand_long.iloc[i-1]) else np.nan
        chand_s = float(chand_short.iloc[i-1]) if np.isfinite(chand_short.iloc[i-1]) else np.nan

        if position is None:
            # Entries on close crossing prior band
            if adx_ok and np.isfinite(up_prev) and px_c > up_prev:
                position = Trade(t, px_c, "long"); trades.append(position); continue
            if not long_only and adx_ok and np.isfinite(lo_prev) and px_c < lo_prev:
                position = Trade(t, px_c, "short"); trades.append(position); continue
        else:
            # Exits: chandelier stop wins over midline exit in same bar
            if position.side == "long":
                stop_hit = np.isfinite(chand_l) and (px_l <= chand_l)
                mid_exit = np.isfinite(mid_prev) and (px_c < mid_prev)
                if stop_hit or mid_exit:
                    exit_price = chand_l if stop_hit else px_c
                    pnl = (exit_price / position.entry_price - 1) * 100
                    pnl = apply_costs(pnl, commission_bps, slippage_bps)
                    position.exit_time, position.exit_price, position.pnl_pct = t, float(exit_price), pnl
                    equity.append(equity[-1] * (1 + pnl/100)); equity_time.append(t)
                    position = None
            else:
                stop_hit = np.isfinite(chand_s) and (px_h >= chand_s)
                mid_exit = np.isfinite(mid_prev) and (px_c > mid_prev)
                if stop_hit or mid_exit:
                    exit_price = chand_s if stop_hit else px_c
                    pnl = (position.entry_price / exit_price - 1) * 100
                    pnl = apply_costs(pnl, commission_bps, slippage_bps)
                    position.exit_time, position.exit_price, position.pnl_pct = t, float(exit_price), pnl
                    equity.append(equity[-1] * (1 + pnl/100)); equity_time.append(t)
                    position = None

    # Close open at last
    if position is not None:
        last_px = float(close.iloc[-1])
        pnl = (last_px / position.entry_price - 1) * 100 if position.side == "long" else (position.entry_price / last_px - 1) * 100
        pnl = apply_costs(pnl, commission_bps, slippage_bps)
        position.exit_time, position.exit_price, position.pnl_pct = close.index[-1], last_px, pnl
        equity.append(equity[-1] * (1 + pnl/100)); equity_time.append(close.index[-1])

    plot_series = {
        "upper": upper, "mid": mid, "lower": lower,
        "adx": adx, "atr": atr, "chand_long": chand_long, "chand_short": chand_short
    }

    return trades, pd.Series(equity, index=pd.to_datetime(equity_time)).sort_index(), plot_series

# ---------- Donchian Breakout (with ADX + Chandelier) ----------

def donchian_channels(df: pd.DataFrame, n: int) -> Tuple[pd.Series, pd.Series]:
    high_n = df["High"].rolling(n, min_periods=n).max()
    low_n = df["Low"].rolling(n, min_periods=n).min()
    return high_n, low_n


def backtest_donchian(df: pd.DataFrame,
                      n_entry: int = 20,
                      n_exit: int = 10,
                      atr_period: int = 20,
                      chand_n: int = 22,
                      chand_mult: float = 3.0,
                      adx_period: int = 14,
                      adx_min: float = 25.0,
                      long_only: bool = False,
                      commission_bps: float = 1.5,
                      slippage_bps: float = 2.0) -> Tuple[List[Trade], pd.Series, dict]:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    atr = compute_atr_wilder(df, atr_period)
    adx = compute_adx(df, adx_period)

    high_entry, low_entry = donchian_channels(df, n_entry)
    high_exit, low_exit = donchian_channels(df, n_exit)
    chand_long, chand_short = chandelier_stops(df, atr, chand_n, chand_mult)

    trades: List[Trade] = []
    equity, equity_time = [1.0], [df.index[0]]
    position: Optional[Trade] = None

    for i in range(1, len(df)):
        t = df.index[i]
        px_h, px_l, px_c = float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])

        # Current (i-1) references
        hi_ent = float(high_entry.iloc[i-1]) if np.isfinite(high_entry.iloc[i-1]) else np.nan
        lo_ent = float(low_entry.iloc[i-1]) if np.isfinite(low_entry.iloc[i-1]) else np.nan
        hi_ex = float(high_exit.iloc[i-1]) if np.isfinite(high_exit.iloc[i-1]) else np.nan
        lo_ex = float(low_exit.iloc[i-1]) if np.isfinite(low_exit.iloc[i-1]) else np.nan
        adx_ok = float(adx.iloc[i-1]) >= adx_min if np.isfinite(adx.iloc[i-1]) else False
        chand_l = float(chand_long.iloc[i-1]) if np.isfinite(chand_long.iloc[i-1]) else np.nan
        chand_s = float(chand_short.iloc[i-1]) if np.isfinite(chand_short.iloc[i-1]) else np.nan

        if position is None:
            if adx_ok and np.isfinite(hi_ent) and px_c > hi_ent:
                position = Trade(t, float(px_c), "long"); trades.append(position); continue
            if not long_only and adx_ok and np.isfinite(lo_ent) and px_c < lo_ent:
                position = Trade(t, float(px_c), "short"); trades.append(position); continue
        else:
            # Manage exits with intrabar logic (stop prioritized)
            if position.side == "long":
                stop_lvl = chand_l if np.isfinite(chand_l) else -np.inf
                exit_lvl = lo_ex if np.isfinite(lo_ex) else -np.inf
                stop_hit = np.isfinite(stop_lvl) and (px_l <= stop_lvl)
                exit_hit = np.isfinite(exit_lvl) and (px_c < exit_lvl)
                if stop_hit or exit_hit:
                    exit_price = stop_lvl if stop_hit else px_c
                    pnl = (exit_price / position.entry_price - 1) * 100
                    pnl = apply_costs(pnl, commission_bps, slippage_bps)
                    position.exit_time, position.exit_price, position.pnl_pct = t, float(exit_price), pnl
                    equity.append(equity[-1] * (1 + pnl / 100)); equity_time.append(t)
                    position = None
            else:
                stop_lvl = chand_s if np.isfinite(chand_s) else np.inf
                exit_lvl = hi_ex if np.isfinite(hi_ex) else np.inf
                stop_hit = np.isfinite(stop_lvl) and (px_h >= stop_lvl)
                exit_hit = np.isfinite(exit_lvl) and (px_c > exit_lvl)
                if stop_hit or exit_hit:
                    exit_price = stop_lvl if stop_hit else px_c
                    pnl = (position.entry_price / exit_price - 1) * 100
                    pnl = apply_costs(pnl, commission_bps, slippage_bps)
                    position.exit_time, position.exit_price, position.pnl_pct = t, float(exit_price), pnl
                    equity.append(equity[-1] * (1 + pnl / 100)); equity_time.append(t)
                    position = None

    if position is not None:
        last_px = float(close.iloc[-1])
        pnl = (last_px / position.entry_price - 1) * 100 if position.side == "long" else (position.entry_price / last_px - 1) * 100
        pnl = apply_costs(pnl, commission_bps, slippage_bps)
        position.exit_time, position.exit_price, position.pnl_pct = close.index[-1], last_px, pnl
        equity.append(equity[-1] * (1 + pnl / 100)); equity_time.append(close.index[-1])

    plot_series = {
        "high_entry": high_entry,
        "low_entry": low_entry,
        "high_exit": high_exit,
        "low_exit": low_exit,
        "chand_long": chand_long,
        "chand_short": chand_short,
        "adx": adx,
        "atr": atr,
    }

    return trades, pd.Series(equity, index=pd.to_datetime(equity_time)).sort_index(), plot_series

# ---------- Metrics ----------

def max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    return float(((eq / eq.cummax()) - 1).min() * 100)


def summarize_trades(trades: List[Trade]) -> Tuple[float, float, int]:
    closed = [t for t in trades if t.pnl_pct is not None]
    if not closed:
        return 0.0, 0.0, 0
    win_rate = sum(1 for t in closed if t.pnl_pct > 0) / len(closed) * 100
    total_ret = 1.0
    for t in closed:
        total_ret *= (1 + t.pnl_pct / 100)
    return win_rate, (total_ret - 1) * 100, len(closed)

# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("Settings")
    end_default = today_utc()
    start_default = end_default - dt.timedelta(days=365 * DEFAULT_YEARS)
    start_date = st.date_input("Start date", value=start_default)
    end_date = st.date_input("End date", value=end_default)

    timeframe = st.selectbox("Timeframe", ["Daily", "Weekly (Fri)"] , index=1)
    long_only = st.checkbox("Long-only (avoid shorts)", value=True)

    strategy = st.selectbox("Strategy", [
        "Donchian Breakout",
        "Keltner Breakout",
    ], index=0)

    # Common risk/costs
    st.subheader("Costs")
    commission_bps = st.slider("Commission (bps)", 0.0, 10.0, 1.5, step=0.5)
    slippage_bps = st.slider("Slippage (bps)", 0.0, 20.0, 2.0, step=0.5)

    st.subheader("Signal Filters")
    adx_min = st.slider("ADX minimum", 5.0, 40.0, 25.0, step=0.5)

    if strategy == "Donchian Breakout":
        n_entry = st.slider("Entry channel N", 10, 60, 26)
        n_exit = st.slider("Exit channel N", 5, 40, 13)
        atr_period = st.slider("ATR period (Wilder)", 5, 40, 20)
        chand_n = st.slider("Chandelier lookback N", 10, 60, 22)
        chand_mult = st.slider("Chandelier ×ATR", 1.0, 5.0, 3.0, step=0.1)
    else:
        ema_len = st.slider("EMA length", 5, 60, 20)
        atr_len = st.slider("ATR length", 5, 60, 20)
        k = st.slider("Keltner multiplier (k)", 0.5, 2.5, 1.25, step=0.05)
        chand_n = st.slider("Chandelier lookback N", 10, 60, 22)
        chand_mult = st.slider("Chandelier ×ATR", 1.0, 5.0, 3.0, step=0.1)

    auto_refresh = st.checkbox("Auto-refresh every 60s (off is fine for daily/weekly)", value=False)

if not auto_refresh:
    st_autorefresh(interval=0, key="data_refresh_off")

if start_date >= end_date:
    st.error("Start date must be before end date."); st.stop()

with st.spinner("Fetching cocoa prices…"):
    prices_daily = get_prices(TICKER, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if prices_daily.empty:
    st.warning("No data returned."); st.stop()

rule = "D" if timeframe.startswith("Daily") else "W-FRI"
prices = resample_ohlc(prices_daily, rule)

# ----------------------------
# Run chosen strategy
# ----------------------------
if strategy == "Donchian Breakout":
    trades, equity, plot_series = backtest_donchian(
        prices,
        n_entry=n_entry,
        n_exit=n_exit,
        atr_period=atr_period,
        chand_n=chand_n,
        chand_mult=chand_mult,
        adx_period=14,
        adx_min=adx_min,
        long_only=long_only,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    # Live signal
    he, le, adx_s = plot_series["high_entry"], plot_series["low_entry"], plot_series["adx"]
    he_prev = float(he.iloc[-2]) if len(he) >= 2 and np.isfinite(he.iloc[-2]) else np.nan
    le_prev = float(le.iloc[-2]) if len(le) >= 2 and np.isfinite(le.iloc[-2]) else np.nan
    adx_ok = float(adx_s.iloc[-2]) >= adx_min if len(adx_s) >= 2 and np.isfinite(adx_s.iloc[-2]) else False
    last_close = float(prices['Close'].iloc[-1])
    signal, reason = "HOLD", ""
    if adx_ok and np.isfinite(he_prev) and last_close > he_prev:
        signal, reason = "BUY", f"Close {last_close:.2f} > {n_entry}-bar high {he_prev:.2f} (ADX≥{adx_min:.0f})"
    elif not long_only and adx_ok and np.isfinite(le_prev) and last_close < le_prev:
        signal, reason = "SELL", f"Close {last_close:.2f} < {n_entry}-bar low {le_prev:.2f} (ADX≥{adx_min:.0f})"

else:  # Keltner
    trades, equity, plot_series = backtest_keltner(
        prices,
        ema_len=ema_len,
        atr_len=atr_len,
        k=k,
        chand_n=chand_n,
        chand_mult=chand_mult,
        adx_period=14,
        adx_min=adx_min,
        long_only=long_only,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    up, mid, low_band, adx_s = plot_series["upper"], plot_series["mid"], plot_series["lower"], plot_series["adx"]
    up_prev = float(up.iloc[-2]) if len(up) >= 2 and np.isfinite(up.iloc[-2]) else np.nan
    lo_prev = float(low_band.iloc[-2]) if len(low_band) >= 2 and np.isfinite(low_band.iloc[-2]) else np.nan
    adx_ok = float(adx_s.iloc[-2]) >= adx_min if len(adx_s) >= 2 and np.isfinite(adx_s.iloc[-2]) else False
    last_close = float(prices['Close'].iloc[-1])
    signal, reason = "HOLD", ""
    if adx_ok and np.isfinite(up_prev) and last_close > up_prev:
        signal, reason = "BUY", f"Close {last_close:.2f} > upper band {up_prev:.2f} (ADX≥{adx_min:.0f})"
    elif not long_only and adx_ok and np.isfinite(lo_prev) and last_close < lo_prev:
        signal, reason = "SELL", f"Close {last_close:.2f} < lower band {lo_prev:.2f} (ADX≥{adx_min:.0f})"

# ----------------------------
# Metrics
# ----------------------------
win_rate, total_return, n_trades = summarize_trades(trades)
mdd = max_drawdown(equity)

bias_text = f"{timeframe.split()[0].upper()} MODE — {'LONG-ONLY' if long_only else 'LONG/SHORT'}"
signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[signal]
st.markdown(f"<h2 style='color:{signal_color};'>📢 Live Signal: {signal}</h2>", unsafe_allow_html=True)
if reason:
    st.caption(reason)

colA, colB, colC, colD, colE = st.columns([1.6, 1, 1, 1, 1.2])
colA.markdown(f"### 📊 {bias_text}")
colB.metric("Win Rate", f"{win_rate:.2f}%")
colC.metric("Total Return (net costs)", f"{total_return:+.2f}%")
colD.metric("Max Drawdown", f"{mdd:.2f}%")
colE.metric("Closed Trades", f"{n_trades}")

# ----------------------------
# Chart
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"], low=prices["Low"], close=prices["Close"], name="Price"))

if strategy == "Donchian Breakout":
    he = plot_series["high_entry"]; le = plot_series["low_entry"]
    hx = plot_series["high_exit"]; lx = plot_series["low_exit"]
    ch_l = plot_series["chand_long"]; ch_s = plot_series["chand_short"]
    fig.add_trace(go.Scatter(x=he.index, y=he, name=f"{n_entry} High", mode="lines"))
    fig.add_trace(go.Scatter(x=le.index, y=le, name=f"{n_entry} Low", mode="lines"))
    fig.add_trace(go.Scatter(x=hx.index, y=hx, name=f"{n_exit} High (exit)", mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=lx.index, y=lx, name=f"{n_exit} Low (exit)", mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=ch_l.index, y=ch_l, name="Chandelier Long", mode="lines"))
    fig.add_trace(go.Scatter(x=ch_s.index, y=ch_s, name="Chandelier Short", mode="lines"))
else:
    up = plot_series["upper"]; mid = plot_series["mid"]; lowb = plot_series["lower"]
    ch_l = plot_series["chand_long"]; ch_s = plot_series["chand_short"]
    fig.add_trace(go.Scatter(x=up.index, y=up, name="Upper", mode="lines"))
    fig.add_trace(go.Scatter(x=mid.index, y=mid, name="EMA/Mid", mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=lowb.index, y=lowb, name="Lower", mode="lines"))
    fig.add_trace(go.Scatter(x=ch_l.index, y=ch_l, name="Chandelier Long", mode="lines"))
    fig.add_trace(go.Scatter(x=ch_s.index, y=ch_s, name="Chandelier Short", mode="lines"))

# Trade markers
long_x, long_y, short_x, short_y, exit_x, exit_y = [], [], [], [], [], []
for t in trades:
    if t.side == "long":
        long_x.append(t.entry_time); long_y.append(t.entry_price)
    else:
        short_x.append(t.entry_time); short_y.append(t.entry_price)
    if t.exit_time and t.exit_price:
        exit_x.append(t.exit_time); exit_y.append(t.exit_price)

if long_x:
    fig.add_trace(go.Scatter(x=long_x, y=long_y, mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=10, color="green")))
if short_x and not long_only:
    fig.add_trace(go.Scatter(x=short_x, y=short_y, mode="markers", name="Sell", marker=dict(symbol="triangle-down", size=10, color="red")))
if exit_x:
    fig.add_trace(go.Scatter(x=exit_x, y=exit_y, mode="markers", name="Exit", marker=dict(symbol="x", size=9, color="gray")))

fig.update_layout(height=760, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=f"Price ({timeframe.split()[0]})")
st.plotly_chart(fig, use_container_width=True)
