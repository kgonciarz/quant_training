# cocoa_dashboard.py
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ----------------------------
# Config
# ----------------------------
TICKER = "CC=F"  # ICE Cocoa futures (delayed)
DEFAULT_YEARS = 3

st.set_page_config(page_title="Cocoa Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------
# Utils
# ----------------------------
def today_utc():
    return dt.datetime.utcnow().date()

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

def rolling_extrema(series: pd.Series, window: int, mode: str) -> pd.Series:
    """
    Mark local highs/lows with a rolling window center.
    mode: 'max' or 'min'
    Returns a series with values at pivot points and NaN elsewhere.
    """
    if window % 2 == 0:
        window += 1  # enforce odd window
    half = window // 2
    if mode == "max":
        roll = series.rolling(window, center=True)
        is_ext = (series == roll.max())
        # Require strictly greater than its immediate neighbors to reduce duplicates
        is_ext &= (series > series.shift(1)) & (series > series.shift(-1))
        piv = series.where(is_ext)
    else:
        roll = series.rolling(window, center=True)
        is_ext = (series == roll.min())
        is_ext &= (series < series.shift(1)) & (series < series.shift(-1))
        piv = series.where(is_ext)
    # Drop leading/trailing centers that can't be evaluated
    piv.iloc[:half] = np.nan
    piv.iloc[-half:] = np.nan
    return piv

def cluster_levels(level_values: List[float], tol_pct: float = 0.6) -> List[float]:
    """
    Merge nearby levels within tol_pct (% of price).
    Returns sorted unique levels (means of clusters).
    """
    if not level_values:
        return []
    level_values = sorted([float(x) for x in level_values if np.isfinite(x)])
    clusters = []
    for lv in level_values:
        if not clusters:
            clusters.append([lv])
        else:
            ref = np.mean(clusters[-1])
            if abs(lv - ref) / ref * 100.0 <= tol_pct:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
    merged = [float(np.mean(c)) for c in clusters]
    return merged

@dataclass
class SRLevels:
    support: List[float]
    resistance: List[float]

def detect_sr(df: pd.DataFrame, window: int = 25, cluster_tol_pct: float = 0.6) -> SRLevels:
    """
    Detect support/resistance using rolling extrema + clustering.
    """
    highs = rolling_extrema(df["High"], window, "max").dropna()
    lows = rolling_extrema(df["Low"], window, "min").dropna()
    res = cluster_levels(list(highs.values), tol_pct=cluster_tol_pct)
    sup = cluster_levels(list(lows.values), tol_pct=cluster_tol_pct)
    # Keep only levels within visible price range
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    sup = [x for x in sup if lo <= x <= hi]
    res = [x for x in res if lo <= x <= hi]
    return SRLevels(support=sup, resistance=res)

def nearest_level(price: float, levels: List[float]) -> Optional[float]:
    if not levels:
        return None
    arr = np.array(levels, dtype=float)
    idx = np.argmin(np.abs(arr - price))
    return float(arr[idx])

def slope(x: pd.Series, period: int = 50) -> pd.Series:
    """
    Simple slope of SMA as trend proxy.
    """
    sma = x.rolling(period, min_periods=period).mean()
    return sma.diff()

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: str  # 'long' or 'short'
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None

def backtest(
    df: pd.DataFrame,
    sr: SRLevels,
    buffer_pct: float = 0.3,
    atr_period: int = 14,
    stop_atr: float = 2.0,
    take_atr: float = 3.0,
) -> Tuple[List[Trade], pd.Series]:
    """
    Trend-following mean-reversion entry near SR.
    - Trend up (SMA50 slope > 0): BUY near nearest support (within buffer_pct).
    - Trend down (SMA50 slope < 0): SHORT near nearest resistance (within buffer_pct).
    Exits: ATR stop or ATR take-profit (simple).
    Returns trades list and equity curve (per-trade stepped).
    """
    close = df["Close"].astype(float)
    atr = compute_atr(df, atr_period)
    trend = slope(close, 50)
    trades: List[Trade] = []
    position: Optional[Trade] = None

    equity = [1.0]
    equity_time = [df.index[0]]

    for t, px in close.items():
        tr_up = (trend.loc[t] or 0) > 0
        tr_dn = (trend.loc[t] or 0) < 0
        ns = nearest_level(px, sr.support)
        nr = nearest_level(px, sr.resistance)

        # Entry logic (only if flat)
        if position is None:
            if tr_up and ns is not None and px <= ns * (1 + buffer_pct / 100.0):
                position = Trade(entry_time=t, entry_price=float(px), side="long")
                trades.append(position)
            elif tr_dn and nr is not None and px >= nr * (1 - buffer_pct / 100.0):
                position = Trade(entry_time=t, entry_price=float(px), side="short")
                trades.append(position)
        else:
            a = float(atr.loc[t]) if np.isfinite(atr.loc[t]) else 0.0
            if a <= 0:
                a = 1e-6  # avoid zero
            if position.side == "long":
                stop = position.entry_price - stop_atr * a
                take = position.entry_price + take_atr * a
                if px <= stop or px >= take:
                    position.exit_time = t
                    position.exit_price = float(px)
                    position.pnl_pct = (position.exit_price / position.entry_price - 1) * 100.0
                    position = None
                    equity.append(equity[-1] * (1 + trades[-1].pnl_pct / 100.0))
                    equity_time.append(t)
            else:
                stop = position.entry_price + stop_atr * a
                take = position.entry_price - take_atr * a
                if px >= stop or px <= take:
                    position.exit_time = t
                    position.exit_price = float(px)
                    position.pnl_pct = (position.entry_price / position.exit_price - 1) * 100.0
                    position = None
                    equity.append(equity[-1] * (1 + trades[-1].pnl_pct / 100.0))
                    equity_time.append(t)

    # If last trade still open, close at last price
    if position is not None:
        last_t = close.index[-1]
        last_px = float(close.iloc[-1])
        if position.side == "long":
            pnl = (last_px / position.entry_price - 1) * 100.0
        else:
            pnl = (position.entry_price / last_px - 1) * 100.0
        position.exit_time = last_t
        position.exit_price = last_px
        position.pnl_pct = pnl
        equity.append(equity[-1] * (1 + pnl / 100.0))
        equity_time.append(last_t)

    eq = pd.Series(equity, index=pd.to_datetime(equity_time)).sort_index()
    return trades, eq

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min() * 100.0)

def summarize_trades(trades: List[Trade]) -> Tuple[float, float]:
    if not trades:
        return 0.0, 0.0
    closed = [t for t in trades if t.pnl_pct is not None]
    if not closed:
        return 0.0, 0.0
    wins = [t for t in closed if t.pnl_pct > 0]
    win_rate = len(wins) / len(closed) * 100.0
    total_return = 1.0
    for t in closed:
        total_return *= (1 + t.pnl_pct / 100.0)
    total_return = (total_return - 1) * 100.0
    return float(win_rate), float(total_return)

@st.cache_data(show_spinner=True, ttl=60 * 10)  # cache for 10 minutes
def fetch_prices(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df

# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("Settings")
    end_default = today_utc()
    start_default = end_default - dt.timedelta(days=365 * DEFAULT_YEARS)
    start_date = st.date_input("Start date", value=start_default)
    end_date = st.date_input("End date", value=end_default)
    sr_window = st.slider("Pivot window (bars)", 15, 61, 25, step=2)
    cluster_tol = st.slider("SR cluster tolerance (%)", 0.1, 1.5, 0.6, step=0.1)
    buffer_pct = st.slider("Entry buffer around SR (%)", 0.0, 1.5, 0.3, step=0.1)
    atr_period = st.slider("ATR period", 5, 30, 14)
    stop_atr = st.slider("Stop Loss (×ATR)", 0.5, 5.0, 2.0, step=0.1)
    take_atr = st.slider("Take Profit (×ATR)", 0.5, 8.0, 3.0, step=0.1)
    refresh = st.button("🔄 Refresh data")

# ----------------------------
# Data Fetch
# ----------------------------
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner("Fetching cocoa prices (delayed) …"):
    if refresh:
        fetch_prices.clear()
    prices = fetch_prices(TICKER, start_date, end_date)

if prices.empty:
    st.warning("No data returned for the selected range.")
    st.stop()

# ----------------------------
# Compute SR, Bias, Backtest
# ----------------------------
sr_levels = detect_sr(prices, window=sr_window, cluster_tol_pct=cluster_tol)
atr = compute_atr(prices, period=atr_period)
trend_slope = slope(prices["Close"], 50)
last_close = float(prices["Close"].iloc[-1])

ns_price = nearest_level(last_close, sr_levels.support)
nr_price = nearest_level(last_close, sr_levels.resistance)

# Safely get the last slope value as a float
last_val = trend_slope.iloc[-1]
if isinstance(last_val, (pd.Series, pd.DataFrame)):
    last_val = last_val.squeeze()

try:
    trend_val = float(last_val)
except Exception:
    trend_val = 0.0

trend_up = trend_val > 0
bias = "LONG" if trend_up else "SHORT"


trades, equity = backtest(
    prices,
    sr_levels,
    buffer_pct=buffer_pct,
    atr_period=atr_period,
    stop_atr=stop_atr,
    take_atr=take_atr,
)

win_rate, total_return = summarize_trades(trades)
mdd = max_drawdown(equity)

# ----------------------------
# Header Metrics (your desired readout)
# ----------------------------
colA, colB, colC, colD, colE, colF = st.columns([1.2, 1, 1, 1, 1, 1.2])
colA.markdown(f"### 📊 Current Bias: **{bias}**")
colB.metric("📉 Nearest Support", f"{ns_price:.2f}" if ns_price else "—")
colC.metric("📈 Nearest Resistance", f"{nr_price:.2f}" if nr_price else "—")
colD.metric("Win Rate", f"{win_rate:.2f}%")
colE.metric("Total Return", f"{total_return:+.2f}%")
colF.metric("Max Drawdown", f"{mdd:.2f}%")

# ----------------------------
# Chart (Plotly Candles + SR + Signals)
# ----------------------------
fig = go.Figure()

# Candles
fig.add_trace(go.Candlestick(
    x=prices.index,
    open=prices["Open"],
    high=prices["High"],
    low=prices["Low"],
    close=prices["Close"],
    name="Price"
))

# SR lines
for lvl in sr_levels.support:
    fig.add_hline(y=lvl, line_width=1, line_dash="dot", line_color="green", opacity=0.5)
for lvl in sr_levels.resistance:
    fig.add_hline(y=lvl, line_width=1, line_dash="dot", line_color="red", opacity=0.5)

# Trade markers
long_x, long_y, short_x, short_y, exit_x, exit_y = [], [], [], [], [], []
for t in trades:
    if t.side == "long":
        long_x.append(t.entry_time)
        long_y.append(t.entry_price)
    else:
        short_x.append(t.entry_time)
        short_y.append(t.entry_price)
    if t.exit_time is not None and t.exit_price is not None:
        exit_x.append(t.exit_time)
        exit_y.append(t.exit_price)

if long_x:
    fig.add_trace(go.Scatter(
        x=long_x, y=long_y, mode="markers",
        name="Buy", marker=dict(symbol="triangle-up", size=10, color="green")
    ))
if short_x:
    fig.add_trace(go.Scatter(
        x=short_x, y=short_y, mode="markers",
        name="Sell", marker=dict(symbol="triangle-down", size=10, color="red")
    ))
if exit_x:
    fig.add_trace(go.Scatter(
        x=exit_x, y=exit_y, mode="markers",
        name="Exit", marker=dict(symbol="x", size=9, color="gray")
    ))

# Bias background strip on the last N bars
N = 100
x0 = prices.index.max() - pd.Timedelta(days=max(5, min(120, len(prices)//5)))
bg_color = "rgba(0, 128, 0, 0.07)" if bias == "LONG" else "rgba(220, 20, 60, 0.07)"
fig.add_vrect(
    x0=x0, x1=prices.index.max(),
    fillcolor=bg_color, line_width=0, layer="below"
)

fig.update_layout(
    height=700,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title=None,
    yaxis_title="Price",
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Trade Log + Downloads
# ----------------------------
closed = [t for t in trades if t.exit_time is not None]
if closed:
    log = pd.DataFrame([{
        "entry_time": t.entry_time, "side": t.side, "entry_price": t.entry_price,
        "exit_time": t.exit_time, "exit_price": t.exit_price, "pnl_pct": t.pnl_pct
    } for t in closed]).sort_values("entry_time")
else:
    log = pd.DataFrame(columns=["entry_time","side","entry_price","exit_time","exit_price","pnl_pct"])

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Trade Log")
    st.dataframe(log, use_container_width=True, height=240)
with c2:
    st.subheader("Equity Curve")
    if not equity.empty:
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity (×)"))
        eq_fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(eq_fig, use_container_width=True)
    else:
        st.write("No closed trades yet for this period/parameters.")

# CSV downloads
st.download_button(
    "⬇️ Download Trade Log (CSV)",
    data=log.to_csv(index=False).encode("utf-8"),
    file_name="trade_log.csv",
    mime="text/csv",
)
nearest_df = pd.DataFrame({
    "nearest_support": [ns_price],
    "nearest_resistance": [nr_price],
    "bias": [bias],
    "asof": [pd.Timestamp.utcnow()]
})
st.download_button(
    "⬇️ Download Nearest Levels (CSV)",
    data=nearest_df.to_csv(index=False).encode("utf-8"),
    file_name="nearest_levels.csv",
    mime="text/csv",
)

# Footer note
st.caption("Data via Yahoo Finance (delayed ~15m). Strategy is for research only, not financial advice.")
