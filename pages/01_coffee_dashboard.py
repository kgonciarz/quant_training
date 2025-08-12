# pages/01_coffee_dashboard.py
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# -------- Settings --------
TICKER = "KC=F"          # Coffee futures (ICE)
DEFAULT_YEARS = 3

# IMPORTANT: don't call st.set_page_config here (keep it in cocoa_dashboard.py only)
st_autorefresh(interval=60000, key="data_refresh_coffee")  # unique key

# -------- Data utils --------
@st.cache_data(show_spinner=False, ttl=3600)
def get_prices(symbol: str, start, end) -> pd.DataFrame:
    import time, datetime as _dt
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
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
        df = df[keep].copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    for _ in range(3):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True,
                         interval="1d", progress=False, threads=False)
        df = _normalize(df)
        if not df.empty:
            return df
        time.sleep(1.2)

    try:
        end_plus = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.Ticker(symbol).history(start=start, end=end_plus, auto_adjust=True, interval="1d")
        df = _normalize(df)
        if not df.empty:
            return df
    except Exception:
        pass

    try:
        days = max(1, (pd.to_datetime(end) - pd.to_datetime(start)).days)
        years = max(1, days // 365)
        df = yf.download(symbol, period=f"{years}y", auto_adjust=True,
                         interval="1d", progress=False, threads=False)
        df = _normalize(df)
        if not df.empty:
            df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
            if not df.empty:
                return df
    except Exception:
        pass

    raise RuntimeError(f"No daily data for {symbol} in {start} → {end}.")

def today_utc():
    return dt.datetime.utcnow().date()

# -------- Indicators & strategy (Donchian + optional filters) --------
def compute_atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

@st.cache_data(show_spinner=False)
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up, down = h.diff(), -l.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: str  # "long" | "short"
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None

def donchian_channels(df: pd.DataFrame, n: int):
    return (
        df["High"].rolling(n, min_periods=n).max(),
        df["Low"].rolling(n, min_periods=n).min(),
    )

def chandelier_stops(df: pd.DataFrame, atr: pd.Series, n: int = 22, m: float = 3.0):
    long_stop  = df["High"].rolling(n, min_periods=n).max() - m * atr
    short_stop = df["Low"].rolling(n, min_periods=n).min() + m * atr
    return long_stop, short_stop

def backtest_donchian(df: pd.DataFrame,
                      n_entry=20, n_exit=10,
                      atr_period=20,
                      chand_n=22, chand_mult=3.0,
                      adx_period=14, adx_min=25.0,
                      use_adx=True, use_chandelier=True,
                      long_only=False, short_trend_gate=True, trend_sma=200):
    c, h, l = df["Close"].astype(float), df["High"].astype(float), df["Low"].astype(float)
    atr = compute_atr_wilder(df, atr_period)
    adx = compute_adx(df, adx_period)
    sma_long = c.rolling(trend_sma, min_periods=trend_sma).mean()

    hi_ent, lo_ent = donchian_channels(df, n_entry)
    hi_ex,  lo_ex  = donchian_channels(df, n_exit)
    chand_l, chand_s = chandelier_stops(df, atr, chand_n, chand_mult)

    trades, equity, equity_time = [], [1.0], [df.index[0]]
    position: Optional[Trade] = None

    for i in range(1, len(df)):
        t = df.index[i]
        px_h, px_l, px_c = float(h.iloc[i]), float(l.iloc[i]), float(c.iloc[i])
        he = float(hi_ent.iloc[i-1]) if np.isfinite(hi_ent.iloc[i-1]) else np.nan
        le = float(lo_ent.iloc[i-1]) if np.isfinite(lo_ent.iloc[i-1]) else np.nan
        hx = float(hi_ex.iloc[i-1])  if np.isfinite(hi_ex.iloc[i-1])  else np.nan
        lx = float(lo_ex.iloc[i-1])  if np.isfinite(lo_ex.iloc[i-1])  else np.nan
        adx_ok = (float(adx.iloc[i-1]) >= adx_min) if (use_adx and np.isfinite(adx.iloc[i-1])) else True
        short_ok = (float(sma_long.iloc[i-1]) > 0 and px_c < float(sma_long.iloc[i-1])) if short_trend_gate else True
        ch_l = float(chand_l.iloc[i-1]) if (use_chandelier and np.isfinite(chand_l.iloc[i-1])) else np.nan
        ch_s = float(chand_s.iloc[i-1]) if (use_chandelier and np.isfinite(chand_s.iloc[i-1])) else np.nan

        if position is None:
            if adx_ok and np.isfinite(he) and px_c > he:
                position = Trade(t, px_c, "long"); trades.append(position); continue
            if (not long_only) and adx_ok and short_ok and np.isfinite(le) and px_c < le:
                position = Trade(t, px_c, "short"); trades.append(position); continue
        else:
            if position.side == "long":
                stop_hit = np.isfinite(ch_l) and (px_l <= ch_l)
                exit_hit = np.isfinite(lx)   and (px_c < lx)
                if stop_hit or exit_hit:
                    exit_price = ch_l if stop_hit else px_c
                    position.exit_time, position.exit_price = t, float(exit_price)
                    position.pnl_pct = (exit_price / position.entry_price - 1) * 100
                    equity.append(equity[-1] * (1 + position.pnl_pct/100)); equity_time.append(t)
                    position = None
            else:
                stop_hit = np.isfinite(ch_s) and (px_h >= ch_s)
                exit_hit = np.isfinite(hx)   and (px_c > hx)
                if stop_hit or exit_hit:
                    exit_price = ch_s if stop_hit else px_c
                    position.exit_time, position.exit_price = t, float(exit_price)
                    position.pnl_pct = (position.entry_price / exit_price - 1) * 100
                    equity.append(equity[-1] * (1 + position.pnl_pct/100)); equity_time.append(t)
                    position = None

    if position is not None:
        last_px = float(c.iloc[-1])
        pnl = (last_px / position.entry_price - 1) * 100 if position.side == "long" else (position.entry_price / last_px - 1) * 100
        position.exit_time, position.exit_price, position.pnl_pct = c.index[-1], last_px, pnl
        equity.append(equity[-1] * (1 + pnl/100)); equity_time.append(c.index[-1])

    plot = {"high_entry": hi_ent, "low_entry": lo_ent, "high_exit": hi_ex, "low_exit": lo_ex,
            "chand_long": chand_l, "chand_short": chand_s, "adx": adx, "atr": atr, "sma_long": sma_long}
    return trades, pd.Series(equity, index=pd.to_datetime(equity_time)).sort_index(), plot

def max_drawdown(eq: pd.Series) -> float:
    return float(((eq / eq.cummax()) - 1).min() * 100) if len(eq) else 0.0

def summarize_trades(trades: List[Trade]):
    closed = [t for t in trades if t.pnl_pct is not None]
    if not closed: return 0.0, 0.0, 0
    win = sum(1 for t in closed if t.pnl_pct > 0) / len(closed) * 100
    total = 1.0
    for t in closed: total *= (1 + t.pnl_pct/100)
    return win, (total - 1) * 100, len(closed)

# -------- Sidebar --------
with st.sidebar:
    st.header("Coffee Settings")
    end_default = today_utc(); start_default = end_default - dt.timedelta(days=365 * DEFAULT_YEARS)
    start_date = st.date_input("Start date", value=start_default, key="coffee_start")
    end_date   = st.date_input("End date",   value=end_default,   key="coffee_end")

    n_entry = st.slider("Donchian entry (N)", 10, 60, 20)
    n_exit  = st.slider("Donchian exit (N)",   5, 40, 10)
    use_adx = st.checkbox("Use ADX filter", value=True)
    adx_min = st.slider("ADX minimum", 0.0, 40.0, 25.0, step=0.5)
    use_chandelier = st.checkbox("Use Chandelier trailing", value=True)
    chand_n   = st.slider("Chandelier lookback (N)", 10, 60, 22)
    chand_mult= st.slider("Chandelier ×ATR",        1.0, 5.0, 3.0, step=0.1)
    long_only = st.checkbox("Long-only", value=False)
    short_gate= st.checkbox("Gate shorts by SMA200 downtrend", value=True)

if start_date >= end_date:
    st.error("Start date must be before end date."); st.stop()

with st.spinner("Fetching coffee prices…"):
    prices = get_prices(TICKER, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
if prices.empty:
    st.warning("No data returned."); st.stop()

# -------- Run strategy & display --------
trades, equity, plot_series = backtest_donchian(
    prices, n_entry=n_entry, n_exit=n_exit,
    chand_n=chand_n, chand_mult=chand_mult,
    adx_min=adx_min, use_adx=use_adx,
    use_chandelier=use_chandelier,
    long_only=long_only, short_trend_gate=short_gate, trend_sma=200,
)

wr, net, n_trades = summarize_trades(trades)
mdd = max_drawdown(equity)

# live signal
last_close = float(prices["Close"].iloc[-1])
he, le, adx_s = plot_series["high_entry"], plot_series["low_entry"], plot_series["adx"]
he_prev = float(he.iloc[-2]) if len(he) >= 2 and np.isfinite(he.iloc[-2]) else np.nan
le_prev = float(le.iloc[-2]) if len(le) >= 2 and np.isfinite(le.iloc[-2]) else np.nan
adx_ok = (float(adx_s.iloc[-2]) >= adx_min) if use_adx and len(adx_s) >= 2 and np.isfinite(adx_s.iloc[-2]) else True
signal, reason = "HOLD", ""
if np.isfinite(he_prev) and adx_ok and last_close > he_prev:
    signal, reason = "BUY", f"Close {last_close:.2f} > {n_entry}-day high {he_prev:.2f}" + (f" (ADX≥{adx_min:.0f})" if use_adx else "")
elif (not long_only) and np.isfinite(le_prev) and adx_ok and last_close < le_prev:
    signal, reason = "SELL", f"Close {last_close:.2f} < {n_entry}-day low {le_prev:.2f}" + (f" (ADX≥{adx_min:.0f})" if use_adx else "")

signal_color = {"BUY":"green","SELL":"red","HOLD":"gray"}[signal]
st.markdown(f"<h2 style='color:{signal_color};'>☕ Coffee Signal: {signal}</h2>", unsafe_allow_html=True)
if reason: st.caption(reason)

colA, colB, colC, colD = st.columns([1.4,1,1,1.2])
colA.markdown("### 📊 Coffee Donchian")
colB.metric("Win Rate", f"{wr:.2f}%")
colC.metric("Total Return", f"{net:+.2f}%")
colD.metric("Max Drawdown", f"{mdd:.2f}%")

fig = go.Figure()
fig.add_trace(go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"],
                             low=prices["Low"], close=prices["Close"], name="Price"))
for name in ["high_entry","low_entry","high_exit","low_exit","chand_long","chand_short","sma_long"]:
    s = plot_series.get(name)
    if s is None: continue
    if name in ["high_exit","low_exit"]:
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name.replace("_"," ").title(), mode="lines", line=dict(dash="dot")))
    else:
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name.replace("_"," ").title(), mode="lines"))

# trade markers
lx, ly, sx, sy, ex, ey = [], [], [], [], [], []
for t in trades:
    (lx if t.side=="long" else sx).append(t.entry_time)
    (ly if t.side=="long" else sy).append(t.entry_price)
    if t.exit_time is not None:
        ex.append(t.exit_time); ey.append(t.exit_price)
if lx: fig.add_trace(go.Scatter(x=lx, y=ly, mode="markers", name="Buy",
                                marker=dict(symbol="triangle-up", size=10, color="green")))
if sx and not long_only:
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="Sell",
                             marker=dict(symbol="triangle-down", size=10, color="red")))
if ex:
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers", name="Exit",
                             marker=dict(symbol="x", size=9, color="gray")))
fig.update_layout(height=720, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
