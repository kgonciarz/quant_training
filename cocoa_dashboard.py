# cocoa_dashboard.py
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
from dataclasses import dataclass
from typing import List, Tuple, Optional


TICKER = "CC=F"
DEFAULT_YEARS = 3
st.set_page_config(page_title="Cocoa Dashboard", layout="wide", initial_sidebar_state="collapsed")
# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="data_refresh")  # 60,000 ms = 60 sec


# ----------------------------
# Utilities
# ----------------------------

@st.cache_data(show_spinner=False, ttl=3600)   # cache for 1 hour
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
            # ('CC=F','Close') … or ('Close','')
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
            end=end,               # Yahoo's end is exclusive (we’ll try +1d later)
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
        df = yf.Ticker(symbol).history(
            start=start, end=end_plus, auto_adjust=True, interval="1d"
        )
        df = _normalize(df)
        if not df.empty:
            return df
    except Exception:
        pass

    # 3) Last resort: period-based fetch then slice
    try:
        days = max(1, (pd.to_datetime(end) - pd.to_datetime(start)).days)
        years = max(1, days // 365)
        df = yf.download(
            symbol, period=f"{years}y", auto_adjust=True, interval="1d",
            progress=False, threads=False
        )
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

# --- Wilder ATR (EMA) ---
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's ATR: True Range smoothed by EMA with alpha=1/period.
    Matches most charting packages more closely than SMA ATR.
    """
    high = df["High"].astype(float)
    low  = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing == EMA with alpha = 1/period
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    return atr


def rolling_extrema(series: pd.Series, window: int, mode: str) -> pd.Series:
    if window % 2 == 0:
        window += 1
    half = window // 2
    roll = series.rolling(window, center=True)
    if mode == "max":
        is_ext = (series == roll.max()) & (series > series.shift(1)) & (series > series.shift(-1))
    else:
        is_ext = (series == roll.min()) & (series < series.shift(1)) & (series < series.shift(-1))
    piv = series.where(is_ext)
    piv.iloc[:half] = np.nan
    piv.iloc[-half:] = np.nan
    return piv

def cluster_levels(level_values: List[float], tol_pct: float = 0.6) -> List[float]:
    if not level_values:
        return []
    vals = sorted([float(x) for x in level_values if np.isfinite(x)])
    clusters = []
    for lv in vals:
        if not clusters:
            clusters.append([lv])
        else:
            ref = np.mean(clusters[-1])
            if abs(lv - ref) / ref * 100.0 <= tol_pct:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
    return [float(np.mean(c)) for c in clusters]

@dataclass
class SRLevels:
    support: List[float]
    resistance: List[float]

@st.cache_data(show_spinner=False, ttl=3600)   # cache SR for 1 hour
def detect_sr(df: pd.DataFrame, window: int = 25, cluster_tol_pct: float = 0.6) -> SRLevels:
    highs = rolling_extrema(df["High"], window, "max").dropna()
    lows = rolling_extrema(df["Low"], window, "min").dropna()
    res = cluster_levels(list(highs.values), cluster_tol_pct)
    sup = cluster_levels(list(lows.values), cluster_tol_pct)
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    sup = [x for x in sup if lo <= x <= hi]
    res = [x for x in res if lo <= x <= hi]
    return SRLevels(support=sup, resistance=res)

def nearest_level(price: float, levels: List[float]) -> Optional[float]:
    if not levels:
        return None
    arr = np.array(levels, dtype=float)
    return float(arr[np.argmin(np.abs(arr - price))])

def slope(x: pd.Series, period: int = 50) -> pd.Series:
    sma = x.rolling(period, min_periods=period).mean()
    return sma.diff()

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: str                            # "long" or "short"
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None      # gross % (no fees/slippage baked in)

def backtest(
    df: pd.DataFrame,
    sr: SRLevels,
    buffer_pct: float = 0.3,
    atr_period: int = 14,
    stop_atr: float = 2.0,
    take_atr: float = 3.0,
    prefer_stop_first: bool = True,   # resolve same-bar stop & take touches
) -> Tuple[List[Trade], pd.Series]:
    """
    Entries:
      - Trend up (SMA50 slope > 0): buy near nearest support (within buffer%).
      - Trend down (SMA50 slope < 0): short near nearest resistance (within buffer%).

    Exits (more realistic):
      - Use bar Low/High to check stop/take. If both touched same bar,
        choose according to `prefer_stop_first`.
    """
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    atr   = compute_atr(df, atr_period)
    trend = slope(close, 50).reindex(df.index)

    trades: List[Trade] = []
    equity = [1.0]
    equity_time = [df.index[0]]

    position: Optional[Trade] = None

    for i, t in enumerate(df.index):
        px = float(close.iloc[i])
        hi = float(high.iloc[i])
        lo = float(low.iloc[i])

        tr_val = float(trend.iloc[i]) if pd.notna(trend.iloc[i]) else 0.0
        tr_up, tr_dn = tr_val > 0, tr_val < 0

        ns = nearest_level(px, sr.support)
        nr = nearest_level(px, sr.resistance)

        # Entry (flat only)
        if position is None:
            if tr_up and ns is not None and px <= ns * (1 + buffer_pct / 100.0):
                position = Trade(entry_time=t, entry_price=px, side="long")
                trades.append(position)
            elif tr_dn and nr is not None and px >= nr * (1 - buffer_pct / 100.0):
                position = Trade(entry_time=t, entry_price=px, side="short")
                trades.append(position)
            continue

        # Exit (intrabar using bar extremes)
        a = float(atr.iloc[i]) if np.isfinite(atr.iloc[i]) else 1e-6

        if position.side == "long":
            stop = position.entry_price - stop_atr * a
            take = position.entry_price + take_atr * a

            stop_hit = lo <= stop
            take_hit = hi >= take

            if stop_hit and take_hit:
                exit_price = stop if prefer_stop_first else take
            elif stop_hit:
                exit_price = stop
            elif take_hit:
                exit_price = take
            else:
                exit_price = None

            if exit_price is not None:
                position.exit_time = t
                position.exit_price = float(exit_price)
                position.pnl_pct = (position.exit_price / position.entry_price - 1) * 100.0
                equity.append(equity[-1] * (1 + position.pnl_pct / 100.0))
                equity_time.append(t)
                position = None

        else:  # short
            stop = position.entry_price + stop_atr * a
            take = position.entry_price - take_atr * a

            stop_hit = hi >= stop
            take_hit = lo <= take

            if stop_hit and take_hit:
                exit_price = stop if prefer_stop_first else take
            elif stop_hit:
                exit_price = stop
            elif take_hit:
                exit_price = take
            else:
                exit_price = None

            if exit_price is not None:
                position.exit_time = t
                position.exit_price = float(exit_price)
                position.pnl_pct = (position.entry_price / position.exit_price - 1) * 100.0
                equity.append(equity[-1] * (1 + position.pnl_pct / 100.0))
                equity_time.append(t)
                position = None

    # Close any open trade on the final close
    if position is not None:
        last_t = df.index[-1]
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


def max_drawdown(eq: pd.Series) -> float:
    if eq.empty: return 0.0
    return float(((eq / eq.cummax()) - 1).min() * 100)

def summarize_trades(
    trades: List[Trade],
    commission_pct_per_side: float = 0.0,   # e.g., 0.02 => 0.02% per side
    slippage_bps_per_side: float = 0.0,     # e.g., 5 => 5 bps per side = 0.05%
) -> Tuple[float, float]:
    """
    Returns (win_rate %, total_return %), with per-side commissions and slippage
    applied to each closed trade. Costs are applied as a simple per-trade
    return haircut of 2 * (commission + slippage).
    """
    closed = [t for t in trades if t.pnl_pct is not None]
    if not closed:
        return 0.0, 0.0

    # Convert bps to pct
    slip_pct_side = slippage_bps_per_side / 10000.0
    side_cost = commission_pct_per_side / 100.0 + slip_pct_side   # in fraction
    total_cost = 2.0 * side_cost                                  # entry + exit

    # win rate based on *net* pnl
    wins = 0
    total_ret = 1.0
    for t in closed:
        gross_r = t.pnl_pct / 100.0
        net_r = gross_r - total_cost
        if net_r > 0:
            wins += 1
        total_ret *= (1.0 + net_r)

    win_rate = 100.0 * wins / len(closed)
    total_return_pct = (total_ret - 1.0) * 100.0
    return float(win_rate), float(total_return_pct)

# ---------- Optimizer helpers ----------
def evaluate_once(prices, sr_window, cluster_tol, buffer_pct, atr_period, stop_atr, take_atr):
    sr = detect_sr(prices, sr_window, cluster_tol)
    trades, equity = backtest(prices, sr, buffer_pct, atr_period, stop_atr, take_atr)
    win, net = summarize_trades(trades)
    mdd = max_drawdown(equity)
    n_trades = len([t for t in trades if t.pnl_pct is not None])
    return {
        "sr_window": sr_window, "cluster_tol": cluster_tol, "buffer_pct": buffer_pct,
        "atr_period": atr_period, "stop_atr": stop_atr, "take_atr": take_atr,
        "win_rate": win, "net": net, "mdd": mdd, "trades": n_trades,
        "trades_list": trades, "equity": equity, "sr_levels": sr
    }

def optimize_params(prices, min_trades=4, dd_penalty=0.7):
    # Small, fast grid (tweak as you like)
    sw_opts   = [15, 19, 25, 31]
    tol_opts  = [0.3, 0.6, 1.0]
    buf_opts  = [0.2, 0.5, 1.0]
    atr_opts  = [10, 14, 20]
    stop_opts = [1.5, 2.0, 2.5]
    take_opts = [2.0, 3.0, 4.0]

    rows = []
    best = None
    for sw in sw_opts:
        for tol in tol_opts:
            # compute SR once per (sw, tol) to speed up
            sr_cache = detect_sr(prices, sw, tol)
            for buf in buf_opts:
                for atrp in atr_opts:
                    for sl in stop_opts:
                        for tp in take_opts:
                            trades, equity = backtest(prices, sr_cache, buf, atrp, sl, tp)
                            win, net = summarize_trades(trades)
                            mdd = max_drawdown(equity)
                            n   = len([t for t in trades if t.pnl_pct is not None])

                            # score: reward profit, penalize drawdown, penalize too few trades
                            score = (net * 2.0) + (win * 0.5) - (max(0, mdd) * dd_penalty)
                            if n < min_trades:
                                score -= 50.0


                            row = {
                                "sr_window": sw, "cluster_tol": tol, "buffer_pct": buf,
                                "atr_period": atrp, "stop_atr": sl, "take_atr": tp,
                                "win_rate": win, "net": net, "mdd": mdd, "trades": n,
                                "score": score
                            }
                            rows.append(row)
                            if best is None or score > best["score"]:
                                best = row.copy()

    grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    return best, grid

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
    optimize_click = st.button("🧪 Optimize (profit ↑ / drawdown ↓)")
    commission_pct_per_side=commission_ps,
    slippage_bps_per_side=slippage_bps,

with st.sidebar:
    commission_ps = st.slider("Commission per side (%)", 0.0, 0.1, 0.02, 0.01)
    slippage_bps  = st.slider("Slippage per side (bps)", 0, 50, 5, 1)

# --- Backtest ---
trades, equity = backtest(
    prices, sr_levels,
    buffer_pct=params["buffer_pct"],
    atr_period=params["atr_period"],
    stop_atr=params["stop_atr"],
    take_atr=params["take_atr"],
)

# --- Summary with costs ---
win_rate, total_return = summarize_trades(
    trades,
    commission_pct_per_side=commission_ps,
    slippage_bps_per_side=slippage_bps,
)
mdd = max_drawdown(equity)


if start_date >= end_date:
    st.error("Start date must be before end date."); st.stop()

with st.spinner("Fetching cocoa prices…"):
    prices = get_prices(TICKER, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

# If user wants to apply the optimizer result to the chart
if "opt_best" in st.session_state and st.session_state["opt_best"]:
    with st.expander("Best parameters found (click 'Apply' to use them)"):
        b = st.session_state["opt_best"]
        st.write(pd.DataFrame([b]))
        if st.button("✅ Apply best params to chart"):
            st.session_state["override_params"] = {
                "sr_window": int(b["sr_window"]),
                "cluster_tol": float(b["cluster_tol"]),
                "buffer_pct": float(b["buffer_pct"]),
                "atr_period": int(b["atr_period"]),
                "stop_atr": float(b["stop_atr"]),
                "take_atr": float(b["take_atr"]),
            }
            st.rerun()

    # Optional: show top 20 candidates
    if "opt_grid" in st.session_state:
        st.dataframe(st.session_state["opt_grid"].head(20), use_container_width=True)


# run optimizer
if optimize_click:
    with st.spinner("Searching best parameters…"):
        best, grid = optimize_params(prices)
    st.session_state["opt_best"] = best
    st.session_state["opt_grid"] = grid
    st.success("Optimization done!")

if prices.empty:
    st.warning("No data returned."); st.stop()


# ----------------------------
# Strategy & Metrics
# ----------------------------
# Use sliders by default; override with optimizer choice if available
params = {
    "sr_window": sr_window,
    "cluster_tol": cluster_tol,
    "buffer_pct": buffer_pct,
    "atr_period": atr_period,
    "stop_atr": stop_atr,
    "take_atr": take_atr,
}
if "override_params" in st.session_state:
    params.update(st.session_state["override_params"])
    st.caption(f"Using optimized params: {params}")

# --- Strategy & Metrics using `params` ---
sr_levels   = detect_sr(prices, params["sr_window"], params["cluster_tol"])
trend_val   = slope(prices["Close"], 50).iloc[-1]
trend_val   = float(trend_val) if pd.notna(trend_val) else 0.0
bias        = "LONG" if trend_val > 0 else "SHORT"
last_close  = float(prices['Close'].iloc[-1])
ns_price    = nearest_level(last_close, sr_levels.support)
nr_price    = nearest_level(last_close, sr_levels.resistance)

# --- LIVE SIGNAL ---
trend_up = trend_val > 0
trend_down = trend_val < 0
signal = "HOLD"
reason = ""

if trend_up and ns_price and last_close <= ns_price * (1 + params["buffer_pct"]/100):
    signal = "BUY"
    reason = f"Price {last_close:.2f} near support {ns_price:.2f} with uptrend"
elif trend_down and nr_price and last_close >= nr_price * (1 - params["buffer_pct"]/100):
    signal = "SELL"
    reason = f"Price {last_close:.2f} near resistance {nr_price:.2f} with downtrend"

# Show the live signal prominently
signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[signal]
st.markdown(
    f"<h2 style='color:{signal_color};'>📢 Live Signal: {signal}</h2>", 
    unsafe_allow_html=True
)
if reason:
    st.caption(reason)


trades, equity = backtest(
    prices, sr_levels,
    buffer_pct=params["buffer_pct"],
    atr_period=params["atr_period"],
    stop_atr=params["stop_atr"],
    take_atr=params["take_atr"],
)
win_rate, total_return = summarize_trades(trades)
mdd = max_drawdown(equity)


colA, colB, colC, colD, colE, colF = st.columns([1.2, 1, 1, 1, 1, 1.2])
colA.markdown(f"### 📊 Current Bias: **{bias}**")
colB.metric("📉 Nearest Support", f"{ns_price:.2f}" if ns_price else "—")
colC.metric("📈 Nearest Resistance", f"{nr_price:.2f}" if nr_price else "—")
colD.metric("Win Rate", f"{win_rate:.2f}%")
colE.metric("Total Return", f"{total_return:+.2f}%")
colF.metric("Max Drawdown", f"{mdd:.2f}%")

# ----------------------------
# Chart
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"],
                             low=prices["Low"], close=prices["Close"], name="Price"))
for lvl in sr_levels.support:
    fig.add_hline(y=lvl, line_width=1, line_dash="dot", line_color="green", opacity=0.5)
for lvl in sr_levels.resistance:
    fig.add_hline(y=lvl, line_width=1, line_dash="dot", line_color="red", opacity=0.5)
long_x, long_y, short_x, short_y, exit_x, exit_y = [], [], [], [], [], []
for t in trades:
    if t.side == "long": long_x.append(t.entry_time); long_y.append(t.entry_price)
    else: short_x.append(t.entry_time); short_y.append(t.entry_price)
    if t.exit_time and t.exit_price: exit_x.append(t.exit_time); exit_y.append(t.exit_price)
if long_x: fig.add_trace(go.Scatter(x=long_x, y=long_y, mode="markers", name="Buy",
                                    marker=dict(symbol="triangle-up", size=10, color="green")))
if short_x: fig.add_trace(go.Scatter(x=short_x, y=short_y, mode="markers", name="Sell",
                                     marker=dict(symbol="triangle-down", size=10, color="red")))
if exit_x: fig.add_trace(go.Scatter(x=exit_x, y=exit_y, mode="markers", name="Exit",
                                    marker=dict(symbol="x", size=9, color="gray")))
bg_color = "rgba(0,128,0,0.07)" if bias == "LONG" else "rgba(220,20,60,0.07)"
fig.add_vrect(x0=prices.index[-min(len(prices), 100)], x1=prices.index.max(),
              fillcolor=bg_color, line_width=0, layer="below")
fig.update_layout(height=700, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
