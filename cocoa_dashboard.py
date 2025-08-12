# cocoa_dashboard.py (Donchian Breakout variant added)
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
DEFAULT_YEARS = 3
st.set_page_config(page_title="Cocoa Dashboard", layout="wide", initial_sidebar_state="collapsed")
# Auto-refresh (toggleable later in sidebar)
st_autorefresh(interval=60000, key="data_refresh")  # 60,000 ms = 60 sec


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

# ---------- Core indicators ----------

def compute_atr_sma(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def compute_atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@st.cache_data(show_spinner=False)
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


def slope(x: pd.Series, period: int = 50) -> pd.Series:
    sma = x.rolling(period, min_periods=period).mean()
    return sma.diff()

# ---------- S/R engine (existing) ----------

@dataclass
class SRLevels:
    support: List[float]
    resistance: List[float]


def rolling_extrema(series: pd.Series, window: int, mode: str) -> pd.Series:
    if window % 2 == 0:
        window += 1
    half = window // 2
    roll = series.rolling(window, center=True)
    if mode == "max":
        is_ext = (series == roll.max()) & (series > series.shift(1)) & (series > series.shift(-1))
    else:
        is_ext = (series == roll.min()) & (series < series.shift(1)) & (series < series.shift(-1))
    piv = series.where(is_ext).copy()
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

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None


# ---------- Original S/R backtest ----------

def backtest_sr(df: pd.DataFrame, sr: SRLevels,
             buffer_pct: float = 0.3, atr_period: int = 14,
             stop_atr: float = 2.0, take_atr: float = 3.0) -> Tuple[List[Trade], pd.Series]:

    close = df["Close"].astype(float)
    atr = compute_atr_sma(df, atr_period)
    trend = slope(close, 50).reindex(df.index)  # align trend

    trades, equity, equity_time = [], [1.0], [df.index[0]]
    position = None

    for i, (t, px) in enumerate(close.items()):
        tr_val = float(trend.iloc[i]) if pd.notna(trend.iloc[i]) else 0.0
        tr_up, tr_dn = tr_val > 0, tr_val < 0
        ns, nr = nearest_level(px, sr.support), nearest_level(px, sr.resistance)

        if position is None:
            if tr_up and ns and px <= ns * (1 + buffer_pct / 100.0):
                position = Trade(t, float(px), "long"); trades.append(position)
            elif tr_dn and nr and px >= nr * (1 - buffer_pct / 100.0):
                position = Trade(t, float(px), "short"); trades.append(position)
        else:
            a = float(atr.iloc[i]) if np.isfinite(atr.iloc[i]) else 1e-6
            if position.side == "long":
                stop, take = position.entry_price - stop_atr * a, position.entry_price + take_atr * a
                if px <= stop or px >= take:
                    position.exit_time, position.exit_price = t, float(px)
                    position.pnl_pct = (px / position.entry_price - 1) * 100
                    equity.append(equity[-1] * (1 + position.pnl_pct / 100)); equity_time.append(t)
                    position = None
            else:
                stop, take = position.entry_price + stop_atr * a, position.entry_price - take_atr * a
                if px >= stop or px <= take:
                    position.exit_time, position.exit_price = t, float(px)
                    position.pnl_pct = (position.entry_price / px - 1) * 100
                    equity.append(equity[-1] * (1 + position.pnl_pct / 100)); equity_time.append(t)
                    position = None

    if position is not None:
        last_px = float(close.iloc[-1])
        pnl = (last_px / position.entry_price - 1) * 100 if position.side == "long" else (position.entry_price / last_px - 1) * 100
        position.exit_time, position.exit_price, position.pnl_pct = close.index[-1], last_px, pnl
        equity.append(equity[-1] * (1 + pnl / 100)); equity_time.append(close.index[-1])

    return trades, pd.Series(equity, index=pd.to_datetime(equity_time)).sort_index()


# ---------- Donchian + Chandelier + ADX backtest ----------

def donchian_channels(df: pd.DataFrame, n: int) -> Tuple[pd.Series, pd.Series]:
    high_n = df["High"].rolling(n, min_periods=n).max()
    low_n = df["Low"].rolling(n, min_periods=n).min()
    return high_n, low_n


def chandelier_stops(df: pd.DataFrame, atr: pd.Series, n: int = 22, m: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    long_stop = df["High"].rolling(n, min_periods=n).max() - m * atr
    short_stop = df["Low"].rolling(n, min_periods=n).min() + m * atr
    return long_stop, short_stop


def backtest_donchian(df: pd.DataFrame,
                      n_entry: int = 20,
                      n_exit: int = 10,
                      atr_period: int = 20,
                      chand_n: int = 22,
                      chand_mult: float = 3.0,
                      adx_period: int = 14,
                      adx_min: float = 25.0,
                      use_wilder_atr: bool = True) -> Tuple[List[Trade], pd.Series, dict]:
    """Donchian breakout with optional Chandelier trailing stop and ADX filter.
    Returns trades, equity curve, and a dict of series (for plotting)."""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    atr = compute_atr_wilder(df, atr_period) if use_wilder_atr else compute_atr_sma(df, atr_period)
    adx = compute_adx(df, adx_period)

    high_entry, low_entry = donchian_channels(df, n_entry)
    high_exit, low_exit = donchian_channels(df, n_exit)
    chand_long, chand_short = chandelier_stops(df, atr, chand_n, chand_mult)

    trades: List[Trade] = []
    equity, equity_time = [1.0], [df.index[0]]
    position: Optional[Trade] = None

    # Use prior-bar signals to avoid look-ahead
    for i in range(1, len(df)):
        t = df.index[i]
        px_o, px_h, px_l, px_c = float(df["Open"].iloc[i]), float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])

        # Current (i-1) references
        hi_ent = float(high_entry.iloc[i-1]) if np.isfinite(high_entry.iloc[i-1]) else np.nan
        lo_ent = float(low_entry.iloc[i-1]) if np.isfinite(low_entry.iloc[i-1]) else np.nan
        hi_ex = float(high_exit.iloc[i-1]) if np.isfinite(high_exit.iloc[i-1]) else np.nan
        lo_ex = float(low_exit.iloc[i-1]) if np.isfinite(low_exit.iloc[i-1]) else np.nan
        adx_ok = float(adx.iloc[i-1]) >= adx_min if np.isfinite(adx.iloc[i-1]) else False
        chand_l = float(chand_long.iloc[i-1]) if np.isfinite(chand_long.iloc[i-1]) else np.nan
        chand_s = float(chand_short.iloc[i-1]) if np.isfinite(chand_short.iloc[i-1]) else np.nan

        if position is None:
            entered = False
            # Long breakout
            if adx_ok and np.isfinite(hi_ent) and px_c > hi_ent:
                position = Trade(t, float(px_c), "long"); trades.append(position); entered = True
            # Short breakout
            elif adx_ok and np.isfinite(lo_ent) and px_c < lo_ent:
                position = Trade(t, float(px_c), "short"); trades.append(position); entered = True
            if entered:
                continue
        else:
            # Manage exits with intrabar logic (stop prioritized if both hit)
            if position.side == "long":
                stop_lvl = chand_l if np.isfinite(chand_l) else -np.inf
                exit_lvl = lo_ex if np.isfinite(lo_ex) else -np.inf
                stop_hit = np.isfinite(stop_lvl) and (px_l <= stop_lvl)
                exit_hit = np.isfinite(exit_lvl) and (px_c < exit_lvl)
                if stop_hit or exit_hit:
                    # Assume stop executes first if both same bar
                    exit_price = stop_lvl if stop_hit else px_c
                    position.exit_time, position.exit_price = t, float(exit_price)
                    position.pnl_pct = (exit_price / position.entry_price - 1) * 100
                    equity.append(equity[-1] * (1 + position.pnl_pct / 100)); equity_time.append(t)
                    position = None
            else:  # short
                stop_lvl = chand_s if np.isfinite(chand_s) else np.inf
                exit_lvl = hi_ex if np.isfinite(hi_ex) else np.inf
                stop_hit = np.isfinite(stop_lvl) and (px_h >= stop_lvl)
                exit_hit = np.isfinite(exit_lvl) and (px_c > exit_lvl)
                if stop_hit or exit_hit:
                    exit_price = stop_lvl if stop_hit else px_c
                    position.exit_time, position.exit_price = t, float(exit_price)
                    position.pnl_pct = (position.entry_price / exit_price - 1) * 100
                    equity.append(equity[-1] * (1 + position.pnl_pct / 100)); equity_time.append(t)
                    position = None

    # Close open at last
    if position is not None:
        last_px = float(close.iloc[-1])
        pnl = (last_px / position.entry_price - 1) * 100 if position.side == "long" else (position.entry_price / last_px - 1) * 100
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


# ---------- Metrics helpers ----------

def max_drawdown(eq: pd.Series) -> float:
    if eq.empty: return 0.0
    return float(((eq / eq.cummax()) - 1).min() * 100)


def summarize_trades(trades: List[Trade]) -> Tuple[float, float]:
    closed = [t for t in trades if t.pnl_pct is not None]
    if not closed: return 0.0, 0.0
    win_rate = sum(1 for t in closed if t.pnl_pct > 0) / len(closed) * 100
    total_ret = 1.0
    for t in closed: total_ret *= (1 + t.pnl_pct / 100)
    return win_rate, (total_ret - 1) * 100

# ---------- Optimizer (unchanged, for SR only) ----------

def evaluate_once(prices, sr_window, cluster_tol, buffer_pct, atr_period, stop_atr, take_atr):
    sr = detect_sr(prices, sr_window, cluster_tol)
    trades, equity = backtest_sr(prices, sr, buffer_pct, atr_period, stop_atr, take_atr)
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
                            trades, equity = backtest_sr(prices, sr_cache, buf, atrp, sl, tp)
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

    # Strategy selector
    strategy = st.selectbox("Strategy", ["S/R Bounce (original)", "Donchian Breakout"], index=1)

    # Common trend viz
    trend_len = st.slider("Trend window for bias (SMA)", 20, 200, 50, step=5)

    # Strategy-specific params
    if strategy == "S/R Bounce (original)":
        st.subheader("S/R Parameters")
        sr_window = st.slider("Pivot window (bars)", 15, 61, 25, step=2)
        cluster_tol = st.slider("SR cluster tolerance (%)", 0.1, 1.5, 0.6, step=0.1)
        buffer_pct = st.slider("Entry buffer around SR (%)", 0.0, 1.5, 0.3, step=0.1)
        atr_period = st.slider("ATR period", 5, 30, 14)
        stop_atr = st.slider("Stop Loss (×ATR)", 0.5, 5.0, 2.0, step=0.1)
        take_atr = st.slider("Take Profit (×ATR)", 0.5, 8.0, 3.0, step=0.1)
        optimize_click = st.button("🧪 Optimize (profit ↑ / drawdown ↓)")
    else:
        st.subheader("Donchian / Chandelier / ADX")
        n_entry = st.slider("Donchian entry (N-high/low)", 10, 60, 20)
        n_exit = st.slider("Donchian exit (N for opposite)", 5, 40, 10)
        atr_period_d = st.slider("ATR period (Wilder)", 5, 40, 20)
        chand_n = st.slider("Chandelier lookback (N)", 10, 60, 22)
        chand_mult = st.slider("Chandelier multiple (×ATR)", 1.0, 5.0, 3.0, step=0.1)
        adx_period = st.slider("ADX period", 7, 30, 14)
        adx_min = st.slider("ADX minimum to trade", 10.0, 40.0, 25.0, step=0.5)
        optimize_click = False  # not wired for Donchian in this build

    # Refresh toggle
    auto_refresh = st.checkbox("Auto-refresh every 60s (daily data doesn’t need it)", value=True)

if not auto_refresh:
    # kill the autorefresh by resetting the key (hacky but works in practice)
    st_autorefresh(interval=0, key="data_refresh_off")

if start_date >= end_date:
    st.error("Start date must be before end date."); st.stop()

with st.spinner("Fetching cocoa prices…"):
    prices = get_prices(TICKER, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if prices.empty:
    st.warning("No data returned."); st.stop()

# ----------------------------
# Strategy & Metrics
# ----------------------------
trend_val = slope(prices["Close"], trend_len).iloc[-1]
trend_val = float(trend_val) if pd.notna(trend_val) else 0.0
bias = "LONG" if trend_val > 0 else "SHORT"
last_close = float(prices['Close'].iloc[-1])

if strategy == "S/R Bounce (original)":
    # Optimizer bits (only for SR)
    refresh = st.sidebar.button("🔄 Refresh data")
    if "opt_best" in st.session_state and st.session_state.get("opt_best"):
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
        if "opt_grid" in st.session_state:
            st.dataframe(st.session_state["opt_grid"].head(20), use_container_width=True)

    # run optimizer
    if optimize_click:
        with st.spinner("Searching best parameters…"):
            best, grid = optimize_params(prices)
        st.session_state["opt_best"] = best
        st.session_state["opt_grid"] = grid
        st.success("Optimization done!")

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

    sr_levels   = detect_sr(prices, params["sr_window"], params["cluster_tol"])
    ns_price    = nearest_level(last_close, sr_levels.support)
    nr_price    = nearest_level(last_close, sr_levels.resistance)

    # LIVE SIGNAL for SR
    trend_up = trend_val > 0
    trend_down = trend_val < 0
    signal = "HOLD"; reason = ""
    if trend_up and ns_price is not None and last_close <= ns_price * (1 + params["buffer_pct"]/100):
        signal = "BUY"; reason = f"Price {last_close:.2f} near support {ns_price:.2f} with uptrend"
    elif trend_down and nr_price is not None and last_close >= nr_price * (1 - params["buffer_pct"]/100):
        signal = "SELL"; reason = f"Price {last_close:.2f} near resistance {nr_price:.2f} with downtrend"

    trades, equity = backtest_sr(
        prices, sr_levels,
        buffer_pct=params["buffer_pct"],
        atr_period=params["atr_period"],
        stop_atr=params["stop_atr"],
        take_atr=params["take_atr"],
    )

    # Metrics
    win_rate, total_return = summarize_trades(trades)
    mdd = max_drawdown(equity)

else:
    # Donchian strategy path
    ns_price = nr_price = None  # not applicable

    trades, equity, plot_series = backtest_donchian(
        prices,
        n_entry=n_entry,
        n_exit=n_exit,
        atr_period=atr_period_d,
        chand_n=chand_n,
        chand_mult=chand_mult,
        adx_period=adx_period,
        adx_min=adx_min,
        use_wilder_atr=True,
    )

    # LIVE SIGNAL for Donchian (use prior-bar channels)
    he, le, adx_s = plot_series["high_entry"], plot_series["low_entry"], plot_series["adx"]
    he_prev = float(he.iloc[-2]) if len(he) >= 2 and np.isfinite(he.iloc[-2]) else np.nan
    le_prev = float(le.iloc[-2]) if len(le) >= 2 and np.isfinite(le.iloc[-2]) else np.nan
    adx_ok = float(adx_s.iloc[-2]) >= adx_min if len(adx_s) >= 2 and np.isfinite(adx_s.iloc[-2]) else False

    signal = "HOLD"; reason = ""
    if adx_ok and np.isfinite(he_prev) and last_close > he_prev:
        signal = "BUY"; reason = f"Close {last_close:.2f} broke {n_entry}-day high {he_prev:.2f} with ADX≥{adx_min:.0f}"
    elif adx_ok and np.isfinite(le_prev) and last_close < le_prev:
        signal = "SELL"; reason = f"Close {last_close:.2f} broke {n_entry}-day low {le_prev:.2f} with ADX≥{adx_min:.0f}"

    # Metrics
    win_rate, total_return = summarize_trades(trades)
    mdd = max_drawdown(equity)

# ----------------------------
# Header signal & metrics
# ----------------------------
signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[signal]
st.markdown(
    f"<h2 style='color:{signal_color};'>📢 Live Signal: {signal}</h2>", 
    unsafe_allow_html=True
)
if reason:
    st.caption(reason)

colA, colB, colC, colD, colE, colF = st.columns([1.2, 1, 1, 1, 1, 1.2])
colA.markdown(f"### 📊 Current Bias: **{bias}**")
colB.metric("📉 Nearest Support", f"{ns_price:.2f}" if ns_price is not None else "—")
colC.metric("📈 Nearest Resistance", f"{nr_price:.2f}" if nr_price is not None else "—")
colD.metric("Win Rate", f"{win_rate:.2f}%")
colE.metric("Total Return", f"{total_return:+.2f}%")
colF.metric("Max Drawdown", f"{mdd:.2f}%")

# ----------------------------
# Chart
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"],
                             low=prices["Low"], close=prices["Close"], name="Price"))

if strategy == "S/R Bounce (original)":
    sr_levels = detect_sr(prices, sr_window if 'sr_window' in locals() else 25, cluster_tol if 'cluster_tol' in locals() else 0.6)
    for lvl in sr_levels.support:
        fig.add_hline(y=lvl, line_width=1, line_dash="dot", line_color="green", opacity=0.5)
    for lvl in sr_levels.resistance:
        fig.add_hline(y=lvl, line_width=1, line_dash="dot", line_color="red", opacity=0.5)
else:
    # Donchian channel & chandelier overlays
    he = plot_series["high_entry"]
    le = plot_series["low_entry"]
    hx = plot_series["high_exit"]
    lx = plot_series["low_exit"]
    ch_l = plot_series["chand_long"]
    ch_s = plot_series["chand_short"]

    fig.add_trace(go.Scatter(x=he.index, y=he, name=f"{n_entry}D High", mode="lines"))
    fig.add_trace(go.Scatter(x=le.index, y=le, name=f"{n_entry}D Low", mode="lines"))
    fig.add_trace(go.Scatter(x=hx.index, y=hx, name=f"{n_exit}D High (exit)", mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=lx.index, y=lx, name=f"{n_exit}D Low (exit)", mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=ch_l.index, y=ch_l, name="Chandelier Long", mode="lines"))
    fig.add_trace(go.Scatter(x=ch_s.index, y=ch_s, name="Chandelier Short", mode="lines"))

# Trade markers
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
fig.update_layout(height=720, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
