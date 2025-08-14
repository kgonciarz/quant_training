# coffee_dashboard.py — Donchian breakout with switchable filters + optimizer
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

TICKER = "KC=F"
DEFAULT_YEARS = 3

st.set_page_config(page_title="Coffee Dashboard", layout="wide", initial_sidebar_state="expanded")
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
            if symbol in df.columns.get_level_values(0):
                df = df.xs(symbol, axis=1, level=0)
            else:
                df = df.droplevel(0, axis=1)
        df = df.rename(columns=lambda c: c.title())
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # 1) Try download()
    for _ in range(3):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, interval="1d", progress=False, threads=False)
        df = _normalize(df)
        if not df.empty:
            return df
        time.sleep(1.2)

    # 2) Try history() with end+1 day
    try:
        end_plus = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.Ticker(symbol).history(start=start, end=end_plus, auto_adjust=True, interval="1d")
        df = _normalize(df)
        if not df.empty:
            return df
    except Exception:
        pass

    # 3) Last resort: period then slice
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
        f"Yahoo returned no daily data for {symbol} in {start} → {end}. Try a shorter range or click Refresh again."
    )


def today_utc():
    return dt.datetime.utcnow().date()

# ----------------------------
# Indicators & helpers
# ----------------------------

def compute_atr_sma(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def compute_atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


@st.cache_data(show_spinner=False)
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


def slope(x: pd.Series, period: int = 50) -> pd.Series:
    sma = x.rolling(period, min_periods=period).mean()
    return sma.diff()

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None

# ----------------------------
# Donchian + optional filters
# ----------------------------

def donchian_channels(df: pd.DataFrame, n: int) -> Tuple[pd.Series, pd.Series]:
    high_n = df["High"].rolling(n, min_periods=n).max()
    low_n = df["Low"].rolling(n, min_periods=n).min()
    return high_n, low_n


def chandelier_stops(df: pd.DataFrame, atr: pd.Series, n: int = 22, m: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    long_stop = df["High"].rolling(n, min_periods=n).max() - m * atr
    short_stop = df["Low"].rolling(n, min_periods=n).min() + m * atr
    return long_stop, short_stop


def backtest_donchian(
    df: pd.DataFrame,
    n_entry: int = 20,
    n_exit: int = 10,
    atr_period: int = 20,
    chand_n: int = 22,
    chand_mult: float = 3.0,
    adx_period: int = 14,
    adx_min: float = 25.0,
    use_adx: bool = True,
    use_chandelier: bool = True,
    long_only: bool = False,
    short_trend_gate: bool = True,
    trend_sma: int = 200,
) -> Tuple[List[Trade], pd.Series, dict]:
    """Donchian breakout with optional ADX filter, Chandelier trailing stop,
    and optional short-only regime gate (price < SMA(trend_sma))."""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    atr = compute_atr_wilder(df, atr_period)
    adx = compute_adx(df, adx_period)
    sma_long = close.rolling(trend_sma, min_periods=trend_sma).mean()

    high_entry, low_entry = donchian_channels(df, n_entry)
    high_exit, low_exit = donchian_channels(df, n_exit)
    chand_long, chand_short = chandelier_stops(df, atr, chand_n, chand_mult)

    trades: List[Trade] = []
    equity, equity_time = [1.0], [df.index[0]]
    position: Optional[Trade] = None

    for i in range(1, len(df)):
        t = df.index[i]
        px_h, px_l, px_c = float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])

        hi_ent = float(high_entry.iloc[i-1]) if np.isfinite(high_entry.iloc[i-1]) else np.nan
        lo_ent = float(low_entry.iloc[i-1]) if np.isfinite(low_entry.iloc[i-1]) else np.nan
        hi_ex = float(high_exit.iloc[i-1]) if np.isfinite(high_exit.iloc[i-1]) else np.nan
        lo_ex = float(low_exit.iloc[i-1]) if np.isfinite(low_exit.iloc[i-1]) else np.nan
        adx_ok = (float(adx.iloc[i-1]) >= adx_min) if (use_adx and np.isfinite(adx.iloc[i-1])) else True
        sma_ok_short = (float(sma_long.iloc[i-1]) > 0 and px_c < float(sma_long.iloc[i-1])) if short_trend_gate else True

        chand_l = float(chand_long.iloc[i-1]) if (use_chandelier and np.isfinite(chand_long.iloc[i-1])) else np.nan
        chand_s = float(chand_short.iloc[i-1]) if (use_chandelier and np.isfinite(chand_short.iloc[i-1])) else np.nan

        if position is None:
            if adx_ok and np.isfinite(hi_ent) and px_c > hi_ent:
                position = Trade(t, float(px_c), "long"); trades.append(position); continue
            if (not long_only) and adx_ok and np.isfinite(lo_ent) and px_c < lo_ent and sma_ok_short:
                position = Trade(t, float(px_c), "short"); trades.append(position); continue
        else:
            # Intrabar exit logic. If Chandelier disabled, use only Donchian exits.
            if position.side == "long":
                stop_hit = np.isfinite(chand_l) and (px_l <= chand_l)
                exit_hit = np.isfinite(lo_ex) and (px_c < lo_ex)
                if stop_hit or exit_hit:
                    exit_price = chand_l if stop_hit else px_c
                    position.exit_time, position.exit_price = t, float(exit_price)
                    position.pnl_pct = (exit_price / position.entry_price - 1) * 100
                    equity.append(equity[-1] * (1 + position.pnl_pct / 100)); equity_time.append(t)
                    position = None
            else:
                stop_hit = np.isfinite(chand_s) and (px_h >= chand_s)
                exit_hit = np.isfinite(hi_ex) and (px_c > hi_ex)
                if stop_hit or exit_hit:
                    exit_price = chand_s if stop_hit else px_c
                    position.exit_time, position.exit_price = t, float(exit_price)
                    position.pnl_pct = (position.entry_price / exit_price - 1) * 100
                    equity.append(equity[-1] * (1 + position.pnl_pct / 100)); equity_time.append(t)
                    position = None

    if position is not None:
        last_px = float(close.iloc[-1])
        pnl = (last_px / position.entry_price - 1) * 100 if position.side == "long" else (position.entry_price / last_px - 1) * 100
        position.exit_time, position.exit_price, position.pnl_pct = close.index[-1], last_px, pnl
        equity.append(equity[-1] * (1 + pnl / 100)); equity_time.append(close.index[-1])

    plot_series = {
        "high_entry": high_entry, "low_entry": low_entry,
        "high_exit": high_exit,   "low_exit": low_exit,
        "chand_long": chand_long, "chand_short": chand_short,
        "adx": adx, "atr": atr, "sma_long": sma_long,
    }
    return trades, pd.Series(equity, index=pd.to_datetime(equity_time)).sort_index(), plot_series

# ----------------------------
# Metrics & optimizer
# ----------------------------

def max_drawdown(eq: pd.Series) -> float:
    if eq.empty: return 0.0
    return float(((eq / eq.cummax()) - 1).min() * 100)


def summarize_trades(trades: List[Trade]) -> Tuple[float, float, int]:
    closed = [t for t in trades if t.pnl_pct is not None]
    if not closed: return 0.0, 0.0, 0
    win_rate = sum(1 for t in closed if t.pnl_pct > 0) / len(closed) * 100
    total_ret = 1.0
    for t in closed: total_ret *= (1 + t.pnl_pct / 100)
    return win_rate, (total_ret - 1) * 100, len(closed)


def directional_breakdown(trades: List[Trade]) -> dict:
    longs = [t for t in trades if t.pnl_pct is not None and t.side == "long"]
    shorts = [t for t in trades if t.pnl_pct is not None and t.side == "short"]
    def _stats(ts):
        if not ts: return {"n":0, "win":0.0, "net":0.0}
        wr = sum(1 for t in ts if t.pnl_pct > 0)/len(ts)*100
        net=1.0
        for t in ts: net *= (1+t.pnl_pct/100)
        return {"n":len(ts), "win":wr, "net":(net-1)*100}
    return {"long": _stats(longs), "short": _stats(shorts)}


def optimize_donchian(prices: pd.DataFrame, *,
                      n_entry_opts=(20,26,39,52), n_exit_opts=(10,13,26),
                      use_adx_opts=(False, True), adx_min_opts=(0,20,25,30),
                      use_chand_opts=(False, True),
                      short_gate_opts=(False, True), long_only_opts=(True, False)):
    rows = []
    best = None
    for ne in n_entry_opts:
        he, le = donchian_channels(prices, ne)  # cache per ne if needed
        for nx in n_exit_opts:
            for ua in use_adx_opts:
                for am in adx_min_opts:
                    for uc in use_chand_opts:
                        for sg in short_gate_opts:
                            for lo in long_only_opts:
                                trades, eq, _ = backtest_donchian(
                                    prices, n_entry=ne, n_exit=nx,
                                    use_adx=ua, adx_min=am,
                                    use_chandelier=uc,
                                    long_only=lo,
                                    short_trend_gate=sg,
                                )
                                wr, net, n = summarize_trades(trades)
                                mdd = max_drawdown(eq)
                                d = directional_breakdown(trades)
                                score = (net*2.0) + (wr*0.3) - (max(0,mdd)*0.7)
                                if n < 4: score -= 40
                                row = {
                                    "n_entry": ne, "n_exit": nx,
                                    "use_adx": ua, "adx_min": am,
                                    "use_chandelier": uc,
                                    "short_trend_gate": sg,
                                    "long_only": lo,
                                    "win_rate": wr, "net": net, "mdd": mdd, "trades": n,
                                    "long_win": d["long"]["win"], "long_net": d["long"]["net"], "long_n": d["long"]["n"],
                                    "short_win": d["short"]["win"], "short_net": d["short"]["net"], "short_n": d["short"]["n"],
                                    "score": score,
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
    st.header("Coffee Settings")   
    end_default = today_utc()
    start_default = end_default - dt.timedelta(days=365 * DEFAULT_YEARS)
    start_date = st.date_input("Start date", value=start_default)
    end_date = st.date_input("End date", value=end_default)

    st.subheader("Strategy")
    strategy = st.selectbox("Strategy", ["Donchian Breakout", "S/R Bounce (original)"], index=0)

    trend_len = st.slider("Trend window for bias (SMA)", 20, 200, 50, step=5)

    if strategy == "Donchian Breakout":
        st.caption("Tip: If shorts underperform, try Long-only or gate shorts by 200SMA downtrend.")
        n_entry = st.slider("Donchian entry (N-high/low)", 10, 60, 20)
        n_exit = st.slider("Donchian exit (opposite N)", 5, 40, 10)
        use_adx = st.checkbox("Use ADX filter", value=True)
        adx_min = st.slider("ADX minimum", 0.0, 40.0, 25.0, step=0.5)
        use_chandelier = st.checkbox("Use Chandelier trailing stop", value=True)
        chand_n = st.slider("Chandelier lookback (N)", 10, 60, 22)
        chand_mult = st.slider("Chandelier ×ATR", 1.0, 5.0, 3.0, step=0.1)
        long_only = st.checkbox("Long-only", value=False)
        short_gate = st.checkbox("Gate shorts by SMA200 downtrend", value=True)
        optimize_d = st.button("🧪 Optimize Donchian (find best filters)")
    else:
        # Original S/R params
        sr_window = st.slider("Pivot window (bars)", 15, 61, 25, step=2)
        cluster_tol = st.slider("SR cluster tolerance (%)", 0.1, 1.5, 0.6, step=0.1)
        buffer_pct = st.slider("Entry buffer around SR (%)", 0.0, 1.5, 0.3, step=0.1)
        atr_period = st.slider("ATR period", 5, 30, 14)
        stop_atr = st.slider("Stop Loss (×ATR)", 0.5, 5.0, 2.0, step=0.1)
        take_atr = st.slider("Take Profit (×ATR)", 0.5, 8.0, 3.0, step=0.1)
        optimize_sr = st.button("🧪 Optimize SR")

    auto_refresh = st.checkbox("Auto-refresh every 60s (daily data doesn’t need it)", value=True)

if not auto_refresh:
    st_autorefresh(interval=0, key="data_refresh_off")

if start_date >= end_date:
    st.error("Start date must be before end date."); st.stop()

with st.spinner("Fetching coffee prices…"):
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

if strategy == "Donchian Breakout":
    trades, equity, plot_series = backtest_donchian(
        prices,
        n_entry=n_entry, n_exit=n_exit,
        atr_period=20, chand_n=chand_n, chand_mult=chand_mult,
        adx_period=14, adx_min=adx_min, use_adx=use_adx,
        use_chandelier=use_chandelier,
        long_only=long_only, short_trend_gate=short_gate, trend_sma=200,
    )

    # LIVE signal
    he, le, adx_s = plot_series["high_entry"], plot_series["low_entry"], plot_series["adx"]
    he_prev = float(he.iloc[-2]) if len(he) >= 2 and np.isfinite(he.iloc[-2]) else np.nan
    le_prev = float(le.iloc[-2]) if len(le) >= 2 and np.isfinite(le.iloc[-2]) else np.nan
    adx_ok = (float(adx_s.iloc[-2]) >= adx_min) if use_adx and len(adx_s) >= 2 and np.isfinite(adx_s.iloc[-2]) else True

    signal = "HOLD"; reason = ""
    if np.isfinite(he_prev) and adx_ok and last_close > he_prev:
        signal, reason = "BUY", f"Close {last_close:.2f} > {n_entry}-day high {he_prev:.2f}" + (f" (ADX≥{adx_min:.0f})" if use_adx else "")
    elif (not long_only) and np.isfinite(le_prev) and adx_ok and last_close < le_prev:
        signal, reason = "SELL", f"Close {last_close:.2f} < {n_entry}-day low {le_prev:.2f}" + (f" (ADX≥{adx_min:.0f})" if use_adx else "")

    # Optimizer
    if 'don_opt_best' not in st.session_state:
        st.session_state['don_opt_best'] = None
    if 'don_opt_grid' not in st.session_state:
        st.session_state['don_opt_grid'] = None

    if 'opt_applied' not in st.session_state:
        st.session_state['opt_applied'] = False

    if st.session_state['don_opt_best']:
        with st.expander("Best Donchian params found (click 'Apply' to use them)"):
            b = st.session_state['don_opt_best']
            st.write(pd.DataFrame([b]))
            if st.button("✅ Apply best params to chart"):
                # apply
                n_entry = int(b['n_entry']); n_exit = int(b['n_exit'])
                use_adx = bool(b['use_adx']); adx_min = float(b['adx_min'])
                use_chandelier = bool(b['use_chandelier'])
                short_gate = bool(b['short_trend_gate']); long_only = bool(b['long_only'])
                st.session_state['opt_applied'] = True
                st.rerun()
        if st.session_state['don_opt_grid'] is not None:
            st.dataframe(st.session_state['don_opt_grid'].head(20), use_container_width=True)

    if optimize_d:
        with st.spinner("Searching best filters & params…"):
            best, grid = optimize_donchian(prices)
        st.session_state['don_opt_best'] = best
        st.session_state['don_opt_grid'] = grid
        st.success("Optimization done!")

    # Metrics
    win_rate, total_return, n_trades = summarize_trades(trades)
    mdd = max_drawdown(equity)
    dir_stats = directional_breakdown(trades)

else:
    # --- Original S/R path (kept as-is so you can compare) ---
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
        piv = series.where(is_ext).copy(); piv.iloc[:half] = np.nan; piv.iloc[-half:] = np.nan
        return piv

    def cluster_levels(level_values: List[float], tol_pct: float = 0.6) -> List[float]:
        if not level_values: return []
        vals = sorted([float(x) for x in level_values if np.isfinite(x)])
        clusters = []
        for lv in vals:
            if not clusters: clusters.append([lv])
            else:
                ref = np.mean(clusters[-1])
                if abs(lv - ref) / ref * 100.0 <= tol_pct: clusters[-1].append(lv)
                else: clusters.append([lv])
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
        if not levels: return None
        arr = np.array(levels, dtype=float)
        return float(arr[np.argmin(np.abs(arr - price))])

    sr_levels   = detect_sr(prices, sr_window, cluster_tol)
    ns_price    = nearest_level(last_close, sr_levels.support)
    nr_price    = nearest_level(last_close, sr_levels.resistance)

    # Live signal for SR
    trend_up = trend_val > 0; trend_down = trend_val < 0
    signal = "HOLD"; reason = ""
    if trend_up and ns_price is not None and last_close <= ns_price * (1 + buffer_pct/100):
        signal = "BUY"; reason = f"Price {last_close:.2f} near support {ns_price:.2f} with uptrend"
    elif trend_down and nr_price is not None and last_close >= nr_price * (1 - buffer_pct/100):
        signal = "SELL"; reason = f"Price {last_close:.2f} near resistance {nr_price:.2f} with downtrend"

    # Simple SR backtest (from your original)
    def backtest_sr(df: pd.DataFrame, sr: SRLevels, buffer_pct: float = 0.3, atr_period: int = 14, stop_atr: float = 2.0, take_atr: float = 3.0):
        close = df["Close"].astype(float)
        atr = compute_atr_sma(df, atr_period)
        trend = slope(close, 50).reindex(df.index)
        trades, equity, equity_time = [], [1.0], [df.index[0]]; position=None
        for i, (t, px) in enumerate(close.items()):
            tr_val = float(trend.iloc[i]) if pd.notna(trend.iloc[i]) else 0.0
            tr_up, tr_dn = tr_val > 0, tr_val < 0
            ns, nr = nearest_level(px, sr.support), nearest_level(px, sr.resistance)
            if position is None:
                if tr_up and ns and px <= ns * (1 + buffer_pct / 100.0): position = Trade(t, float(px), "long"); trades.append(position)
                elif tr_dn and nr and px >= nr * (1 - buffer_pct / 100.0): position = Trade(t, float(px), "short"); trades.append(position)
            else:
                a = float(atr.iloc[i]) if np.isfinite(atr.iloc[i]) else 1e-6
                if position.side == "long":
                    stop, take = position.entry_price - stop_atr * a, position.entry_price + take_atr * a
                    if px <= stop or px >= take:
                        position.exit_time, position.exit_price = t, float(px); position.pnl_pct = (px / position.entry_price - 1) * 100
                        equity.append(equity[-1] * (1 + position.pnl_pct / 100)); equity_time.append(t); position = None
                else:
                    stop, take = position.entry_price + stop_atr * a, position.entry_price - take_atr * a
                    if px >= stop or px <= take:
                        position.exit_time, position.exit_price = t, float(px); position.pnl_pct = (position.entry_price / px - 1) * 100
                        equity.append(equity[-1] * (1 + position.pnl_pct / 100)); equity_time.append(t); position = None
        if position is not None:
            last_px = float(close.iloc[-1])
            pnl = (last_px / position.entry_price - 1) * 100 if position.side == "long" else (position.entry_price / last_px - 1) * 100
            position.exit_time, position.exit_price, position.pnl_pct = close.index[-1], last_px, pnl
            equity.append(equity[-1] * (1 + pnl / 100)); equity_time.append(close.index[-1])
        return trades, pd.Series(equity, index=pd.to_datetime(equity_time)).sort_index()

    trades, equity = backtest_sr(prices, sr_levels, buffer_pct, atr_period, stop_atr, take_atr)
    win_rate, total_return, n_trades = summarize_trades(trades)
    mdd = max_drawdown(equity)
    dir_stats = directional_breakdown(trades)

# ----------------------------
# Header signal & metrics
# ----------------------------
signal_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[signal]
st.markdown(f"<h2 style='color:{signal_color};'>📢 Live Signal: {signal}</h2>", unsafe_allow_html=True)
if 'reason' in locals() and reason:
    st.caption(reason)

colA, colB, colC, colD, colE, colF = st.columns([1.4, 1, 1, 1, 1, 1.2])
colA.markdown(f"### 📊 Current Bias: **{bias}**")
colB.metric("Win Rate", f"{win_rate:.2f}%")
colC.metric("Total Return", f"{total_return:+.2f}%")
colD.metric("Max Drawdown", f"{mdd:.2f}%")
colE.metric("Trades", f"{n_trades}")

# Long vs Short breakdown
st.write("**Directional breakdown**")
st.dataframe(pd.DataFrame({
    "side": ["long","short"],
    "n": [dir_stats["long"]["n"], dir_stats["short"]["n"]],
    "win%": [round(dir_stats["long"]["win"],2), round(dir_stats["short"]["win"],2)],
    "net%": [round(dir_stats["long"]["net"],2), round(dir_stats["short"]["net"],2)],
}), use_container_width=True)

# ----------------------------
# Chart
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"], low=prices["Low"], close=prices["Close"], name="Price"))

if strategy == "Donchian Breakout":
    he = plot_series["high_entry"]; le = plot_series["low_entry"]
    hx = plot_series["high_exit"]; lx = plot_series["low_exit"]
    ch_l = plot_series["chand_long"]; ch_s = plot_series["chand_short"]
    sma_long = plot_series["sma_long"]
    fig.add_trace(go.Scatter(x=he.index, y=he, name=f"{n_entry}D High", mode="lines"))
    fig.add_trace(go.Scatter(x=le.index, y=le, name=f"{n_entry}D Low", mode="lines"))
    fig.add_trace(go.Scatter(x=hx.index, y=hx, name=f"{n_exit}D High (exit)", mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=lx.index, y=lx, name=f"{n_exit}D Low (exit)", mode="lines", line=dict(dash="dot")))
    if use_chandelier:
        fig.add_trace(go.Scatter(x=ch_l.index, y=ch_l, name="Chandelier Long", mode="lines"))
        fig.add_trace(go.Scatter(x=ch_s.index, y=ch_s, name="Chandelier Short", mode="lines"))
    if short_gate:
        fig.add_trace(go.Scatter(x=sma_long.index, y=sma_long, name="SMA200", mode="lines"))
else:
    # show SR levels as in your original (optional)
    pass

# Trade markers
long_x, long_y, short_x, short_y, exit_x, exit_y = [], [], [], [], [], []
for t in trades:
    if t.side == "long": long_x.append(t.entry_time); long_y.append(t.entry_price)
    else: short_x.append(t.entry_time); short_y.append(t.entry_price)
    if t.exit_time and t.exit_price: exit_x.append(t.exit_time); exit_y.append(t.exit_price)
if long_x: fig.add_trace(go.Scatter(x=long_x, y=long_y, mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=10, color="green")))
if short_x and not (strategy=="Donchian Breakout" and long_only): fig.add_trace(go.Scatter(x=short_x, y=short_y, mode="markers", name="Sell", marker=dict(symbol="triangle-down", size=10, color="red")))
if exit_x: fig.add_trace(go.Scatter(x=exit_x, y=exit_y, mode="markers", name="Exit", marker=dict(symbol="x", size=9, color="gray")))

bg_color = "rgba(0,128,0,0.07)" if bias == "LONG" else "rgba(220,20,60,0.07)"
fig.add_vrect(x0=prices.index[-min(len(prices), 100)], x1=prices.index.max(), fillcolor=bg_color, line_width=0, layer="below")
fig.update_layout(height=720, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

