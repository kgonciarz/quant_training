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
TRADING_DAYS = 252  # for Sharpe

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
    debounce_bars: int = 1,             # NEW
    confirm_mode: str = "none",         # NEW: "none" or "close-confirm"
) -> Tuple[List[Trade], pd.Series]:

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    atr   = compute_atr(df, atr_period)
    trend = slope(close, 50).reindex(df.index)  # align

    trades: List[Trade] = []
    equity, equity_time = [1.0], [df.index[0]]
    position: Optional[Trade] = None

    # Debounce state
    long_touch_count  = 0
    short_touch_count = 0
    long_armed  = False   # set true after N consecutive touches near support
    short_armed = False   # set true after N consecutive touches near resistance
    last_ns = None
    last_nr = None

    for i, (t, px) in enumerate(close.items()):
        tr_val = float(trend.iloc[i]) if pd.notna(trend.iloc[i]) else 0.0
        tr_up, tr_dn = tr_val > 0, tr_val < 0
        ns = nearest_level(px, sr.support)
        nr = nearest_level(px, sr.resistance)
        last_ns, last_nr = ns, nr  # keep for confirm step when price moves away

        # Touch conditions (near a level within buffer %)
        long_touch  = (tr_up and ns is not None and px <= ns * (1 + buffer_pct / 100.0))
        short_touch = (tr_dn and nr is not None and px >= nr * (1 - buffer_pct / 100.0))

        # Debounce counters (only when flat)
        if position is None:
            if long_touch:
                long_touch_count += 1
            else:
                long_touch_count = 0
                long_armed = False

            if short_touch:
                short_touch_count += 1
            else:
                short_touch_count = 0
                short_armed = False

            # Arm after N consecutive touches
            if long_touch_count >= debounce_bars:
                long_armed = True
            if short_touch_count >= debounce_bars:
                short_armed = True

            enter_long = False
            enter_short = False

            if confirm_mode == "none":
                # Enter immediately once armed on a touching bar
                enter_long = long_armed and long_touch
                enter_short = short_armed and short_touch

            elif confirm_mode == "close-confirm":
                # Require *subsequent* close moving away from level:
                # - long: after touches near support, wait for close > support
                # - short: after touches near resistance, wait for close < resistance
                if long_armed and ns is not None and px > ns:
                    enter_long = True
                if short_armed and nr is not None and px < nr:
                    enter_short = True

            # Execute entry
            if position is None and enter_long:
                position = Trade(t, float(px), "long")
                trades.append(position)
                # reset arms/counters
                long_armed = False
                long_touch_count = 0
                short_touch_count = 0

            elif position is None and enter_short:
                position = Trade(t, float(px), "short")
                trades.append(position)
                short_armed = False
                short_touch_count = 0
                long_touch_count = 0

        else:
            # Manage open position using bar extremes (more realistic)
            a = float(atr.iloc[i]) if np.isfinite(atr.iloc[i]) else 1e-6
            if position.side == "long":
                stop = position.entry_price - stop_atr * a
                take = position.entry_price + take_atr * a

                # Assume stop hits before take when both touched (configurable rule)
                hit_stop = low.iloc[i]  <= stop
                hit_take = high.iloc[i] >= take

                if hit_stop or hit_take:
                    exit_px = stop if hit_stop else take
                    position.exit_time  = t
                    position.exit_price = float(exit_px)
                    position.pnl_pct    = (position.exit_price / position.entry_price - 1) * 100.0
                    equity.append(equity[-1] * (1 + position.pnl_pct / 100.0))
                    equity_time.append(t)
                    position = None
                    # reset debounce state after exit
                    long_touch_count = short_touch_count = 0
                    long_armed = short_armed = False

            else:  # short
                stop = position.entry_price + stop_atr * a
                take = position.entry_price - take_atr * a

                hit_stop = high.iloc[i] >= stop
                hit_take = low.iloc[i]  <= take

                if hit_stop or hit_take:
                    exit_px = stop if hit_stop else take
                    position.exit_time  = t
                    position.exit_price = float(exit_px)
                    position.pnl_pct    = (position.entry_price / position.exit_price - 1) * 100.0
                    equity.append(equity[-1] * (1 + position.pnl_pct / 100.0))
                    equity_time.append(t)
                    position = None
                    short_touch_count = long_touch_count = 0
                    long_armed = short_armed = False

    # Force-close last trade at final close
    if position is not None:
        last_t = close.index[-1]
        last_px = float(close.iloc[-1])
        if position.side == "long":
            pnl = (last_px / position.entry_price - 1) * 100.0
        else:
            pnl = (position.entry_price / last_px - 1) * 100.0
        position.exit_time  = last_t
        position.exit_price = last_px
        position.pnl_pct    = pnl
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

# ---------- Overfitting guardrails: helpers ----------

def equity_daily_from_trades(trades: List[Trade], prices: pd.DataFrame) -> pd.Series:
    """Build a daily equity line (×) forward-filled between exits."""
    if not trades:
        return pd.Series(dtype=float)
    # start at 1.0 and step on exits
    steps = [(prices.index[0], 1.0)]
    eq = 1.0
    for t in trades:
        if t.exit_time is None or t.pnl_pct is None:
            continue
        eq *= (1 + t.pnl_pct / 100.0)
        steps.append((pd.to_datetime(t.exit_time), eq))
    s = pd.Series(dict(steps)).sort_index()
    # resample to daily and ffill
    daily = s.resample("1D").ffill()
    # ensure daily index covers the selected price range
    idx = pd.date_range(prices.index[0].normalize(),
                        prices.index[-1].normalize(), freq="D")
    daily = daily.reindex(idx).ffill()
    return daily

def sharpe_from_equity_daily(eq_daily: pd.Series, risk_free=0.0) -> float:
    """Daily Sharpe (simple): mean(daily_ret - rf) / std(daily_ret) * sqrt(252)."""
    if eq_daily is None or eq_daily.empty or eq_daily.size < 5:
        return 0.0
    ret = eq_daily.pct_change().dropna()
    if ret.std() == 0 or np.isnan(ret.std()):
        return 0.0
    return float(((ret - risk_free/252).mean() / ret.std()) * np.sqrt(252))

def perf_summary(prices: pd.DataFrame,
                 trades: List[Trade],
                 equity: pd.Series,
                 commission_pct_per_side: float = 0.0,
                 slippage_bps_per_side: int = 0) -> dict:
    """Pack IS metrics + daily Sharpe into one dict (uses your summarize_trades)."""
    win, net = summarize_trades(
        trades,
        commission_pct_per_side=commission_pct_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    mdd = max_drawdown(equity)
    eq_daily = equity_daily_from_trades(trades, prices)
    sharpe = sharpe_from_equity_daily(eq_daily)
    ntr = len([t for t in trades if t.pnl_pct is not None])
    return {
        "win_rate": win,
        "net": net,
        "mdd": mdd,
        "sharpe": sharpe,
        "trades": ntr,
        "eq_daily": eq_daily,
    }

def split_prices(prices: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split."""
    cut = max(5, int(len(prices) * train_ratio))
    return prices.iloc[:cut].copy(), prices.iloc[cut:].copy()

def timeseries_folds(prices: pd.DataFrame, n_folds: int = 3):
    """Simple walk-forward generator of (train_df, test_df) tuples."""
    n = len(prices)
    if n_folds < 2:
        yield prices.iloc[: -1], prices.iloc[-1:]
        return
    fold_size = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        split = fold_size * k
        train = prices.iloc[:split].copy()
        test  = prices.iloc[split: split + fold_size].copy()
        if len(test) < 5:  # skip tiny tails
            continue
        yield train, test

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

def _grid_by_breadth(breadth: str):
    if breadth == "Fast":
        sw = [19, 25]
        tol = [0.4, 0.8]
        buf = [0.3, 0.6]
        atrp = [10, 14]
        sl = [1.5, 2.0]
        tp = [2.5, 3.0]
    elif breadth == "Wide":
        sw = [15, 19, 25, 31, 37]
        tol = [0.3, 0.6, 1.0, 1.2]
        buf = [0.2, 0.5, 0.8, 1.0]
        atrp = [10, 14, 18, 22]
        sl = [1.2, 1.5, 2.0, 2.5]
        tp = [2.0, 2.5, 3.0, 4.0]
    else:  # Balanced
        sw = [19, 25, 31]
        tol = [0.3, 0.6, 1.0]
        buf = [0.3, 0.6, 0.9]
        atrp = [10, 14, 20]
        sl = [1.5, 2.0, 2.5]
        tp = [2.5, 3.0, 4.0]
    return sw, tol, buf, atrp, sl, tp

def optimize_oos(prices: pd.DataFrame,
                 breadth: str,
                 dd_penalty: float,
                 use_holdout: bool,
                 train_ratio: float,
                 use_walkforward: bool,
                 n_folds: int,
                 min_trades_guard: int,
                 max_mdd_guard: float,
                 min_sharpe_guard: float,
                 commission_pct_per_side: float = 0.0,
                 slippage_bps_per_side: int = 0):
    """Optimize parameters on OOS performance with guardrails."""
    sw_opts, tol_opts, buf_opts, atr_opts, stop_opts, take_opts = _grid_by_breadth(breadth)

    rows = []
    best = None

    def score_row(oos_net, oos_mdd, oos_sharpe, ntr):
        # base score — profit minus DD penalty; reward Sharpe
        s = oos_net - max(0, oos_mdd) * dd_penalty + 25.0 * oos_sharpe
        # guards
        if ntr < min_trades_guard:
            s -= 100.0
        if oos_mdd > max_mdd_guard:
            s -= 100.0
        if oos_sharpe < min_sharpe_guard:
            s -= 50.0
        return s

    # Build folds
    folds = []
    if use_walkforward:
        folds = list(timeseries_folds(prices, n_folds))
    elif use_holdout:
        folds = [split_prices(prices, train_ratio)]
    else:
        # No OOS; treat entire range as both (not recommended, but keeps code paths simple)
        folds = [(prices, prices)]

    for sw in sw_opts:
        for tol in tol_opts:
            # SR for each segment (we’ll recompute per segment anyway)
            for buf in buf_opts:
                for atrp in atr_opts:
                    for sl in stop_opts:
                        for tp in take_opts:
                            # accumulate OOS metrics over folds
                            oos_net_all, oos_mdd_all, oos_sharpe_all, oos_trades_all = [], [], [], []
                            is_net_all, is_mdd_all, is_sharpe_all = [], [], []
                            for train_df, test_df in folds:
                                # In-sample (train) perf
                                sr_train = detect_sr(train_df, sw, tol)
                                tr_train, eq_train = backtest(
                                    train_df, sr_train,
                                    buffer_pct=buf, atr_period=atrp,
                                    stop_atr=sl, take_atr=tp
                                )
                                is_perf = perf_summary(
                                    train_df, tr_train, eq_train,
                                    commission_pct_per_side=commission_pct_per_side,
                                    slippage_bps_per_side=slippage_bps_per_side,
                                )
                                is_net_all.append(is_perf["net"])
                                is_mdd_all.append(is_perf["mdd"])
                                is_sharpe_all.append(is_perf["sharpe"])

                                # Out-of-sample (test) perf
                                sr_test = detect_sr(test_df, sw, tol)
                                tr_test, eq_test = backtest(
                                    test_df, sr_test,
                                    buffer_pct=buf, atr_period=atrp,
                                    stop_atr=sl, take_atr=tp
                                )
                                oos_perf = perf_summary(
                                    test_df, tr_test, eq_test,
                                    commission_pct_per_side=commission_pct_per_side,
                                    slippage_bps_per_side=slippage_bps_per_side,
                                )
                                oos_net_all.append(oos_perf["net"])
                                oos_mdd_all.append(oos_perf["mdd"])
                                oos_sharpe_all.append(oos_perf["sharpe"])
                                oos_trades_all.append(oos_perf["trades"])

                            # average across folds
                            oos_net = float(np.mean(oos_net_all)) if oos_net_all else 0.0
                            oos_mdd = float(np.mean(oos_mdd_all)) if oos_mdd_all else 0.0
                            oos_sharpe = float(np.mean(oos_sharpe_all)) if oos_sharpe_all else 0.0
                            ntr = int(np.sum(oos_trades_all)) if oos_trades_all else 0
                            is_net = float(np.mean(is_net_all)) if is_net_all else 0.0
                            is_mdd = float(np.mean(is_mdd_all)) if is_mdd_all else 0.0
                            is_sharpe = float(np.mean(is_sharpe_all)) if is_sharpe_all else 0.0

                            s = score_row(oos_net, oos_mdd, oos_sharpe, ntr)
                            row = {
                                "sr_window": sw, "cluster_tol": tol, "buffer_pct": buf,
                                "atr_period": atrp, "stop_atr": sl, "take_atr": tp,
                                "IS_net": is_net, "IS_mdd": is_mdd, "IS_sharpe": is_sharpe,
                                "OOS_net": oos_net, "OOS_mdd": oos_mdd, "OOS_sharpe": oos_sharpe,
                                "OOS_trades": ntr, "score": s
                            }
                            rows.append(row)
                            if best is None or s > best["score"]:
                                best = row.copy()

    grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    return best, grid



def daily_equity(equity_step: pd.Series, price_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill the stepwise equity onto the price index (trading days).
    Ensures the first day starts at 1.0.
    """
    if equity_step.empty:
        return pd.Series(dtype=float)
    # Align to trading days and forward-fill between exits
    eq = equity_step.reindex(price_index, method="ffill")
    # If equity started after the first price date, fill the very first with 1.0
    if not eq.empty and pd.isna(eq.iloc[0]):
        eq.iloc[0] = 1.0
        eq = eq.ffill()
    return eq.astype(float)

def max_drawdown_from_equity(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    return float(dd.min() * 100.0)

def sharpe_from_equity(eq: pd.Series) -> float:
    # Daily log or simple returns—use simple here
    rets = eq.pct_change().dropna()
    if rets.empty or rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * np.sqrt(TRADING_DAYS))

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
    commission_ps = st.slider("Commission per side (%)", 0.0, 0.1, 0.02, 0.01, key="commission_ps")
    slippage_bps  = st.slider("Slippage per side (bps)", 0, 50, 5, 1, key="slippage_bps")
    refresh = st.button("🔄 Refresh data")
    optimize_click = st.button("🧪 Optimize (profit ↑ / drawdown ↓)")
    debounce_bars = st.slider("Debounce bars (hold condition)", 1, 5, 2, 1)
    confirm_mode  = st.selectbox("Signal confirmation", ["none", "close-confirm"], index=1)
    
with st.sidebar.expander("🛡️ Overfitting guardrails", expanded=False):
    use_holdout = st.checkbox("Use train/test split (hold-out)", value=True)
    train_ratio = st.slider("Train ratio", 0.5, 0.9, 0.7, 0.05)
    use_walkforward = st.checkbox("Use walk-forward CV", value=False)
    n_folds = st.slider("Walk-forward folds", 2, 6, 3, 1, disabled=not use_walkforward)

    # Search breadth & quality caps
    min_trades_guard = st.number_input("Min closed trades (OOS)", 0, 100, 6, 1)
    max_mdd_guard = st.slider("Max drawdown cap (OOS, %)", 0.0, 80.0, 40.0, 1.0)
    min_sharpe_guard = st.slider("Min daily Sharpe (OOS)", 0.0, 3.0, 0.2, 0.05)

    breadth = st.selectbox("Grid breadth", ["Fast", "Balanced", "Wide"], index=1)
    dd_penalty = st.slider("Drawdown penalty in score", 0.0, 2.0, 0.7, 0.1)

if start_date >= end_date:
    st.error("Start date must be before end date."); st.stop()

with st.spinner("Fetching cocoa prices…"):
    prices = get_prices(TICKER, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if optimize_click:
    with st.spinner("Searching best OOS parameters…"):
        best, grid = optimize_oos(
            prices=prices,
            breadth=breadth,
            dd_penalty=dd_penalty,
            use_holdout=use_holdout,
            train_ratio=train_ratio,
            use_walkforward=use_walkforward,
            n_folds=n_folds,
            min_trades_guard=min_trades_guard,
            max_mdd_guard=max_mdd_guard,
            min_sharpe_guard=min_sharpe_guard,
            commission_pct_per_side=commission_ps,     # ← from your sidebar
            slippage_bps_per_side=slippage_bps,        # ← from your sidebar
        )
    st.session_state["opt_best"] = best
    st.session_state["opt_grid"] = grid
    st.success("Optimization done! Showing OOS-scored leaderboard…")

# Show leaderboard + apply button (unique keys to avoid duplicate widget IDs)
if "opt_best" in st.session_state and st.session_state["opt_best"]:
    with st.expander("Best parameters (OOS-scored) — click 'Apply' to use them"):
        b = st.session_state["opt_best"]
        st.write(pd.DataFrame([b]))
        if st.button("✅ Apply best params to chart", key="apply_oos"):
            st.session_state["override_params"] = {
                "sr_window": int(b["sr_window"]),
                "cluster_tol": float(b["cluster_tol"]),
                "buffer_pct": float(b["buffer_pct"]),
                "atr_period": int(b["atr_period"]),
                "stop_atr": float(b["stop_atr"]),
                "take_atr": float(b["take_atr"]),
            }
            st.rerun()
    if "opt_grid" in st.session_state and not st.session_state["opt_grid"].empty:
        st.dataframe(
            st.session_state["opt_grid"].head(30)[
                ["sr_window","cluster_tol","buffer_pct","atr_period","stop_atr","take_atr",
                 "IS_net","IS_mdd","IS_sharpe","OOS_net","OOS_mdd","OOS_sharpe","OOS_trades","score"]
            ],
            use_container_width=True
        )


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
    "debounce_bars": debounce_bars,
    "confirm_mode": confirm_mode,
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
# Distance to nearest levels (in % of last price)
dist_support = None if ns_price is None else (last_close - ns_price) / last_close * 100.0
dist_resist  = None if nr_price is None else (nr_price - last_close) / last_close * 100.0


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


trades, equity_step = backtest(
    prices, sr_levels,
    buffer_pct=params["buffer_pct"],
    atr_period=params["atr_period"],
    stop_atr=params["stop_atr"],
    take_atr=params["take_atr"],
    debounce_bars=params["debounce_bars"],       
    confirm_mode=params["confirm_mode"],
)

# Costs-aware summary (you already wired commission_ps & slippage_bps in the sidebar)
win_rate, total_return = summarize_trades(
    trades,
    commission_pct_per_side=commission_ps,
    slippage_bps_per_side=slippage_bps,
)

# --- NEW: daily equity for MDD/Sharpe ---
equity_daily = daily_equity(equity_step, prices.index)
mdd = max_drawdown_from_equity(equity_daily)
sharpe = sharpe_from_equity(equity_daily)




colA, colB, colC, colD, colE, colF = st.columns([1.2, 1, 1, 1, 1, 1.2])
colA.markdown(f"### 📊 Current Bias: **{bias}**")
colB.metric("📉 Nearest Support", f"{ns_price:.2f}" if ns_price else "—")
colC.metric("📈 Nearest Resistance", f"{nr_price:.2f}" if nr_price else "—")
colD.metric("Win Rate", f"{win_rate:.2f}%")
colE.metric("Total Return (net)", f"{total_return:+.2f}%")
colF.metric("Max Drawdown", f"{mdd:.2f}%")
st.caption(f"Sharpe (daily): **{sharpe:.2f}**")
# Tiny captions showing distance-to-levels
st.caption(
    f"Distance: {('—' if dist_support is None else f'{dist_support:+.2f}%')} from support · "
    f"{('—' if dist_resist  is None else f'{dist_resist:+.2f}%')} from resistance"
)

st.subheader("Equity Curve (daily)")
if not equity_daily.empty:
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=equity_daily.index,
        y=equity_daily.values,
        mode="lines",
        name="Equity (×)"
    ))
    eq_fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(eq_fig, use_container_width=True)
else:
    st.write("No closed trades yet for this period/parameters.")


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
