"""
Tech-Rising Signal Options Backtest - Streamlit App
--------------------------------------------------
Interactive web app for backtesting options strategies on the Tech-Rising signal.

Run with: streamlit run streamlit_options_backtest.py
"""

import math
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Tech-Rising Options Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Helper Functions (same as original)
# ----------------------------

def wilder_rsi(close: pd.Series, length: int) -> pd.Series:
    """Wilder's RSI implementation on daily closes."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()

    rs = roll_up / roll_down.replace({0: np.nan})
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

@st.cache_data
def fetch_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    """Cached function to fetch price data."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.index = pd.to_datetime(df.index)
    return df

def run_stock_strategy(trips, S, initial_capital=10000):
    """Run buy-and-hold stock strategy during signal periods"""
    trades = []
    for (d_in, d_out) in trips:
        if d_in not in S.index or d_out not in S.index:
            continue
        
        entry_price = float(S.loc[d_in])
        exit_price = float(S.loc[d_out])
        
        # Calculate shares we can buy with initial capital
        shares = initial_capital / entry_price
        
        # P&L calculation
        pnl = (exit_price - entry_price) * shares
        pnl_pct = (exit_price - entry_price) / entry_price
        
        trades.append(Trade(
            d_in, d_out, entry_price, exit_price, pnl, pnl_pct, 
            f"Shares={shares:.1f}, Capital=${initial_capital}"
        ))
    return trades

def run_0dte_call(trips, S, iv, rf, max_days, otm_pct, slippage, commission, contract_mult):
    """Run 0DTE call strategy - very short-term, high gamma plays"""
    trades = []
    for (d_in, d_out) in trips:
        if d_in not in S.index or d_out not in S.index:
            continue
        
        # For 0DTE, we limit the holding period regardless of signal
        actual_exit = min(d_out, d_in + pd.Timedelta(days=max_days))
        if actual_exit not in S.index:
            # Find next available trading day
            available_dates = S.index[S.index > actual_exit]
            if len(available_dates) == 0:
                continue
            actual_exit = available_dates[0]
        
        S_in, S_out = float(S.loc[d_in]), float(S.loc[actual_exit])
        
        # Very short DTE
        T_in = annualize_days(max_days) if max_days > 0 else 1/365  # Minimum time value
        r_in = float(rf.loc[d_in])
        sig_in = float(iv.loc[d_in])
        
        # Slightly OTM strike for 0DTE (smaller moves expected)
        K = nearest_strike(S_in * (1 + otm_pct))
        entry_price = black_scholes_call_price(S_in, K, T_in, r_in, sig_in)
        
        # Exit pricing - either intrinsic value or small time value
        days_elapsed = max((actual_exit - d_in).days, 0)
        T_out = annualize_days(max(max_days - days_elapsed, 0))
        r_out = float(rf.loc[actual_exit])
        sig_out = float(iv.loc[actual_exit])
        
        if T_out <= 0:
            # At expiration, only intrinsic value
            exit_price = max(S_out - K, 0.0)
        else:
            exit_price = black_scholes_call_price(S_out, K, T_out, r_out, sig_out)

        pnl = (exit_price - entry_price) * contract_mult - 2*(slippage + commission)
        pnl_pct = (exit_price - entry_price) / max(entry_price, 1e-9)

        hold_days = (actual_exit - d_in).days
        trades.append(Trade(d_in, actual_exit, entry_price, exit_price, pnl, pnl_pct, 
                          f"K={K}, OTM={otm_pct:.1%}, Hold={hold_days}d"))
    return trades
    """Return explanation text for each strategy"""
    explanations = {
        "ATM Call": """
**ATM Call Strategy Rules:**
• **Entry:** When Tech-Rising signal activates, buy ATM call options
• **Strike:** Nearest strike to current price
• **Expiration:** Fixed DTE (Days To Expiration)
• **Exit:** Sell calls when signal deactivates
• **Risk:** Limited to premium paid, benefits from upward moves and volatility
• **Leverage:** ~50 delta provides moderate leverage to underlying moves
        """,
        
        "OTM Call": """
**OTM Call Strategy Rules:**
• **Entry:** When Tech-Rising signal activates, buy OTM call options
• **Strike:** Set percentage above current price
• **Expiration:** Fixed DTE (Days To Expiration)  
• **Exit:** Sell calls when signal deactivates
• **Risk:** Higher chance of expiring worthless, but lower cost
• **Leverage:** Higher gamma provides more explosive gains if underlying moves favorably
        """,
        
        "Stock": """
**Stock Strategy Rules:**
• **Entry:** When Tech-Rising signal activates, buy shares
• **Position Size:** Fixed dollar amount invested each trade
• **Hold:** Maintain position while signal remains active
• **Exit:** Sell shares when signal deactivates
• **Risk:** Full exposure to stock price movements, no time decay
• **Leverage:** 1:1 exposure (except TQQQ which is 3x leveraged ETF)
        """
    }
    return explanations.get(strategy_name, "No explanation available.")

def black_scholes_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from math import erf, sqrt, exp
    N = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    call = S * N(d1) - K * math.exp(-r * T) * N(d2)
    return call

def black_scholes_put_price(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from math import erf, sqrt, exp
    N = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    put = K * math.exp(-r * T) * N(-d2) - S * N(-d1)
    return put

def bs_call_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    from math import erf, sqrt
    N = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return N(d1)

def bs_put_delta(S, K, T, r, sigma):
    return bs_call_delta(S, K, T, r, sigma) - 1.0

def strike_for_target_call_delta(S, T, r, sigma, target_delta, tol=1e-4, max_iter=60):
    K_low = S * 0.3
    K_high = S * 1.7
    for _ in range(max_iter):
        K_mid = 0.5 * (K_low + K_high)
        delta = bs_call_delta(S, K_mid, T, r, sigma)
        if delta > target_delta:
            K_low, K_high = K_low, K_mid
        else:
            K_low, K_high = K_mid, K_high
        if abs(delta - target_delta) < tol:
            return K_mid
    return K_mid

def strike_for_target_put_delta(S, T, r, sigma, target_put_delta, tol=1e-4, max_iter=60):
    K_low = S * 0.3
    K_high = S * 1.7
    for _ in range(max_iter):
        K_mid = 0.5 * (K_low + K_high)
        delta = bs_put_delta(S, K_mid, T, r, sigma)
        if delta < target_put_delta:
            K_low, K_high = K_mid, K_high
        else:
            K_low, K_high = K_low, K_mid
        if abs(delta - target_put_delta) < tol:
            return K_mid
    return K_mid

def nearest_strike(x, step=1.0):
    return round(x / step) * step

def annualize_days(days: int) -> float:
    return max(days, 0) / 365.0

@st.cache_data
def load_iv_proxy(target: str, start: str, end: Optional[str]) -> pd.Series:
    idx = "^VIX" if target.upper() == "SPY" else "^VXN"
    iv_df = yf.download(idx, start=start, end=end, auto_adjust=False, progress=False)["Close"]
    iv = (iv_df / 100.0).reindex(iv_df.index)
    iv.name = "iv_proxy"
    return iv

@st.cache_data
def load_rf_proxy(start: str, end: Optional[str]) -> pd.Series:
    rf_df = yf.download("^IRX", start=start, end=end, auto_adjust=False, progress=False)["Close"]
    rf = (rf_df / 100.0).reindex(rf_df.index)
    rf.name = "rf_proxy"
    return rf

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    notes: str

def build_signal(start, end, rsi_len) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tickers = ["XLK", "KMLM", "IGIB"]
    px = fetch_prices(tickers, start, end)
    
    rsi = {}
    for t in tickers:
        rsi[t] = wilder_rsi(px[t], rsi_len)
    rsi = pd.DataFrame(rsi)
    cond = (rsi["XLK"] > rsi["KMLM"]) & (rsi["IGIB"] > rsi["XLK"])
    cond = cond.astype(bool)

    entries = (cond & (~cond.shift(1).fillna(False)))
    exits = (~cond & (cond.shift(1).fillna(False)))

    df = pd.DataFrame({
        "cond": cond,
        "entry": entries,
        "exit": exits
    }, index=px.index)

    return df, px

def build_trips(signal_df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    entries = list(signal_df.index[signal_df["entry"]])
    exits   = list(signal_df.index[signal_df["exit"]])

    trips = []
    ei = xi = 0
    active_entry = None
    for d in signal_df.index:
        if ei < len(entries) and d == entries[ei]:
            active_entry = d
            ei += 1
        if xi < len(exits) and d == exits[xi] and active_entry is not None:
            trips.append((active_entry, d))
            active_entry = None
            xi += 1
    if active_entry is not None:
        trips.append((active_entry, signal_df.index[-1]))
    return trips

def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = -np.inf
    dd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = min(dd, x - peak)
    return dd

def summarize_trades(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades])
    out = {
        "n_trades": [len(df)],
        "hit_rate": [(df["pnl"] > 0).mean()],
        "avg_pnl_%": [df["pnl_pct"].mean()],
        "avg_win_%": [df.loc[df["pnl"]>0, "pnl_pct"].mean() if (df["pnl"]>0).any() else np.nan],
        "avg_loss_%": [df.loc[df["pnl"]<=0, "pnl_pct"].mean() if (df["pnl"]<=0).any() else np.nan],
        "profit_factor": [df.loc[df["pnl"]>0, "pnl"].sum() / abs(df.loc[df["pnl"]<=0, "pnl"].sum()) if (df["pnl"]<=0).any() else np.inf],
        "total_pnl_$": [df["pnl"].sum()],
        "max_drawdown_$": [max_drawdown(np.cumsum(df["pnl"]).values)],
    }
    return pd.DataFrame(out)

def equity_curve(trades: List[Trade]) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    df = pd.DataFrame([t.__dict__ for t in trades])
    ec = df.groupby("exit_date")["pnl"].sum().cumsum()
    ec.name = "equity"
    return ec

# Strategy implementations (simplified versions of original functions)
def run_atm_call(trips, S, iv, rf, dte, slippage, commission, contract_mult):
    trades = []
    for (d_in, d_out) in trips:
        if d_in not in S.index or d_out not in S.index:
            continue
        S_in, S_out = float(S.loc[d_in]), float(S.loc[d_out])
        T_in = annualize_days(dte)
        r_in = float(rf.loc[d_in])
        sig_in = float(iv.loc[d_in])
        
        K = nearest_strike(S_in)
        entry_price = black_scholes_call_price(S_in, K, T_in, r_in, sig_in)

        days_elapsed = max((d_out - d_in).days, 0)
        T_out = annualize_days(max(dte - days_elapsed, 0))
        r_out = float(rf.loc[d_out])
        sig_out = float(iv.loc[d_out])
        exit_price = black_scholes_call_price(S_out, K, T_out, r_out, sig_out)

        pnl = (exit_price - entry_price) * contract_mult - 2*(slippage + commission)
        pnl_pct = (exit_price - entry_price) / max(entry_price, 1e-9)

        trades.append(Trade(d_in, d_out, entry_price, exit_price, pnl, pnl_pct, f"K={K}, DTE={dte}"))
    return trades

def run_otm_call(trips, S, iv, rf, dte, otm_pct, slippage, commission, contract_mult):
    """Run slightly OTM call strategy - lower cost, higher leverage"""
    trades = []
    for (d_in, d_out) in trips:
        if d_in not in S.index or d_out not in S.index:
            continue
        S_in, S_out = float(S.loc[d_in]), float(S.loc[d_out])
        T_in = annualize_days(dte)
        r_in = float(rf.loc[d_in])
        sig_in = float(iv.loc[d_in])
        
        # OTM strike (e.g., 2-5% above current price)
        K = nearest_strike(S_in * (1 + otm_pct))
        entry_price = black_scholes_call_price(S_in, K, T_in, r_in, sig_in)

        days_elapsed = max((d_out - d_in).days, 0)
        T_out = annualize_days(max(dte - days_elapsed, 0))
        r_out = float(rf.loc[d_out])
        sig_out = float(iv.loc[d_out])
        exit_price = black_scholes_call_price(S_out, K, T_out, r_out, sig_out)

        pnl = (exit_price - entry_price) * contract_mult - 2*(slippage + commission)
        pnl_pct = (exit_price - entry_price) / max(entry_price, 1e-9)

        trades.append(Trade(d_in, d_out, entry_price, exit_price, pnl, pnl_pct, f"K={K}, OTM={otm_pct:.1%}, DTE={dte}"))
    return trades

# ----------------------------
# Streamlit App
# ----------------------------

def main():
    st.title("📈 Tech-Rising Options Backtest")
    st.markdown("Interactive backtesting of options strategies on the Tech-Rising signal")
    
    # Sidebar for parameters
    st.sidebar.header("Configuration")
    
    # Basic parameters
    target = st.sidebar.selectbox("Target Symbol", ["QQQ", "SPY", "TQQQ"], index=0)
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    rsi_length = st.sidebar.slider("RSI Length", 5, 20, 10)
    
    # Options parameters
    st.sidebar.subheader("Options Parameters")
    dte_atm = st.sidebar.slider("ATM Call DTE", 7, 30, 14)
    dte_otm = st.sidebar.slider("OTM Call DTE", 7, 30, 14)
    otm_percentage = st.sidebar.slider("OTM Call % Above Spot", 0.01, 0.10, 0.05, format="%.2f")
    
    # 0DTE parameters
    st.sidebar.subheader("0DTE Parameters")
    dte_0dte = st.sidebar.slider("0DTE Call Days", 0, 2, 0, help="0=Same day exit, 1=Next day exit, 2=Two day max hold")
    otm_0dte_pct = st.sidebar.slider("0DTE OTM % Above Spot", 0.005, 0.03, 0.01, format="%.3f", help="Typically smaller moves for 0DTE")
    
    # Cost parameters
    st.sidebar.subheader("Cost Parameters")
    initial_capital = st.sidebar.number_input("Starting Capital ($)", 1000, 100000, 10000, step=1000)
    slippage_per_leg = st.sidebar.number_input("Slippage per Leg ($)", 0.0, 0.10, 0.02)
    commission_per_leg = st.sidebar.number_input("Commission per Leg ($)", 0.0, 1.0, 0.0)
    contract_mult = st.sidebar.number_input("Contract Multiplier", 50, 200, 100)
    
    # Strategy selection
    st.sidebar.subheader("Strategies to Run")
    run_atm = st.sidebar.checkbox("ATM Call", True)
    run_otm = st.sidebar.checkbox("OTM Call", True)
    run_0dte = st.sidebar.checkbox("0DTE Call", True)
    
    st.sidebar.subheader("Stock Comparison Backtest")
    compare_ticker = st.sidebar.selectbox("Compare with Stock/ETF", ["QQQ", "QLD", "TQQQ"], index=0)
    run_stock = st.sidebar.checkbox("Include Stock Comparison", True)
    
    st.sidebar.info("💡 **Strategy Comparison:**\n\n**ATM Calls:** Higher cost, lower risk, steady gains\n\n**OTM Calls:** Lower cost, higher leverage, more volatile\n\n**0DTE Calls:** Extreme leverage, very short-term, high gamma\n\n**Stock/ETF:** Direct exposure, no time decay, unlimited holding period")
    
    if st.sidebar.button("Run Backtest", type="primary"):
        with st.spinner("Building signal and fetching data..."):
            # Build signal
            sig_df, signal_px = build_signal(start_date.strftime("%Y-%m-%d"), 
                                           end_date.strftime("%Y-%m-%d"), 
                                           rsi_length)
            trips = build_trips(sig_df)
            
            # Load target data
            S = fetch_prices([target], start_date.strftime("%Y-%m-%d"), 
                           end_date.strftime("%Y-%m-%d"))[target]
            iv = load_iv_proxy(target, start_date.strftime("%Y-%m-%d"), 
                             end_date.strftime("%Y-%m-%d"))
            rf = load_rf_proxy(start_date.strftime("%Y-%m-%d"), 
                             end_date.strftime("%Y-%m-%d"))
            
            # Align data
            idx = sig_df.index.intersection(S.index).intersection(iv.index).intersection(rf.index)
            S = S.reindex(idx)
            iv = iv.reindex(idx)
            rf = rf.reindex(idx)
            sig_df = sig_df.reindex(idx)
            
        st.success(f"Found {len(trips)} risk-on episodes")
        
        # Display signal chart
        st.header("Signal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI chart
            tickers = ["XLK", "KMLM", "IGIB"]
            rsi_data = {}
            for t in tickers:
                if t in signal_px.columns:
                    rsi_data[t] = wilder_rsi(signal_px[t], rsi_length)
            rsi_df = pd.DataFrame(rsi_data)
            
            fig_rsi = go.Figure()
            for col in rsi_df.columns:
                fig_rsi.add_trace(go.Scatter(x=rsi_df.index, y=rsi_df[col], 
                                           name=f"RSI({col})", mode='lines'))
            fig_rsi.update_layout(title="RSI Components", xaxis_title="Date", yaxis_title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            # Signal periods
            fig_signal = go.Figure()
            fig_signal.add_trace(go.Scatter(x=S.index, y=S, name=target, mode='lines'))
            
            # Highlight signal periods
            for i, (start_trip, end_trip) in enumerate(trips):
                fig_signal.add_vrect(x0=start_trip, x1=end_trip, 
                                   fillcolor="green", opacity=0.2, 
                                   annotation_text=f"Signal {i+1}" if i < 5 else "",
                                   annotation_position="top left")
            
            fig_signal.update_layout(title=f"{target} Price with Signal Periods", 
                                   xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_signal, use_container_width=True)
        
        # Run strategies
        strat_trades = {}
        summaries = []
        
        with st.spinner("Running strategies..."):
            if run_atm:
                atm_trades = run_atm_call(trips, S, iv, rf, dte_atm, 
                                        slippage_per_leg, commission_per_leg, contract_mult)
                strat_trades["ATM Call"] = atm_trades
            
            if run_otm:
                otm_trades = run_otm_call(trips, S, iv, rf, dte_otm, otm_percentage,
                                        slippage_per_leg, commission_per_leg, contract_mult)
                strat_trades["OTM Call"] = otm_trades
            
            if run_0dte:
                dte_0dte_trades = run_0dte_call(trips, S, iv, rf, dte_0dte, otm_0dte_pct,
                                              slippage_per_leg, commission_per_leg, contract_mult)
                strat_trades["0DTE Call"] = dte_0dte_trades
                try:
                    with st.spinner(f"Loading {compare_ticker} data..."):
                        # Get comparison ticker data
                        compare_S = fetch_prices([compare_ticker], start_date.strftime("%Y-%m-%d"), 
                                               end_date.strftime("%Y-%m-%d"))[compare_ticker]
                        compare_S = compare_S.reindex(idx)
                        
                        stock_trades = run_stock_strategy(trips, compare_S)
                        strat_trades[f"{compare_ticker} Stock"] = stock_trades
                except Exception as e:
                    st.error(f"Error running stock strategy: {str(e)}")
                    st.warning("Continuing with options strategies only...")
        
        # Results
        if strat_trades:
            st.header("Backtest Results")
            
            # Summary table
            for name, trades in strat_trades.items():
                if trades:
                    summ = summarize_trades(trades)
                    if not summ.empty:
                        summ.insert(0, "strategy", name)
                        summaries.append(summ)
            
            if summaries:
                all_summ = pd.concat(summaries, ignore_index=True)
                
                # Format summary for display
                display_summ = all_summ.copy()
                for col in ["hit_rate", "avg_pnl_%", "avg_win_%", "avg_loss_%"]:
                    if col in display_summ.columns:
                        display_summ[col] = (display_summ[col] * 100).round(2)
                
                st.subheader("Performance Summary")
                st.dataframe(display_summ, use_container_width=True)
                
                # Equity curves
                st.subheader("Equity Curves")
                fig_equity = go.Figure()
                
                for name, trades in strat_trades.items():
                    if trades:
                        ec = equity_curve(trades)
                        if not ec.empty:
                            fig_equity.add_trace(go.Scatter(x=ec.index, y=ec.values, 
                                                          name=name, mode='lines'))
                
                fig_equity.update_layout(title="Strategy Equity Curves", 
                                       xaxis_title="Date", yaxis_title="Cumulative P&L ($)")
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Trade details
                st.subheader("Strategy Details & Trade Results")
                for name, trades in strat_trades.items():
                    if trades:
                        with st.expander(f"{name} Strategy ({len(trades)} total trades)"):
                            # Show strategy explanation
                            try:
                                if "Call" in name:
                                    if "0DTE" in name:
                                        explanation = """
**0DTE Call Strategy Rules:**
• **Entry:** When Tech-Rising signal activates, buy very short-term OTM calls
• **Strike:** Small percentage above current price (0.5-3%)
• **Expiration:** 0-2 days maximum hold period
• **Exit:** Sell within max hold period OR when signal deactivates (whichever comes first)
• **Risk:** Extremely high chance of total loss, but very low cost
• **Leverage:** Maximum gamma exposure - explosive gains on small moves
                                        """
                                    elif "OTM" in name:
                                        explanation = """
**OTM Call Strategy Rules:**
• **Entry:** When Tech-Rising signal activates, buy OTM call options
• **Strike:** Set percentage above current price
• **Expiration:** Fixed DTE (Days To Expiration)  
• **Exit:** Sell calls when signal deactivates
• **Risk:** Higher chance of expiring worthless, but lower cost
• **Leverage:** Higher gamma provides more explosive gains if underlying moves favorably
                                        """
                                    else:  # ATM Call
                                        explanation = """
**ATM Call Strategy Rules:**
• **Entry:** When Tech-Rising signal activates, buy ATM call options
• **Strike:** Nearest strike to current price
• **Expiration:** Fixed DTE (Days To Expiration)
• **Exit:** Sell calls when signal deactivates
• **Risk:** Limited to premium paid, benefits from upward moves and volatility
• **Leverage:** ~50 delta provides moderate leverage to underlying moves
                                        """
                                else:  # Stock strategy
                                    explanation = """
**Stock Strategy Rules:**
• **Entry:** When Tech-Rising signal activates, buy shares
• **Position Size:** Fixed dollar amount invested each trade
• **Hold:** Maintain position while signal remains active
• **Exit:** Sell shares when signal deactivates
• **Risk:** Full exposure to stock price movements, no time decay
• **Leverage:** 1:1 exposure (except TQQQ which is 3x leveraged ETF)
                                    """
                                st.markdown(explanation)
                            except Exception as e:
                                st.warning(f"Could not load strategy explanation: {e}")
                            
                            st.markdown("---")
                            st.markdown("**Trade History:**")
                            
                            trades_df = pd.DataFrame([t.__dict__ for t in trades])
                            trades_df["pnl_pct"] = (trades_df["pnl_pct"] * 100).round(2)
                            trades_df["pnl"] = trades_df["pnl"].round(2)
                            st.dataframe(trades_df, use_container_width=True)
                            
                            # Download CSV
                            csv = trades_df.to_csv(index=False)
                            st.download_button(
                                label=f"Download {name} trades CSV",
                                data=csv,
                                file_name=f"{name.replace(' ', '_')}_trades_{target}.csv",
                                mime="text/csv"
                            )
            else:
                st.warning("No trades generated. Check your parameters or date range.")
        else:
            st.warning("No strategies selected or no trades generated.")

if __name__ == "__main__":
    main()
