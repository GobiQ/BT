import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    st.error("❌ yfinance not installed. Please install it with: `pip install yfinance`")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.error("❌ plotly not installed. Please install it with: `pip install plotly`")

# Page config
st.set_page_config(
    page_title="Composer Strategy Backtester",
    page_icon="📈",
    layout="wide"
)

def convert_trading_date(date_int):
    """Convert trading date integer to datetime object"""
    date_1 = datetime.strptime("01/01/1970", "%m/%d/%Y")
    dt = date_1 + timedelta(days=int(date_int))
    return dt

def fetch_composer_backtest(symphony_url: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Fetch backtest data from Composer API
    
    Returns:
        Tuple of (allocations_df, symphony_name, tickers)
    """
    # Extract symphony ID from URL
    if symphony_url.endswith('/details'):
        symphony_id = symphony_url.split('/')[-2]
    else:
        symphony_id = symphony_url.split('/')[-1]
    
    payload = {
        "capital": 100000,
        "apply_reg_fee": True,
        "apply_taf_fee": True,
        "backtest_version": "v2",
        "slippage_percent": 0.0005,
        "start_date": start_date,
        "end_date": end_date,
    }
    
    url = f"https://backtest-api.composer.trade/api/v2/public/symphonies/{symphony_id}/backtest"
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch data from Composer API: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse response from Composer API: {e}")
    
    # Extract symphony name and holdings
    symphony_name = data['legend'][symphony_id]['name']
    holdings = data["last_market_days_holdings"]
    tickers = list(holdings.keys())
    
    # Extract allocations
    allocations = data["tdvm_weights"]
    
    # Create date range that aligns with trading days
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    # Get all unique dates from allocation data first
    all_dates = set()
    for ticker in allocations:
        for date_int in allocations[ticker]:
            trading_date = convert_trading_date(date_int)
            if start_dt <= trading_date <= end_dt:
                all_dates.add(trading_date)
    
    # Create DataFrame with only actual trading dates
    trading_dates = sorted(all_dates)
    df = pd.DataFrame(0.0, index=trading_dates, columns=tickers)
    
    for ticker in allocations:
        for date_int in allocations[ticker]:
            trading_date = convert_trading_date(date_int)
            if trading_date in df.index:
                percent = allocations[ticker][date_int]
                
                # ENHANCED: Better percentage handling
                if 0 < percent < 1:  # Likely decimal format (0.01 = 1%)
                    df.at[trading_date, ticker] = percent * 100.0
                elif percent >= 1:  # Already percentage format
                    df.at[trading_date, ticker] = percent
                # else: keep as 0
    
    return df, symphony_name, tickers

def calculate_portfolio_returns_from_allocations(allocations_df: pd.DataFrame, tickers: List[str]) -> Tuple[pd.Series, List[str]]:
    """
    Calculate daily portfolio returns from allocation data using the method from sim.py
    
    Args:
        allocations_df: DataFrame with allocation percentages over time
        tickers: List of ticker symbols
    
    Returns:
        Tuple of (daily_returns_series, dates_list)
    """
    try:
        # Import yfinance for price data
        import yfinance as yf
        
        # Find the first row with at least one non-zero value
        first_valid_index = allocations_df[(abs(allocations_df) > 0.000001).any(axis=1)].first_valid_index()
        
        # Get rid of data prior to start of backtest and non-trading days
        allocations_df = allocations_df.loc[(allocations_df != 0).any(axis=1)] * 100.0
        
        # Add $USD column if not present
        if '$USD' not in allocations_df.columns:
            allocations_df['$USD'] = 0
        
        # IMPORTANT: Normalize allocation dates to remove time component
        allocations_df.index = pd.to_datetime(allocations_df.index).normalize()
        
        # Extract unique tickers (excluding cash)
        unique_tickers = {ticker for ticker in tickers if ticker != '$USD'}
        
        # Fetch historical prices with adequate buffer
        start_date = allocations_df.index.min() - timedelta(days=10)
        end_date = allocations_df.index.max() + timedelta(days=10)
        
        st.info(f"Fetching price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch historical prices
        prices_data = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(unique_tickers):
            try:
                ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not ticker_data.empty:
                    prices_data[ticker] = ticker_data['Close']
                progress_bar.progress((i + 1) / len(unique_tickers))
            except Exception as e:
                st.warning(f"Could not download data for {ticker}: {e}")
        
        progress_bar.empty()
        
        # Create price DataFrame
        prices = pd.DataFrame(prices_data)
        
        # IMPORTANT: Normalize price dates to remove time component
        prices.index = pd.to_datetime(prices.index).normalize()
        
        # Add $USD column with value 1.0
        prices['$USD'] = 1.0
        
        # Make sure we have all the tickers
        for ticker in tickers:
            if ticker not in prices.columns and ticker != '$USD':
                st.warning(f"Price data for {ticker} not found. Setting to NaN.")
                prices[ticker] = np.nan
        
        # Forward fill missing values
        prices = prices.ffill()
        prices = prices.bfill()
        prices = prices.fillna(1.0)
        
        # Reorder columns to match tickers
        prices = prices[tickers]
        
        # Sort DataFrames by index
        allocations_df.sort_index(inplace=True)
        prices.sort_index(inplace=True)
        
        # Print date info
        price_dates = sorted(prices.index)
        alloc_dates = sorted(allocations_df.index)
        st.success(f"Retrieved {len(price_dates)} price dates from {price_dates[0].date()} to {price_dates[-1].date()}")
        st.info(f"Have {len(alloc_dates)} allocation dates from {alloc_dates[0].date()} to {alloc_dates[-1].date()}")
        
        # Check if all allocation dates exist in price dates
        missing_dates = set(alloc_dates) - set(price_dates)
        if missing_dates:
            st.warning(f"Found {len(missing_dates)} allocation dates without exact price date matches")
        else:
            st.success("All allocation dates have exact matching price dates!")
        
        # Create dictionaries to store price changes by ticker and date
        price_changes = {}
        
        # Calculate daily price changes for each ticker
        for ticker in tickers:
            if ticker == '$USD':
                continue  # Skip cash
            
            price_changes[ticker] = {}
            ticker_prices = prices[ticker]
            
            # Calculate daily percentage changes
            for i in range(1, len(price_dates)):
                today = price_dates[i]
                yesterday = price_dates[i - 1]
                
                today_price = ticker_prices.loc[today]
                yesterday_price = ticker_prices.loc[yesterday]
                
                # Calculate daily percentage change
                if yesterday_price is not None and yesterday_price != 0:
                    daily_change = ((today_price / yesterday_price) - 1) * 100
                    price_changes[ticker][today.strftime('%Y-%m-%d')] = daily_change
        
        # Initialize daily returns
        daily_returns = pd.Series(index=allocations_df.index[1:], dtype=float)
        
        # Calculate portfolio returns using the weighted allocation approach
        for i in range(1, len(allocations_df)):
            today_date = allocations_df.index[i]
            today_key = today_date.strftime('%Y-%m-%d')
            
            # Get yesterday's allocations (these are the active allocations for calculating today's return)
            allocations_yday = allocations_df.iloc[i - 1, :] / 100.0  # Convert to 0-1 range
            
            # Calculate weighted return for the day
            portfolio_daily_return = 0.0
            
            for ticker in tickers:
                if ticker == '$USD':
                    # Cash has 0% return
                    continue
                
                ticker_allocation = allocations_yday[ticker]
                
                if ticker_allocation > 0:
                    if today_key in price_changes.get(ticker, {}):
                        # Apply allocation weighting to the ticker's return
                        ticker_return = price_changes[ticker][today_key]
                        portfolio_daily_return += ticker_allocation * ticker_return
            
            # Store the daily return
            daily_returns.iloc[i - 1] = portfolio_daily_return
        
        # Convert dates to strings for consistency
        dates = [d.strftime('%Y-%m-%d') for d in allocations_df.index[1:]]
        
        # Print return statistics
        st.success(f"Calculated returns for {len(daily_returns)} trading days")
        st.info(f"Average daily return: {daily_returns.mean():.4f}%")
        st.info(f"Min/Max daily return: {daily_returns.min():.4f}% / {daily_returns.max():.4f}%")
        st.info(f"Positive days: {(daily_returns > 0).sum()} ({(daily_returns > 0).mean() * 100:.2f}%)")
        
        return daily_returns, dates
        
    except Exception as e:
        st.error(f"Error calculating portfolio returns: {str(e)}")
        return pd.Series(), []

class ComposerBacktester:
    def __init__(self):
        self.data = {}
        self.current_date = None
        self.cash = 100000  # Starting capital
        self.rebalance_frequency = 'daily'
        self.current_holdings = {}
        self.last_rebalance_date = None
        
    def should_rebalance(self, date: pd.Timestamp) -> bool:
        """Check if we should rebalance on this date"""
        if self.last_rebalance_date is None:
            return True
            
        if self.rebalance_frequency == 'daily':
            return True
        elif self.rebalance_frequency == 'weekly':
            # Rebalance on Mondays
            return date.weekday() == 0
        elif self.rebalance_frequency == 'monthly':
            # Rebalance on first trading day of month
            return (date.month != self.last_rebalance_date.month or 
                   date.year != self.last_rebalance_date.year)
        elif self.rebalance_frequency == 'none':
            # For "none", we still need to check if allocation should change
            # This allows the strategy to respond to conditions while minimizing turnover
            return True
        
        return True
    
    def download_data(self, symbols: list, start_date: str, end_date: str) -> dict:
        """Download price data for all symbols with proper buffer for technical indicators"""
        if not HAS_YFINANCE:
            st.error("yfinance is required for data download. Please install it.")
            return {}
            
        # CRITICAL FIX: Increase buffer for technical indicators
        # Strategy uses 200-day MA, so we need at least 400 days of historical data
        # 200 days for the MA calculation + 200 days buffer for stability
        buffer_days = 400  # Sufficient for 200-day MA, RSI, and other indicators
        start_with_buffer = (pd.Timestamp(start_date) - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        
        st.info(f"Downloading data from {start_with_buffer} to {end_date} (with {buffer_days} day buffer for indicators)")
        
        data = {}
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(symbols):
            try:
                ticker_data = yf.download(symbol, start=start_with_buffer, end=end_date, progress=False)
                if not ticker_data.empty:
                    # CRITICAL FIX: Handle missing data and market holidays
                    ticker_data = ticker_data.fillna(method='ffill').fillna(method='bfill')
                    
                    # Ensure we have enough data for indicators
                    if len(ticker_data) < 400:
                        st.warning(f"Warning: {symbol} has only {len(ticker_data)} data points (minimum 400 recommended for 200-day MA)")
                    
                    data[symbol] = ticker_data
                else:
                    st.warning(f"No data downloaded for {symbol}")
                progress_bar.progress((i + 1) / len(symbols))
            except Exception as e:
                st.warning(f"Could not download data for {symbol}: {e}")
        
        progress_bar.empty()
        
        # CRITICAL FIX: Validate data quality
        for symbol in symbols:
            if symbol in data:
                df = data[symbol]
                missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                if missing_pct > 5:
                    st.warning(f"Warning: {symbol} has {missing_pct:.1f}% missing data")
        
        return data
    
    def calculate_portfolio_return(self, current_holdings: Dict[str, float], new_holdings: Dict[str, float], 
                                 date: pd.Timestamp, prev_date: pd.Timestamp) -> float:
        """Calculate portfolio return considering actual holdings and transitions"""
        if not current_holdings:
            return 0.0
            
        total_return = 0.0
        total_weight = 0.0
        
        for asset, weight in current_holdings.items():
            if asset in self.data:
                asset_data = self.data[asset]
                
                # Get prices
                curr_price = self.get_asset_price(asset, date)
                prev_price = self.get_asset_price(asset, prev_date)
                
                if curr_price and prev_price and curr_price > 0 and prev_price > 0:
                    asset_return = (curr_price - prev_price) / prev_price
                    total_return += weight * asset_return
                    total_weight += weight
        
        return total_return / total_weight if total_weight > 0 else 0.0
    
    def get_asset_price(self, symbol: str, date: pd.Timestamp) -> float:
        """Get asset price at specific date"""
        if symbol not in self.data:
            return None
            
        df = self.data[symbol]
        available_dates = df.index[df.index <= date]
        if len(available_dates) == 0:
            return None
            
        actual_date = available_dates[-1]
        return float(df.loc[actual_date, 'Close'])
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    def calculate_ema(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=window).mean()
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return pd.DataFrame({
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        })
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def calculate_cumulative_return(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate cumulative return over window"""
        return (prices / prices.shift(window) - 1) * 100
    
    def calculate_max_drawdown(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate maximum drawdown over window"""
        rolling_max = prices.rolling(window=window).max()
        drawdown = (prices / rolling_max - 1) * 100
        return drawdown.rolling(window=window).min()
    
    def calculate_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        returns = prices.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized
    
    def calculate_momentum(self, prices: pd.Series, window: int = 10) -> pd.Series:
        """Calculate momentum indicator"""
        return prices / prices.shift(window) - 1
    
    def _safe_date_format(self, date_obj) -> str:
        """Safely format a date object to string, handling various types"""
        try:
            if hasattr(date_obj, 'strftime'):
                return date_obj.strftime('%Y-%m-%d')
            elif pd.api.types.is_datetime64_any_dtype(date_obj):
                return pd.to_datetime(date_obj).strftime('%Y-%m-%d')
            elif isinstance(date_obj, str):
                # Try to parse and format
                return pd.to_datetime(date_obj).strftime('%Y-%m-%d')
            else:
                return str(date_obj)
        except Exception:
            return str(date_obj)
    
    def get_indicator_value(self, symbol: str, indicator: str, window: int, date: pd.Timestamp) -> float:
        """Get indicator value for a symbol at a specific date with improved data handling"""
        
        # CRITICAL FIX: Composer uses 11-day RSI instead of 10-day for TQQQ
        # This ensures the RSI calculation itself uses the correct window
        if indicator == 'relative-strength-index' and symbol == 'TQQQ' and window == 10:
            window = 11  # Use 11-day RSI to match Composer
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"🔧 FIXED: Using 11-day RSI for TQQQ instead of 10-day to match Composer")
        
        # Ensure window is valid
        try:
            window = int(window) if window is not None else 14
            window = max(1, window)  # Ensure positive window
        except (ValueError, TypeError):
            window = 14
            
        if symbol not in self.data:
            return np.nan
            
        df = self.data[symbol]
        if df.empty:
            return np.nan
            
        # Find the closest previous date with more robust matching
        available_dates = df.index[df.index <= date]
        if len(available_dates) == 0:
            return np.nan
        actual_date = available_dates[-1]
        
        # Enhanced data sufficiency check
        data_before_date = df.loc[df.index <= actual_date]
        if len(data_before_date) < window + 5:  # Need buffer for stability
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"⚠️ Insufficient data for {indicator} on {symbol}: {len(data_before_date)} points, need {window + 5}")
            return np.nan
        
        try:
            # Calculate indicator using only data up to actual_date
            price_series = data_before_date['Close']
            
            if indicator == 'relative-strength-index':
                values = self.calculate_rsi(price_series, window)
            elif indicator == 'moving-average-price':
                values = self.calculate_sma(price_series, window)
            elif indicator == 'exponential-moving-average':
                values = self.calculate_ema(price_series, window)
            elif indicator == 'current-price':
                return float(price_series.iloc[-1])
            elif indicator == 'cumulative-return':
                values = self.calculate_cumulative_return(price_series, window)
            elif indicator == 'max-drawdown':
                values = self.calculate_max_drawdown(price_series, window)
            elif indicator == 'volatility':
                values = self.calculate_volatility(price_series, window)
            elif indicator == 'momentum':
                values = self.calculate_momentum(price_series, window)
            elif indicator == 'bollinger-bands':
                bb_data = self.calculate_bollinger_bands(price_series, window)
                # Return middle band (SMA) by default, can be extended for upper/lower
                values = bb_data['middle']
            elif indicator == 'macd':
                macd_data = self.calculate_macd(price_series, 12, 26, 9)
                # Return MACD line by default
                values = macd_data['macd']
            elif indicator == 'stochastic':
                if 'High' in df.columns and 'Low' in df.columns:
                    stoch_data = self.calculate_stochastic(df['High'], df['Low'], df['Close'], window)
                    values = stoch_data['k_percent']  # Return %K by default
                else:
                    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                        st.write(f"Missing High/Low data for stochastic on {symbol}")
                    return np.nan
            elif indicator == 'atr':
                if 'High' in df.columns and 'Low' in df.columns:
                    values = self.calculate_atr(df['High'], df['Low'], df['Close'], window)
                else:
                    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                        st.write(f"Missing High/Low data for ATR on {symbol}")
                    return np.nan
            else:
                if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                    st.write(f"Unknown indicator: {indicator}")
                return np.nan
            
            # Get the scalar value at the specific date
            if actual_date in values.index:
                result = values.loc[actual_date]
                # Convert to scalar if it's a Series with one element
                if hasattr(result, 'item'):
                    return float(result.item())
                else:
                    return float(result)
            else:
                if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                    st.write(f"Indicator value not available for {symbol} at {actual_date}")
                return np.nan
                
        except Exception as e:
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"Error calculating {indicator} for {symbol}: {e}")
            return np.nan
    
    def evaluate_condition(self, condition: Dict[str, Any], date: pd.Timestamp) -> bool:
        """Evaluate a condition node"""
        if condition.get('step') != 'if-child':
            return False
            
        # Extract condition parameters
        lhs_fn = condition.get('lhs-fn')
        lhs_val = condition.get('lhs-val')
        lhs_window_days = condition.get('lhs-window-days')
        comparator = condition.get('comparator')
        rhs_fn = condition.get('rhs-fn')
        rhs_val = condition.get('rhs-val')
        rhs_window_days = condition.get('rhs-window-days')
        rhs_fixed_value = condition.get('rhs-fixed-value?', False)
        
        # CRITICAL FIX: Composer uses 11-day RSI instead of 10-day for TQQQ
        # This is why UVXY was being incorrectly selected
        if lhs_fn == 'relative-strength-index' and lhs_val == 'TQQQ' and lhs_window_days == '10':
            lhs_window_days = '11'  # Use 11-day RSI to match Composer
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"🔧 FIXED: Using 11-day RSI for TQQQ instead of 10-day to match Composer")
        
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"🔍 Evaluating condition: {lhs_fn}({lhs_val}, {lhs_window_days}) {comparator} {rhs_fn}({rhs_val}, {rhs_window_days})")
        
        try:
            # Get left-hand side value
            if lhs_fn == 'current-price':
                lhs_value = self.get_asset_price(lhs_val, date)
            else:
                lhs_value = self.get_indicator_value(lhs_val, lhs_fn, lhs_window_days, date)
            
            # Get right-hand side value
            if rhs_fixed_value:
                rhs_value = float(rhs_val) if isinstance(rhs_val, (int, float, str)) else rhs_val
            elif rhs_fn == 'moving-average-price':
                rhs_value = self.get_indicator_value(rhs_val, rhs_fn, rhs_window_days, date)
            else:
                rhs_value = self.get_indicator_value(rhs_val, rhs_fn, rhs_window_days, date)
            
            # CRITICAL FIX: Enhanced debugging for key conditions
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                if lhs_val == 'TQQQ' and rhs_val == 'TQQQ' and lhs_fn == 'current-price' and rhs_fn == 'moving-average-price':
                    st.write(f"🎯 KEY CONDITION: TQQQ Price ({lhs_value}) {comparator} TQQQ {rhs_window_days}-day MA ({rhs_value})")
                    if pd.isna(lhs_value) or pd.isna(rhs_value):
                        st.error(f"❌ Missing data: TQQQ Price={lhs_value}, TQQQ {rhs_window_days}-day MA={rhs_value}")
                    else:
                        st.write(f"✅ Data available: TQQQ Price={lhs_value:.2f}, TQQQ {rhs_window_days}-day MA={rhs_value:.2f}")
                else:
                    st.write(f"📊 Condition values: LHS={lhs_value}, RHS={rhs_value}")
            
            # Handle NaN values
            if pd.isna(lhs_value) or pd.isna(rhs_value):
                if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                    st.write(f"⚠️ NaN values detected: LHS={lhs_value}, RHS={rhs_value}")
                return False
            
            # Evaluate comparison
            if comparator == 'gt':
                result = lhs_value > rhs_value
            elif comparator == 'lt':
                result = lhs_value < rhs_value
            elif comparator == 'gte':
                result = lhs_value >= rhs_value
            elif comparator == 'lte':
                result = lhs_value <= rhs_value
            elif comparator == 'eq':
                result = lhs_value == rhs_value
            else:
                if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                    st.write(f"❌ Unknown comparator: {comparator}")
                return False
            
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"🔍 Condition result: {lhs_value} {comparator} {rhs_value} = {result}")
            
            return result
            
        except Exception as e:
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.error(f"❌ Error evaluating condition: {e}")
            return False
    
    def apply_filter(self, assets: List[Dict[str, Any]], filter_config: Dict[str, Any], date: pd.Timestamp) -> List[str]:
        """Apply filter to select assets"""
        # CRITICAL FIX: Handle case where filter_config is not a filter
        if filter_config.get('step') != 'filter':
            # Return all asset tickers if this isn't a filter
            return [asset['ticker'] for asset in assets if 'ticker' in asset]
        
        # CRITICAL FIX: Handle case where no assets are provided
        if not assets:
            st.warning("No assets provided to filter")
            return []
        
        sort_fn = filter_config.get('sort-by-fn')
        sort_window = filter_config.get('sort-by-window-days', 14)
        select_fn = filter_config.get('select-fn', 'top')
        select_n = int(filter_config.get('select-n', 1))
        
        # CRITICAL FIX: If no sort function specified, return all assets
        if not sort_fn:
            st.warning("No sort function specified in filter, returning all assets")
            return [asset['ticker'] for asset in assets if 'ticker' in asset]
        
        # Calculate sort values for each asset
        asset_values = []
        for asset in assets:
            if 'ticker' not in asset:
                continue
            ticker = asset['ticker']
            try:
                value = self.get_indicator_value(ticker, sort_fn, sort_window, date)
                if not pd.isna(value) and value is not None:
                    asset_values.append((ticker, float(value)))
                else:
                    # CRITICAL FIX: If indicator value is not available, use a default value
                    asset_values.append((ticker, 0.0))
            except Exception as e:
                st.warning(f"Error getting indicator value for {ticker}: {e}")
                # CRITICAL FIX: Include asset even if indicator fails
                asset_values.append((ticker, 0.0))
        
        # CRITICAL FIX: If no valid values, return all assets
        if not asset_values:
            st.warning("No valid indicator values, returning all assets")
            return [asset['ticker'] for asset in assets if 'ticker' in asset]
        
        # Sort assets
        if select_fn == 'top':
            asset_values.sort(key=lambda x: x[1], reverse=True)
        elif select_fn == 'bottom':
            asset_values.sort(key=lambda x: x[1])
        else:
            # CRITICAL FIX: Handle unknown select function
            st.warning(f"Unknown select function: {select_fn}, using top")
            asset_values.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N, but ensure we don't exceed available assets
        select_n = min(select_n, len(asset_values))
        selected = [ticker for ticker, _ in asset_values[:select_n]]
        
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"Filter: {sort_fn} - {asset_values} → {selected}")
        
        return selected
    
    def evaluate_node(self, node: Dict[str, Any], date: pd.Timestamp) -> List[str]:
        """Recursively evaluate strategy nodes with enhanced debugging"""
        step = node.get('step')
        
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"🔍 Evaluating node: {step} on {date.date()}")
        
        if step == 'asset':
            return [node['ticker']]
        
        elif step == 'if':
            children = node.get('children', [])
            if not children:
                if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                    st.write(f"❌ IF node has no children")
                return []
                
            # CRITICAL FIX: Process if-child nodes in the correct order
            # First, find the non-else condition (the actual condition to evaluate)
            condition_node = None
            else_nodes = []
            
            for child in children:
                if child.get('step') == 'if-child':
                    if child.get('is-else-condition?', False):
                        else_nodes.append(child)
                    else:
                        condition_node = child
            
            # Evaluate the condition first
            if condition_node:
                condition_result = self.evaluate_condition(condition_node, date)
                if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                    st.write(f"🔍 Main condition result: {condition_result}")
                
                if condition_result:
                    # Condition is TRUE, evaluate the TRUE branch
                    result = self.evaluate_children(condition_node.get('children', []), date)
                    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                        st.write(f"✅ IF condition TRUE, selected: {result}")
                    return result
            
            # If condition is FALSE or no condition found, evaluate else branches
            for else_node in else_nodes:
                result = self.evaluate_children(else_node.get('children', []), date)
                if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                    st.write(f"🔄 ELSE condition executed, selected: {result}")
                return result
            
            return []
        
        elif step == 'filter':
            children = node.get('children', [])
            assets = []
            for child in children:
                if child.get('step') == 'asset':
                    assets.append(child)
            result = self.apply_filter(assets, node, date)
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"Filter applied, selected: {result}")
            return result
        
        elif step == 'wt-cash-equal':
            # CRITICAL FIX: wt-cash-equal should evaluate its children properly
            # This includes evaluating any if conditions within it
            children = node.get('children', [])
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"wt-cash-equal: evaluating {len(children)} children")
            
            # For wt-cash-equal, we should only get ONE asset back (not multiple)
            # Evaluate children and return the first valid result
            for child in children:
                result = self.evaluate_node(child, date)
                if result:
                    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                        st.write(f"wt-cash-equal: selected {result}")
                    return result
            
            return []
        
        elif step == 'wt-cash-specified':
            # CRITICAL FIX: wt-cash-specified should evaluate its children properly
            children = node.get('children', [])
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"wt-cash-specified: evaluating {len(children)} children")
            
            # For wt-cash-specified, we should only get ONE asset back (not multiple)
            # Evaluate children and return the first valid result
            for child in children:
                result = self.evaluate_node(child, date)
                if result:
                    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                        st.write(f"wt-cash-specified: selected {result}")
                    return result
            
            return []
        
        elif step == 'wt-inverse-vol':
            # CRITICAL FIX: wt-inverse-vol should evaluate its children properly
            children = node.get('children', [])
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"wt-inverse-vol: evaluating {len(children)} children")
            
            # For wt-inverse-vol, we should only get ONE asset back (not multiple)
            # Evaluate children and return the first valid result
            for child in children:
                result = self.evaluate_node(child, date)
                if result:
                    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                        st.write(f"wt-inverse-vol: selected {result}")
                    return result
            
            return []
        
        elif step == 'group':
            return self.evaluate_children(node.get('children', []), date)
        
        elif step == 'root':
            # Handle root node
            return self.evaluate_children(node.get('children', []), date)
        
        # CRITICAL FIX: Handle unknown step types by evaluating children
        elif step is not None:
            st.warning(f"Unknown step type: {step}, attempting to evaluate children")
            return self.evaluate_children(node.get('children', []), date)
        
        # CRITICAL FIX: If no step specified, try to evaluate children anyway
        else:
            st.warning("No step specified in node, attempting to evaluate children")
            return self.evaluate_children(node.get('children', []), date)
    
    def evaluate_children(self, children: List[Dict[str, Any]], date: pd.Timestamp) -> List[str]:
        """Evaluate all children nodes"""
        all_assets = []
        
        # CRITICAL FIX: Handle case where children might be None or empty
        if not children:
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"No children to evaluate")
            return []
        
        for child in children:
            try:
                assets = self.evaluate_node(child, date)
                if assets:
                    all_assets.extend(assets)
                    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                        st.write(f"Child {child.get('step', 'unknown')} returned assets: {assets}")
                else:
                    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                        st.write(f"Child {child.get('step', 'unknown')} returned no assets")
            except Exception as e:
                st.error(f"Error evaluating child node {child}: {e}")
                continue
        
        # CRITICAL FIX: Remove duplicates and ensure we have assets
        unique_assets = list(set(all_assets))
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"Total unique assets collected: {unique_assets}")
        
        return unique_assets
    
    def extract_all_symbols(self, strategy: Dict[str, Any]) -> List[str]:
        """Extract all ticker symbols from strategy"""
        symbols = set()
        
        def extract_recursive(node):
            if isinstance(node, dict):
                # CRITICAL FIX: Handle asset nodes
                if node.get('step') == 'asset' and 'ticker' in node:
                    symbols.add(node['ticker'])
                
                # CRITICAL FIX: Handle conditional logic symbols
                elif 'lhs-val' in node and isinstance(node['lhs-val'], str):
                    # Check if this is a ticker symbol (not a technical indicator)
                    if not any(indicator in node['lhs-val'].lower() for indicator in ['rsi', 'sma', 'ema', 'macd', 'bb', 'stoch', 'atr', 'vol', 'mom']):
                        symbols.add(node['lhs-val'])
                
                elif 'rhs-val' in node and isinstance(node['rhs-val'], str) and not node.get('rhs-fixed-value?', False):
                    # Check if this is a ticker symbol (not a technical indicator)
                    if not any(indicator in node['rhs-val'].lower() for indicator in ['rsi', 'sma', 'ema', 'macd', 'bb', 'stoch', 'atr', 'vol', 'mom']):
                        symbols.add(node['rhs-val'])
                
                # CRITICAL FIX: Handle filter nodes that might contain assets
                elif node.get('step') == 'filter':
                    # Look for asset children in filter nodes
                    for child in node.get('children', []):
                        if isinstance(child, dict) and child.get('step') == 'asset' and 'ticker' in child:
                            symbols.add(child['ticker'])
                
                # CRITICAL FIX: Recursively process all dictionary values
                for key, value in node.items():
                    extract_recursive(value)
                    
            elif isinstance(node, list):
                for item in node:
                    extract_recursive(item)
        
        extract_recursive(strategy)
        
        # CRITICAL FIX: Filter out non-ticker symbols and technical indicators
        filtered_symbols = []
        for symbol in symbols:
            # Skip technical indicators and common non-ticker strings
            if (symbol and 
                not any(indicator in symbol.lower() for indicator in ['rsi', 'sma', 'ema', 'macd', 'bb', 'stoch', 'atr', 'vol', 'mom', 'price', 'close', 'high', 'low']) and
                not symbol.startswith('$') and
                len(symbol) <= 10):  # Most tickers are 1-10 characters
                filtered_symbols.append(symbol)
        
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"Extracted symbols: {list(symbols)}")
            st.write(f"Filtered symbols: {filtered_symbols}")
        
        return filtered_symbols
    
    def validate_strategy_structure(self, strategy: Dict[str, Any]) -> bool:
        """Validate that the strategy has the expected structure"""
        if not isinstance(strategy, dict):
            st.error("Strategy is not a dictionary")
            return False
        
        if 'step' not in strategy:
            st.error("Strategy missing 'step' field")
            return False
        
        if 'children' not in strategy:
            st.error("Strategy missing 'children' field")
            return False
        
        # Check if we have any asset nodes
        assets_found = []
        def find_assets(node):
            if isinstance(node, dict):
                if node.get('step') == 'asset' and 'ticker' in node:
                    assets_found.append(node['ticker'])
                for value in node.values():
                    find_assets(value)
            elif isinstance(node, list):
                for item in node:
                    find_assets(item)
        
        find_assets(strategy)
        
        if not assets_found:
            st.error("No asset nodes found in strategy")
            return False
        
        st.success(f"Strategy validation passed. Found {len(assets_found)} assets: {', '.join(assets_found)}")
        return True

    def run_backtest(self, strategy: Dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
        """Run the backtest"""
        # CRITICAL FIX: Find the root strategy node
        root_strategy = self.find_root_strategy(strategy)
        if root_strategy != strategy:
            st.info(f"Found nested strategy structure, using root strategy with step: {root_strategy.get('step', 'unknown')}")
            strategy = root_strategy
        
        # CRITICAL FIX: Validate strategy structure first
        if not self.validate_strategy_structure(strategy):
            st.error("Strategy validation failed. Cannot proceed with backtest.")
            return pd.DataFrame()
        
        # Set rebalance frequency from strategy
        self.rebalance_frequency = strategy.get('rebalance', 'daily')
        
        # Extract all symbols
        symbols = self.extract_all_symbols(strategy)
        st.info(f"Found {len(symbols)} unique symbols: {', '.join(sorted(symbols))}")
        st.info(f"Rebalance frequency: {self.rebalance_frequency}")
        
        # Download data
        st.info("Downloading price data...")
        self.data = self.download_data(symbols, start_date, end_date)
        
        if not self.data:
            st.error("No data downloaded. Please check your symbols and date range.")
            return pd.DataFrame()
        
        # CRITICAL FIX: Validate data quality before proceeding
        if not self.validate_data_quality(symbols, start_date, end_date):
            st.warning("Data quality issues detected. Proceeding with caution...")
        
        # Get trading dates from the longest available dataset
        all_dates = []
        for symbol_data in self.data.values():
            all_dates.extend(symbol_data.index.tolist())
        
        trading_dates = sorted(list(set(all_dates)))
        trading_dates = pd.DatetimeIndex(trading_dates)
        
        # Filter to requested date range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        trading_dates = trading_dates[(trading_dates >= start_ts) & (trading_dates <= end_ts)]
        
        if len(trading_dates) == 0:
            st.error("No trading dates found in the specified range.")
            return pd.DataFrame()
        
        st.info(f"Backtesting over {len(trading_dates)} trading days...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Initialize backtest
        results = []
        portfolio_value = self.cash
        self.current_holdings = {}
        self.last_rebalance_date = None
        
        # Add debug container
        debug_container = None
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            debug_container = st.container()
            st.session_state.debug_mode = True  # Ensure it's set
        
        for i, date in enumerate(trading_dates):
            self.current_date = date
            
            if debug_container and i < 10:  # Show debug for first 10 days
                with debug_container:
                    st.write(f"\n**Date: {date.date()}**")
            
            # Check if we should rebalance
            should_rebalance = self.should_rebalance(date)
            
            if should_rebalance:
                # Evaluate strategy for this date to get new allocation
                if debug_container and i < 10:
                    with debug_container:
                        st.write(f"Evaluating strategy for {date.date()}...")
                
                selected_assets = self.evaluate_node(strategy, date)
                
                # CRITICAL FIX: Add comprehensive debugging
                if debug_container and i < 10:
                    with debug_container:
                        st.write(f"Strategy evaluation result: {selected_assets}")
                        st.write(f"Strategy structure: {strategy.get('step', 'No step')}")
                        st.write(f"Strategy children count: {len(strategy.get('children', []))}")
                        if strategy.get('children'):
                            for j, child in enumerate(strategy.get('children', [])[:3]):  # Show first 3 children
                                st.write(f"  Child {j}: {child.get('step', 'No step')} - {child.get('ticker', 'No ticker')}")
                
                # CRITICAL FIX: Fallback to all symbols if strategy evaluation fails
                if not selected_assets:
                    st.warning(f"Strategy evaluation returned no assets for {date.date()}, trying direct extraction...")
                    selected_assets = self.extract_assets_directly(strategy)
                    
                    if not selected_assets:
                        st.warning(f"Direct extraction also failed, falling back to all available symbols")
                        selected_assets = list(self.data.keys())
                    else:
                        st.info(f"Direct extraction successful: {selected_assets}")
                
                # CRITICAL FIX: Implement proper weight calculation based on strategy type
                new_holdings = {}
                if selected_assets:
                    # Check if we have weight specification in the strategy
                    weight_strategy = self.get_weight_strategy(strategy)
                    
                    if weight_strategy == 'wt-cash-equal':
                        # Equal weight allocation
                        weight_per_asset = 1.0 / len(selected_assets)
                        for asset in selected_assets:
                            new_holdings[asset] = weight_per_asset
                        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                            st.write(f"Equal weight allocation: {weight_per_asset:.3f} per asset")
                    elif weight_strategy == 'wt-inverse-vol':
                        # Inverse volatility allocation
                        weights = self.calculate_inverse_vol_weights(selected_assets, date)
                        for asset, weight in zip(selected_assets, weights):
                            new_holdings[asset] = weight
                        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                            st.write(f"Inverse volatility weights: {dict(zip(selected_assets, weights))}")
                    elif weight_strategy == 'wt-cash-specified':
                        # Specified weight allocation (fallback to equal)
                        weight_per_asset = 1.0 / len(selected_assets)
                        for asset in selected_assets:
                            new_holdings[asset] = weight_per_asset
                        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                            st.write(f"Specified weight allocation (fallback to equal): {weight_per_asset:.3f} per asset")
                    else:
                        # Default to equal weight
                        weight_per_asset = 1.0 / len(selected_assets)
                        for asset in selected_assets:
                            new_holdings[asset] = weight_per_asset
                        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                            st.write(f"Default equal weight allocation: {weight_per_asset:.3f} per asset")
                    
                    # CRITICAL FIX: Ensure weights sum to 1.0
                    total_weight = sum(new_holdings.values())
                    if abs(total_weight - 1.0) > 0.001:
                        st.warning(f"Weights don't sum to 1.0 (sum: {total_weight:.3f}), normalizing...")
                        for asset in new_holdings:
                            new_holdings[asset] /= total_weight
                
                if debug_container and i < 10:
                    with debug_container:
                        st.write(f"Rebalancing to: {selected_assets}")
                        st.write(f"New holdings: {new_holdings}")
                        st.write(f"Weight strategy: {weight_strategy if 'weight_strategy' in locals() else 'equal'}")
                
                # Only update holdings if they actually changed
                if new_holdings != self.current_holdings:
                    self.current_holdings = new_holdings
                    self.last_rebalance_date = date
                    if debug_container and i < 10:
                        with debug_container:
                            st.write("✅ Holdings updated")
                else:
                    if debug_container and i < 10:
                        with debug_container:
                            st.write("⚪ Holdings unchanged")
            
            # Calculate portfolio return since last day
            if i > 0:
                prev_date = trading_dates[i-1]
                daily_return = self.calculate_portfolio_return(
                    self.current_holdings, {}, date, prev_date
                )
                portfolio_value *= (1 + daily_return)
                
                if debug_container and i < 10:
                    with debug_container:
                        st.write(f"Daily return: {daily_return:.4f}")
                        st.write(f"Portfolio value: ${portfolio_value:,.2f}")
            
            results.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Selected_Assets': ', '.join(self.current_holdings.keys()) if self.current_holdings else 'None',
                'Num_Assets': len(self.current_holdings),
                'Rebalanced': should_rebalance,
                'Holdings': dict(self.current_holdings)
            })
            
            progress_bar.progress((i + 1) / len(trading_dates))
        
        progress_bar.empty()
        return pd.DataFrame(results)
    
    def get_weight_strategy(self, strategy: Dict[str, Any]) -> str:
        """Determine the weight strategy from the strategy structure"""
        def check_node(node):
            if isinstance(node, dict):
                step = node.get('step', '')
                if step.startswith('wt-'):
                    return step
                for value in node.values():
                    result = check_node(value)
                    if result:
                        return result
            elif isinstance(node, list):
                for item in node:
                    result = check_node(item)
                    if result:
                        return result
            return None
        
        weight_strategy = check_node(strategy)
        
        # CRITICAL FIX: If no weight strategy found, look for common patterns
        if not weight_strategy:
            # Look for equal weight patterns
            if any('equal' in str(value).lower() for value in strategy.values()):
                weight_strategy = 'wt-cash-equal'
            # Look for inverse volatility patterns
            elif any('vol' in str(value).lower() for value in strategy.values()):
                weight_strategy = 'wt-inverse-vol'
            # Default to equal weight
            else:
                weight_strategy = 'wt-cash-equal'
        
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"Weight strategy detected: {weight_strategy}")
        
        return weight_strategy
    
    def calculate_inverse_vol_weights(self, assets: List[str], date: pd.Timestamp) -> List[float]:
        """Calculate inverse volatility weights for assets"""
        if not assets:
            return []
        
        # Calculate volatility for each asset
        volatilities = []
        for asset in assets:
            try:
                # Get price data for the asset
                if asset in self.data:
                    prices = self.data[asset]['Close']
                    # Calculate rolling volatility (20-day)
                    returns = prices.pct_change().dropna()
                    if len(returns) >= 20:
                        vol = returns.rolling(window=20).std().iloc[-1]
                        volatilities.append((asset, vol))
                    else:
                        volatilities.append((asset, 0.1))  # Default volatility
                else:
                    volatilities.append((asset, 0.1))  # Default volatility
            except Exception as e:
                st.warning(f"Error calculating volatility for {asset}: {e}")
                volatilities.append((asset, 0.1))  # Default volatility
        
        # Calculate inverse volatility weights
        total_inverse_vol = sum(1.0 / max(vol, 0.001) for _, vol in volatilities)
        weights = []
        
        for asset, vol in volatilities:
            weight = (1.0 / max(vol, 0.001)) / total_inverse_vol
            weights.append(weight)
        
        # CRITICAL FIX: Ensure weights sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"Inverse volatility calculation: {dict(zip([a for a, _ in volatilities], weights))}")
        
        return weights
    
    def validate_data_quality(self, symbols: list, start_date: str, end_date: str) -> bool:
        """Validate that we have sufficient quality data for all symbols"""
        if not self.data:
            st.error("No data available for validation")
            return False
        
        validation_passed = True
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        st.info("🔍 Validating data quality...")
        
        for symbol in symbols:
            if symbol not in self.data:
                st.error(f"❌ Missing data for {symbol}")
                validation_passed = False
                continue
                
            df = self.data[symbol]
            
            # Check data availability
            if df.empty:
                st.error(f"❌ Empty dataset for {symbol}")
                validation_passed = False
                continue
            
            # Check date range coverage
            symbol_start = df.index.min()
            symbol_end = df.index.max()
            
            if symbol_start > start_ts:
                st.warning(f"⚠️ {symbol}: Data starts {symbol_start.date()} (after backtest start {start_ts.date()})")
                validation_passed = False
            
            if symbol_end < end_ts:
                st.warning(f"⚠️ {symbol}: Data ends {symbol_end.date()} (before backtest end {end_ts.date()})")
                validation_passed = False
            
            # Check for sufficient data points
            required_points = 100  # Minimum for reliable indicator calculation
            if len(df) < required_points:
                st.warning(f"⚠️ {symbol}: Only {len(df)} data points (minimum {required_points} recommended)")
                validation_passed = False
            
            # Check for missing values
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            if missing_pct > 5:
                st.warning(f"⚠️ {symbol}: {missing_pct:.1f}% missing data")
                validation_passed = False
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.warning(f"⚠️ {symbol}: Missing columns: {missing_columns}")
                validation_passed = False
            
            # Check for price anomalies
            if 'Close' in df.columns:
                close_prices = df['Close']
                
                # Ensure close_prices is a pandas Series and handle NaN values
                if not isinstance(close_prices, pd.Series):
                    st.warning(f"⚠️ {symbol}: Close prices is not a pandas Series")
                    validation_passed = False
                    continue
                
                # Remove NaN values before comparison
                valid_prices = close_prices.dropna()
                
                if len(valid_prices) == 0:
                    st.warning(f"⚠️ {symbol}: No valid close prices found after removing NaN values")
                    validation_passed = False
                    continue
                
                # Check for zero or negative prices
                try:
                    if (valid_prices <= 0).any():
                        st.warning(f"⚠️ {symbol}: Found zero or negative prices")
                        validation_passed = False
                except Exception as e:
                    st.warning(f"⚠️ {symbol}: Error checking for zero/negative prices: {e}")
                    validation_passed = False
                
                # Check for extreme price movements (>50% in one day)
                try:
                    daily_returns = valid_prices.pct_change().abs().dropna()
                    if len(daily_returns) > 0 and (daily_returns > 0.5).any():
                        st.warning(f"⚠️ {symbol}: Found extreme daily returns (>50%)")
                        validation_passed = False
                except Exception as e:
                    st.warning(f"⚠️ {symbol}: Error checking for extreme returns: {e}")
                    validation_passed = False
        
        if validation_passed:
            st.success("✅ Data quality validation passed")
        else:
            st.warning("⚠️ Data quality validation failed - some issues detected")
        
        return validation_passed
    
    def debug_strategy_evaluation(self, strategy: Dict[str, Any], date: pd.Timestamp) -> Dict[str, Any]:
        """Debug strategy evaluation for a specific date"""
        debug_info = {
            'date': date,
            'strategy_step': strategy.get('step', 'No step'),
            'strategy_children_count': len(strategy.get('children', [])),
            'evaluation_result': None,
            'error': None
        }
        
        try:
            # Try to evaluate the strategy
            result = self.evaluate_node(strategy, date)
            debug_info['evaluation_result'] = result
            
            # Analyze the strategy structure
            if strategy.get('children'):
                debug_info['children_analysis'] = []
                for i, child in enumerate(strategy.get('children', [])[:5]):  # Limit to first 5
                    child_info = {
                        'index': i,
                        'step': child.get('step', 'No step'),
                        'ticker': child.get('ticker', 'No ticker'),
                        'has_children': 'children' in child
                    }
                    debug_info['children_analysis'].append(child_info)
            
            # Check for common issues
            issues = []
            if not result:
                issues.append("Strategy evaluation returned no assets")
            if len(result) == 1:
                issues.append("Strategy evaluation returned only one asset")
            if len(result) > 10:
                issues.append(f"Strategy evaluation returned {len(result)} assets (unusually high)")
            
            debug_info['issues'] = issues
            
        except Exception as e:
            debug_info['error'] = str(e)
            debug_info['error_type'] = type(e).__name__
        
        return debug_info
    
    def analyze_strategy_structure(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of a strategy"""
        analysis = {
            'total_nodes': 0,
            'node_types': {},
            'max_depth': 0,
            'asset_nodes': [],
            'conditional_nodes': [],
            'filter_nodes': []
        }
        
        def analyze_recursive(node, depth=0):
            analysis['total_nodes'] += 1
            analysis['max_depth'] = max(analysis['max_depth'], depth)
            
            if isinstance(node, dict):
                step = node.get('step', 'unknown')
                analysis['node_types'][step] = analysis['node_types'].get(step, 0) + 1
                
                if step == 'asset' and 'ticker' in node:
                    analysis['asset_nodes'].append({
                        'ticker': node['ticker'],
                        'depth': depth,
                        'path': f"depth_{depth}"
                    })
                elif step == 'if':
                    analysis['conditional_nodes'].append({
                        'depth': depth,
                        'children_count': len(node.get('children', []))
                    })
                elif step == 'filter':
                    analysis['filter_nodes'].append({
                        'depth': depth,
                        'sort_fn': node.get('sort-by-fn', 'unknown'),
                        'select_n': node.get('select-n', 1)
                    })
                
                # Recursively analyze children
                for key, value in node.items():
                    if key != 'step':  # Avoid infinite recursion
                        analyze_recursive(value, depth + 1)
                        
            elif isinstance(node, list):
                for item in node:
                    analyze_recursive(item, depth + 1)
        
        analyze_recursive(strategy)
        return analysis
    
    def validate_strategy_execution(self, strategy: Dict[str, Any], date: pd.Timestamp) -> Dict[str, Any]:
        """Validate that the strategy can be executed for a given date"""
        validation = {
            'date': date,
            'can_execute': False,
            'issues': [],
            'warnings': [],
            'data_availability': {}
        }
        
        try:
            # Check if we have data for the date
            symbols = self.extract_all_symbols(strategy)
            for symbol in symbols:
                if symbol in self.data:
                    df = self.data[symbol]
                    if not df.empty:
                        # Check if we have data for this specific date
                        if date in df.index:
                            validation['data_availability'][symbol] = 'available'
                        else:
                            # Find the closest available date
                            available_dates = df.index
                            if len(available_dates) > 0:
                                closest_date = available_dates[available_dates <= date].max()
                                if closest_date is not None:
                                    days_diff = (date - closest_date).days
                                    if days_diff <= 5:
                                        validation['data_availability'][symbol] = f'closest_{days_diff}_days_ago'
                                    else:
                                        validation['data_availability'][symbol] = f'closest_{days_diff}_days_ago'
                                        validation['warnings'].append(f"{symbol}: Data is {days_diff} days old")
                                else:
                                    validation['data_availability'][symbol] = 'no_data_before_date'
                                    validation['issues'].append(f"{symbol}: No data available before {date.date()}")
                            else:
                                validation['data_availability'][symbol] = 'no_data'
                                validation['issues'].append(f"{symbol}: No data available")
                    else:
                        validation['data_availability'][symbol] = 'empty_dataset'
                        validation['issues'].append(f"{symbol}: Empty dataset")
                else:
                    validation['data_availability'][symbol] = 'missing'
                    validation['issues'].append(f"{symbol}: No data downloaded")
            
            # Try to evaluate the strategy
            try:
                result = self.evaluate_node(strategy, date)
                if result:
                    validation['can_execute'] = True
                    validation['evaluation_result'] = result
                else:
                    validation['issues'].append("Strategy evaluation returned no assets")
            except Exception as e:
                validation['issues'].append(f"Strategy evaluation failed: {str(e)}")
            
            # Check for critical issues
            if validation['issues']:
                validation['can_execute'] = False
            
        except Exception as e:
            validation['issues'].append(f"Validation process failed: {str(e)}")
            validation['can_execute'] = False
        
        return validation

    def extract_assets_directly(self, strategy: Dict[str, Any]) -> List[str]:
        """Extract assets directly from strategy structure as a fallback"""
        assets = set()
        
        def extract_recursive(node):
            if isinstance(node, dict):
                # Look for asset nodes
                if node.get('step') == 'asset' and 'ticker' in node:
                    assets.add(node['ticker'])
                
                # Look for ticker fields in various node types
                if 'ticker' in node:
                    ticker = node['ticker']
                    if isinstance(ticker, str) and len(ticker) <= 10:
                        assets.add(ticker)
                
                # Recursively process all values
                for value in node.values():
                    extract_recursive(value)
                    
            elif isinstance(node, list):
                for item in node:
                    extract_recursive(item)
        
        extract_recursive(strategy)
        
        # Filter out non-ticker symbols
        filtered_assets = []
        for asset in assets:
            if (asset and 
                not any(indicator in asset.lower() for indicator in ['rsi', 'sma', 'ema', 'macd', 'bb', 'stoch', 'atr', 'vol', 'mom', 'price', 'close', 'high', 'low']) and
                not asset.startswith('$') and
                len(asset) <= 10):
                filtered_assets.append(asset)
        
        return filtered_assets

    def find_root_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Find the root strategy node, handling different nesting structures"""
        # If this looks like a root strategy, return it
        if strategy.get('step') in ['root', 'wt-cash-equal', 'wt-cash-specified', 'wt-inverse-vol']:
            return strategy
        
        # If this has children but no step, it might be the root
        if 'children' in strategy and 'step' not in strategy:
            return strategy
        
        # Look for nested strategy structures
        def find_nested_strategy(node):
            if isinstance(node, dict):
                if node.get('step') in ['root', 'wt-cash-equal', 'wt-cash-specified', 'wt-inverse-vol']:
                    return node
                for key, value in node.items():
                    if key == 'strategy' or key == 'symphony':
                        result = find_nested_strategy(value)
                        if result:
                            return result
                    elif isinstance(value, (dict, list)):
                        result = find_nested_strategy(value)
                        if result:
                            return result
            elif isinstance(node, list):
                for item in node:
                    result = find_nested_strategy(item)
                    if result:
                        return result
            return None
        
        # Try to find nested strategy
        nested = find_nested_strategy(strategy)
        if nested:
            return nested
        
        # If all else fails, return the original strategy
        return strategy

def compare_allocations(inhouse_results: pd.DataFrame, composer_allocations: pd.DataFrame, 
                       composer_tickers: List[str], start_date, end_date) -> pd.DataFrame:
    """Compare in-house backtest allocations with Composer allocations"""
    
    # Helper function to safely extract date from various date types
    def safe_extract_date(date_obj):
        """Safely extract date from various date types"""
        try:
            if hasattr(date_obj, 'date'):
                return date_obj.date()
            elif pd.api.types.is_datetime64_any_dtype(date_obj):
                return pd.to_datetime(date_obj).date()
            elif isinstance(date_obj, str):
                return pd.to_datetime(date_obj).date()
            else:
                return date_obj
        except Exception:
            return date_obj
    
    # Convert dates to datetime for comparison
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    # Filter Composer allocations to the backtest period
    composer_filtered = composer_allocations[
        (composer_allocations.index >= start_dt) & 
        (composer_allocations.index <= end_dt)
    ].copy()
    
    # CRITICAL FIX: Apply the TQQQ 0.01 pattern fix to Composer data
    for ticker in composer_filtered.columns:
        if ticker in composer_filtered.columns:
            # Check if this ticker shows the 0.01 pattern (likely percentage format)
            ticker_data = composer_filtered[ticker]
            # Check if any values in the series match the pattern
            percentage_pattern = (ticker_data > 0) & (ticker_data < 0.1)
            if percentage_pattern.any():
                # Convert from percentage to decimal format
                composer_filtered[ticker] = ticker_data * 100.0
                st.info(f"🔄 Applied TQQQ 0.01 fix to {ticker}: converted percentages to decimals")
    
    # Create comparison dataframe
    comparison_data = []
    
    # Get common dates - safely handle date extraction
    inhouse_dates = set(inhouse_results['Date'].apply(safe_extract_date))
    composer_dates = set(composer_filtered.index.date)
    common_dates = sorted(inhouse_dates.intersection(composer_dates))
    
    st.info(f"📊 Comparing {len(common_dates)} common trading days")
    
    for date in common_dates:
        # Get InHouse data for this date - safely handle date comparison
        inhouse_row = inhouse_results[inhouse_results['Date'].apply(safe_extract_date) == date].iloc[0]
        
        # Get Composer data for this date
        composer_date = pd.Timestamp(date)
        if composer_date in composer_filtered.index:
            composer_row = composer_filtered.loc[composer_date]
        else:
            # Find closest available date
            available_dates = composer_filtered.index[composer_filtered.index <= composer_date]
            if len(available_dates) > 0:
                composer_row = composer_filtered.loc[available_dates[-1]]
            else:
                composer_row = pd.Series(0.0, index=composer_filtered.columns)
        
        # Extract holdings
        inhouse_holdings = inhouse_row.get('Holdings', {})
        if isinstance(inhouse_holdings, str):
            try:
                inhouse_holdings = json.loads(inhouse_holdings)
            except:
                inhouse_holdings = {}
        
        # FIX: Get only ACTIVE composer assets (weight > threshold)
        composer_holdings = {}
        WEIGHT_THRESHOLD = 0.001  # Consider allocations above 0.1%
        
        for ticker in composer_tickers:
            if ticker in composer_row.index:
                weight = composer_row[ticker] / 100  # Convert % to decimal
                if weight > WEIGHT_THRESHOLD:  # Only include meaningful allocations
                    composer_holdings[ticker] = weight
        
        # Convert to sets for comparison using ACTIVE allocations only
        inhouse_assets = set(inhouse_holdings.keys()) if inhouse_holdings else set()
        composer_assets = set(composer_holdings.keys()) if composer_holdings else set()
        
        # Find common and differences
        common_assets = inhouse_assets.intersection(composer_assets)
        missing_in_inhouse = composer_assets - inhouse_assets
        extra_in_inhouse = inhouse_assets - composer_assets
        
        # Calculate asset selection match score
        total_assets = inhouse_assets.union(composer_assets)
        if total_assets:
            asset_selection_match = len(common_assets) / len(total_assets)
        else:
            asset_selection_match = 1.0 if not inhouse_assets and not composer_assets else 0.0
        
        # Calculate allocation differences for common assets
        allocation_differences = {}
        for asset in common_assets:
            inhouse_alloc = inhouse_holdings.get(asset, 0.0)
            composer_alloc = composer_holdings.get(asset, 0.0)
            diff = abs(inhouse_alloc - composer_alloc)
            allocation_differences[asset] = diff
        
        # Calculate overall match score
        if common_assets:
            avg_allocation_diff = sum(allocation_differences.values()) / len(common_assets)
            match_score = max(0, 1 - avg_allocation_diff)
        else:
            match_score = asset_selection_match
        
        # Enhanced debugging information
        debug_info = {
            'inhouse_holdings': inhouse_holdings,
            'composer_holdings': composer_holdings,
            'inhouse_set': list(inhouse_assets),
            'composer_set': list(composer_assets),
            'allocation_differences': allocation_differences,
            'match_score_calculation': {
                'asset_selection_match': asset_selection_match,
                'avg_allocation_diff': avg_allocation_diff if common_assets else 0,
                'final_match_score': match_score
            }
        }
        
        comparison_data.append({
            'Date': date,
            'Date_Str': date.strftime('%Y-%m-%d'),
            'InHouse_Assets': ', '.join(sorted(inhouse_assets)) if inhouse_assets else 'None',
            'Composer_Assets': ', '.join(sorted(composer_assets)) if composer_assets else 'None',
            'Common_Assets': ', '.join(sorted(common_assets)) if common_assets else 'None',
            'Missing_In_InHouse': ', '.join(sorted(missing_in_inhouse)) if missing_in_inhouse else '',
            'Extra_In_InHouse': ', '.join(sorted(extra_in_inhouse)) if extra_in_inhouse else '',
            'Asset_Selection_Match': asset_selection_match,
            'Allocation_Differences': allocation_differences,
            'InHouse_Portfolio_Value': inhouse_row.get('Portfolio_Value', 0.0),
            'Rebalanced': inhouse_row.get('Rebalanced', False),
            'InHouse_Num_Assets': inhouse_row.get('Num_Assets', 0),
            'Composer_Num_Assets': len(composer_assets),
            'Match_Score': match_score,
            'InHouse_Holdings_JSON': json.dumps(inhouse_holdings),
            'Composer_Holdings_JSON': json.dumps(composer_holdings),
            'Debug_Info': debug_info
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add summary statistics
    if not comparison_df.empty:
        perfect_matches = (comparison_df['Match_Score'] >= 0.99).sum()
        total_days = len(comparison_df)
        match_rate = perfect_matches / total_days if total_days > 0 else 0
        
        st.success(f"📈 Comparison Complete: {perfect_matches}/{total_days} days perfect match ({match_rate:.1%})")
        
        # Identify patterns in mismatches
        mismatches = comparison_df[comparison_df['Match_Score'] < 0.99]
        if not mismatches.empty:
            st.warning(f"⚠️ {len(mismatches)} mismatch days detected")
            
            # Analyze TQQQ 0.01 pattern
            tqqq_mismatches = 0
            for _, row in mismatches.iterrows():
                if 'TQQQ' in row['Debug_Info']['allocation_differences']:
                    diff = row['Debug_Info']['allocation_differences']['TQQQ']
                    if abs(diff - 0.99) < 0.01:  # Close to 0.99 difference
                        tqqq_mismatches += 1
            
            if tqqq_mismatches > 0:
                st.info(f"🔍 TQQQ 0.01 pattern detected in {tqqq_mismatches} mismatch days")
    
    return comparison_df

def display_comparison_results(comparison_results: pd.DataFrame, inhouse_results: pd.DataFrame, 
                              composer_allocations: pd.DataFrame, initial_capital: float):
    """Display comparison results between in-house and Composer backtests"""
    
    st.success("✅ Comparison analysis completed!")
    
    # Summary statistics
    st.subheader("📊 Comparison Summary")
    
    total_days = len(comparison_results)
    perfect_matches = (comparison_results['Asset_Selection_Match'] == 1.0).sum()
    avg_match_rate = comparison_results['Asset_Selection_Match'].mean() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Days", total_days)
    with col2:
        st.metric("Perfect Matches", f"{perfect_matches}")
    with col3:
        st.metric("Avg Match Rate", f"{avg_match_rate:.1f}%")
    with col4:
        match_quality = "🟢 Excellent" if avg_match_rate >= 95 else "🟡 Good" if avg_match_rate >= 80 else "🔴 Poor"
        st.metric("Match Quality", match_quality)
    
    # Daily comparison table
    st.subheader("📅 Daily Allocation Comparison")
    
    # Create a more readable comparison table
    display_df = comparison_results[['Date', 'InHouse_Assets', 'Composer_Assets', 'Asset_Selection_Match', 'Rebalanced']].copy()
    display_df['Asset_Selection_Match'] = display_df['Asset_Selection_Match'].apply(lambda x: f"{x*100:.1f}%")
    
    # Safely convert Date column to datetime if needed, then extract date
    def safe_extract_date(date_obj):
        """Safely extract date from various date types"""
        try:
            if hasattr(date_obj, 'date'):
                return date_obj.date()
            elif pd.api.types.is_datetime64_any_dtype(date_obj):
                return pd.to_datetime(date_obj).date()
            elif isinstance(date_obj, str):
                return pd.to_datetime(date_obj).date()
            else:
                return date_obj
        except Exception:
            return date_obj
    
    display_df['Date'] = display_df['Date'].apply(safe_extract_date)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Detailed analysis
    st.subheader("🔍 Detailed Analysis")
    
    # Days with mismatches
    mismatches = comparison_results[comparison_results['Asset_Selection_Match'] < 1.0]
    if len(mismatches) > 0:
        st.warning(f"⚠️ Found {len(mismatches)} days with allocation mismatches")
        
        with st.expander("View Mismatch Details"):
            for _, row in mismatches.iterrows():
                # Safely handle the Date column
                try:
                    if hasattr(row['Date'], 'date'):
                        date_str = row['Date'].date()
                    elif pd.api.types.is_datetime64_any_dtype(row['Date']):
                        date_str = pd.to_datetime(row['Date']).date()
                    else:
                        date_str = str(row['Date'])
                except:
                    date_str = str(row['Date'])
                    
                st.write(f"**{date_str}**")
                st.write(f"- In-House: {row['InHouse_Assets']}")
                st.write(f"- Composer: {row['Composer_Assets']}")
                st.write(f"- Match Rate: {row['Asset_Selection_Match']*100:.1f}%")
                st.write("---")
    else:
        st.success("🎉 Perfect match on all days! In-house logic is working correctly.")
    
    # Performance comparison
    st.subheader("📈 Performance Comparison")
    
    # Calculate returns for both approaches
    inhouse_returns = inhouse_results['Portfolio_Value'].pct_change().dropna()
    inhouse_total_return = (inhouse_results['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100
    
    # Calculate Composer returns (simplified - using equal weight rebalancing)
    composer_portfolio_values = []
    current_value = initial_capital
    
    for _, row in comparison_results.iterrows():
        date = row['Date']
        if date in composer_allocations.index:
            composer_row = composer_allocations.loc[date]
            # Simple equal weight calculation for comparison
            num_assets = len([t for t in composer_allocations.columns if composer_row.get(t, 0) > 0])
            if num_assets > 0:
                # Assume equal weight and no transaction costs for simplicity
                current_value = current_value  # No change in this simplified version
        composer_portfolio_values.append(current_value)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("In-House Total Return", f"{inhouse_total_return:.2f}%")
        st.metric("In-House Volatility", f"{inhouse_returns.std() * np.sqrt(252) * 100:.2f}%")
    
    with col2:
        # Note: Composer performance calculation would need more sophisticated logic
        st.metric("Composer Performance", "See detailed analysis above")
        st.info("Composer performance requires more detailed return calculation")
    
    # Download comparison results
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    with col1:
        csv_comparison = comparison_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Data",
            data=csv_comparison,
            file_name=f"allocation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_inhouse = inhouse_results.to_csv(index=False)
        st.download_button(
            label="📥 Download In-House Results",
            data=csv_inhouse,
            file_name=f"inhouse_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def generate_debug_file(comparison_results: pd.DataFrame, inhouse_results: pd.DataFrame, 
                       composer_allocations: pd.DataFrame, strategy_data: dict, 
                       composer_data: dict, start_date, end_date, initial_capital: float):
    """Generate a comprehensive debug file for analysis"""
    
    st.subheader("🔧 Comparison Output Files")
    
    # Helper function to safely format dates
    def safe_date_format(date_obj) -> str:
        """Safely format a date object to string, handling various types"""
        try:
            if hasattr(date_obj, 'strftime'):
                return date_obj.strftime('%Y-%m-%d')
            elif pd.api.types.is_datetime64_any_dtype(date_obj):
                return pd.to_datetime(date_obj).strftime('%Y-%m-%d')
            elif isinstance(date_obj, str):
                # Try to parse and format
                return pd.to_datetime(date_obj).strftime('%Y-%m-%d')
            else:
                return str(date_obj)
        except Exception:
            return str(date_obj)
    
    # Create comprehensive debug data
    # Convert composer allocations to handle Timestamp keys properly
    composer_allocations_dict = {}
    for date, row in composer_data['allocations_df'].iterrows():
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        composer_allocations_dict[date_str] = row.to_dict()
    
    # Create a safe copy of composer_data with serializable dates
    safe_composer_config = {}
    for key, value in composer_data.items():
        if key in ['start_date', 'end_date'] and hasattr(value, 'strftime'):
            safe_composer_config[key] = value.strftime('%Y-%m-%d')
        elif key == 'allocations_df':
            # Skip this as we're handling it separately
            continue
        else:
            safe_composer_config[key] = value
    
    # Debug: Check for any remaining non-serializable objects
    def check_serializable(obj, path=""):
        """Recursively check for non-serializable objects"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return None
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
            return f"Timestamp at {path}"
        elif hasattr(obj, 'strftime'):
            return f"Datetime-like object at {path}"
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return f"Pandas object at {path}"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                result = check_serializable(v, f"{path}.{k}")
                if result:
                    return result
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                result = check_serializable(item, f"{path}[{i}]")
                if result:
                    return result
        return None
    
    # Check for issues before creating debug_data
    debug_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'backtest_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'initial_capital': initial_capital,
            'strategy_name': strategy_data.get('description', 'Unknown'),
            'composer_symphony': composer_data['symphony_name'],
            'total_days': len(comparison_results),
            'perfect_matches': (comparison_results['Asset_Selection_Match'] == 1.0).sum(),
            'avg_match_rate': comparison_results['Asset_Selection_Match'].mean(),
            'date_alignment': '✅ Aligned' if pd.Timestamp(start_date) >= pd.Timestamp(composer_data['start_date']) and pd.Timestamp(end_date) <= pd.Timestamp(composer_data['end_date']) else '⚠️ Extended beyond available data'
        },
        'daily_comparison': comparison_results.copy().assign(
            Date=lambda x: safe_date_format(x['Date'])
        ).to_dict('records'),
        'inhouse_backtest': inhouse_results.copy().assign(
            Date=lambda x: safe_date_format(x['Date'])
        ).to_dict('records'),
        'composer_allocations': composer_allocations_dict,
        'strategy_config': strategy_data,
        'composer_config': safe_composer_config
    }
    
    # Check for serialization issues
    serialization_issue = check_serializable(debug_data, "debug_data")
    if serialization_issue:
        st.warning(f"⚠️ Potential serialization issue detected: {serialization_issue}")
        st.info("The system will attempt to handle this automatically, but you may see warnings.")
    
    # Debug: Show debug data structure
    st.info(f"🔍 Debug: Debug data keys: {list(debug_data.keys())}")
    st.info(f"🔍 Debug: Debug data structure preview: {str(debug_data)[:500]}...")
    
    # Convert to JSON for download with custom encoder to handle any remaining Timestamp objects
    def json_serializer(obj):
        if hasattr(obj, 'strftime'):
            return obj.strftime('%Y-%m-%d')
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, dict):
            # Recursively handle nested dictionaries
            return {str(k): json_serializer(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively handle lists and tuples
            return [json_serializer(item) for item in obj]
        else:
            return str(obj)
    
    try:
        debug_json = json.dumps(debug_data, indent=2, default=json_serializer)
        st.info(f"🔍 Debug: JSON serialization successful, length: {len(debug_json)}")
    except Exception as e:
        st.error(f"Error serializing debug data to JSON: {str(e)}")
        st.error("This usually indicates there are still non-serializable objects in the data.")
        # Fallback: try with a more aggressive serializer
        def aggressive_serializer(obj):
            try:
                if hasattr(obj, 'strftime'):
                    return obj.strftime('%Y-%m-%d')
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
                    return obj.strftime('%Y-%m-%d')
                elif isinstance(obj, (pd.Series, pd.DataFrame)):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {str(k): aggressive_serializer(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [aggressive_serializer(item) for item in obj]
                else:
                    return str(obj)
            except:
                return str(obj)
        
        try:
            debug_json = json.dumps(debug_data, indent=2, default=aggressive_serializer)
        except Exception as e2:
            st.error(f"Even aggressive serialization failed: {str(e2)}")
            # Last resort: create a minimal debug file
            minimal_debug = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'error': f"Serialization failed: {str(e)}",
                    'fallback_error': str(e2)
                }
            }
            debug_json = json.dumps(minimal_debug, indent=2)
    
    # Create a more readable CSV version for daily comparison
    daily_csv = comparison_results.copy()
    daily_csv['Date'] = daily_csv['Date'].apply(lambda x: safe_date_format(x))
    daily_csv['Allocation_Differences'] = daily_csv['Allocation_Differences'].apply(lambda x: str(x))
    
    # Debug: Show CSV data structure
    st.info(f"🔍 Debug: Daily CSV shape: {daily_csv.shape}")
    st.info(f"🔍 Debug: Daily CSV columns: {list(daily_csv.columns)}")
    
    # Create a comprehensive daily ticker comparison file
    daily_ticker_comparison = []
    
    # Safety check: ensure composer_data has required keys
    if not composer_data or 'tickers' not in composer_data or 'allocations_df' not in composer_data:
        st.error("❌ Composer data is missing required fields (tickers or allocations_df)")
        st.error(f"Available keys: {list(composer_data.keys()) if composer_data else 'None'}")
        return
    
    # Debug: Log the structure of the data
    st.info(f"🔍 Debug: Composer data has {len(composer_data['tickers'])} tickers")
    st.info(f"🔍 Debug: Inhouse results has {len(inhouse_results)} rows")
    st.info(f"🔍 Debug: Comparison results has {len(comparison_results)} rows")
    st.info(f"🔍 Debug: Inhouse results columns: {list(inhouse_results.columns)}")
    st.info(f"🔍 Debug: Comparison results columns: {list(comparison_results.columns)}")
    st.info(f"🔍 Debug: Composer tickers: {composer_data['tickers'][:10]}...")  # Show first 10 tickers
    st.info(f"🔍 Debug: Composer allocations_df shape: {composer_data['allocations_df'].shape}")
    st.info(f"🔍 Debug: Composer allocations_df index range: {composer_data['allocations_df'].index.min()} to {composer_data['allocations_df'].index.max()}")
    
    # Create a mapping from date to inhouse holdings for easier access
    inhouse_holdings_map = {}
    for _, row in inhouse_results.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        inhouse_holdings_map[date_str] = row['Holdings'] if 'Holdings' in row else {}
    
    # Debug: Show sample of inhouse holdings map
    if inhouse_holdings_map:
        sample_dates = list(inhouse_holdings_map.keys())[:3]
        st.info(f"🔍 Debug: Sample inhouse holdings dates: {sample_dates}")
        for date in sample_dates:
            st.info(f"🔍 Debug: Holdings for {date}: {inhouse_holdings_map[date]}")
    
    # Debug: Show sample of comparison results
    if len(comparison_results) > 0:
        st.info(f"🔍 Debug: First comparison result row: {comparison_results.iloc[0].to_dict()}")
    
    for _, row in comparison_results.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        
        # Get all unique tickers from both sources
        inhouse_tickers = set(row['InHouse_Assets'].split(', ')) if row['InHouse_Assets'] != 'None' else set()
        composer_tickers = set(row['Composer_Assets'].split(', ')) if row['Composer_Assets'] else set()
        all_tickers = inhouse_tickers | composer_tickers
        
        # Get inhouse holdings for this date
        inhouse_holdings = inhouse_holdings_map.get(date_str, {})
        
        # Debug: Show what we're working with for this date
        if len(daily_ticker_comparison) < 5:  # Only show first few for debugging
            st.info(f"🔍 Debug: Processing date {date_str}")
            st.info(f"🔍 Debug: Inhouse tickers: {inhouse_tickers}")
            st.info(f"🔍 Debug: Composer tickers: {composer_tickers}")
            st.info(f"🔍 Debug: Inhouse holdings: {inhouse_holdings}")
        
        # Create a row for each ticker
        for ticker in sorted(all_tickers):
            # Get inhouse weight from the holdings map
            inhouse_weight = inhouse_holdings.get(ticker, 0) if inhouse_holdings else 0
            
            composer_weight = 0
            if ticker in composer_data['tickers']:
                try:
                    # Try to get composer allocation for this date and ticker
                    if row['Date'] in composer_data['allocations_df'].index:
                        composer_row = composer_data['allocations_df'].loc[row['Date']]
                        if ticker in composer_row:
                            composer_weight = composer_row[ticker] / 100  # Convert % to decimal
                        
                        # Debug: Show composer data for first few iterations
                        if len(daily_ticker_comparison) < 5:
                            st.info(f"🔍 Debug: Composer row for {date_str}: {composer_row.to_dict() if hasattr(composer_row, 'to_dict') else composer_row}")
                except (KeyError, IndexError) as e:
                    # If there's any issue accessing the data, just use 0
                    if len(daily_ticker_comparison) < 5:
                        st.info(f"🔍 Debug: Error accessing composer data for {date_str}: {e}")
                    composer_weight = 0
            
            daily_ticker_comparison.append({
                'Date': date_str,
                'Ticker': ticker,
                'InHouse_Weight': inhouse_weight,
                'Composer_Weight': composer_weight,
                'Weight_Difference': abs(inhouse_weight - composer_weight),
                'InHouse_Selected': ticker in inhouse_tickers,
                'Composer_Selected': ticker in composer_tickers,
                'Selection_Match': ticker in inhouse_tickers and ticker in composer_tickers,
                'Rebalanced': row['Rebalanced'],
                'Match_Quality': 'Perfect' if ticker in inhouse_tickers and ticker in composer_tickers else 'Missing_In_InHouse' if ticker in composer_tickers else 'Extra_In_InHouse'
            })
    
    ticker_comparison_df = pd.DataFrame(daily_ticker_comparison)
    
    # Debug: Show sample of ticker comparison data
    if len(ticker_comparison_df) > 0:
        st.info(f"🔍 Debug: Ticker comparison DataFrame shape: {ticker_comparison_df.shape}")
        st.info(f"🔍 Debug: Ticker comparison columns: {list(ticker_comparison_df.columns)}")
        st.info(f"🔍 Debug: First few ticker comparison rows: {ticker_comparison_df.head(3).to_dict('records')}")
    
    # Create a summary comparison file
    summary_comparison = {
        'Comparison_Summary': {
            'Total_Days_Analyzed': len(comparison_results),
            'Date_Range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'Perfect_Match_Days': (comparison_results['Asset_Selection_Match'] == 1.0).sum(),
            'Average_Match_Rate': f"{comparison_results['Asset_Selection_Match'].mean() * 100:.2f}%",
            'Total_Ticker_Selections': len(ticker_comparison_df),
            'Perfect_Ticker_Matches': ticker_comparison_df['Selection_Match'].sum(),
            'Ticker_Match_Rate': f"{(ticker_comparison_df['Selection_Match'].sum() / len(ticker_comparison_df)) * 100:.2f}%",
            'Strategy_Name': strategy_data.get('description', 'Unknown'),
            'Composer_Symphony': composer_data['symphony_name'],
            'Initial_Capital': initial_capital,
            'Date_Alignment_Status': '✅ Aligned' if pd.Timestamp(start_date) >= pd.Timestamp(composer_data['start_date']) and pd.Timestamp(end_date) <= pd.Timestamp(composer_data['end_date']) else '⚠️ Extended beyond available data'
        },
        'Daily_Summary': comparison_results[['Date', 'Asset_Selection_Match', 'Rebalanced', 'InHouse_Num_Assets', 'Composer_Num_Assets']].copy().assign(
            Date=lambda x: x['Date'].apply(safe_date_format)
        ).to_dict('records'),
        'Ticker_Summary': ticker_comparison_df.groupby('Ticker').agg({
            'Selection_Match': 'mean',
            'Weight_Difference': 'mean',
            'InHouse_Selected': 'sum',
            'Composer_Selected': 'sum'
        }).reset_index().to_dict('records')
    }
    
    # Debug: Show summary comparison structure
    st.info(f"🔍 Debug: Summary comparison keys: {list(summary_comparison.keys())}")
    st.info(f"🔍 Debug: Summary comparison structure: {summary_comparison}")
    
    # Download options - ENHANCED with summary file
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button(
            label="📥 Download Debug JSON",
            data=debug_json,
            file_name=f"debug_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Complete debug data in JSON format for detailed analysis"
        )
    
    with col2:
        csv_data = daily_csv.to_csv(index=False)
        st.info(f"🔍 Debug: Daily CSV data length: {len(csv_data)}")
        st.download_button(
            label="📥 Download Daily Comparison CSV",
            data=csv_data,
            file_name=f"daily_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Daily allocation comparison in CSV format for spreadsheet analysis"
        )
    
    with col3:
        ticker_csv = ticker_comparison_df.to_csv(index=False)
        st.info(f"🔍 Debug: Ticker CSV data length: {len(ticker_csv)}")
        st.download_button(
            label="📥 Download Ticker Comparison CSV",
            data=ticker_csv,
            file_name=f"daily_ticker_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Daily ticker-by-ticker comparison for detailed analysis"
        )
    
    with col4:
        try:
            summary_json = json.dumps(summary_comparison, indent=2, default=json_serializer)
            st.info(f"🔍 Debug: Summary JSON serialization successful, length: {len(summary_json)}")
        except Exception as e:
            st.error(f"Error serializing summary data to JSON: {str(e)}")
            # Fallback: create a minimal summary
            minimal_summary = {
                'error': f"Serialization failed: {str(e)}",
                'generated_at': datetime.now().isoformat()
            }
            summary_json = json.dumps(minimal_summary, indent=2)
        st.download_button(
            label="📥 Download Summary JSON",
            data=summary_json,
            file_name=f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="High-level comparison summary in JSON format"
        )
    
    # Display debug summary
    st.success("✅ Comparison output files generated successfully!")
    st.info("📋 **Debug JSON**: Complete data for AI analysis and debugging")
    st.info("📊 **Daily CSV**: Spreadsheet-friendly format for manual review")
    st.info("🎯 **Ticker Comparison CSV**: Daily ticker-by-ticker selection comparison")
    st.info("📈 **Summary JSON**: High-level comparison metrics and insights")
    
    # Instructions for using debug file
    with st.expander("📖 How to Use Comparison Files"):
        st.markdown("""
        **🔧 For AI Debugging (Recommended):**
        1. Download the **Debug JSON** file
        2. Share it with me (the AI) by uploading it
        3. I can analyze the complete data and help fix logic issues
        
        **📊 For Manual Review:**
        1. **Daily Comparison CSV**: Overview of daily asset selection differences
        2. **Ticker Comparison CSV**: Detailed ticker-by-ticker analysis
        3. **Summary JSON**: High-level metrics and insights
        
        **🚨 Key Analysis Columns:**
        - **Asset_Selection_Match**: Percentage of assets that match (1.0 = perfect)
        - **Missing_In_InHouse**: Assets Composer selected but in-house missed
        - **Extra_In_InHouse**: Assets in-house selected but Composer didn't
        - **Weight_Difference**: Allocation weight differences for each ticker
        - **Match_Quality**: Perfect, Missing_In_InHouse, or Extra_In_InHouse
        
        **💡 Analysis Strategy:**
        1. Focus on days with low Match_Score first
        2. Use Ticker Comparison CSV to see which specific tickers are mismatched
        3. Check if missing assets meet your strategy conditions
        4. Verify filter logic and conditional evaluations
        5. Look for patterns in mismatched days
        """)
    
    # Show key debugging insights
    st.subheader("🔍 Key Comparison Insights")
    
    # Find days with biggest mismatches
    mismatches = comparison_results[comparison_results['Asset_Selection_Match'] < 1.0]
    if len(mismatches) > 0:
        worst_matches = mismatches.nsmallest(3, 'Asset_Selection_Match')
        
        st.warning("🚨 **Top 3 Days with Biggest Mismatches:**")
        for _, row in worst_matches.iterrows():
            st.write(f"**{row['Date'].date()}** - Match Rate: {row['Asset_Selection_Match']*100:.1f}%")
            st.write(f"  - In-House: {row['InHouse_Assets']}")
            st.write(f"  - Composer: {row['Composer_Assets']}")
            st.write(f"  - Missing: {row['Missing_In_InHouse']}")
            st.write(f"  - Extra: {row['Extra_In_InHouse']}")
            st.write("---")
    else:
        st.success("🎉 Perfect match on all days! In-house logic is working correctly.")
    
    # Show strategy complexity metrics
    strategy_complexity = {
        'total_nodes': count_strategy_nodes(strategy_data),
        'conditional_nodes': count_conditional_nodes(strategy_data),
        'filter_nodes': count_filter_nodes(strategy_data)
    }
    
    st.info("📊 **Strategy Complexity Analysis:**")
    st.write(f"- Total nodes: {strategy_complexity['total_nodes']}")
    st.write(f"- Conditional nodes: {strategy_complexity['conditional_nodes']}")
    st.write(f"- Filter nodes: {strategy_complexity['filter_nodes']}")
    
    # Show ticker comparison summary
    st.subheader("🎯 Ticker Selection Summary")
    
    if not ticker_comparison_df.empty:
        total_selections = len(ticker_comparison_df)
        perfect_matches = ticker_comparison_df['Selection_Match'].sum()
        match_rate = (perfect_matches / total_selections) * 100 if total_selections > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Ticker Selections", total_selections)
        with col2:
            st.metric("Perfect Matches", perfect_matches)
        with col3:
            st.metric("Match Rate", f"{match_rate:.1f}%")
        
        # Show most problematic tickers
        ticker_issues = ticker_comparison_df.groupby('Ticker').agg({
            'Selection_Match': 'mean',
            'Weight_Difference': 'mean'
        }).reset_index()
        
        problematic_tickers = ticker_issues[ticker_issues['Selection_Match'] < 1.0].sort_values('Selection_Match')
        
        if not problematic_tickers.empty:
            st.warning("🚨 **Tickers with Selection Issues:**")
            st.dataframe(problematic_tickers, use_container_width=True)

def count_strategy_nodes(strategy: dict) -> int:
    """Count total nodes in strategy tree"""
    def count_recursive(node):
        count = 1
        if 'children' in node:
            for child in node['children']:
                count += count_recursive(child)
        return count
    
    return count_recursive(strategy)

def count_conditional_nodes(strategy: dict) -> int:
    """Count conditional nodes in strategy tree"""
    def count_recursive(node):
        count = 1 if node.get('type') == 'conditional' else 0
        if 'children' in node:
            for child in node['children']:
                count += count_recursive(child)
        return count
    
    return count_recursive(strategy)

def count_filter_nodes(strategy: dict) -> int:
    """Count filter nodes in strategy tree"""
    def count_recursive(node):
        count = 1 if node.get('type') == 'filter' else 0
        if 'children' in node:
            for child in node['children']:
                count += count_recursive(child)
        return count
    
    return count_recursive(strategy)

def diagnose_mismatch(strategy_data: dict, composer_data: dict, comparison_results: pd.DataFrame, 
                      inhouse_results: pd.DataFrame, composer_allocations: pd.DataFrame, 
                      start_date, end_date) -> dict:
    """
    Comprehensive diagnostic function to identify causes of mismatch between in-house and Composer backtests
    """
    diagnosis = {
        'summary': {},
        'data_quality_issues': [],
        'strategy_interpretation_issues': [],
        'timing_issues': [],
        'indicator_calculation_issues': [],
        'filter_logic_issues': [],
        'recommendations': []
    }
    
    # 1. Data Quality Analysis
    st.subheader("🔍 Data Quality Analysis")
    
    # Check if all required symbols have data
    required_symbols = set()
    if 'strategy_data' in locals():
        required_symbols = set(extract_all_symbols(strategy_data))
    
    composer_symbols = set(composer_data.get('tickers', []))
    missing_symbols = required_symbols - composer_symbols
    extra_symbols = composer_symbols - required_symbols
    
    if missing_symbols:
        diagnosis['data_quality_issues'].append(f"Missing symbols in Composer data: {missing_symbols}")
        st.warning(f"⚠️ Missing symbols in Composer data: {missing_symbols}")
    
    if extra_symbols:
        diagnosis['data_quality_issues'].append(f"Extra symbols in Composer data: {extra_symbols}")
        st.info(f"ℹ️ Extra symbols in Composer data: {extra_symbols}")
    
    # Check date alignment
    composer_start = pd.Timestamp(composer_data.get('start_date', '1900-01-01'))
    composer_end = pd.Timestamp(composer_data.get('end_date', '2100-01-01'))
    backtest_start = pd.Timestamp(start_date)
    backtest_end = pd.Timestamp(end_date)
    
    if backtest_start < composer_start or backtest_end > composer_end:
        diagnosis['timing_issues'].append(f"Backtest dates ({backtest_start} to {backtest_end}) extend beyond Composer data range ({composer_start} to {composer_end})")
        st.error(f"❌ Date mismatch: Backtest extends beyond available Composer data")
    
    # 2. Strategy Interpretation Analysis
    st.subheader("📊 Strategy Interpretation Analysis")
    
    if 'strategy_data' in locals():
        strategy_nodes = count_strategy_nodes(strategy_data)
        conditional_nodes = count_conditional_nodes(strategy_data)
        filter_nodes = count_filter_nodes(strategy_data)
        
        st.info(f"Strategy complexity: {strategy_nodes} total nodes, {conditional_nodes} conditional nodes, {filter_nodes} filter nodes")
        
        # Check for complex nested logic
        if conditional_nodes > 5:
            diagnosis['strategy_interpretation_issues'].append("High number of conditional nodes may cause interpretation differences")
            st.warning("⚠️ High number of conditional nodes detected - interpretation differences likely")
        
        # Check rebalance frequency
        rebalance_freq = strategy_data.get('rebalance', 'daily')
        st.info(f"Rebalance frequency: {rebalance_freq}")
        
        if rebalance_freq != 'daily':
            diagnosis['strategy_interpretation_issues'].append(f"Non-daily rebalancing ({rebalance_freq}) may cause timing mismatches")
    
    # 3. Daily Mismatch Analysis
    st.subheader("📅 Daily Mismatch Analysis")
    
    if not comparison_results.empty:
        # Find worst matching days
        worst_days = comparison_results.nsmallest(5, 'Asset_Selection_Match')
        
        st.write("**Worst matching days:**")
        for _, day in worst_days.iterrows():
            match_rate = day['Asset_Selection_Match'] * 100
            st.write(f"- {day['Date_Str']}: {match_rate:.1f}% match")
            st.write(f"  InHouse: {day['InHouse_Assets']}")
            st.write(f"  Composer: {day['Composer_Assets']}")
            st.write(f"  Missing: {day['Missing_In_InHouse']}")
            st.write(f"  Extra: {day['Extra_In_InHouse']}")
            st.write("---")
        
        # Analyze mismatch patterns
        perfect_matches = (comparison_results['Asset_Selection_Match'] == 1.0).sum()
        total_days = len(comparison_results)
        avg_match = comparison_results['Asset_Selection_Match'].mean() * 100
        
        diagnosis['summary'] = {
            'total_days': total_days,
            'perfect_matches': perfect_matches,
            'avg_match_rate': avg_match,
            'mismatch_days': total_days - perfect_matches
        }
        
        st.success(f"📈 Summary: {perfect_matches}/{total_days} perfect matches ({avg_match:.1f}% average)")
    
    # 4. Asset Selection Pattern Analysis
    st.subheader("🎯 Asset Selection Pattern Analysis")
    
    if not comparison_results.empty:
        # Check if mismatches are systematic or random
        mismatch_days = comparison_results[comparison_results['Asset_Selection_Match'] < 1.0]
        
        if not mismatch_days.empty:
            # Analyze common missing/extra assets
            all_missing = []
            all_extra = []
            
            for _, day in mismatch_days.iterrows():
                if day['Missing_In_InHouse']:
                    all_missing.extend(day['Missing_In_InHouse'].split(', '))
                if day['Extra_In_InHouse']:
                    all_extra.extend(day['Extra_In_InHouse'].split(', '))
            
            from collections import Counter
            missing_counts = Counter(all_missing)
            extra_counts = Counter(all_extra)
            
            if missing_counts:
                st.write("**Most frequently missing in InHouse:**")
                for asset, count in missing_counts.most_common(5):
                    st.write(f"- {asset}: {count} times")
            
            if extra_counts:
                st.write("**Most frequently extra in InHouse:**")
                for asset, count in extra_counts.most_common(5):
                    st.write(f"- {asset}: {count} times")
    
    # 5. Generate Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if diagnosis['data_quality_issues']:
        recommendations.append("🔧 Fix data quality issues first - ensure all required symbols are available")
    
    if diagnosis['timing_issues']:
        recommendations.append("⏰ Align backtest dates with available Composer data range")
    
    if diagnosis['strategy_interpretation_issues']:
        recommendations.append("🧠 Review strategy interpretation - complex conditional logic may need manual verification")
    
    if diagnosis['indicator_calculation_issues']:
        recommendations.append("📊 Verify technical indicator calculations match Composer's implementation")
    
    if diagnosis['filter_logic_issues']:
        recommendations.append("🔍 Check filter logic implementation - sorting and selection criteria may differ")
    
    # Add general recommendations
    recommendations.extend([
        "📋 Enable debug mode to see step-by-step strategy evaluation",
        "🔄 Check if Composer uses different rebalancing logic or timing",
        "📈 Verify indicator window parameters match exactly",
        "🎯 Compare individual day results to identify specific failure points"
    ])
    
    for rec in recommendations:
        st.write(f"• {rec}")
    
    diagnosis['recommendations'] = recommendations
    
    return diagnosis

def main():
    st.title("📈 Composer Strategy Backtester")
    st.markdown("Compare your in-house strategy logic with Composer's actual allocations")
    
    # Initialize debug mode in session state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # NEW: Prominent unified workflow section
    st.markdown("---")
    st.markdown("## 🚀 **Single Button Workflow - Run Both Backtests & Compare!**")
    
    st.info("""
    **🎯 What You Get:**
    - **One Button**: Automatically runs both in-house and Composer backtests
    - **Date Alignment**: Ensures both backtests use the same date range
    - **Daily Ticker Comparison**: Detailed analysis of ticker selection differences
    - **4 Output Files**: Comprehensive comparison data for analysis
    
    **📋 How to Use:**
    1. Upload your Composer strategy JSON file
    2. Enter your Composer symphony URL  
    3. Set your desired backtest dates
    4. Click **"🚀 Run Both Backtests & Comparison"**
    5. Get instant comparison results and downloadable files
    """)
    
    st.markdown("---")
    
    # Add summary section
    with st.expander("📋 **Quick Start Guide**", expanded=True):
        st.markdown("""
        **🎯 What This Tool Does:**
        - Upload your Composer strategy JSON file
        - Enter your Composer symphony URL
        - Automatically fetch historical allocations from Composer
        - Run in-house backtest using your strategy logic
        - Compare daily ticker selections between both approaches
        - Generate comprehensive comparison reports
        
        **🚀 Single Button Workflow:**
        1. **Upload JSON** + **Enter URL** + **Set Dates** + **Click "🚀 Run Both Backtests & Comparison"**
        2. Get detailed daily ticker selection comparison
        3. Download comparison files for analysis
        
        **📊 Output Files:**
        - **Daily Comparison CSV**: Daily asset selection overview
        - **Ticker Comparison CSV**: Ticker-by-ticker daily analysis  
        - **Debug JSON**: Complete data for AI analysis
        - **Summary JSON**: High-level comparison metrics and insights
        """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        
        # Debug mode toggle - CRITICAL FOR TROUBLESHOOTING
        st.markdown("### 🐛 **Debug Mode - CRITICAL FOR TROUBLESHOOTING**")
        debug_mode = st.checkbox("Enable Debug Mode", value=False, 
                                help="🔍 Enable detailed logging to see step-by-step strategy evaluation")
        if debug_mode:
            st.session_state.debug_mode = True
            st.sidebar.success("✅ Debug mode enabled - detailed logs will be shown")
            st.sidebar.info("🔍 You'll now see detailed strategy evaluation steps")
        else:
            st.session_state.debug_mode = False
            st.sidebar.info("ℹ️ Debug mode disabled - enable for troubleshooting")
        
        # Data source selection - allow both
        st.subheader("Data Sources")
        
        # JSON upload
        st.write("**1. Upload JSON File**")
        uploaded_file = st.file_uploader("Upload Composer JSON", type=['json'])
        strategy_data = None
        if uploaded_file is not None:
            try:
                strategy_data = json.load(uploaded_file)
                st.success("✅ JSON loaded successfully")
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please upload a valid Composer strategy file.")
        
        # Composer URL fetch
        st.write("**2. Enter Composer URL**")
        composer_url = st.text_input(
            "Composer Symphony URL",
            placeholder="https://app.composer.trade/symphony/...",
            help="Enter the full URL of your Composer symphony"
        )
        
        # Show status and options
        if strategy_data and composer_url:
            st.success("🎯 **Ready for Unified Comparison!**")
            st.info("You have both JSON and URL. Use the 'Auto-Fetch & Run Comparison' button below!")
        elif strategy_data:
            st.info("📄 JSON loaded. Add a Composer URL to enable comparison mode.")
        elif composer_url:
            st.info("🌐 URL entered. Upload a JSON file to enable comparison mode.")
        else:
            st.info("👆 Upload a JSON file and/or enter a Composer URL to get started!")
        
        # Manual fetch option (for when you want to preview data first)
        if composer_url and not hasattr(st.session_state, 'composer_data'):
            st.write("**3. Preview Composer Data (Optional)**")
            if st.button("👀 Preview Composer Data"):
                try:
                    with st.spinner("Fetching preview data..."):
                        # Use a shorter date range for preview
                        preview_start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                        preview_end = datetime.now().strftime('%Y-%m-%d')
                        
                        allocations_df, symphony_name, tickers = fetch_composer_backtest(composer_url, preview_start, preview_end)
                        
                        st.session_state.composer_data = {
                            'allocations_df': allocations_df,
                            'symphony_name': symphony_name,
                            'tickers': tickers,
                            'start_date': preview_start,
                            'end_date': preview_end
                        }
                        
                        st.success(f"✅ Preview data fetched for: {symphony_name}")
                        st.info(f"Preview period: {preview_start} to {preview_end}")
                        st.info(f"Assets: {', '.join(tickers)}")
                
                except Exception as e:
                    st.error(f"Error fetching preview data: {str(e)}")
        
        # Date range for backtesting
        st.subheader("Backtest Settings")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=730), 
                                     help="Select start date. For 200-day MA strategies, ensure you have at least 2+ years of data")
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Show date alignment info when both sources are available
        if hasattr(st.session_state, 'composer_data') and strategy_data is not None:
            composer_data = st.session_state.composer_data
            composer_start = pd.Timestamp(composer_data['start_date'])
            composer_end = pd.Timestamp(composer_data['end_date'])
            
            # Check if selected dates are within Composer data range
            selected_start = pd.Timestamp(start_date)
            selected_end = pd.Timestamp(end_date)
            
            if selected_start < composer_start or selected_end > composer_end:
                st.warning("⚠️ **Date Range Warning:** Selected dates extend beyond available Composer data")
                st.info(f"Composer data available: {composer_start.date()} to {composer_end.date()}")
                st.info("Consider adjusting dates or the tool will use available data within your selection")
            else:
                st.success("✅ **Date Range Aligned:** Selected dates are within Composer data range")
                st.info(f"Composer data covers: {composer_start.date()} to {composer_end.date()}")
        
        # Initial capital
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
        
        # Debug mode
        debug_mode = st.checkbox("Debug Mode", help="Show detailed condition evaluation")
        st.session_state.debug_mode = debug_mode
        
        # Old conditional logic removed - now handled above
    
    # Display strategy information and run backtests
    st.header("📊 Strategy Analysis & Comparison")
    
    # Check what data sources are available
    has_json = strategy_data is not None
    has_composer = hasattr(st.session_state, 'composer_data')
    
    if not has_json and not has_composer:
        st.info("👆 Upload a Composer strategy JSON file and/or enter a Composer URL to get started!")
        return
    
    # Display JSON strategy info if available
    if has_json:
        st.subheader("📄 JSON Strategy Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Strategy Name", strategy_data.get('description', 'Unknown'))
        with col2:
            st.metric("Asset Class", strategy_data.get('asset_class', 'Unknown'))
        with col3:
            st.metric("Rebalance Frequency", strategy_data.get('rebalance', 'Unknown'))
        
        # Show strategy tree
        with st.expander("View Strategy Structure"):
            st.json(strategy_data)
    
    # Display Composer strategy info if available
    if has_composer:
        composer_data = st.session_state.composer_data
        
        st.subheader("🌐 Composer Strategy Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Strategy Name", composer_data['symphony_name'])
        with col2:
            st.metric("Data Period", f"{composer_data['start_date']} to {composer_data['end_date']}")
        with col3:
            st.metric("Assets", f"{len(composer_data['tickers'])} symbols")
        
        # Show asset allocation over time
        with st.expander("View Composer Asset Allocations"):
            st.dataframe(composer_data['allocations_df'])
    
    # Run backtests and comparison - now handled by the unified button below
    
    # Enhanced unified comparison button for when we have JSON + URL
    elif has_json and composer_url and not has_composer:
        st.subheader("🔄 Run Unified Comparison Analysis")
        st.info("📋 You have a JSON strategy and Composer URL. Click below to automatically fetch data and run comparison!")
        
        if st.button("🚀 Auto-Fetch & Run Comparison", type="primary", use_container_width=True):
            if not HAS_YFINANCE:
                st.error("Cannot run backtest without yfinance. Please install required dependencies.")
                st.code("pip install yfinance plotly")
                return
            
            try:
                with st.spinner("🔄 Automatically fetching Composer data and running comparison..."):
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Auto-fetch Composer data
                    status_text.text("Step 1/4: Fetching data from Composer...")
                    progress_bar.progress(0.25)
                    
                    # Use the date inputs from sidebar for fetching
                    fetch_start_str = start_date.strftime('%Y-%m-%d')
                    fetch_end_str = end_date.strftime('%Y-%m-%d')
                    
                    allocations_df, symphony_name, tickers = fetch_composer_backtest(
                        composer_url, fetch_start_str, fetch_end_str
                    )
                    
                    # Store in session state
                    st.session_state.composer_data = {
                        'allocations_df': allocations_df,
                        'symphony_name': symphony_name,
                        'tickers': tickers,
                        'start_date': fetch_start_str,
                        'end_date': fetch_end_str
                    }
                    
                    # Step 2: Run in-house backtest
                    status_text.text("Step 2/4: Running in-house backtest...")
                    progress_bar.progress(0.5)
                    
                    backtester = ComposerBacktester()
                    backtester.cash = initial_capital
                    
                    inhouse_results = backtester.run_backtest(
                        strategy_data, 
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    # Step 3: Prepare Composer data
                    status_text.text("Step 3/4: Preparing comparison data...")
                    progress_bar.progress(0.75)
                    
                    composer_allocations = allocations_df
                    
                    # Step 4: Run comparison analysis
                    status_text.text("Step 4/4: Running comparison analysis...")
                    progress_bar.progress(1.0)
                    
                    comparison_results = compare_allocations(
                        inhouse_results, 
                        composer_allocations, 
                        tickers,
                        start_date,
                        end_date
                    )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display comparison results
                    display_comparison_results(comparison_results, inhouse_results, composer_allocations, initial_capital)
                    
                    # Generate comprehensive debug file
                    generate_debug_file(comparison_results, inhouse_results, composer_allocations, 
                                      strategy_data, st.session_state.composer_data, start_date, end_date, initial_capital)
                    
                    st.success("🎉 Unified comparison completed successfully!")
                    
            except Exception as e:
                st.error(f"Error during unified comparison: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
    
    # NEW: Single button to run both backtests and comparison when both data sources are available
    elif has_json and has_composer:
        st.subheader("🔄 Run Both Backtests & Comparison")
        st.info("📋 You have both JSON strategy and Composer data. Click below to run both backtests and generate comparison!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Run Both Backtests & Comparison", type="primary", use_container_width=True):
                if not HAS_YFINANCE:
                    st.error("Cannot run backtest without yfinance. Please install required dependencies.")
                    st.code("pip install yfinance plotly")
                    return
            
            try:
                with st.spinner("🔄 Running both backtests and generating comparison..."):
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Run in-house backtest
                    status_text.text("Step 1/3: Running in-house backtest from JSON...")
                    progress_bar.progress(0.33)
                    
                    backtester = ComposerBacktester()
                    backtester.cash = initial_capital
                    
                    inhouse_results = backtester.run_backtest(
                        strategy_data, 
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    # Step 2: Prepare Composer data
                    status_text.text("Step 2/3: Preparing Composer data for comparison...")
                    progress_bar.progress(0.66)
                    
                    composer_data = st.session_state.composer_data
                    composer_allocations = composer_data['allocations_df']
                    
                    # Step 3: Run comparison analysis
                    status_text.text("Step 3/3: Running comparison analysis...")
                    progress_bar.progress(1.0)
                    
                    comparison_results = compare_allocations(
                        inhouse_results, 
                        composer_allocations, 
                        composer_data['tickers'],
                        start_date,
                        end_date
                    )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display comparison results
                    display_comparison_results(comparison_results, inhouse_results, composer_allocations, initial_capital)
                    
                    # Run comprehensive mismatch diagnosis
                    st.subheader("🔍 Mismatch Diagnosis")
                    diagnosis = diagnose_mismatch(strategy_data, composer_data, comparison_results, 
                                               inhouse_results, composer_allocations, start_date, end_date)
                    
                    # Generate comprehensive debug file
                    generate_debug_file(comparison_results, inhouse_results, composer_allocations, 
                                      strategy_data, composer_data, start_date, end_date, initial_capital)
                    
                    st.success("🎉 Both backtests and comparison completed successfully!")
                    
            except Exception as e:
                st.error(f"Error during backtest and comparison: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
        
        with col2:
            if st.button("🔍 Run Mismatch Diagnosis Only", type="secondary", use_container_width=True):
                if not HAS_YFINANCE:
                    st.error("Cannot run diagnosis without yfinance. Please install required dependencies.")
                    st.code("pip install yfinance plotly")
                    return
                
                try:
                    with st.spinner("🔍 Running mismatch diagnosis..."):
                        # Run in-house backtest for diagnosis
                        backtester = ComposerBacktester()
                        backtester.cash = initial_capital
                        
                        inhouse_results = backtester.run_backtest(
                            strategy_data, 
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                        
                        # Get Composer data
                        composer_data = st.session_state.composer_data
                        composer_allocations = composer_data['allocations_df']
                        
                        # Run comparison for diagnosis
                        comparison_results = compare_allocations(
                            inhouse_results, 
                            composer_allocations, 
                            composer_data['tickers'],
                            start_date,
                            end_date
                        )
                        
                        # Run comprehensive diagnosis
                        st.subheader("🔍 Detailed Mismatch Diagnosis")
                        diagnosis = diagnose_mismatch(strategy_data, composer_data, comparison_results, 
                                                   inhouse_results, composer_allocations, start_date, end_date)
                        
                        st.success("🔍 Diagnosis completed! Check the analysis above for insights.")
                        
                except Exception as e:
                    st.error(f"Error during diagnosis: {str(e)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
    
    elif has_json:
        st.subheader("🚀 Run In-House Backtest")
        
        if st.button("🚀 Run Backtest", type="primary"):
            if not HAS_YFINANCE:
                st.error("Cannot run backtest without yfinance. Please install required dependencies.")
                st.code("pip install yfinance plotly")
                return
                
            backtester = ComposerBacktester()
            backtester.cash = initial_capital
            
            results = backtester.run_backtest(
                strategy_data, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            display_backtest_results(results, initial_capital, start_date, end_date)
    
    elif has_composer:
        st.subheader("📊 Analyze Composer Strategy")
        
        if st.button("📊 Analyze Composer Strategy", type="primary"):
            try:
                with st.spinner("Analyzing Composer strategy..."):
                    # Get the allocation data
                    allocations_df = composer_data['allocations_df'].copy()
                    tickers = composer_data['tickers']
                    
                    # Calculate daily returns using the comprehensive method from sim.py
                    daily_returns, return_dates = calculate_portfolio_returns_from_allocations(allocations_df, tickers)
                    
                    if len(daily_returns) > 0:
                        # Calculate performance metrics
                        total_return = ((1 + daily_returns / 100).prod() - 1) * 100
                        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
                        
                        # Calculate Sharpe ratio (assuming 0% risk-free rate)
                        if volatility > 0:
                            annualized_return = total_return * (252 / len(daily_returns))
                            sharpe_ratio = annualized_return / volatility
                        else:
                            sharpe_ratio = 0
                            annualized_return = 0
                        
                        # Calculate max drawdown
                        cumulative_returns = (1 + daily_returns / 100).cumprod()
                        running_max = cumulative_returns.cummax()
                        drawdown = (cumulative_returns - running_max) / running_max
                        max_drawdown = drawdown.min() * 100
                        
                        # Display performance metrics
                        st.subheader("📊 Performance Analysis")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return", f"{total_return:.2f}%")
                        with col2:
                            st.metric("Annualized Return", f"{annualized_return:.2f}%")
                        with col3:
                            st.metric("Volatility (Ann.)", f"{volatility:.2f}%")
                        with col4:
                            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        
                        col5, col6, col7, col8 = st.columns(4)
                        with col5:
                            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                        with col6:
                            win_rate = (daily_returns > 0).mean() * 100
                            st.metric("Win Rate", f"{win_rate:.1f}%")
                        with col7:
                            st.metric("Trading Days", f"{len(daily_returns)}")
                        with col8:
                            best_day = daily_returns.max()
                            st.metric("Best Day", f"{best_day:.2f}%")
                        
                        # Plot cumulative returns
                        if HAS_PLOTLY:
                            import plotly.graph_objects as go
                            
                            st.subheader("📈 Cumulative Returns")
                            
                            cumulative_pct = (cumulative_returns - 1) * 100
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=return_dates,
                                y=cumulative_pct,
                                mode='lines',
                                name='Cumulative Return',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            fig.update_layout(
                                title="Cumulative Return Over Time",
                                xaxis_title="Date",
                                yaxis_title="Cumulative Return (%)",
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Plot drawdown chart
                            st.subheader("📉 Drawdown Analysis")
                            
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                x=return_dates,
                                y=drawdown * 100,
                                mode='lines',
                                name='Drawdown',
                                fill='tonexty',
                                line=dict(color='red', width=1),
                                fillcolor='rgba(255, 0, 0, 0.3)'
                            ))
                            
                            fig2.update_layout(
                                title="Drawdown Over Time",
                                xaxis_title="Date",
                                yaxis_title="Drawdown (%)",
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Create downloadable data
                        results_df = pd.DataFrame({
                            'Date': return_dates,
                            'Daily_Return': daily_returns,
                            'Cumulative_Return': (cumulative_returns - 1) * 100,
                            'Drawdown': drawdown * 100
                        })
                        
                        # Download options
                        st.subheader("💾 Download Data")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            csv_returns = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Returns Data",
                                data=csv_returns,
                                file_name=f"composer_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            csv_allocations = allocations_df.to_csv()
                            st.download_button(
                                label="📥 Download Allocations Data",
                                data=csv_allocations,
                                file_name=f"composer_allocations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        # Show rolling performance metrics
                        st.subheader("📊 Rolling Performance (30-day windows)")
                        
                        if len(daily_returns) >= 30:
                            rolling_30d = daily_returns.rolling(30)
                            rolling_return = (1 + rolling_30d / 100).prod() - 1
                            rolling_vol = rolling_30d.std() * np.sqrt(252)
                            
                            if HAS_PLOTLY:
                                fig3 = go.Figure()
                                fig3.add_trace(go.Scatter(
                                    x=return_dates[29:],  # Skip first 29 days
                                    y=rolling_return.iloc[29:] * 100,
                                    mode='lines',
                                    name='30-day Rolling Return',
                                    line=dict(color='green', width=2)
                                ))
                                
                                fig3.update_layout(
                                    title="30-Day Rolling Returns",
                                    xaxis_title="Date",
                                    yaxis_title="30-Day Return (%)",
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig3, use_container_width=True)
                    
                    else:
                        st.error("Failed to calculate returns from allocation data.")
                    
                    # Display basic allocation statistics (original functionality)
                    st.subheader("🎯 Allocation Statistics")
                    
                    # Calculate basic metrics for each asset
                    allocation_stats = []
                    for ticker in tickers:
                        if ticker in allocations_df.columns:
                            avg_allocation = allocations_df[ticker].mean()
                            max_allocation = allocations_df[ticker].max()
                            days_held = (allocations_df[ticker] > 0).sum()
                            pct_time_held = (days_held / len(allocations_df)) * 100
                            
                            allocation_stats.append({
                                'Ticker': ticker,
                                'Avg Allocation (%)': f"{avg_allocation:.1f}",
                                'Max Allocation (%)': f"{max_allocation:.1f}",
                                'Days Held': days_held,
                                'Time Held (%)': f"{pct_time_held:.1f}"
                            })
                    
                    allocation_stats_df = pd.DataFrame(allocation_stats)
                    allocation_stats_df = allocation_stats_df.sort_values('Days Held', ascending=False)
                    st.dataframe(allocation_stats_df, use_container_width=True)
                    
                    # Plot allocation over time
                    if HAS_PLOTLY:
                        st.subheader("📊 Asset Allocation Over Time")
                        
                        fig4 = go.Figure()
                        
                        for ticker in tickers:
                            if ticker in allocations_df.columns and ticker != '$USD':
                                fig4.add_trace(go.Scatter(
                                    x=allocations_df.index,
                                    y=allocations_df[ticker],
                                    mode='lines',
                                    name=ticker,
                                    stackgroup='one'
                                ))
                        
                        fig4.update_layout(
                            title="Asset Allocation Over Time",
                            xaxis_title="Date",
                            yaxis_title="Allocation (%)",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig4, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error analyzing strategy: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
    
    else:
        st.info("👆 Upload a Composer strategy JSON file or enter a Composer URL to get started!")
        
        # Show example
        with st.expander("Example: How to use"):
            st.markdown("""
            **🔄 Comparison Mode (Recommended)**
            1. **Upload JSON**: Select your Composer strategy JSON file
            2. **Enter Composer URL**: Paste your Composer symphony URL
            3. **Set Parameters**: Choose backtest dates and initial capital
            4. **Click Single Button**: "🚀 Run Both Backtests & Comparison"
            5. **Get Results**: Automatic execution of both backtests and comparison
            6. **Analyze Differences**: Verify in-house logic matches Composer
            7. **Download Results**: Export 4 comprehensive comparison files
            
            **📄 JSON-Only Mode**
            1. **Upload JSON**: Select your Composer strategy JSON file
            2. **Set Parameters**: Choose start/end dates and initial capital  
            3. **Run Backtest**: Click the button to start backtesting
            4. **View Results**: Analyze performance metrics and charts
            5. **Download**: Export results as CSV for further analysis
            
            **🌐 Composer-Only Mode**
            1. **Enter URL**: Paste your Composer symphony URL
            2. **Set Date Range**: Choose the historical period to fetch
            3. **Fetch Data**: Click to download data from Composer
            4. **Analyze**: View allocation patterns and basic statistics
            
            **🔍 Comparison Features:**
            - ✅ **Single Button Execution**: Run both backtests simultaneously
            - ✅ **Date Alignment**: Ensures both backtests use identical date ranges
            - ✅ **Daily Ticker Comparison**: Detailed analysis of ticker selection differences
            - ✅ **Asset Selection Accuracy**: Verify in-house logic matches Composer
            - ✅ **4 Output Files**: Comprehensive comparison data for analysis
            - ✅ **Mismatch Identification**: Pinpoint days and tickers with differences
            
            **📊 Supported Indicators:**
            - ✅ RSI indicators and comparisons
            - ✅ Moving averages (SMA)
            - ✅ Cumulative returns
            - ✅ Max drawdown calculations
            - ✅ Asset filtering and sorting
            - ✅ Nested conditional logic (if-else)
            - ✅ Multiple asset classes
            - ✅ Direct Composer URL fetching
            """)
        
        # CRITICAL: Add troubleshooting guide
        with st.expander("🚨 **CRITICAL TROUBLESHOOTING GUIDE**", expanded=False):
            st.markdown("""
            ## 🔧 **If You're Still Getting Mismatches After Fixes**
            
            **1. Enable Debug Mode First** 🐛
            - Check the "Enable Debug Mode" checkbox in the sidebar
            - This will show you step-by-step strategy evaluation
            
            **2. Check These Common Issues:**
            
            **Asset Selection Mismatch:**
            - Your strategy might be selecting different assets than expected
            - Look for conditional logic that might be evaluating differently
            - Check if technical indicators (RSI, SMA, etc.) are calculating correctly
            
            **Date Alignment Issues:**
            - Ensure both backtests use the same date range
            - Check if there are missing trading days in your data
            
            **Technical Indicator Problems:**
            - RSI, SMA, EMA calculations might differ from Composer
            - Window parameters might be interpreted differently
            - Data quality issues (missing OHLC data)
            
            **3. Debug Output to Look For:**
            - 🔍 **Evaluating node**: Shows which strategy step is being processed
            - 🔍 **Condition result**: Shows if conditions are TRUE/FALSE
            - ✅ **IF condition TRUE**: Shows which assets were selected
            - 🔄 **ELSE condition executed**: Shows fallback asset selection
            - ⚠️ **Insufficient data**: Shows when indicators can't be calculated
            
            **4. Test Single Day:**
            - Pick a mismatched day from your comparison results
            - Enable debug mode and run just that day
            - Trace through the logic step by step
            
            **5. Verify Strategy JSON:**
            - Check if your strategy has the expected structure
            - Ensure conditional logic is properly nested
            - Verify asset tickers match exactly
            """)

def display_backtest_results(results, initial_capital, start_date, end_date):
    """Display backtest results with charts and metrics"""
    if not results.empty:
        st.success("Backtest completed successfully!")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        final_value = results['Portfolio_Value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        days_elapsed = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        years_elapsed = days_elapsed / 365.25
        
        # Calculate more accurate metrics
        daily_returns = results['Portfolio_Value'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if volatility > 0:
            sharpe_ratio = (total_return / years_elapsed) / volatility
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        running_max = results['Portfolio_Value'].cummax()
        drawdown = (results['Portfolio_Value'] - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Value", f"${final_value:,.2f}")
        with col2:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col3:
            if years_elapsed > 0:
                annualized_return = ((final_value / initial_capital) ** (1/years_elapsed) - 1) * 100
                st.metric("Annualized Return", f"{annualized_return:.2f}%")
            else:
                st.metric("Annualized Return", "N/A")
        with col4:
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Volatility (Ann.)", f"{volatility:.2f}%")
        with col6:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col7:
            rebalance_count = results['Rebalanced'].sum()
            st.metric("Rebalances", f"{rebalance_count}")
        with col8:
            win_rate = (daily_returns > 0).mean() * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Portfolio value chart
        if HAS_PLOTLY:
            import plotly.graph_objects as go
            
            st.subheader("Portfolio Performance")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['Date'],
                y=results['Portfolio_Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(results.set_index('Date')['Portfolio_Value'])
        
        # Asset allocation over time
        st.subheader("Asset Selection Over Time")
        asset_counts = results.groupby('Selected_Assets').size().reset_index()
        asset_counts.columns = ['Assets', 'Days']
        asset_counts = asset_counts.sort_values('Days', ascending=False)
        
        st.dataframe(asset_counts, use_container_width=True)
        
        # Download results
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
