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
    date_range = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(0.0, index=date_range, columns=tickers)
    
    for ticker in allocations:
        for date_int in allocations[ticker]:
            trading_date = convert_trading_date(date_int)
            percent = allocations[ticker][date_int]
            if trading_date in df.index:
                df.at[trading_date, ticker] = percent
    
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
        """Download price data for all symbols"""
        if not HAS_YFINANCE:
            st.error("yfinance is required for data download. Please install it.")
            return {}
            
        data = {}
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(symbols):
            try:
                ticker_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not ticker_data.empty:
                    data[symbol] = ticker_data
                progress_bar.progress((i + 1) / len(symbols))
            except Exception as e:
                st.warning(f"Could not download data for {symbol}: {e}")
        
        progress_bar.empty()
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
    
    def calculate_cumulative_return(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate cumulative return over window"""
        return (prices / prices.shift(window) - 1) * 100
    
    def calculate_max_drawdown(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate maximum drawdown over window"""
        rolling_max = prices.rolling(window=window).max()
        drawdown = (prices / rolling_max - 1) * 100
        return drawdown.rolling(window=window).min()
    
    def get_indicator_value(self, symbol: str, indicator: str, window: int, date: pd.Timestamp) -> float:
        """Get indicator value for a symbol at a specific date"""
        if symbol not in self.data:
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"Symbol {symbol} not found in data")
            return np.nan
            
        df = self.data[symbol]
        if df.empty:
            return np.nan
            
        # Find the closest previous date
        available_dates = df.index[df.index <= date]
        if len(available_dates) == 0:
            return np.nan
        actual_date = available_dates[-1]
        
        prices = df['Close']
        
        try:
            if indicator == 'relative-strength-index':
                values = self.calculate_rsi(prices, window)
            elif indicator == 'moving-average-price':
                values = self.calculate_sma(prices, window)
            elif indicator == 'current-price':
                return float(prices.loc[actual_date])
            elif indicator == 'cumulative-return':
                values = self.calculate_cumulative_return(prices, window)
            elif indicator == 'max-drawdown':
                values = self.calculate_max_drawdown(prices, window)
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
                return np.nan
                
        except Exception as e:
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"Error calculating {indicator} for {symbol}: {e}")
            return np.nan
    
    def evaluate_condition(self, condition: Dict[str, Any], date: pd.Timestamp) -> bool:
        """Evaluate a condition node"""
        if condition.get('step') != 'if-child':
            return False
            
        # Skip else conditions in this function
        if condition.get('is-else-condition?', False):
            return False
            
        lhs_fn = condition.get('lhs-fn')
        rhs_fn = condition.get('rhs-fn') 
        lhs_val = condition.get('lhs-val')
        rhs_val = condition.get('rhs-val')
        comparator = condition.get('comparator')
        lhs_window = condition.get('lhs-window-days', condition.get('lhs-fn-params', {}).get('window', 14))
        rhs_window = condition.get('rhs-window-days', condition.get('rhs-fn-params', {}).get('window', 14))
        rhs_fixed = condition.get('rhs-fixed-value?', False)
        
        # Convert string windows to integers
        if isinstance(lhs_window, str):
            lhs_window = int(lhs_window)
        if isinstance(rhs_window, str):
            rhs_window = int(rhs_window)
        
        # Get left-hand side value
        lhs_value = self.get_indicator_value(lhs_val, lhs_fn, lhs_window, date)
        
        # Get right-hand side value
        if rhs_fixed:
            try:
                rhs_value = float(rhs_val)
            except (ValueError, TypeError):
                return False
        else:
            rhs_value = self.get_indicator_value(rhs_val, rhs_fn, rhs_window, date)
        
        # Ensure we have scalar values
        if pd.isna(lhs_value) or pd.isna(rhs_value):
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write(f"Debug: Missing data - LHS: {lhs_value}, RHS: {rhs_value}")
            return False
        
        # Convert to float to ensure scalar comparison
        try:
            lhs_value = float(lhs_value)
            rhs_value = float(rhs_value)
        except (ValueError, TypeError):
            return False
        
        # Debug output
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"Debug: {date.date()} - {lhs_fn}({lhs_val}, {lhs_window}) = {lhs_value:.2f} {comparator} {rhs_value:.2f}")
        
        # Compare values
        if comparator == 'gt':
            result = lhs_value > rhs_value
        elif comparator == 'lt':
            result = lhs_value < rhs_value
        elif comparator == 'gte':
            result = lhs_value >= rhs_value
        elif comparator == 'lte':
            result = lhs_value <= rhs_value
        elif comparator == 'eq':
            result = abs(lhs_value - rhs_value) < 0.0001
        else:
            result = False
        
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"  → Result: {result}")
        
        return bool(result)
    
    def apply_filter(self, assets: List[Dict[str, Any]], filter_config: Dict[str, Any], date: pd.Timestamp) -> List[str]:
        """Apply filter to select assets"""
        if filter_config['step'] != 'filter':
            return [asset['ticker'] for asset in assets if 'ticker' in asset]
        
        sort_fn = filter_config.get('sort-by-fn')
        sort_window = filter_config.get('sort-by-window-days', 14)
        select_fn = filter_config.get('select-fn', 'top')
        select_n = int(filter_config.get('select-n', 1))
        
        # Calculate sort values for each asset
        asset_values = []
        for asset in assets:
            if 'ticker' not in asset:
                continue
            ticker = asset['ticker']
            value = self.get_indicator_value(ticker, sort_fn, sort_window, date)
            if not pd.isna(value):
                asset_values.append((ticker, float(value)))
        
        if not asset_values:
            return []
        
        # Sort assets
        if select_fn == 'top':
            asset_values.sort(key=lambda x: x[1], reverse=True)
        elif select_fn == 'bottom':
            asset_values.sort(key=lambda x: x[1])
        
        # Select top N
        selected = [ticker for ticker, _ in asset_values[:select_n]]
        
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"Filter: {sort_fn} - {asset_values} → {selected}")
        
        return selected
    
    def evaluate_node(self, node: Dict[str, Any], date: pd.Timestamp) -> List[str]:
        """Recursively evaluate strategy nodes"""
        step = node.get('step')
        
        if step == 'asset':
            return [node['ticker']]
        
        elif step == 'if':
            children = node.get('children', [])
            if not children:
                return []
                
            # Process if-child nodes in order
            else_child = None
            for child in children:
                if child.get('step') == 'if-child':
                    is_else = child.get('is-else-condition?', False)
                    
                    if is_else:
                        # Store else condition for later
                        else_child = child
                    else:
                        # Evaluate condition
                        if self.evaluate_condition(child, date):
                            result = self.evaluate_children(child.get('children', []), date)
                            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                                st.write(f"IF condition TRUE, selected: {result}")
                            return result
            
            # If no condition matched and we have an else, execute it
            if else_child:
                result = self.evaluate_children(else_child.get('children', []), date)
                if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                    st.write(f"ELSE condition executed, selected: {result}")
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
        
        elif step in ['wt-cash-equal', 'wt-cash-specified', 'wt-inverse-vol', 'group']:
            return self.evaluate_children(node.get('children', []), date)
        
        elif step == 'root':
            # Handle root node
            return self.evaluate_children(node.get('children', []), date)
        
        return []
    
    def evaluate_children(self, children: List[Dict[str, Any]], date: pd.Timestamp) -> List[str]:
        """Evaluate all children nodes"""
        all_assets = []
        for child in children:
            assets = self.evaluate_node(child, date)
            all_assets.extend(assets)
        return list(set(all_assets))  # Remove duplicates
    
    def extract_all_symbols(self, strategy: Dict[str, Any]) -> List[str]:
        """Extract all ticker symbols from strategy"""
        symbols = set()
        
        def extract_recursive(node):
            if isinstance(node, dict):
                if node.get('step') == 'asset' and 'ticker' in node:
                    symbols.add(node['ticker'])
                elif 'lhs-val' in node and isinstance(node['lhs-val'], str):
                    symbols.add(node['lhs-val'])
                elif 'rhs-val' in node and isinstance(node['rhs-val'], str) and not node.get('rhs-fixed-value?', False):
                    symbols.add(node['rhs-val'])
                
                for key, value in node.items():
                    extract_recursive(value)
            elif isinstance(node, list):
                for item in node:
                    extract_recursive(item)
        
        extract_recursive(strategy)
        return list(symbols)
    
    def run_backtest(self, strategy: Dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
        """Run the backtest"""
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
                
                # Create new holdings (equal weight)
                new_holdings = {}
                if selected_assets:
                    weight_per_asset = 1.0 / len(selected_assets)
                    for asset in selected_assets:
                        new_holdings[asset] = weight_per_asset
                
                if debug_container and i < 10:
                    with debug_container:
                        st.write(f"Rebalancing to: {selected_assets}")
                        st.write(f"New holdings: {new_holdings}")
                
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

def compare_allocations(inhouse_results: pd.DataFrame, composer_allocations: pd.DataFrame, 
                       composer_tickers: List[str], start_date, end_date) -> pd.DataFrame:
    """Compare in-house backtest allocations with Composer allocations"""
    
    # Convert dates to datetime for comparison
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    # Filter Composer allocations to the backtest period
    composer_filtered = composer_allocations[
        (composer_allocations.index >= start_dt) & 
        (composer_allocations.index <= end_dt)
    ].copy()
    
    # Create comparison dataframe
    comparison_data = []
    
    for _, row in inhouse_results.iterrows():
        date = row['Date']
        inhouse_assets = row['Selected_Assets']
        inhouse_holdings = row['Holdings']
        
        # Get Composer allocations for this date
        if date in composer_filtered.index:
            composer_row = composer_filtered.loc[date]
            
            # Extract Composer assets (non-zero allocations)
            composer_assets = []
            composer_holdings = {}
            for ticker in composer_tickers:
                if ticker in composer_row and composer_row[ticker] > 0:
                    composer_assets.append(ticker)
                    composer_holdings[ticker] = composer_row[ticker] / 100  # Convert % to decimal
            
            # Compare asset selections
            inhouse_set = set(inhouse_assets.split(', ')) if inhouse_assets != 'None' else set()
            composer_set = set(composer_assets)
            
            # Calculate differences
            missing_in_inhouse = composer_set - inhouse_set
            extra_in_inhouse = inhouse_set - composer_set
            common_assets = inhouse_set & composer_set
            
            # Calculate allocation differences for common assets
            allocation_diffs = {}
            for asset in common_assets:
                inhouse_weight = inhouse_holdings.get(asset, 0)
                composer_weight = composer_holdings.get(asset, 0)
                diff = abs(inhouse_weight - composer_weight)
                allocation_diffs[asset] = diff
            
            comparison_data.append({
                'Date': date,
                'InHouse_Assets': inhouse_assets,
                'Composer_Assets': ', '.join(composer_assets),
                'Common_Assets': ', '.join(common_assets),
                'Missing_In_InHouse': ', '.join(missing_in_inhouse),
                'Extra_In_InHouse': ', '.join(extra_in_inhouse),
                'Asset_Selection_Match': len(common_assets) / max(len(inhouse_set), len(composer_set)) if max(len(inhouse_set), len(composer_set)) > 0 else 1.0,
                'Allocation_Differences': allocation_diffs,
                'InHouse_Portfolio_Value': row['Portfolio_Value'],
                'Rebalanced': row['Rebalanced']
            })
    
    return pd.DataFrame(comparison_data)

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
    display_df['Date'] = display_df['Date'].dt.date
    
    st.dataframe(display_df, use_container_width=True)
    
    # Detailed analysis
    st.subheader("🔍 Detailed Analysis")
    
    # Days with mismatches
    mismatches = comparison_results[comparison_results['Asset_Selection_Match'] < 1.0]
    if len(mismatches) > 0:
        st.warning(f"⚠️ Found {len(mismatches)} days with allocation mismatches")
        
        with st.expander("View Mismatch Details"):
            for _, row in mismatches.iterrows():
                st.write(f"**{row['Date'].date()}**")
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

def main():
    st.title("📈 Composer Strategy Backtester")
    st.markdown("Upload your Composer strategy JSON files or fetch directly from Composer URL.")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        
        # Data source selection - allow both
        st.subheader("Data Sources")
        st.info("You can use both sources to compare results!")
        
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
        st.write("**2. Fetch from Composer URL**")
        composer_url = st.text_input(
            "Composer Symphony URL",
            placeholder="https://app.composer.trade/symphony/...",
            help="Enter the full URL of your Composer symphony"
        )
        
        if composer_url:
            # Date range for fetching
            col1, col2 = st.columns(2)
            with col1:
                fetch_start_date = st.date_input("Fetch Start Date", datetime.now() - timedelta(days=365*3))
            with col2:
                fetch_end_date = st.date_input("Fetch End Date", datetime.now())
            
            if st.button("🔄 Fetch Strategy Data"):
                try:
                    with st.spinner("Fetching data from Composer..."):
                        # Convert dates to string format
                        start_str = fetch_start_date.strftime('%Y-%m-%d')
                        end_str = fetch_end_date.strftime('%Y-%m-%d')
                        
                        # Fetch the backtest data
                        allocations_df, symphony_name, tickers = fetch_composer_backtest(composer_url, start_str, end_str)
                        
                        # Store in session state for later use
                        st.session_state.composer_data = {
                            'allocations_df': allocations_df,
                            'symphony_name': symphony_name,
                            'tickers': tickers,
                            'start_date': start_str,
                            'end_date': end_str
                        }
                        
                        st.success(f"✅ Successfully fetched data for: {symphony_name}")
                        st.info(f"Date range: {start_str} to {end_str}")
                        st.info(f"Assets: {', '.join(tickers)}")
                
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
        
        # Date range for backtesting (for both sources)
        st.subheader("Backtest Settings")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
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
    
    # Run backtests and comparison
    if has_json and has_composer:
        st.subheader("🔄 Run Both Backtests & Compare")
        
        if st.button("🚀 Run Comparison Analysis", type="primary"):
            if not HAS_YFINANCE:
                st.error("Cannot run backtest without yfinance. Please install required dependencies.")
                st.code("pip install yfinance plotly")
                return
            
            # Run in-house backtest
            st.info("Running in-house backtest from JSON...")
            backtester = ComposerBacktester()
            backtester.cash = initial_capital
            
            inhouse_results = backtester.run_backtest(
                strategy_data, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Get Composer data for comparison
            composer_data = st.session_state.composer_data
            composer_allocations = composer_data['allocations_df']
            
            # Run comparison analysis
            comparison_results = compare_allocations(
                inhouse_results, 
                composer_allocations, 
                composer_data['tickers'],
                start_date,
                end_date
            )
            
            # Display comparison results
            display_comparison_results(comparison_results, inhouse_results, composer_allocations, initial_capital)
    
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
            3. **Fetch Data**: Download historical allocations from Composer
            4. **Set Parameters**: Choose backtest dates and initial capital
            5. **Run Comparison**: Execute both backtests and compare results
            6. **Analyze Differences**: Verify in-house logic matches Composer
            7. **Download Results**: Export comparison data for analysis
            
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
            - ✅ Daily allocation matching verification
            - ✅ Asset selection accuracy analysis
            - ✅ Performance comparison metrics
            - ✅ Mismatch identification and analysis
            - ✅ Downloadable comparison reports
            
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

def display_backtest_results(results, initial_capital, start_date, end_date):
    """Display backtest results with charts and metrics"""
    if not results.empty:
        st.success("Backtest completed successfully!")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        final_value = results['Portfolio_Value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        days_elapsed = (end_date - start_date).days
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
