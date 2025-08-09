import streamlit as st
import pandas as pd
import numpy as np
import json
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
            return False
        
        return True
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
        if condition['step'] != 'if-child':
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
                            if result:  # Only return if we got assets
                                return result
            
            # If no condition matched and we have an else, execute it
            if else_child:
                return self.evaluate_children(else_child.get('children', []), date)
            
            return []
        
        elif step == 'filter':
            children = node.get('children', [])
            assets = []
            for child in children:
                if child.get('step') == 'asset':
                    assets.append(child)
            return self.apply_filter(assets, node, date)
        
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
        
        # Initialize backtest
        results = []
        portfolio_value = self.cash
        self.current_holdings = {}
        self.last_rebalance_date = None
        
        progress_bar = st.progress(0)
        
        # Add debug container
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            debug_container = st.container()
        else:
            debug_container = None
        
        for i, date in enumerate(trading_dates):
            self.current_date = date
            
            if debug_container and i < 10:  # Show debug for first 10 days
                with debug_container:
                    st.write(f"\n**Date: {date.date()}**")
            
            # Check if we should rebalance
            should_rebalance = self.should_rebalance(date)
            
            if should_rebalance:
                # Evaluate strategy for this date to get new allocation
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
                
                self.current_holdings = new_holdings
                self.last_rebalance_date = date
            
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

def main():
    st.title("📈 Composer Strategy Backtester")
    st.markdown("Upload your Composer strategy JSON files and backtest them with historical data.")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Composer JSON", type=['json'])
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Initial capital
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
        
        # Debug mode
        debug_mode = st.checkbox("Debug Mode", help="Show detailed condition evaluation")
    
    if uploaded_file is not None:
        try:
            # Load strategy
            strategy_data = json.load(uploaded_file)
            
            # Display strategy info
            st.subheader("Strategy Information")
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
            
            # Run backtest button
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
                
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid Composer strategy file.")
        except Exception as e:
            st.error(f"Error processing strategy: {str(e)}")
    
    else:
        st.info("👆 Upload a Composer strategy JSON file to get started!")
        
        # Show example
        with st.expander("Example: How to use"):
            st.markdown("""
            1. **Upload JSON**: Select your Composer strategy JSON file
            2. **Set Parameters**: Choose start/end dates and initial capital  
            3. **Run Backtest**: Click the button to start backtesting
            4. **View Results**: Analyze performance metrics and charts
            5. **Download**: Export results as CSV for further analysis
            
            **Supported Features:**
            - ✅ RSI indicators and comparisons
            - ✅ Moving averages (SMA)
            - ✅ Cumulative returns
            - ✅ Max drawdown calculations
            - ✅ Asset filtering and sorting
            - ✅ Nested conditional logic (if-else)
            - ✅ Multiple asset classes
            """)

if __name__ == "__main__":
    main()
