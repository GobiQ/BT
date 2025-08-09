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
        
    def download_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
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
            return np.nan
            
        df = self.data[symbol]
        if date not in df.index:
            # Find the closest previous date
            available_dates = df.index[df.index <= date]
            if len(available_dates) == 0:
                return np.nan
            date = available_dates[-1]
        
        prices = df['Close']
        
        if indicator == 'relative-strength-index':
            values = self.calculate_rsi(prices, window)
        elif indicator == 'moving-average-price':
            values = self.calculate_sma(prices, window)
        elif indicator == 'current-price':
            return prices.loc[date]
        elif indicator == 'cumulative-return':
            values = self.calculate_cumulative_return(prices, window)
        elif indicator == 'max-drawdown':
            values = self.calculate_max_drawdown(prices, window)
        else:
            return np.nan
            
        return values.loc[date] if date in values.index else np.nan
    
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
        
        # Debug output
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.write(f"Debug: {date.date()} - {lhs_fn}({lhs_val}, {lhs_window}) = {lhs_value:.2f} {comparator} {rhs_value:.2f}")
        
        # Compare values
        if pd.isna(lhs_value) or pd.isna(rhs_value):
            return False
            
        if comparator == 'gt':
            return lhs_value > rhs_value
        elif comparator == 'lt':
            return lhs_value < rhs_value
        elif comparator == 'gte':
            return lhs_value >= rhs_value
        elif comparator == 'lte':
            return lhs_value <= rhs_value
        elif comparator == 'eq':
            return abs(lhs_value - rhs_value) < 0.0001
        
        return False
    
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
                asset_values.append((ticker, value))
        
        # Sort assets
        if select_fn == 'top':
            asset_values.sort(key=lambda x: x[1], reverse=True)
        elif select_fn == 'bottom':
            asset_values.sort(key=lambda x: x[1])
        
        # Select top N
        selected = [ticker for ticker, _ in asset_values[:select_n]]
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
        # Extract all symbols
        symbols = self.extract_all_symbols(strategy)
        st.info(f"Found {len(symbols)} unique symbols: {', '.join(sorted(symbols))}")
        
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
        
        # Run backtest
        results = []
        portfolio_value = self.cash
        prev_portfolio_value = self.cash
        
        progress_bar = st.progress(0)
        
        # Add debug container
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            debug_container = st.container()
        else:
            debug_container = None
        
        for i, date in enumerate(trading_dates):
            self.current_date = date
            
            if debug_container and i < 5:  # Show debug for first 5 days
                with debug_container:
                    st.write(f"\n**Date: {date.date()}**")
            
            # Evaluate strategy for this date
            selected_assets = self.evaluate_node(strategy, date)
            
            # Calculate portfolio value (simplified - equal weight)
            if selected_assets and i > 0:
                daily_returns = []
                for asset in selected_assets:
                    if asset in self.data:
                        # Find the closest available date
                        asset_data = self.data[asset]
                        available_dates = asset_data.index[asset_data.index <= date]
                        if len(available_dates) > 0:
                            current_date = available_dates[-1]
                            
                            # Find previous date
                            prev_available_dates = asset_data.index[asset_data.index < current_date]
                            if len(prev_available_dates) > 0:
                                prev_date = prev_available_dates[-1]
                                
                                curr_price = asset_data.loc[current_date, 'Close']
                                prev_price = asset_data.loc[prev_date, 'Close']
                                daily_return = (curr_price - prev_price) / prev_price
                                daily_returns.append(daily_return)
                
                if daily_returns:
                    avg_return = np.mean(daily_returns)
                    portfolio_value = prev_portfolio_value * (1 + avg_return)
                else:
                    portfolio_value = prev_portfolio_value
            
            prev_portfolio_value = portfolio_value
            
            results.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Selected_Assets': ', '.join(selected_assets) if selected_assets else 'None',
                'Num_Assets': len(selected_assets)
            })
            
            if debug_container and i < 5:  # Show debug for first 5 days
                with debug_container:
                    st.write(f"Selected: {selected_assets}")
                    st.write(f"Portfolio Value: ${portfolio_value:,.2f}")
            
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
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Value", f"${final_value:,.2f}")
                    with col2:
                        st.metric("Total Return", f"{total_return:.2f}%")
                    with col3:
                        st.metric("Annualized Return", f"{(total_return / ((end_date - start_date).days / 365)):.2f}%")
                    with col4:
                        max_value = results['Portfolio_Value'].max()
                        max_dd = ((results['Portfolio_Value'] / results['Portfolio_Value'].cummax() - 1).min() * 100)
                        st.metric("Max Drawdown", f"{max_dd:.2f}%")
                    
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
