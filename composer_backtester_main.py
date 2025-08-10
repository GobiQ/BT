import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import math
warnings.filterwarnings('ignore')

# Try to import scipy, fallback to basic implementations if not available
try:
    from scipy import stats
    from scipy.stats import ttest_1samp, binom_test
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ scipy not available. Using basic statistical implementations.")

def basic_t_test(data, mu=0):
    """Basic t-test implementation when scipy is not available"""
    n = len(data)
    if n <= 1:
        return 0, 1
    
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    
    if std_data == 0:
        return float('inf') if mean_data != mu else 0, 1
    
    t_stat = (mean_data - mu) / (std_data / math.sqrt(n))
    
    # Approximate p-value using normal distribution for large samples
    if n >= 30:
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    else:
        # Very rough approximation for small samples
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    
    return t_stat, p_value

def basic_binom_test(successes, trials, p=0.5):
    """Basic binomial test implementation when scipy is not available"""
    if trials == 0:
        return 1.0
    
    observed_rate = successes / trials
    expected_rate = p
    
    # Use normal approximation for large samples
    if trials >= 30:
        std_error = math.sqrt(p * (1 - p) / trials)
        z_score = (observed_rate - expected_rate) / std_error
        # Approximate p-value
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))
    else:
        # For small samples, use a simple approximation
        p_value = 2 * min(observed_rate, 1 - observed_rate) if observed_rate != 0.5 else 1.0
    
    return p_value

# Page configuration
st.set_page_config(
    page_title="Portfolio Comparison Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 12px;
        padding-right: 12px;
    }
    .significance-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .win-rate-box {
        background-color: #e8f5e8;
        border-left: 4px solid #2ca02c;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def fetch_portfolio_data(inhouse_symbols, composer_symbols, start_date, end_date, inhouse_weights=None, composer_weights=None):
    """Fetch portfolio data from yfinance and calculate comparison metrics"""
    try:
        # Default equal weights if not provided
        if inhouse_weights is None:
            inhouse_weights = [1/len(inhouse_symbols)] * len(inhouse_symbols)
        if composer_weights is None:
            composer_weights = [1/len(composer_symbols)] * len(composer_symbols)
        
        # Fetch price data
        all_symbols = list(set(inhouse_symbols + composer_symbols))
        data = yf.download(all_symbols, start=start_date, end=end_date)['Adj Close']
        
        if len(all_symbols) == 1:
            data = data.to_frame(all_symbols[0])
        
        # Calculate portfolio values
        inhouse_portfolio = pd.Series(0, index=data.index)
        composer_portfolio = pd.Series(0, index=data.index)
        
        # InHouse portfolio
        for symbol, weight in zip(inhouse_symbols, inhouse_weights):
            if symbol in data.columns:
                inhouse_portfolio += data[symbol] * weight
        
        # Composer portfolio
        for symbol, weight in zip(composer_symbols, composer_weights):
            if symbol in data.columns:
                composer_portfolio += data[symbol] * weight
        
        # Normalize to start at 100,000
        initial_value = 100000
        inhouse_portfolio = (inhouse_portfolio / inhouse_portfolio.iloc[0]) * initial_value
        composer_portfolio = (composer_portfolio / composer_portfolio.iloc[0]) * initial_value
        
        # Create comparison dataframe
        df = pd.DataFrame({
            'Date': data.index,
            'InHouse_Portfolio_Value': inhouse_portfolio.values,
            'Composer_Portfolio_Value': composer_portfolio.values,
            'InHouse_Assets': ','.join(inhouse_symbols),
            'Composer_Assets': ','.join(composer_symbols),
            'Common_Assets': ','.join(set(inhouse_symbols) & set(composer_symbols)),
            'InHouse_Num_Assets': len(inhouse_symbols),
            'Composer_Num_Assets': len(composer_symbols)
        })
        
        # Calculate asset selection match (Jaccard similarity)
        inhouse_set = set(inhouse_symbols)
        composer_set = set(composer_symbols)
        intersection = len(inhouse_set & composer_set)
        union = len(inhouse_set | composer_set)
        asset_selection_match = intersection / union if union > 0 else 0
        df['Asset_Selection_Match'] = asset_selection_match
        
        # Calculate allocation differences and match scores
        allocation_diffs = []
        match_scores = []
        
        for _, row in df.iterrows():
            # Simple allocation difference calculation
            # For now, assume equal weight difference
            if len(inhouse_symbols) == len(composer_symbols) and set(inhouse_symbols) == set(composer_symbols):
                # Same assets, calculate weight differences
                weight_diff = sum(abs(w1 - w2) for w1, w2 in zip(inhouse_weights, composer_weights))
                allocation_diffs.append(weight_diff)
                match_scores.append(1 - weight_diff)  # Higher score for lower differences
            else:
                # Different assets
                allocation_diffs.append(1.0)  # Max difference
                match_scores.append(asset_selection_match * 0.5)  # Penalize for different assets
        
        df['Allocation_Difference'] = allocation_diffs
        df['Match_Score'] = match_scores
        df['Rebalanced'] = True  # Assume daily rebalancing for simplicity
        
        # Calculate relative performance
        df['Relative_Performance'] = df['InHouse_Portfolio_Value'] - df['Composer_Portfolio_Value']
        df['Outperformance_Days'] = df['Relative_Performance'] > 0
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_statistical_metrics(df):
    """Calculate comprehensive statistical metrics including win rates and significance tests"""
    metrics = {}
    
    # Calculate daily returns for both portfolios
    df['InHouse_Daily_Return'] = df['InHouse_Portfolio_Value'].pct_change()
    df['Composer_Daily_Return'] = df['Composer_Portfolio_Value'].pct_change()
    df['Relative_Daily_Return'] = df['InHouse_Daily_Return'] - df['Composer_Daily_Return']
    
    inhouse_returns = df['InHouse_Daily_Return'].dropna()
    composer_returns = df['Composer_Daily_Return'].dropna()
    relative_returns = df['Relative_Daily_Return'].dropna()
    
    # Win Rate Analysis - InHouse Portfolio
    positive_returns = (inhouse_returns > 0).sum()
    total_returns = len(inhouse_returns)
    win_rate = positive_returns / total_returns if total_returns > 0 else 0
    
    metrics['inhouse_win_rate'] = win_rate
    metrics['inhouse_positive_days'] = positive_returns
    metrics['inhouse_negative_days'] = total_returns - positive_returns
    metrics['total_trading_days'] = total_returns
    
    # Win Rate Analysis - Composer Portfolio
    composer_positive_returns = (composer_returns > 0).sum()
    composer_win_rate = composer_positive_returns / len(composer_returns) if len(composer_returns) > 0 else 0
    
    metrics['composer_win_rate'] = composer_win_rate
    metrics['composer_positive_days'] = composer_positive_returns
    metrics['composer_negative_days'] = len(composer_returns) - composer_positive_returns
    
    # Outperformance Analysis
    outperformance_days = (relative_returns > 0).sum()
    outperformance_rate = outperformance_days / len(relative_returns) if len(relative_returns) > 0 else 0
    
    metrics['outperformance_rate'] = outperformance_rate
    metrics['outperformance_days'] = outperformance_days
    metrics['underperformance_days'] = len(relative_returns) - outperformance_days
    
    # Statistical Significance Tests
    if len(inhouse_returns) > 1 and len(composer_returns) > 1:
        # Test if InHouse mean return is significantly different from zero
        if SCIPY_AVAILABLE:
            t_stat_ih, p_value_ih = ttest_1samp(inhouse_returns, 0)
        else:
            t_stat_ih, p_value_ih = basic_t_test(inhouse_returns, 0)
        
        metrics['inhouse_return_t_stat'] = t_stat_ih
        metrics['inhouse_return_p_value'] = p_value_ih
        metrics['inhouse_return_significant'] = p_value_ih < 0.05
        
        # Test if Composer mean return is significantly different from zero
        if SCIPY_AVAILABLE:
            t_stat_comp, p_value_comp = ttest_1samp(composer_returns, 0)
        else:
            t_stat_comp, p_value_comp = basic_t_test(composer_returns, 0)
        
        metrics['composer_return_t_stat'] = t_stat_comp
        metrics['composer_return_p_value'] = p_value_comp
        metrics['composer_return_significant'] = p_value_comp < 0.05
        
        # Test if relative performance is significantly different from zero
        if SCIPY_AVAILABLE:
            t_stat_rel, p_value_rel = ttest_1samp(relative_returns, 0)
        else:
            t_stat_rel, p_value_rel = basic_t_test(relative_returns, 0)
        
        metrics['relative_return_t_stat'] = t_stat_rel
        metrics['relative_return_p_value'] = p_value_rel
        metrics['relative_return_significant'] = p_value_rel < 0.05
        
        # Test if InHouse win rate is significantly different from 50%
        if SCIPY_AVAILABLE:
            p_value_winrate_ih = binom_test(positive_returns, total_returns, 0.5)
        else:
            p_value_winrate_ih = basic_binom_test(positive_returns, total_returns, 0.5)
        
        metrics['inhouse_winrate_p_value'] = p_value_winrate_ih
        metrics['inhouse_winrate_significant'] = p_value_winrate_ih < 0.05
        
        # Test if Composer win rate is significantly different from 50%
        if SCIPY_AVAILABLE:
            p_value_winrate_comp = binom_test(composer_positive_returns, len(composer_returns), 0.5)
        else:
            p_value_winrate_comp = basic_binom_test(composer_positive_returns, len(composer_returns), 0.5)
        
        metrics['composer_winrate_p_value'] = p_value_winrate_comp
        metrics['composer_winrate_significant'] = p_value_winrate_comp < 0.05
        
        # Test if outperformance rate is significantly different from 50%
        if SCIPY_AVAILABLE:
            p_value_outperf = binom_test(outperformance_days, len(relative_returns), 0.5)
        else:
            p_value_outperf = basic_binom_test(outperformance_days, len(relative_returns), 0.5)
        
        metrics['outperformance_p_value'] = p_value_outperf
        metrics['outperformance_significant'] = p_value_outperf < 0.05
        
        # Additional performance metrics
        metrics['inhouse_mean_return'] = inhouse_returns.mean()
        metrics['composer_mean_return'] = composer_returns.mean()
        metrics['relative_mean_return'] = relative_returns.mean()
        
        metrics['inhouse_volatility'] = inhouse_returns.std() * np.sqrt(252)
        metrics['composer_volatility'] = composer_returns.std() * np.sqrt(252)
        
        # Sharpe ratios (assuming 0% risk-free rate)
        metrics['inhouse_sharpe'] = (inhouse_returns.mean() * 252) / (inhouse_returns.std() * np.sqrt(252)) if inhouse_returns.std() > 0 else 0
        metrics['composer_sharpe'] = (composer_returns.mean() * 252) / (composer_returns.std() * np.sqrt(252)) if composer_returns.std() > 0 else 0
        
        # Information Ratio
        metrics['information_ratio'] = (relative_returns.mean() * 252) / (relative_returns.std() * np.sqrt(252)) if relative_returns.std() > 0 else 0
    
    return metrics

def main():
    st.markdown('<div class="main-header">📊 Portfolio Comparison Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for portfolio configuration
    st.sidebar.header("🔧 Portfolio Configuration")
    
    # Portfolio setup
    st.sidebar.subheader("InHouse Portfolio")
    inhouse_input = st.sidebar.text_input("InHouse Symbols (comma-separated)", "AAPL,GOOGL,MSFT")
    inhouse_symbols = [s.strip().upper() for s in inhouse_input.split(',') if s.strip()]
    
    st.sidebar.subheader("Composer Portfolio")
    composer_input = st.sidebar.text_input("Composer Symbols (comma-separated)", "SPY,QQQ")
    composer_symbols = [s.strip().upper() for s in composer_input.split(',') if s.strip()]
    
    # Date range
    st.sidebar.subheader("📅 Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date, end_date),
        min_value=datetime(2010, 1, 1),
        max_value=end_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    
    # Load data button
    if st.sidebar.button("🚀 Load Portfolio Data"):
        with st.spinner("Fetching data from Yahoo Finance..."):
            df = fetch_portfolio_data(inhouse_symbols, composer_symbols, start_date, end_date)
            st.session_state['portfolio_data'] = df
    
    # Check if data is loaded
    if 'portfolio_data' not in st.session_state:
        st.info("👆 Configure your portfolios in the sidebar and click 'Load Portfolio Data' to get started!")
        st.stop()
    
    df = st.session_state['portfolio_data']
    if df is None or df.empty:
        st.error("Failed to load portfolio data. Please check your symbols and try again.")
        st.stop()
    
    # Calculate statistical metrics
    metrics = calculate_statistical_metrics(df)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 InHouse Win Rate",
            value=f"{metrics['inhouse_win_rate']:.1%}",
            delta=f"{metrics['inhouse_win_rate'] - 0.5:.1%} vs 50%"
        )
    
    with col2:
        st.metric(
            label="📈 Outperformance Rate",
            value=f"{metrics['outperformance_rate']:.1%}",
            delta=f"{metrics['outperformance_rate'] - 0.5:.1%} vs 50%"
        )
    
    with col3:
        st.metric(
            label="💰 InHouse Portfolio Value",
            value=f"${df['InHouse_Portfolio_Value'].iloc[-1]:,.2f}",
            delta=f"${df['InHouse_Portfolio_Value'].iloc[-1] - 100000:,.2f}"
        )
    
    with col4:
        st.metric(
            label="🏆 Information Ratio",
            value=f"{metrics['information_ratio']:.3f}",
            delta="Higher is better"
        )
    
    # Statistical significance summary
    st.markdown("## 📊 Statistical Significance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="win-rate-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Win Rate Analysis")
        
        significance_symbol_ih = "✅" if metrics['inhouse_winrate_significant'] else "❌"
        significance_symbol_out = "✅" if metrics['outperformance_significant'] else "❌"
        
        st.markdown(f"""
        **InHouse Win Rate:** {metrics['inhouse_win_rate']:.1%} 
        - Positive days: {metrics['inhouse_positive_days']} / {metrics['total_trading_days']}
        - Statistical significance: {significance_symbol_ih} (p-value: {metrics['inhouse_winrate_p_value']:.4f})
        
        **Outperformance Rate:** {metrics['outperformance_rate']:.1%}
        - Better days: {metrics['outperformance_days']} / {len(df)-1}
        - Statistical significance: {significance_symbol_out} (p-value: {metrics['outperformance_p_value']:.4f})
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="significance-box">', unsafe_allow_html=True)
        st.markdown("### 📈 Return Significance")
        
        ih_sig_symbol = "✅" if metrics['inhouse_return_significant'] else "❌"
        rel_sig_symbol = "✅" if metrics['relative_return_significant'] else "❌"
        
        st.markdown(f"""
        **InHouse Returns vs Zero:** {ih_sig_symbol}
        - Mean daily return: {metrics['inhouse_mean_return']:.4%}
        - t-statistic: {metrics['inhouse_return_t_stat']:.3f}
        - p-value: {metrics['inhouse_return_p_value']:.4f}
        
        **Relative Returns vs Zero:** {rel_sig_symbol}
        - Mean relative return: {metrics['relative_mean_return']:.4%}
        - t-statistic: {metrics['relative_return_t_stat']:.3f}
        - p-value: {metrics['relative_return_p_value']:.4f}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance Comparison", "🎯 Win Rate Analysis", "📊 Statistical Tests", 
        "💹 Risk Metrics", "🔍 Detailed Data"
    ])
    
    with tab1:
        st.header("Portfolio Performance Comparison")
        
        # Portfolio values over time
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['Date'],
            y=df['InHouse_Portfolio_Value'],
            mode='lines',
            name='InHouse Portfolio',
            line=dict(color='#1f77b4', width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Composer_Portfolio_Value'],
            mode='lines',
            name='Composer Portfolio',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig1.update_layout(
            title="Portfolio Value Comparison Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Relative performance
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Relative_Performance'],
            mode='lines',
            name='InHouse - Composer',
            line=dict(color='#2ca02c', width=2),
            fill='tonexty'
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig2.update_layout(
            title="Relative Performance (InHouse - Composer)",
            xaxis_title="Date",
            yaxis_title="Relative Performance ($)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("Win Rate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Win rate comparison chart
            win_rates = [
                metrics['inhouse_win_rate'],
                metrics['composer_win_rate'],
                metrics['outperformance_rate']
            ]
            labels = ['InHouse Win Rate', 'Composer Win Rate', 'Outperformance Rate']
            
            fig3 = go.Figure(data=[
                go.Bar(x=labels, y=win_rates, 
                       marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ])
            fig3.add_hline(y=0.5, line_dash="dash", line_color="red", 
                          annotation_text="50% Baseline")
            fig3.update_layout(
                title="Win Rates Comparison",
                yaxis_title="Win Rate",
                yaxis=dict(tickformat='.1%'),
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Rolling win rate
            window = 30
            df['Rolling_InHouse_WinRate'] = df['InHouse_Daily_Return'].rolling(window).apply(
                lambda x: (x > 0).mean()
            )
            df['Rolling_Outperformance_Rate'] = df['Relative_Daily_Return'].rolling(window).apply(
                lambda x: (x > 0).mean()
            )
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Rolling_InHouse_WinRate'],
                mode='lines',
                name=f'{window}-Day InHouse Win Rate',
                line=dict(color='#1f77b4')
            ))
            fig4.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Rolling_Outperformance_Rate'],
                mode='lines',
                name=f'{window}-Day Outperformance Rate',
                line=dict(color='#2ca02c')
            ))
            fig4.add_hline(y=0.5, line_dash="dash", line_color="red")
            fig4.update_layout(
                title=f"Rolling {window}-Day Win Rates",
                xaxis_title="Date",
                yaxis_title="Win Rate",
                yaxis=dict(tickformat='.1%'),
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.header("Statistical Significance Tests")
        
        # Create significance results table
        significance_data = [
            {
                'Test': 'InHouse Returns vs Zero',
                'Statistic': f"{metrics['inhouse_return_t_stat']:.3f}",
                'P-Value': f"{metrics['inhouse_return_p_value']:.4f}",
                'Significant (α=0.05)': '✅ Yes' if metrics['inhouse_return_significant'] else '❌ No',
                'Interpretation': 'Returns significantly different from zero' if metrics['inhouse_return_significant'] else 'Returns not significantly different from zero'
            },
            {
                'Test': 'Composer Returns vs Zero',
                'Statistic': f"{metrics['composer_return_t_stat']:.3f}",
                'P-Value': f"{metrics['composer_return_p_value']:.4f}",
                'Significant (α=0.05)': '✅ Yes' if metrics['composer_return_significant'] else '❌ No',
                'Interpretation': 'Returns significantly different from zero' if metrics['composer_return_significant'] else 'Returns not significantly different from zero'
            },
            {
                'Test': 'Relative Returns vs Zero',
                'Statistic': f"{metrics['relative_return_t_stat']:.3f}",
                'P-Value': f"{metrics['relative_return_p_value']:.4f}",
                'Significant (α=0.05)': '✅ Yes' if metrics['relative_return_significant'] else '❌ No',
                'Interpretation': 'Significant outperformance' if metrics['relative_return_significant'] and metrics['relative_mean_return'] > 0 else 'No significant outperformance'
            },
            {
                'Test': 'InHouse Win Rate vs 50%',
                'Statistic': 'Binomial Test',
                'P-Value': f"{metrics['inhouse_winrate_p_value']:.4f}",
                'Significant (α=0.05)': '✅ Yes' if metrics['inhouse_winrate_significant'] else '❌ No',
                'Interpretation': 'Win rate significantly different from 50%' if metrics['inhouse_winrate_significant'] else 'Win rate not significantly different from 50%'
            },
            {
                'Test': 'Outperformance Rate vs 50%',
                'Statistic': 'Binomial Test',
                'P-Value': f"{metrics['outperformance_p_value']:.4f}",
                'Significant (α=0.05)': '✅ Yes' if metrics['outperformance_significant'] else '❌ No',
                'Interpretation': 'Outperformance rate significantly different from 50%' if metrics['outperformance_significant'] else 'Outperformance rate not significantly different from 50%'
            }
        ]
        
        significance_df = pd.DataFrame(significance_data)
        st.dataframe(significance_df, use_container_width=True)
        
        # P-value visualization
        p_values = [
            metrics['inhouse_return_p_value'],
            metrics['composer_return_p_value'], 
            metrics['relative_return_p_value'],
            metrics['inhouse_winrate_p_value'],
            metrics['outperformance_p_value']
        ]
        test_names = [
            'InHouse\nReturns', 'Composer\nReturns', 'Relative\nReturns',
            'InHouse\nWin Rate', 'Outperformance\nRate'
        ]
        
        fig5 = go.Figure(data=[
            go.Bar(x=test_names, y=p_values, 
                   marker_color=['red' if p < 0.05 else 'blue' for p in p_values])
        ])
        fig5.add_hline(y=0.05, line_dash="dash", line_color="red", 
                      annotation_text="α = 0.05 significance level")
        fig5.update_layout(
            title="P-Values for Statistical Tests",
            yaxis_title="P-Value",
            height=400
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab4:
        st.header("Risk and Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Metrics")
            perf_metrics = pd.DataFrame({
                'Metric': [
                    'Mean Daily Return',
                    'Annualized Return',
                    'Volatility (Annualized)',
                    'Sharpe Ratio',
                    'Win Rate'
                ],
                'InHouse': [
                    f"{metrics['inhouse_mean_return']:.4%}",
                    f"{metrics['inhouse_mean_return'] * 252:.2%}",
                    f"{metrics['inhouse_volatility']:.2%}",
                    f"{metrics['inhouse_sharpe']:.3f}",
                    f"{metrics['inhouse_win_rate']:.1%}"
                ],
                'Composer': [
                    f"{metrics['composer_mean_return']:.4%}",
                    f"{metrics['composer_mean_return'] * 252:.2%}",
                    f"{metrics['composer_volatility']:.2%}",
                    f"{metrics['composer_sharpe']:.3f}",
                    f"{metrics['composer_win_rate']:.1%}"
                ]
            })
            st.dataframe(perf_metrics, use_container_width=True)
        
        with col2:
            st.subheader("📈 Relative Performance")
            relative_metrics = pd.DataFrame({
                'Metric': [
                    'Mean Relative Return',
                    'Annualized Relative Return',
                    'Information Ratio',
                    'Outperformance Rate',
                    'Best Relative Day',
                    'Worst Relative Day'
                ],
                'Value': [
                    f"{metrics['relative_mean_return']:.4%}",
                    f"{metrics['relative_mean_return'] * 252:.2%}",
                    f"{metrics['information_ratio']:.3f}",
                    f"{metrics['outperformance_rate']:.1%}",
                    f"{df['Relative_Daily_Return'].max():.2%}",
                    f"{df['Relative_Daily_Return'].min():.2%}"
                ]
            })
            st.dataframe(relative_metrics, use_container_width=True)
        
        # Return distributions
        fig6 = make_subplots(rows=1, cols=2, subplot_titles=('InHouse Returns', 'Relative Returns'))
        
        fig6.add_trace(
            go.Histogram(x=df['InHouse_Daily_Return'].dropna(), name='InHouse', 
                        marker_color='#1f77b4', opacity=0.7),
            row=1, col=1
        )
        
        fig6.add_trace(
            go.Histogram(x=df['Relative_Daily_Return'].dropna(), name='Relative', 
                        marker_color='#2ca02c', opacity=0.7),
            row=1, col=2
        )
        
        fig6.update_layout(title="Return Distributions", height=400)
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab5:
        st.header("Detailed Portfolio Data")
        
        # Display configuration
        st.subheader("📋 Current Configuration")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write("**InHouse Portfolio:**", ', '.join(inhouse_symbols))
            st.write("**Date Range:**", f"{start_date} to {end_date}")
        
        with config_col2:
            st.write("**Composer Portfolio:**", ', '.join(composer_symbols))
            st.write("**Total Trading Days:**", len(df))
        
        # Data table
        st.subheader("📊 Portfolio Data")
        display_columns = [
            'Date', 'InHouse_Portfolio_Value', 'Composer_Portfolio_Value',
            'Relative_Performance', 'InHouse_Daily_Return', 'Composer_Daily_Return',
            'Relative_Daily_Return', 'Outperformance_Days'
        ]
        
        display_df = df[display_columns].copy()
        display_df['InHouse_Daily_Return'] = display_df['InHouse_Daily_Return'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        display_df['Composer_Daily_Return'] = display_df['Composer_Daily_Return'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        display_df['Relative_Daily_Return'] = display_df['Relative_Daily_Return'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        summary_stats = df[['InHouse_Portfolio_Value', 'Composer_Portfolio_Value', 'Relative_Performance']].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Download data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Portfolio Data as CSV",
            data=csv,
            file_name=f"portfolio_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Footer with additional insights
    st.markdown("---")
    st.markdown("## 🎯 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("### 📊 Performance Summary")
        
        if metrics['relative_mean_return'] > 0:
            performance_text = f"✅ InHouse portfolio outperformed by {metrics['relative_mean_return']:.4%} daily on average"
        else:
            performance_text = f"❌ InHouse portfolio underperformed by {abs(metrics['relative_mean_return']):.4%} daily on average"
        
        st.markdown(f"""
        - {performance_text}
        - InHouse win rate: {metrics['inhouse_win_rate']:.1%}
        - Outperformance rate: {metrics['outperformance_rate']:.1%}
        - Information ratio: {metrics['information_ratio']:.3f}
        """)
    
    with insights_col2:
        st.markdown("### 🔬 Statistical Confidence")
        
        confidence_items = []
        if metrics['relative_return_significant']:
            if metrics['relative_mean_return'] > 0:
                confidence_items.append("✅ Outperformance is statistically significant")
            else:
                confidence_items.append("❌ Underperformance is statistically significant")
        else:
            confidence_items.append("⚠️ Performance difference is not statistically significant")
        
        if metrics['inhouse_winrate_significant']:
            confidence_items.append("✅ Win rate is significantly different from 50%")
        else:
            confidence_items.append("⚠️ Win rate is not significantly different from 50%")
        
        if metrics['outperformance_significant']:
            confidence_items.append("✅ Outperformance rate is significantly different from 50%")
        else:
            confidence_items.append("⚠️ Outperformance rate is not significantly different from 50%")
        
        for item in confidence_items:
            st.markdown(f"- {item}")

    # Technical notes
    with st.expander("📝 Technical Notes"):
        st.markdown("""
        **Statistical Tests Performed:**
        - **T-tests**: Test if mean returns are significantly different from zero
        - **Binomial tests**: Test if win rates are significantly different from 50%
        - **Information Ratio**: Measures risk-adjusted outperformance
        
        **Definitions:**
        - **Win Rate**: Percentage of days with positive returns
        - **Outperformance Rate**: Percentage of days InHouse beats Composer
        - **Information Ratio**: Annualized relative return / relative return volatility
        - **Statistical Significance**: p-value < 0.05 (95% confidence level)
        
        **Data Source**: Yahoo Finance via yfinance library
        **Assumptions**: Equal weighting within portfolios, daily rebalancing
        """)

if __name__ == "__main__":
    main()
