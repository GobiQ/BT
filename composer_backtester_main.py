import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import json
import ast
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the CSV data"""
    try:
        df = pd.read_csv('daily_comparison_20250809_170850.csv')
        
        # Convert date columns
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date_Str'] = pd.to_datetime(df['Date_Str'])
        
        # Convert boolean column
        df['Rebalanced'] = df['Rebalanced'].map({'True': True, 'False': False})
        
        # Sort by date
        df = df.sort_values('Date')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def parse_json_column(df, column_name):
    """Parse JSON-like strings in a column"""
    parsed_data = []
    for value in df[column_name]:
        try:
            if pd.isna(value) or value == '':
                parsed_data.append({})
            else:
                # Handle the JSON string format
                cleaned = value.replace('""', '"')
                parsed_data.append(json.loads(cleaned))
        except:
            parsed_data.append({})
    return parsed_data

def safe_eval_dict(s):
    """Safely evaluate dictionary strings"""
    if pd.isna(s) or s == '':
        return {}
    try:
        # Handle numpy float64 references in string
        s = str(s).replace('np.float64(', '').replace(')', '')
        return ast.literal_eval(s)
    except:
        return {}

def main():
    st.markdown('<div class="main-header">📊 Portfolio Comparison Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("📅 Filters")
    
    # Date range filter
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                        (df['Date'] <= pd.to_datetime(end_date))]
    else:
        filtered_df = df
    
    # Rebalanced filter
    rebalanced_filter = st.sidebar.selectbox(
        "Rebalanced Status",
        options=['All', 'True', 'False'],
        index=0
    )
    
    if rebalanced_filter != 'All':
        filtered_df = filtered_df[filtered_df['Rebalanced'] == (rebalanced_filter == 'True')]
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_match_score = filtered_df['Match_Score'].mean()
        st.metric(
            label="📈 Average Match Score",
            value=f"{avg_match_score:.4f}",
            delta=f"{avg_match_score - df['Match_Score'].mean():.4f}"
        )
    
    with col2:
        avg_portfolio_value = filtered_df['InHouse_Portfolio_Value'].mean()
        st.metric(
            label="💰 Average Portfolio Value",
            value=f"${avg_portfolio_value:,.2f}",
            delta=f"${avg_portfolio_value - df['InHouse_Portfolio_Value'].mean():,.2f}"
        )
    
    with col3:
        avg_asset_match = filtered_df['Asset_Selection_Match'].mean()
        st.metric(
            label="🎯 Average Asset Selection Match",
            value=f"{avg_asset_match:.2f}",
            delta=f"{avg_asset_match - df['Asset_Selection_Match'].mean():.2f}"
        )
    
    with col4:
        rebalanced_count = filtered_df['Rebalanced'].sum()
        st.metric(
            label="🔄 Rebalanced Days",
            value=f"{rebalanced_count}",
            delta=f"{rebalanced_count - df['Rebalanced'].sum()}"
        )
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series", "🎯 Match Analysis", "💰 Portfolio Performance", 
        "📊 Asset Analysis", "🔍 Detailed Data"
    ])
    
    with tab1:
        st.header("Time Series Analysis")
        
        # Portfolio value over time
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['InHouse_Portfolio_Value'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        fig1.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Match scores over time
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig2.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Match_Score'],
                mode='lines+markers',
                name='Match Score',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=4)
            ),
            secondary_y=False,
        )
        
        fig2.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Asset_Selection_Match'],
                mode='lines+markers',
                name='Asset Selection Match',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=4)
            ),
            secondary_y=True,
        )
        
        fig2.update_xaxes(title_text="Date")
        fig2.update_yaxes(title_text="Match Score", secondary_y=False)
        fig2.update_yaxes(title_text="Asset Selection Match", secondary_y=True)
        fig2.update_layout(
            title="Match Scores Over Time",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("Match Score Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Match score distribution
            fig3 = px.histogram(
                filtered_df,
                x='Match_Score',
                nbins=30,
                title="Match Score Distribution",
                labels={'Match_Score': 'Match Score', 'count': 'Frequency'}
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Asset selection match distribution
            fig4 = px.histogram(
                filtered_df,
                x='Asset_Selection_Match',
                nbins=20,
                title="Asset Selection Match Distribution",
                labels={'Asset_Selection_Match': 'Asset Selection Match', 'count': 'Frequency'}
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Correlation heatmap
        numeric_cols = ['Match_Score', 'Asset_Selection_Match', 'InHouse_Portfolio_Value', 
                       'InHouse_Num_Assets', 'Composer_Num_Assets']
        correlation_matrix = filtered_df[numeric_cols].corr()
        
        fig5 = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab3:
        st.header("Portfolio Performance Analysis")
        
        # Calculate returns
        filtered_df['Daily_Return'] = filtered_df['InHouse_Portfolio_Value'].pct_change()
        filtered_df['Cumulative_Return'] = (1 + filtered_df['Daily_Return']).cumprod() - 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cumulative returns
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Cumulative_Return'] * 100,
                mode='lines',
                name='Cumulative Return',
                line=dict(color='#1f77b4', width=2)
            ))
            fig6.update_layout(
                title="Cumulative Returns (%)",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=400
            )
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            # Daily returns distribution
            fig7 = px.histogram(
                filtered_df.dropna(subset=['Daily_Return']),
                x='Daily_Return',
                nbins=50,
                title="Daily Returns Distribution",
                labels={'Daily_Return': 'Daily Return', 'count': 'Frequency'}
            )
            fig7.update_layout(height=400)
            st.plotly_chart(fig7, use_container_width=True)
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            total_return = filtered_df['Cumulative_Return'].iloc[-1] if not filtered_df.empty else 0
            st.metric("Total Return", f"{total_return:.2%}")
        
        with perf_col2:
            volatility = filtered_df['Daily_Return'].std() * np.sqrt(252) if not filtered_df['Daily_Return'].isna().all() else 0
            st.metric("Annualized Volatility", f"{volatility:.2%}")
        
        with perf_col3:
            max_value = filtered_df['InHouse_Portfolio_Value'].max()
            min_value = filtered_df['InHouse_Portfolio_Value'].min()
            max_drawdown = (min_value - max_value) / max_value
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        
        with perf_col4:
            avg_daily_return = filtered_df['Daily_Return'].mean()
            sharpe_ratio = avg_daily_return / filtered_df['Daily_Return'].std() * np.sqrt(252) if filtered_df['Daily_Return'].std() != 0 else 0
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with tab4:
        st.header("Asset Analysis")
        
        # Number of assets over time
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['InHouse_Num_Assets'],
            mode='lines+markers',
            name='InHouse Assets',
            line=dict(color='#1f77b4', width=2)
        ))
        fig8.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Composer_Num_Assets'],
            mode='lines+markers',
            name='Composer Assets',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig8.update_layout(
            title="Number of Assets Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Assets",
            height=400
        )
        st.plotly_chart(fig8, use_container_width=True)
        
        # Asset count distributions
        col1, col2 = st.columns(2)
        
        with col1:
            fig9 = px.box(
                filtered_df,
                y='InHouse_Num_Assets',
                title="InHouse Assets Distribution"
            )
            fig9.update_layout(height=400)
            st.plotly_chart(fig9, use_container_width=True)
        
        with col2:
            fig10 = px.box(
                filtered_df,
                y='Composer_Num_Assets',
                title="Composer Assets Distribution"
            )
            fig10.update_layout(height=400)
            st.plotly_chart(fig10, use_container_width=True)
    
    with tab5:
        st.header("Detailed Data View")
        
        # Search and filter options
        st.subheader("🔍 Search and Filter")
        
        search_col1, search_col2 = st.columns(2)
        
        with search_col1:
            search_assets = st.text_input("Search Assets (InHouse or Composer)", "")
        
        with search_col2:
            sort_column = st.selectbox(
                "Sort by",
                options=['Date', 'Match_Score', 'InHouse_Portfolio_Value', 'Asset_Selection_Match'],
                index=0
            )
        
        # Apply search filter
        display_df = filtered_df.copy()
        if search_assets:
            mask = (display_df['InHouse_Assets'].str.contains(search_assets, case=False, na=False) |
                   display_df['Composer_Assets'].str.contains(search_assets, case=False, na=False))
            display_df = display_df[mask]
        
        # Sort data
        display_df = display_df.sort_values(sort_column, ascending=False)
        
        # Display options
        show_all = st.checkbox("Show all columns", value=False)
        
        if show_all:
            st.dataframe(display_df, use_container_width=True, height=500)
        else:
            # Show selected columns
            key_columns = [
                'Date', 'InHouse_Assets', 'Composer_Assets', 'Common_Assets',
                'Asset_Selection_Match', 'Match_Score', 'InHouse_Portfolio_Value',
                'Rebalanced', 'InHouse_Num_Assets', 'Composer_Num_Assets'
            ]
            st.dataframe(display_df[key_columns], use_container_width=True, height=500)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        st.dataframe(filtered_df.describe(), use_container_width=True)
        
        # Download filtered data
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"portfolio_comparison_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "📊 **Portfolio Comparison Dashboard** | "
        f"Data Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')} | "
        f"Total Records: {len(df)}"
    )

if __name__ == "__main__":
    main()
