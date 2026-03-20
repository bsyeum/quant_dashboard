"""
BSQuant Dashboard — Streamlit App
===================================
Multi-Strategy Ensemble for Global Asset Allocation
by Bosun Yeum

Usage:
    streamlit run bsquant_dashboard.py

Requirements:
    pip install streamlit plotly pandas numpy yfinance pandas_datareader
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Import CAA engine
from bsquant_strategy import (
    download_prices, download_unemployment_rate, get_month_end_dates,
    strategy_baa, strategy_faa, strategy_raa, strategy_paa, strategy_laa, strategy_haa,
    combine_caa, create_leveraged_prices, backtest, calc_performance,
    CORE_ASSETS, RISK_ASSETS, SAFE_ASSETS, LEVERAGE_2X, LEVERAGE_3X,
)

# ─── Page Config ───
st.set_page_config(
    page_title="BSQuant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expander - white text on dark bar (updated selector) */
    [data-testid="stExpander"] summary {
        color: #FFFFFF !important;
        background: #1E293B !important;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Header bar - visible Deploy button */
    [data-testid="stHeader"] { background: #F8FAFC !important; }
    [data-testid="stHeader"] * { color: #0F172A !important; }

    .stApp {
        background: #F8FAFC;
        font-family: 'Outfit', sans-serif;
        color: #1E293B;
    }

    /* Force ALL text dark on light theme */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp .stMarkdown, .stApp .stText {
        color: #1E293B;
    }

    /* Sidebar - force dark bg + light text */
    [data-testid="stSidebar"] {
        background: #0F172A !important;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background: #1E293B !important;
        color: #F1F5F9 !important;
        border: 1px solid #334155 !important;
    }

    /* Tabs - dark text on light bg */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #F1F5F9;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 500;
        color: #334155 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0F172A !important;
        font-weight: 700;
    }

    /* DataFrame / table text - force dark */
    .stDataFrame, .stDataFrame td, .stDataFrame th,
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        color: #1E293B !important;
    }

    /* Expander content */
    .streamlit-expanderContent, [data-testid="stExpander"] > div {
        color: #1E293B !important;
    }

    /* Selectbox / dropdown - force dark text */
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div {
        color: #1E293B !important;
    }

    /* Download button - white text on blue */
    [data-testid="stDownloadButton"] button {
        color: #FFFFFF !important;
        background: #2563EB !important;
        border: none !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #1D4ED8 !important;
    }

    /* DataFrame toolbar - search bar icons */
    [data-testid="stDataFrame"] [data-testid="stDataFrameToolbar"],
    [data-testid="stDataFrame"] input,
    [data-testid="stDataFrame"] button {
        color: #FFFFFF !important;
        background: #1E293B !important;
    }
    [data-testid="stDataFrame"] svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    /* Plotly chart text - force readable */
    .js-plotly-plot .plotly text { fill: #334155 !important; }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
    }
    .metric-value.positive { color: #059669 !important; }
    .metric-value.negative { color: #DC2626 !important; }
    .metric-value.neutral { color: #1E293B !important; }
    .metric-sub {
        font-size: 12px;
        color: #94A3B8 !important;
        margin-top: 4px;
    }

    .dashboard-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
    }
    .dashboard-title {
        font-family: 'Outfit', sans-serif;
        font-size: 28px;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .dashboard-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #64748B !important;
        margin-top: 4px;
    }

    .strategy-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        margin: 2px;
    }
    .badge-offensive { background: #ECFDF5; color: #059669 !important; border: 1px solid #A7F3D0; }
    .badge-defensive { background: #FEF2F2; color: #DC2626 !important; border: 1px solid #FECACA; }

    .signal-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .signal-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
            
    /* Sidebar selectbox value - light text */
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
    }

    /* Markdown headers - force dark text on main area */
    .stMarkdown h4, .stMarkdown h3, .stMarkdown h2 {
        color: #0F172A !important;
    }


</style>
""", unsafe_allow_html=True)


# ─── Caching: Download + Compute Once ───
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(start_date, backtest_start):
    """Download and cache all price data + run all strategies."""
    prices = download_prices(start=start_date)
    unrate = download_unemployment_rate()
    rebal_dates = get_month_end_dates(prices)

    # Run 6 strategies
    w_baa = strategy_baa(prices, rebal_dates)
    w_faa = strategy_faa(prices, rebal_dates)
    w_raa = strategy_raa(prices, rebal_dates, unrate)
    w_paa = strategy_paa(prices, rebal_dates)
    w_laa = strategy_laa(prices, rebal_dates, unrate)
    w_haa = strategy_haa(prices, rebal_dates)

    # Combine
    w_caa = combine_caa([w_baa, w_faa, w_raa, w_paa, w_laa, w_haa])

    # Backtest period
    bt_prices = prices[prices.index >= backtest_start]
    bt_rebal = rebal_dates[rebal_dates >= backtest_start]

    # Backtests
    commission = 0.0021
    eq_caa, ret_caa = backtest(bt_prices, w_caa, bt_rebal, commission_pct=commission)

    # Individual (zero commission)
    strategies = {}
    for name, w in [('BAA', w_baa), ('FAA', w_faa), ('RAA', w_raa),
                     ('PAA', w_paa), ('LAA', w_laa), ('HAA', w_haa)]:
        eq, ret = backtest(bt_prices, w, bt_rebal, commission_pct=0)
        strategies[name] = {'equity': eq, 'returns': ret, 'weights': w}

    # Benchmarks
    spy_eq = bt_prices['SPY'] / bt_prices['SPY'].iloc[0]
    tlt_eq = bt_prices['TLT'] / bt_prices['TLT'].iloc[0]

    # Leverage
    leverage_eq = {}
    for lev in [2, 3]:
        lev_prices = create_leveraged_prices(prices, leverage=lev)
        bt_lev = lev_prices[lev_prices.index >= backtest_start]
        eq_l, _ = backtest(bt_lev, w_caa, bt_rebal, commission_pct=commission)
        leverage_eq[f'BSQ {lev}x'] = eq_l

    return {
        'prices': prices,
        'unrate': unrate,
        'rebal_dates': rebal_dates,
        'bt_rebal': bt_rebal,
        'strategies': strategies,
        'w_caa': w_caa,
        'eq_caa': eq_caa,
        'ret_caa': ret_caa,
        'spy_eq': spy_eq,
        'tlt_eq': tlt_eq,
        'leverage_eq': leverage_eq,
    }


# ─── Helper: Build Performance Table ───
def build_perf_table(data):
    """Build a performance comparison DataFrame."""
    rows = []
    eq = data['eq_caa']
    ret = data['ret_caa']
    rows.append(calc_performance(eq, ret, 'BSQ 1x'))

    for name, s in data['strategies'].items():
        rows.append(calc_performance(s['equity'], s['returns'], name))

    spy_ret = data['spy_eq'].pct_change().fillna(0)
    rows.append(calc_performance(data['spy_eq'], spy_ret, 'S&P500'))
    tlt_ret = data['tlt_eq'].pct_change().fillna(0)
    rows.append(calc_performance(data['tlt_eq'], tlt_ret, 'US LT Bond'))

    for lev_name, eq_l in data['leverage_eq'].items():
        ret_l = eq_l.pct_change().fillna(0)
        rows.append(calc_performance(eq_l, ret_l, lev_name))

    return pd.DataFrame(rows).set_index('Name')


def monthly_returns_df(equity):
    """Generate month x year return matrix (Jan-Dec order)."""
    monthly = equity.resample('ME').last()
    monthly_rets = monthly.pct_change().dropna() * 100
    table = pd.DataFrame()
    for dt, ret in monthly_rets.items():
        table.loc[dt.year, dt.month] = round(ret, 1)
    # Force column order: 1-12
    month_cols = list(range(1, 13))
    table = table.reindex(columns=month_cols)
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    table = table.rename(columns=month_names)
    # Annual
    annual = equity.resample('YE').last()
    annual_rets = annual.pct_change().dropna() * 100
    for dt, ret in annual_rets.items():
        table.loc[dt.year, 'Year'] = round(ret, 1)
    table.index.name = 'Year'
    return table


def annual_returns_df(equity_dict):
    """Build annual returns for multiple strategies."""
    table = pd.DataFrame()
    for name, eq in equity_dict.items():
        annual = eq.resample('YE').last()
        annual_rets = annual.pct_change().dropna() * 100
        for dt, ret in annual_rets.items():
            table.loc[dt.year, name] = round(ret, 1)
    table.index.name = 'Year'
    return table


def weight_history_df(w_caa, rebal_dates):
    """Build monthly weight history."""
    w = w_caa.loc[w_caa.index.isin(rebal_dates)].copy()
    w = w[CORE_ASSETS].copy()
    w = w.fillna(0)
    w = (w * 100).round(1)
    w.index = w.index.strftime('%Y-%m')
    w.index.name = 'Date'
    return w


# ─── Plotly Chart Builders ───
COLORS = {
    'BSQ 1x': '#3B82F6', 'BSQ 2x': '#06B6D4', 'BSQ 3x': '#8B5CF6',
    'S&P500': '#EF4444', 'US LT Bond': '#94A3B8',
    'BAA': '#F59E0B', 'FAA': '#10B981', 'RAA': '#8B5CF6',
    'PAA': '#3B82F6', 'LAA': '#D97706', 'HAA': '#EF4444',
}

# Strategy display names (hide internal strategy codes)
STRATEGY_DISPLAY = {
    'BAA': '🔴 Regime Detection',
    'FAA': '📊 Multi-Factor',
    'RAA': '🌍 Macro Cycle',
    'PAA': '🛡️ Drawdown Shield',
    'LAA': '⚓ Structural Hedge',
    'HAA': '🌡️ Inflation Switch',
}
STRATEGY_SHORT = {
    'BAA': 'RD', 'FAA': 'MF', 'RAA': 'MC',
    'PAA': 'DS', 'LAA': 'SH', 'HAA': 'IS',
}
ASSET_COLORS = {
    'SPY': '#EF4444', 'QQQ': '#7C3AED', 'SMH': '#EC4899', 
    'VGK': '#3B82F6', 'FXI': '#F59E0B', 'EWY': '#10B981', 'IEMG': '#14B8A6',
    'USDU': '#8B5CF6', 'GLD': '#EAB308', 'PDBC': '#D97706',
    'TLT': '#06B6D4', 'IEF': '#6366F1', 'BIL': '#94A3B8',
}

CHART_LAYOUT = dict(
    template='plotly_white',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#FFFFFF',
    font=dict(family='JetBrains Mono, monospace', size=12, color='#334155'),
    title_font=dict(family='Outfit, sans-serif', size=18, color='#0F172A'),
    legend=dict(bgcolor='rgba(255,255,255,0.9)', borderwidth=0, font=dict(size=11, color='#334155')),
    margin=dict(l=60, r=30, t=60, b=40),
    xaxis=dict(gridcolor='#E2E8F0', showgrid=True, tickfont=dict(color='#64748B')),
)


def _hex_to_rgba(hex_color, alpha=0.15):
    """Convert hex color to rgba string for Plotly."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def chart_cumulative(data, show_leverage=False, show_individual=False):
    """Cumulative performance chart (log scale)."""
    fig = go.Figure()

    # Core: CAA, SPY, TLT
    for name, eq, dash in [
        ('BSQ 1x', data['eq_caa'], None),
        ('S&P500', data['spy_eq'], 'dot'),
        ('US LT Bond', data['tlt_eq'], 'dash'),
    ]:
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=name, mode='lines',
            line=dict(color=COLORS.get(name, '#999'), width=2.5 if 'CAA' in name else 1.5, dash=dash),
        ))

    if show_leverage:
        for lev_name, eq_l in data['leverage_eq'].items():
            fig.add_trace(go.Scatter(
                x=eq_l.index, y=eq_l.values,
                name=lev_name, mode='lines',
                line=dict(color=COLORS.get(lev_name, '#999'), width=2),
            ))

    if show_individual:
        for name, s in data['strategies'].items():
            fig.add_trace(go.Scatter(
                x=s['equity'].index, y=s['equity'].values,
                name=name, mode='lines',
                line=dict(color=COLORS.get(name, '#999'), width=1, dash='dot'),
                visible='legendonly',
            ))

    fig.update_layout(
        **CHART_LAYOUT,
        title='Cumulative Performance (Growth of $1)',
        yaxis=dict(type='log', dtick=1, gridcolor='#E2E8F0', tickfont=dict(color='#64748B')),
        yaxis_title='Growth of $1 (log scale)',
        hovermode='x unified',
        height=520,
    )
    return fig


def chart_drawdown(data):
    """Drawdown comparison chart."""
    fig = go.Figure()

    for name, eq, color in [
        ('BSQ', data['eq_caa'], COLORS['BSQ 1x']),
        ('S&P500', data['spy_eq'], COLORS['S&P500']),
    ]:
        dd = (eq / eq.cummax() - 1) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=name, mode='lines',
            fill='tozeroy',
            line=dict(color=color, width=1),
            fillcolor=_hex_to_rgba(color, 0.15),
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title='Drawdown Comparison',
        yaxis_title='Drawdown (%)',
        height=350,
        hovermode='x unified',
    )
    return fig


def chart_weights_area(data):
    """Stacked area chart of CAA weights over time."""
    w = data['w_caa'].loc[data['w_caa'].index.isin(data['bt_rebal'])].copy()
    w = w[CORE_ASSETS].fillna(0)

    fig = go.Figure()
    for asset in CORE_ASSETS:
        fig.add_trace(go.Scatter(
            x=w.index, y=w[asset].values * 100,
            name=asset, mode='lines',
            stackgroup='one',
            line=dict(width=0.5, color=ASSET_COLORS.get(asset, '#666')),
            fillcolor=ASSET_COLORS.get(asset, '#666'),
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title='Portfolio Allocation Over Time',
        yaxis_title='Weight (%)',
        yaxis=dict(range=[0, 105], gridcolor='#E2E8F0', tickfont=dict(color='#64748B')),
        height=420,
        hovermode='x unified',
    )
    return fig


def chart_annual_bars(annual_df, strategies_to_show):
    """Grouped bar chart of annual returns."""
    fig = go.Figure()
    for col in strategies_to_show:
        if col in annual_df.columns:
            colors = [COLORS.get('BSQ 1x') if v >= 0 else COLORS.get('S&P500') for v in annual_df[col].fillna(0)]
            fig.add_trace(go.Bar(
                x=annual_df.index, y=annual_df[col],
                name=col,
                marker_color=COLORS.get(col, '#3B82F6'),
                opacity=0.85,
            ))

    fig.update_layout(
        **CHART_LAYOUT,
        title='Annual Returns Comparison (%)',
        barmode='group',
        yaxis_title='Return (%)',
        height=420,
        hovermode='x unified',
    )
    return fig


def style_monthly_table(df):
    """Color-code monthly returns: green for positive, red for negative."""
    def color_cell(val):
        if pd.isna(val):
            return 'color: #CBD5E1; background-color: #F8FAFC;'
        elif val > 0:
            intensity = min(abs(val) / 8.0, 1.0)
            return f'color: #064E3B; background-color: rgba(16, 185, 129, {0.08 + intensity * 0.3});'
        elif val < 0:
            intensity = min(abs(val) / 8.0, 1.0)
            return f'color: #7F1D1D; background-color: rgba(239, 68, 68, {0.08 + intensity * 0.3});'
        else:
            return 'color: #64748B; background-color: #FFFFFF;'
    return df.style.applymap(color_cell).format('{:.1f}', na_rep='—')


# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    start_year = st.selectbox("Backtest Start", [1997, 2000, 2005, 2010, 2015], index=0)
    backtest_start = f"{start_year}-02-01"
    data_start = f"{start_year - 2}-01-01"

    leverage_option = st.selectbox("Leverage", ["1x", "2x", "3x", "All"], index=0)
    show_individual = st.checkbox("Show Individual Strategies", value=False)
    show_drawdown = st.checkbox("Show Drawdown", value=True)

    st.markdown("---")
    st.markdown("### 📋 Strategy Info")
    st.markdown("""
    **BSQuant Engine** — 6 Modules:
    - 🔴 **Regime Detection** — Cross-asset momentum regime switching
    - 📊 **Multi-Factor** — Return / Volatility / Correlation scoring
    - 🌍 **Macro Cycle** — Economic indicator-based allocation
    - 🛡️ **Drawdown Shield** — Breadth momentum tail risk management
    - ⚓ **Structural Hedge** — Permanent portfolio with dual-side hedging
    - 🌡️ **Inflation Switch** — Real yield signal-driven rotation

    **Universe**: 13 global ETFs

    **Rebalancing**: Monthly (EOM)

    **Transaction cost**: 0.21%
    """)

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─── MAIN: LOAD DATA ───
with st.spinner("📡 Downloading data & computing strategies..."):
    data = load_data(data_start, backtest_start)

perf = build_perf_table(data)
last_date = data['eq_caa'].index[-1].strftime('%Y-%m-%d')

# ─── HEADER ───
st.markdown(f"""
<div class="dashboard-header">
    <p class="dashboard-title">📊 BSQuant Dashboard</p>
    <p class="dashboard-subtitle">Multi-Strategy Ensemble for Global Asset Allocation &nbsp;|&nbsp; Data through {last_date} &nbsp;|&nbsp; Backtest from {start_year}</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI CARDS ───
caa_perf = perf.loc['BSQ 1x'] if 'BSQ 1x' in perf.index else perf.iloc[0]
spy_perf = perf.loc['S&P500'] if 'S&P500' in perf.index else None

cols = st.columns(5)
kpis = [
    ("CAGR", f"{caa_perf['CAGR(%)']:.1f}%", "positive" if caa_perf['CAGR(%)'] > 0 else "negative", f"S&P500: {spy_perf['CAGR(%)']:.1f}%" if spy_perf is not None else ""),
    ("VOL", f"{caa_perf['VOL(%)']:.1f}%", "neutral", f"S&P500: {spy_perf['VOL(%)']:.1f}%" if spy_perf is not None else ""),
    ("MDD", f"{caa_perf['MDD(%)']:.1f}%", "negative", f"S&P500: {spy_perf['MDD(%)']:.1f}%" if spy_perf is not None else ""),
    ("SHARPE", f"{caa_perf['Sharpe']:.2f}", "positive" if caa_perf['Sharpe'] > 1 else "neutral", f"S&P500: {spy_perf['Sharpe']:.2f}" if spy_perf is not None else ""),
    ("CALMAR", f"{caa_perf['Calmar']:.2f}", "positive" if caa_perf['Calmar'] > 0.8 else "neutral", f"S&P500: {spy_perf['Calmar']:.2f}" if spy_perf is not None else ""),
]

for col, (label, value, css_class, sub) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <p class="metric-value {css_class}">{value}</p>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── LIVE SIGNAL ───
with st.expander("🔔 Current Signal (Latest Rebalancing)", expanded=True):
    latest_rebal = data['bt_rebal'][-1]
    signal_cols = st.columns(6)

    strategy_names = ['BAA', 'FAA', 'RAA', 'PAA', 'LAA', 'HAA']
    for i, (name, sc) in enumerate(zip(strategy_names, signal_cols)):
        w = data['strategies'][name]['weights']
        display_name = STRATEGY_DISPLAY.get(name, name)
        if latest_rebal in w.index:
            row = w.loc[latest_rebal]
            active = row[row > 0.001].sort_values(ascending=False)
            has_risk = any(a in RISK_ASSETS for a in active.index)
            badge_class = "badge-offensive" if has_risk else "badge-defensive"
            regime = "RISK-ON" if has_risk else "DEFENSIVE"
            positions = " / ".join([f"{a} {v:.0%}" for a, v in active.head(3).items()])
            with sc:
                st.markdown(f"""
                <div class="signal-box">
                    <div class="signal-title">{display_name}</div>
                    <span class="strategy-badge {badge_class}">{regime}</span>
                    <div style="font-size:11px; color:#64748B; margin-top:6px;">{positions}</div>
                </div>
                """, unsafe_allow_html=True)

    # CAA combined
    st.markdown("<br>", unsafe_allow_html=True)
    caa_row = data['w_caa'].loc[latest_rebal]
    active_caa = caa_row[caa_row > 0.001].sort_values(ascending=False)

    wcols = st.columns(len(active_caa))
    for j, (asset, wt) in enumerate(active_caa.items()):
        emoji = "🟢" if asset in RISK_ASSETS else "🔵"
        with wcols[j]:
            st.markdown(f"""
            <div class="metric-card" style="padding:12px 8px;">
                <div style="font-size:20px;">{emoji}</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:700; color:#0F172A;">{asset}</div>
                <div style="font-size:18px; font-weight:700; color:#2563EB;">{wt:.1%}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── MAIN TABS ───
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Cumulative Performance",
    "📊 Annual Returns (%)",
    "📅 Monthly Returns (%)",
    "⚖️ MP Weights History",
    "🔍 Strategy Detail",
])

# ── Tab 1: Cumulative Performance ──
with tab1:
    show_lev = leverage_option in ['2x', '3x', 'All']

    fig_cum = chart_cumulative(data, show_leverage=show_lev, show_individual=show_individual)
    st.plotly_chart(fig_cum, use_container_width=True)

    if show_drawdown:
        fig_dd = chart_drawdown(data)
        st.plotly_chart(fig_dd, use_container_width=True)

    # Performance table
    st.markdown("#### Performance Summary")
    display_perf = perf.copy()
    display_perf = display_perf.style.format({
        'CAGR(%)': '{:.2f}', 'VOL(%)': '{:.2f}', 'MDD(%)': '{:.2f}',
        'Sharpe': '{:.4f}', 'Calmar': '{:.4f}'
    })
    st.dataframe(display_perf, use_container_width=True, height=450)

# ── Tab 2: Annual Returns ──
with tab2:
    equity_dict = {'BSQ 1x': data['eq_caa'], 'S&P500': data['spy_eq']}
    for name, s in data['strategies'].items():
        display = STRATEGY_SHORT.get(name, name)
        equity_dict[display] = s['equity']
    for lev_name, eq_l in data['leverage_eq'].items():
        equity_dict[lev_name] = eq_l

    annual_df = annual_returns_df(equity_dict)

    # Bar chart
    bar_strategies = ['BSQ 1x', 'S&P500']
    if leverage_option == '2x':
        bar_strategies.append('BSQ 2x')
    elif leverage_option == '3x':
        bar_strategies.append('BSQ 3x')
    elif leverage_option == 'All':
        bar_strategies.extend(['BSQ 2x', 'BSQ 3x'])

    fig_annual = chart_annual_bars(annual_df, bar_strategies)
    st.plotly_chart(fig_annual, use_container_width=True)

    # Table
    st.markdown("#### Annual Returns Table (%)")
    styled_annual = annual_df.style.format('{:.1f}', na_rep='—').background_gradient(
        cmap='RdYlGn', vmin=-30, vmax=30, axis=None
    )
    st.dataframe(styled_annual, use_container_width=True, height=600)

# ── Tab 3: Monthly Returns ──
with tab3:
    st.markdown("#### BSQ 1x Monthly Returns (%)")
    mr_caa = monthly_returns_df(data['eq_caa'])
    st.dataframe(style_monthly_table(mr_caa), use_container_width=True, height=600)

    if leverage_option in ['2x', 'All'] and 'BSQ 2x' in data['leverage_eq']:
        st.markdown("#### BSQ 2x Monthly Returns (%)")
        mr_2x = monthly_returns_df(data['leverage_eq']['BSQ 2x'])
        st.dataframe(style_monthly_table(mr_2x), use_container_width=True, height=600)

    if leverage_option in ['3x', 'All'] and 'BSQ 3x' in data['leverage_eq']:
        st.markdown("#### BSQ 3x Monthly Returns (%)")
        mr_3x = monthly_returns_df(data['leverage_eq']['BSQ 3x'])
        st.dataframe(style_monthly_table(mr_3x), use_container_width=True, height=600)

# ── Tab 4: MP Weights History ──
with tab4:
    st.markdown("#### Asset Allocation Over Time")
    fig_weights = chart_weights_area(data)
    st.plotly_chart(fig_weights, use_container_width=True)

    st.markdown("#### Monthly Portfolio Weights (%, Downloadable)")
    w_hist = weight_history_df(data['w_caa'], data['bt_rebal'])

    # Download button
    csv = w_hist.to_csv()
    st.download_button(
        label="📥 Download Weights CSV",
        data=csv,
        file_name=f"caa_weights_{last_date}.csv",
        mime="text/csv",
    )

    # Display table (reverse chronological)
    w_display = w_hist.iloc[::-1]
    st.dataframe(
        w_display.style.format('{:.1f}').background_gradient(
            cmap='Blues', vmin=0, vmax=50, axis=None
        ),
        use_container_width=True,
        height=600,
    )


# ─── FOOTER ───
st.markdown(f"""
<div style="text-align:center; padding:30px; color:#94A3B8; font-size:11px; font-family:'JetBrains Mono',monospace;">
    BSQuant Dashboard v1.0 &nbsp;|&nbsp; Multi-Strategy Ensemble by Bosun Yeum
    <br>Data: Yahoo Finance &nbsp;|&nbsp; Last updated: {last_date}
    <br>⚠️ This is a research tool, not investment advice.
</div>
""", unsafe_allow_html=True)
