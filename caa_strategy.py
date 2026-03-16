"""
CAA (Combination of Asset Allocation) — Python Implementation
==============================================================
Based on: DB금융투자 "자산배분의 다음 단계" (2023.04.03, 강현기)

6 strategies combined with equal weight (1/6):
  1. BAA (Bold Asset Allocation)
  2. FAA (Flexible Asset Allocation)
  3. RAA (Resilient Asset Allocation)
  4. PAA (Protective Asset Allocation)
  5. LAA (Lethargic Asset Allocation)
  6. HAA (Hybrid Asset Allocation)

Unified Universe (10 assets + 5 canary/timing):
  Risk:  SPY, VGK, FXI, EWY
  Safe:  USDU, GLD, PDBC, TLT, IEF, BIL
  Canary: SPY(=R1), VEA, VWO, BND, TIP
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Diagnostic: print yfinance version (API changed significantly across versions)
print(f"[INFO] yfinance version: {yf.__version__}")


# =============================================================================
# 1. DATA LOADER
# =============================================================================

# Ticker fallback chain: matches report R code exactly
# R code: "S1=USDU+DX-Y.NYB", "S3=PDBC+PCRAX+CVX", "Z2=VEA+VTMGX+FDIVX", etc.
# Order: [primary ETF, fallback1, fallback2, ...]
# Download tries each in order; splices earlier data from later tickers in the chain.
TICKER_CHAIN = {
    # --- 10 Core Assets ---
    'SPY':  ['SPY', 'VFINX'],               # R1: 미국 주식
    'VGK':  ['VGK', 'VEURX'],               # R2: 유럽 주식
    'FXI':  ['FXI', 'VEIEX'],               # R3: 중국 주식
    'EWY':  ['EWY', 'VEIEX'],               # R4: 한국 주식
    'USDU': ['USDU', 'DX-Y.NYB', 'DX=F', 'UUP'],  # S1: 달러 (multiple fallbacks for reliability)
    'GLD':  ['GLD', 'FKRCX'],               # S2: 금
    'PDBC': ['PDBC', 'PCRAX', 'CVX'],       # S3: 원자재 (3단계 chain)
    'TLT':  ['TLT', 'VUSTX', 'EDV'],           # S4: 미국 장기국채 (EDV = Vanguard Extended Duration)
    'IEF':  ['IEF', 'VFITX'],               # S5: 미국 중기국채
    'BIL':  ['BIL', 'VFISX'],               # S6: 미국 초단기국채
    # --- Canary / Timing ---
    'VEA':  ['VEA', 'VTMGX', 'FDIVX'],      # Z2: 선진국 주식 (3단계)
    'VWO':  ['VWO', 'VEIEX'],               # Z3: 신흥국 주식
    'BND':  ['BND', 'VBMFX'],               # Z4: 미국 종합채권
    'TIP':  ['TIP', 'VIPSX', 'LSGSX'],      # Z5: 미국 물가연동채 (3단계)
}

# Leverage ETF mapping
LEVERAGE_2X = {
    'SPY': 'SSO', 'VGK': 'UPV', 'FXI': 'XPP', 'EWY': 'EET',
    'USDU': 'USDU', 'GLD': 'GLD', 'PDBC': 'PDBC',  # 1x 유지 (PTP 회피)
    'TLT': 'UBT', 'IEF': 'UST', 'BIL': 'BIL',       # BIL 1x 유지
}
LEVERAGE_3X = {
    'SPY': 'SPXL', 'VGK': 'EURL', 'FXI': 'YINN', 'EWY': 'EDC',
    'USDU': 'USDU', 'GLD': 'GLD', 'PDBC': 'PDBC',
    'TLT': 'TMF', 'IEF': 'TYD', 'BIL': 'BIL',
}

CORE_ASSETS = ['SPY', 'VGK', 'FXI', 'EWY', 'USDU', 'GLD', 'PDBC', 'TLT', 'IEF', 'BIL']
RISK_ASSETS = ['SPY', 'VGK', 'FXI', 'EWY']
SAFE_ASSETS = ['USDU', 'GLD', 'PDBC', 'TLT', 'IEF']
CANARY_TICKERS = ['VEA', 'VWO', 'BND', 'TIP']
ALL_TICKERS = CORE_ASSETS + CANARY_TICKERS


def _extract_close(df):
    """
    Extract close price Series from yfinance download result.
    Handles both old API (Adj Close column) and new API (Close, possibly multi-level).
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Flatten multi-level columns if present (yfinance >= 0.2.31)
    if isinstance(df.columns, pd.MultiIndex):
        # e.g. ('Close', 'SPY') → just grab 'Close' level
        if 'Adj Close' in df.columns.get_level_values(0):
            df = df['Adj Close']
        elif 'Close' in df.columns.get_level_values(0):
            df = df['Close']
        # If it's now a DataFrame with one column, squeeze to Series
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        return df

    # Single-level columns (older yfinance or single ticker)
    if 'Adj Close' in df.columns:
        return df['Adj Close']
    elif 'Close' in df.columns:
        return df['Close']
    else:
        # Last resort: take last column
        return df.iloc[:, -1]


def _download_single(ticker, start, end, max_retries=3):
    """Download a single ticker with retry, return Series or empty Series on failure."""
    import time
    for attempt in range(max_retries):
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            series = _extract_close(raw)
            if isinstance(series, pd.Series) and len(series) > 0:
                return series
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
            pass
    return pd.Series(dtype=float)


def _splice_series(primary, fallback):
    """
    Splice two price series: use fallback data before primary inception,
    scaling fallback to match primary at the overlap point.
    """
    if len(primary) == 0:
        return fallback
    if len(fallback) == 0:
        return primary

    primary_start = primary.index[0]
    fallback_before = fallback[fallback.index < primary_start]

    if len(fallback_before) == 0:
        return primary

    # Scale fallback to match primary at splice point
    if primary_start in fallback.index:
        scale = float(primary.iloc[0]) / float(fallback.loc[primary_start])
    else:
        scale = float(primary.iloc[0]) / float(fallback_before.iloc[-1])

    fallback_scaled = fallback_before * scale
    combined = pd.concat([fallback_scaled, primary])
    combined = combined[~combined.index.duplicated(keep='last')]
    return combined


def download_prices(start='1995-01-01', end=None):
    """
    Download daily adjusted close prices for all tickers.
    Uses chain fallback: tries each ticker in TICKER_CHAIN in order,
    splicing earlier data from later tickers (matching report R code logic).
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    prices = pd.DataFrame()

    for name in ALL_TICKERS:
        chain = TICKER_CHAIN[name]

        # Download all tickers in the chain
        chain_data = {}
        for ticker in chain:
            series = _download_single(ticker, start, end)
            if len(series) > 0:
                chain_data[ticker] = series

        if not chain_data:
            print(f"  WARNING: No data for {name} (tried: {chain})")
            continue

        # Splice chain: start from the LAST (oldest) fallback, layer on top
        # E.g., for ['PDBC', 'PCRAX', 'CVX']:
        #   1. Start with CVX (oldest)
        #   2. Splice PCRAX on top (scales to match)
        #   3. Splice PDBC on top (scales to match)
        reversed_chain = list(reversed(chain))
        combined = pd.Series(dtype=float)

        for ticker in reversed_chain:
            if ticker in chain_data:
                if len(combined) == 0:
                    combined = chain_data[ticker]
                else:
                    combined = _splice_series(chain_data[ticker], combined)

        prices[name] = combined
        chain_str = ' → '.join([f"{t}({len(chain_data.get(t, []))})" for t in chain])
        print(f"  ✓ {name:5s}: {len(combined)} rows  [{chain_str}]")

    prices = prices.sort_index()

    # Track which assets need backfilling (short history)
    for col in prices.columns:
        first_valid = prices[col].first_valid_index()
        if first_valid and first_valid > prices.index[100]:  # More than 100 rows of leading NaN
            n_missing = prices.index.get_loc(first_valid)
            print(f"  ⚠️ {col}: backfilling {n_missing} early days (data starts {first_valid.strftime('%Y-%m-%d')})")

    prices = prices.ffill()
    prices = prices.bfill()  # Fill early gaps with first known value
    prices = prices.dropna()
    return prices


def download_unemployment_rate(start='1990-01-01'):
    """
    Download US Unemployment Rate from FRED via pandas_datareader or yfinance.
    Returns monthly Series with 2-month publication lag applied.
    """
    try:
        import pandas_datareader.data as web
        unrate = web.DataReader('UNRATE', 'fred', start=start)['UNRATE']
    except Exception:
        try:
            from fredapi import Fred
            fred = Fred(api_key='YOUR_FRED_API_KEY')  # Replace with your key
            unrate = fred.get_series('UNRATE', observation_start=start)
        except Exception:
            print("WARNING: Cannot download UNRATE. Using fallback.")
            print("Please install pandas_datareader or fredapi and set FRED API key.")
            return None

    # Apply 2-month publication lag (data available with ~2 month delay)
    unrate.index = unrate.index + pd.DateOffset(months=2)
    return unrate


def get_month_end_dates(prices):
    """Get month-end rebalancing dates from price DataFrame."""
    return prices.groupby([prices.index.year, prices.index.month]).tail(1).index


# =============================================================================
# 2. MOMENTUM CALCULATORS
# =============================================================================

def mom_13612w(prices):
    """
    13612W Momentum (BAA Canary, RAA Trend Timing)
    = 1M*12 + 3M*4 + 6M*2 + 12M*1
    """
    p = prices
    return (p / p.shift(20) - 1) * 12 + \
           (p / p.shift(60) - 1) * 4 + \
           (p / p.shift(120) - 1) * 2 + \
           (p / p.shift(240) - 1) * 1


def mom_3612(prices):
    """
    3+6+12M Momentum (BAA Risk Asset Selection)
    """
    p = prices
    return (p / p.shift(60) - 1) + \
           (p / p.shift(120) - 1) + \
           (p / p.shift(240) - 1)


def mom_13612(prices):
    """
    1+3+6+12M Momentum (HAA)
    """
    p = prices
    return (p / p.shift(20) - 1) + \
           (p / p.shift(60) - 1) + \
           (p / p.shift(120) - 1) + \
           (p / p.shift(240) - 1)


def sma_ratio(prices, window):
    """Price / SMA(window) - used for BAA safe (120), PAA (240)"""
    return prices / prices.rolling(window).mean()


def ntop(scores, n):
    """
    Select top n assets by score, equal weight.
    Returns DataFrame of weights (each selected asset = 1/n).
    """
    ranks = scores.rank(axis=1, ascending=False, method='first')
    selected = (ranks <= n).astype(float)
    row_sums = selected.sum(axis=1)
    weights = selected.div(row_sums, axis=0)
    weights = weights.fillna(0)
    return weights


# =============================================================================
# 3. STRATEGY IMPLEMENTATIONS
# =============================================================================

def strategy_baa(prices, rebal_dates):
    """
    Strategy #1: Bold Asset Allocation (BAA)
    
    Canary: SPY, VEA, VWO, BND (4개)
    - All canary 13612W > 0 → Risk 100%
    - Any canary 13612W <= 0 → Safe 100%
    
    Risk: Top 2 of (SPY, VGK, FXI, EWY) by mom_3612, equal weight
    Safe: Top 3 of (USDU, GLD, PDBC, TLT, IEF) by price/SMA(120)
          If selected safe has SMA ratio < 1, replace with BIL
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=CORE_ASSETS)

    # Canary signals
    canary_prices = prices[['SPY', 'VEA', 'VWO', 'BND']]
    canary_mom = mom_13612w(canary_prices)
    # 1 if any canary <= 0 (defensive), 0 if all > 0 (offensive)
    is_defensive = (canary_mom <= 0).any(axis=1).astype(int)

    # Risk asset scores
    risk_prices = prices[RISK_ASSETS]
    risk_scores = mom_3612(risk_prices)
    risk_weights = ntop(risk_scores, 2)  # Top 2, equal weight

    # Safe asset scores
    safe_prices = prices[SAFE_ASSETS]
    safe_scores = sma_ratio(safe_prices, 120)  # price / SMA(120)
    safe_weights_raw = ntop(safe_scores, 3)    # Top 3, equal weight

    # Absolute momentum filter for safe: if SMA ratio < 1, replace with BIL
    safe_am = (sma_ratio(safe_prices, 120) - 1) > 0  # True if positive momentum
    safe_weights_filtered = safe_weights_raw * safe_am.astype(float)
    # BIL gets the weight of replaced assets
    bil_extra = safe_weights_raw.sum(axis=1) - safe_weights_filtered.sum(axis=1)

    for dt in rebal_dates:
        if dt not in prices.index:
            continue
        if pd.isna(is_defensive.get(dt, np.nan)):
            continue

        if is_defensive.loc[dt] == 0:
            # Offensive → Risk assets
            for asset in RISK_ASSETS:
                if asset in risk_weights.columns and dt in risk_weights.index:
                    weights.loc[dt, asset] = risk_weights.loc[dt, asset]
        else:
            # Defensive → Safe assets
            for asset in SAFE_ASSETS:
                if asset in safe_weights_filtered.columns and dt in safe_weights_filtered.index:
                    weights.loc[dt, asset] = safe_weights_filtered.loc[dt, asset]
            weights.loc[dt, 'BIL'] = bil_extra.loc[dt] if dt in bil_extra.index else 0

    # Forward fill weights on rebal dates only
    weights = weights.loc[rebal_dates].ffill()
    return weights


def strategy_faa(prices, rebal_dates, lookback=80, wR=1.0, wV=0.5, wC=0.5, bestN=4):
    """
    Strategy #2: Flexible Asset Allocation (FAA)
    
    Generalized Momentum = wR*rank(return) + wV*rank(vol_inv) + wC*rank(corr_inv)
    - 4-month (80 trading days) lookback
    - Top 4 by generalized momentum, equal weight
    - If selected asset has negative 4M return → replace with BIL
    """
    assets = [a for a in CORE_ASSETS if a != 'BIL']  # 9 assets
    asset_prices = prices[assets]
    weights = pd.DataFrame(0.0, index=prices.index, columns=CORE_ASSETS)

    for dt in rebal_dates:
        if dt not in prices.index:
            continue
        idx = prices.index.get_loc(dt)
        if idx < lookback + 10:
            continue

        # Slice lookback window
        window = asset_prices.iloc[idx - lookback:idx + 1]
        if window.shape[0] < lookback:
            continue

        current_prices = window.iloc[-1]
        start_prices = window.iloc[0]

        # 1. Return rank: higher return → higher rank number → higher GM
        returns = (current_prices / start_prices - 1)
        rank_r = returns.rank(ascending=True)  # high return → high rank

        # 2. Volatility rank: lower vol → higher rank number → higher GM
        daily_rets = window.pct_change().dropna()
        vols = daily_rets.std()
        rank_v = vols.rank(ascending=False)  # low vol → high rank

        # 3. Correlation rank: lower corr → higher rank number → higher GM
        # For each asset: correlation with equal-weighted portfolio of the rest
        corrs = pd.Series(dtype=float)
        for asset in assets:
            others = [a for a in assets if a != asset]
            other_rets = daily_rets[others].mean(axis=1)
            asset_rets = daily_rets[asset]
            if asset_rets.std() == 0 or other_rets.std() == 0:
                corrs[asset] = 0
            else:
                corrs[asset] = asset_rets.corr(other_rets)
        rank_c = corrs.rank(ascending=False)  # low corr → high rank

        # Generalized Momentum
        gm = wR * rank_r + wV * rank_v + wC * rank_c

        # Select top N
        top_assets = gm.nlargest(bestN).index.tolist()

        # Equal weight, but replace negative return assets with BIL
        w = 1.0 / bestN
        bil_weight = 0.0
        for asset in top_assets:
            if returns[asset] < 0:
                bil_weight += w
            else:
                weights.loc[dt, asset] = w
        weights.loc[dt, 'BIL'] = bil_weight

    weights = weights.loc[rebal_dates].ffill()
    return weights


def strategy_raa(prices, rebal_dates, unrate):
    """
    Strategy #3: Resilient Asset Allocation (RAA)
    
    Offensive (All Seasons): 40% SPY + 20% IEF + 20% TLT + 20% GLD
    Defensive: 50% IEF + 50% TLT
    
    Switch condition:
    - Growth improving OR Market bull → Offensive
    - Growth worsening AND Market bear → Defensive
    
    Growth Timing: UNRATE < UNRATE(2 months ago) → improving
    Trend Timing: VWO, BND 13612W momentum, both >= 0 → bull
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=CORE_ASSETS)

    # Trend Timing: Canary = VWO, BND
    canary_prices = prices[['VWO', 'BND']]
    canary_mom = mom_13612w(canary_prices)
    # Bear if any canary < 0
    is_market_bear = (canary_mom < 0).any(axis=1)

    for dt in rebal_dates:
        if dt not in prices.index:
            continue

        # Growth Timing
        growth_worsening = True  # Default: worsening
        if unrate is not None:
            ur_avail = unrate[unrate.index <= dt]
            if len(ur_avail) >= 3:
                current_ur = ur_avail.iloc[-1]
                prev_ur = ur_avail.iloc[-3] if len(ur_avail) >= 3 else ur_avail.iloc[0]
                growth_worsening = (current_ur >= prev_ur)

        # Trend Timing
        market_bear = is_market_bear.get(dt, True)
        if pd.isna(market_bear):
            market_bear = True

        # Decision: Defensive only if growth worsening AND market bear
        if growth_worsening and market_bear:
            # Defensive Portfolio
            weights.loc[dt, 'IEF'] = 0.50
            weights.loc[dt, 'TLT'] = 0.50
        else:
            # Offensive (All Seasons)
            weights.loc[dt, 'SPY'] = 0.40
            weights.loc[dt, 'IEF'] = 0.20
            weights.loc[dt, 'TLT'] = 0.20
            weights.loc[dt, 'GLD'] = 0.20

    weights = weights.loc[rebal_dates].ffill()
    return weights


def strategy_paa(prices, rebal_dates, top_n=3, a=0):
    """
    Strategy #4: Protective Asset Allocation (PAA)
    
    9 assets (excl BIL). Momentum = price / SMA(240)
    BF = (N - n) / N  where N=9, n=# assets with momentum > 1
    BF → BIL, (1-BF) → top 3 equal weight
    
    Note: Report uses a=0 (not original PAA a=2)
    """
    assets = [a_ for a_ in CORE_ASSETS if a_ != 'BIL']  # 9 assets
    N = len(assets)
    weights = pd.DataFrame(0.0, index=prices.index, columns=CORE_ASSETS)

    asset_prices = prices[assets]
    scores = sma_ratio(asset_prices, 240)  # price / SMA(240)

    for dt in rebal_dates:
        if dt not in prices.index:
            continue
        if dt not in scores.index:
            continue

        score_row = scores.loc[dt]
        if score_row.isna().all():
            continue

        # Count "good" assets (momentum > 1, i.e., price > SMA)
        n_good = (score_row > 1).sum()

        # Bond Fraction
        if a == 0:
            bf = (N - n_good) / N
        else:
            bf = (N - n_good) / (N - a * N / 4)
        bf = min(bf, 1.0)
        bf = max(bf, 0.0)

        # BIL weight
        weights.loc[dt, 'BIL'] = bf

        # Risk budget: (1 - BF) split among top 3
        if bf < 1.0:
            risk_budget = 1.0 - bf
            top_assets = score_row.nlargest(top_n).index.tolist()
            w_each = risk_budget / top_n
            for asset in top_assets:
                weights.loc[dt, asset] = w_each

    weights = weights.loc[rebal_dates].ffill()
    return weights


def strategy_laa(prices, rebal_dates, unrate):
    """
    Strategy #5: Lethargic Asset Allocation (LAA)
    
    Offensive Permanent: 50% SPY + 25% GLD + 25% TLT
    Defensive Permanent: 25% SPY + 25% GLD + 25% TLT + 25% BIL
    
    Growth Timing: UNRATE < UNRATE(2 months ago) → improving
    Trend Timing: SPY > EMA(160 trading days) → bull
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=CORE_ASSETS)

    # Trend Timing: SPY vs 8-month (160 trading day) EMA
    spy_ema = prices['SPY'].ewm(span=160, adjust=False).mean()

    for dt in rebal_dates:
        if dt not in prices.index:
            continue

        # Growth Timing
        growth_worsening = True
        if unrate is not None:
            ur_avail = unrate[unrate.index <= dt]
            if len(ur_avail) >= 3:
                current_ur = ur_avail.iloc[-1]
                prev_ur = ur_avail.iloc[-3] if len(ur_avail) >= 3 else ur_avail.iloc[0]
                growth_worsening = (current_ur >= prev_ur)

        # Trend Timing
        spy_price = prices['SPY'].get(dt, np.nan)
        spy_ema_val = spy_ema.get(dt, np.nan)
        market_bear = True
        if not pd.isna(spy_price) and not pd.isna(spy_ema_val):
            market_bear = (spy_price <= spy_ema_val)

        # Decision
        if growth_worsening and market_bear:
            # Defensive
            weights.loc[dt, 'SPY'] = 0.25
            weights.loc[dt, 'GLD'] = 0.25
            weights.loc[dt, 'TLT'] = 0.25
            weights.loc[dt, 'BIL'] = 0.25
        else:
            # Offensive
            weights.loc[dt, 'SPY'] = 0.50
            weights.loc[dt, 'GLD'] = 0.25
            weights.loc[dt, 'TLT'] = 0.25

    weights = weights.loc[rebal_dates].ffill()
    return weights


def strategy_haa(prices, rebal_dates):
    """
    Strategy #6: Hybrid Asset Allocation (HAA)
    
    Canary: TIP (1개)
    Momentum = 1M + 3M + 6M + 12M
    
    TIP momentum > 0 → Invest 100%
      → Top 4 from (SPY, VGK, FXI, EWY, USDU, GLD, PDBC, TLT), equal weight
    TIP momentum <= 0 → Safe 100%
      → Top 1 from (IEF, BIL)
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=CORE_ASSETS)

    # TIP canary
    tip_mom = mom_13612(prices[['TIP']])['TIP']

    # Invest universe: 8 assets
    invest_assets = ['SPY', 'VGK', 'FXI', 'EWY', 'USDU', 'GLD', 'PDBC', 'TLT']
    invest_prices = prices[invest_assets]
    invest_scores = mom_13612(invest_prices)

    # Safe universe: 2 assets
    safe_assets = ['IEF', 'BIL']
    safe_prices = prices[safe_assets]
    safe_scores = mom_13612(safe_prices)

    for dt in rebal_dates:
        if dt not in prices.index:
            continue

        tip_signal = tip_mom.get(dt, np.nan)
        if pd.isna(tip_signal):
            continue

        if tip_signal > 0:
            # Invest: top 4
            if dt in invest_scores.index:
                score_row = invest_scores.loc[dt].dropna()
                if len(score_row) >= 4:
                    top4 = score_row.nlargest(4).index.tolist()
                    for asset in top4:
                        weights.loc[dt, asset] = 0.25
        else:
            # Safe: top 1
            if dt in safe_scores.index:
                score_row = safe_scores.loc[dt].dropna()
                if len(score_row) >= 1:
                    top1 = score_row.nlargest(1).index.tolist()
                    weights.loc[dt, top1[0]] = 1.0

    weights = weights.loc[rebal_dates].ffill()
    return weights


# =============================================================================
# 4. CAA COMBINATION ENGINE
# =============================================================================

def combine_caa(strategy_weights_list):
    """
    Combine 6 strategy weights with equal weight (1/6 average).
    Each element: DataFrame with columns = CORE_ASSETS, index = rebal_dates
    """
    combined = strategy_weights_list[0].copy() * 0
    for w in strategy_weights_list:
        # Align indices
        w_aligned = w.reindex(combined.index, method='ffill').fillna(0)
        combined = combined + w_aligned
    combined = combined / len(strategy_weights_list)
    return combined


# =============================================================================
# 4b. LEVERAGE ENGINE
# =============================================================================

# Which assets stay at 1x (PTP avoidance + stability)
LEVERAGE_1X_ASSETS = {'USDU', 'GLD', 'PDBC', 'BIL'}

def create_leveraged_prices(prices, leverage=2):
    """
    Create synthetic leveraged price series from 1x prices.
    
    The report uses the same CAA weights (computed from 1x) but applies them
    to leveraged ETF prices. Since leveraged ETFs didn't exist before ~2006,
    we synthesize them: daily_return_lev = leverage * daily_return_1x.
    
    Assets in LEVERAGE_1X_ASSETS stay at 1x (PTP avoidance / stability).
    
    Args:
        prices: DataFrame of 1x daily prices
        leverage: 2 or 3
    
    Returns:
        DataFrame of synthetic leveraged prices
    """
    daily_rets = prices[CORE_ASSETS].pct_change().fillna(0)

    lev_rets = daily_rets.copy()
    for col in lev_rets.columns:
        if col not in LEVERAGE_1X_ASSETS:
            lev_rets[col] = lev_rets[col] * leverage

    # Reconstruct price series from leveraged returns
    lev_prices = (1 + lev_rets).cumprod()
    # Scale so first valid price = original first price (cosmetic)
    for col in lev_prices.columns:
        first_valid = prices[col].first_valid_index()
        if first_valid is not None and first_valid in lev_prices.index:
            lev_prices[col] = lev_prices[col] / lev_prices[col].loc[first_valid] * prices[col].loc[first_valid]

    # Keep canary tickers unchanged (they're only used for signal, not execution)
    for col in prices.columns:
        if col not in CORE_ASSETS:
            lev_prices[col] = prices[col]

    return lev_prices

def backtest(prices, weights, rebal_dates, commission_pct=0.0):
    """
    Simple backtest engine.
    
    Args:
        prices: DataFrame of daily adjusted close prices (CORE_ASSETS columns)
        weights: DataFrame of target weights at each rebal date
        rebal_dates: DatetimeIndex of rebalancing dates
        commission_pct: One-way transaction cost as decimal (e.g., 0.0021 for 0.21%)
    
    Returns:
        equity_curve: Series of portfolio value (starting at 1.0)
        daily_returns: Series of daily portfolio returns
    """
    asset_prices = prices[CORE_ASSETS].copy()
    daily_returns = asset_prices.pct_change().fillna(0)

    # Initialize
    equity = pd.Series(1.0, index=asset_prices.index)
    current_weights = pd.Series(0.0, index=CORE_ASSETS)
    prev_value = 1.0

    for i in range(1, len(asset_prices)):
        dt = asset_prices.index[i]
        prev_dt = asset_prices.index[i - 1]

        # Daily return from holding
        port_return = (current_weights * daily_returns.iloc[i]).sum()
        new_value = prev_value * (1 + port_return)

        # Rebalance at month end
        if prev_dt in rebal_dates:
            if prev_dt in weights.index:
                new_weights = weights.loc[prev_dt]
                if not new_weights.isna().all():
                    # Transaction cost: proportional to turnover
                    turnover = (new_weights - current_weights).abs().sum()
                    cost = turnover * commission_pct
                    new_value *= (1 - cost)
                    current_weights = new_weights.copy()

        equity.iloc[i] = new_value
        prev_value = new_value

    daily_rets = equity.pct_change().fillna(0)
    return equity, daily_rets


# =============================================================================
# 6. PERFORMANCE ANALYTICS
# =============================================================================

def calc_performance(equity, daily_returns, name='Strategy'):
    """Calculate key performance metrics."""
    total_days = len(equity)
    total_years = total_days / 252

    # CAGR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / total_years) - 1

    # VOL (annualized)
    vol = daily_returns.std() * np.sqrt(252)

    # MDD
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    mdd = drawdown.min()

    # Sharpe (simplified: CAGR / VOL)
    sharpe = cagr / vol if vol > 0 else 0

    # Calmar (CAGR / |MDD|)
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    return {
        'Name': name,
        'CAGR(%)': round(cagr * 100, 2),
        'VOL(%)': round(vol * 100, 2),
        'MDD(%)': round(mdd * 100, 2),
        'Sharpe': round(sharpe, 4),
        'Calmar': round(calmar, 4),
    }


def monthly_return_table(equity):
    """Generate monthly return table (year x month), matching report format."""
    monthly = equity.resample('ME').last()
    monthly_rets = monthly.pct_change().dropna()

    table = pd.DataFrame()
    for dt, ret in monthly_rets.items():
        year = dt.year
        month = dt.month
        table.loc[year, month] = round(ret * 100, 1)

    # Ensure columns 1-12 in order
    month_cols = sorted([c for c in table.columns if isinstance(c, int)])
    table = table.reindex(columns=month_cols)

    # Annual return (compounded from monthly)
    annual = equity.resample('YE').last()
    annual_rets = annual.pct_change().dropna()
    for dt, ret in annual_rets.items():
        table.loc[dt.year, 'Year'] = round(ret * 100, 1)

    # Annual MaxDD
    for year in table.index:
        year_eq = equity[equity.index.year == year]
        if len(year_eq) > 0:
            rolling_max = year_eq.cummax()
            dd = year_eq / rolling_max - 1
            table.loc[year, 'MaxDD'] = round(dd.min() * 100, 1)

    # Rename month columns to match report format
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    table = table.rename(columns=month_names)
    return table


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def run_caa(start_date='1996-01-01', backtest_start='1997-02-01', commission=0.0021):
    """
    Full CAA pipeline:
    1. Download data
    2. Run 6 individual strategies
    3. Combine with 1/6 equal weight
    4. Backtest and report performance
    
    Args:
        start_date: Data download start (need buffer for lookback)
        backtest_start: Backtest evaluation start
        commission: Transaction cost (default 0.21% per the report)
    """
    print("=" * 70)
    print("CAA (Combination of Asset Allocation) — Python Implementation")
    print("Based on DB금융투자 Report (2023.04.03)")
    print("=" * 70)

    # --- Step 1: Download Data ---
    print("\n[1/7] Downloading price data...")
    prices = download_prices(start=start_date)
    print(f"  Data range: {prices.index[0].strftime('%Y-%m-%d')} ~ {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Assets: {list(prices.columns)}")
    print(f"  Total trading days: {len(prices)}")

    print("\n[2/7] Downloading FRED unemployment rate...")
    unrate = download_unemployment_rate()
    if unrate is not None:
        print(f"  UNRATE range: {unrate.index[0].strftime('%Y-%m-%d')} ~ {unrate.index[-1].strftime('%Y-%m-%d')}")

    # --- Step 2: Get rebalancing dates ---
    rebal_dates = get_month_end_dates(prices)
    print(f"\n  Rebalancing dates: {len(rebal_dates)} month-ends")

    # --- Step 3: Run 6 Strategies ---
    print("\n[3/7] Running BAA (Bold Asset Allocation)...")
    w_baa = strategy_baa(prices, rebal_dates)
    print("  ✓ BAA complete")

    print("[4/7] Running FAA (Flexible Asset Allocation)...")
    w_faa = strategy_faa(prices, rebal_dates)
    print("  ✓ FAA complete")

    print("[5/7] Running RAA (Resilient Asset Allocation)...")
    w_raa = strategy_raa(prices, rebal_dates, unrate)
    print("  ✓ RAA complete")

    print("       Running PAA (Protective Asset Allocation)...")
    w_paa = strategy_paa(prices, rebal_dates)
    print("  ✓ PAA complete")

    print("       Running LAA (Lethargic Asset Allocation)...")
    w_laa = strategy_laa(prices, rebal_dates, unrate)
    print("  ✓ LAA complete")

    print("       Running HAA (Hybrid Asset Allocation)...")
    w_haa = strategy_haa(prices, rebal_dates)
    print("  ✓ HAA complete")

    # --- Step 4: Combine ---
    print("\n[6/7] Combining 6 strategies (1/6 equal weight)...")
    w_caa = combine_caa([w_baa, w_faa, w_raa, w_paa, w_laa, w_haa])
    print("  ✓ CAA weights computed")

    # --- Step 5: Backtest ---
    print("\n[7/7] Running backtests...")

    # Filter to backtest period
    bt_prices = prices[prices.index >= backtest_start]
    bt_rebal = rebal_dates[rebal_dates >= backtest_start]

    # CAA backtest (with commission)
    eq_caa, ret_caa = backtest(bt_prices, w_caa, bt_rebal, commission_pct=commission)

    # Individual strategy backtests (zero commission per report methodology)
    eq_baa, ret_baa = backtest(bt_prices, w_baa, bt_rebal, commission_pct=0)
    eq_faa, ret_faa = backtest(bt_prices, w_faa, bt_rebal, commission_pct=0)
    eq_raa, ret_raa = backtest(bt_prices, w_raa, bt_rebal, commission_pct=0)
    eq_paa, ret_paa = backtest(bt_prices, w_paa, bt_rebal, commission_pct=0)
    eq_laa, ret_laa = backtest(bt_prices, w_laa, bt_rebal, commission_pct=0)
    eq_haa, ret_haa = backtest(bt_prices, w_haa, bt_rebal, commission_pct=0)

    # SPY benchmark
    spy_eq = bt_prices['SPY'] / bt_prices['SPY'].iloc[0]
    spy_ret = spy_eq.pct_change().fillna(0)

    # TLT benchmark
    tlt_eq = bt_prices['TLT'] / bt_prices['TLT'].iloc[0]
    tlt_ret = tlt_eq.pct_change().fillna(0)

    # --- Step 6: Performance Report ---
    print("\n" + "=" * 70)
    print("PERFORMANCE REPORT")
    print("=" * 70)

    results = []
    results.append(calc_performance(eq_baa, ret_baa, 'BAA'))
    results.append(calc_performance(eq_faa, ret_faa, 'FAA'))
    results.append(calc_performance(eq_raa, ret_raa, 'RAA'))
    results.append(calc_performance(eq_paa, ret_paa, 'PAA'))
    results.append(calc_performance(eq_laa, ret_laa, 'LAA'))
    results.append(calc_performance(eq_haa, ret_haa, 'HAA'))
    results.append(calc_performance(eq_caa, ret_caa, 'CAA'))
    results.append(calc_performance(spy_eq, spy_ret, 'S&P500'))
    results.append(calc_performance(tlt_eq, tlt_ret, 'US LT Bond'))

    df_results = pd.DataFrame(results).set_index('Name')
    print("\n--- Individual Strategies (commission=0) + CAA (commission=0.21%) ---")
    print(df_results.to_string())

    # Monthly table
    print("\n--- CAA Monthly Returns (%) ---")
    mt = monthly_return_table(eq_caa)
    print(mt.to_string())

    # --- Return objects for further analysis ---
    return {
        'prices': prices,
        'weights': {
            'BAA': w_baa, 'FAA': w_faa, 'RAA': w_raa,
            'PAA': w_paa, 'LAA': w_laa, 'HAA': w_haa, 'CAA': w_caa,
        },
        'equity': {
            'BAA': eq_baa, 'FAA': eq_faa, 'RAA': eq_raa,
            'PAA': eq_paa, 'LAA': eq_laa, 'HAA': eq_haa, 'CAA': eq_caa,
            'SPY': spy_eq, 'TLT': tlt_eq,
        },
        'performance': df_results,
        'rebal_dates': rebal_dates,
        'unrate': unrate,
        'backtest_start': backtest_start,
    }


def run_leverage_backtest(results, commission=0.0021):
    """
    Run leverage 2x and 3x backtests using pre-computed CAA weights.
    
    The report methodology:
    - Weights are identical to 1x CAA (computed from 1x prices)
    - Execution uses leveraged ETF prices (synthetic for backtest)
    - Some assets stay 1x: USDU, GLD, PDBC, BIL (PTP avoidance)
    
    Args:
        results: dict returned by run_caa()
        commission: transaction cost (default 0.21%)
    """
    prices = results['prices']
    w_caa = results['weights']['CAA']
    backtest_start = results['backtest_start']
    rebal_dates = results['rebal_dates']

    bt_rebal = rebal_dates[rebal_dates >= backtest_start]

    print("\n" + "=" * 70)
    print("LEVERAGE BACKTEST")
    print("=" * 70)

    all_equity = {'CAA 1x': results['equity']['CAA']}
    all_results = []

    for lev in [2, 3]:
        print(f"\n  Creating synthetic {lev}x leveraged prices...")
        lev_prices = create_leveraged_prices(prices, leverage=lev)
        bt_lev_prices = lev_prices[lev_prices.index >= backtest_start]

        # CAA leveraged
        eq_lev, ret_lev = backtest(bt_lev_prices, w_caa, bt_rebal, commission_pct=commission)
        perf = calc_performance(eq_lev, ret_lev, f'CAA {lev}x')
        all_results.append(perf)
        all_equity[f'CAA {lev}x'] = eq_lev

        # SPY leveraged benchmark
        spy_lev_rets = prices['SPY'].pct_change().fillna(0) * lev
        spy_lev_eq = (1 + spy_lev_rets).cumprod()
        spy_lev_eq = spy_lev_eq[spy_lev_eq.index >= backtest_start]
        spy_lev_eq = spy_lev_eq / spy_lev_eq.iloc[0]
        spy_lev_ret = spy_lev_eq.pct_change().fillna(0)
        perf_spy = calc_performance(spy_lev_eq, spy_lev_ret, f'S&P500 {lev}x')
        all_results.append(perf_spy)
        all_equity[f'S&P500 {lev}x'] = spy_lev_eq

        print(f"  ✓ CAA {lev}x: CAGR={perf['CAGR(%)']:.1f}%, MDD={perf['MDD(%)']:.1f}%")

    # Add 1x for comparison
    all_results.insert(0, calc_performance(results['equity']['CAA'],
                        results['equity']['CAA'].pct_change().fillna(0), 'CAA 1x'))
    all_results.insert(1, calc_performance(results['equity']['SPY'],
                        results['equity']['SPY'].pct_change().fillna(0), 'S&P500 1x'))

    df_lev = pd.DataFrame(all_results).set_index('Name')
    print(f"\n--- Leverage Comparison (commission={commission*100:.2f}%) ---")
    print(df_lev.to_string())

    # Monthly tables for leverage versions
    for lev in [2, 3]:
        print(f"\n--- CAA {lev}x Monthly Returns (%) ---")
        mt = monthly_return_table(all_equity[f'CAA {lev}x'])
        print(mt.to_string())

    results['equity'].update(all_equity)
    results['leverage_performance'] = df_lev
    return results


# =============================================================================
# 8. LIVE SIGNAL GENERATOR
# =============================================================================

def generate_live_signal(prices=None, unrate=None):
    """
    Generate current month's CAA signal (latest rebalancing weights).
    Can be called standalone or integrated with Telegram briefing.
    
    Returns:
        dict with keys: date, strategy_weights (6 dicts), caa_weights, actions
    """
    if prices is None:
        print("Downloading latest price data...")
        prices = download_prices(start='2023-01-01')

    if unrate is None:
        unrate = download_unemployment_rate()

    rebal_dates = get_month_end_dates(prices)
    latest_date = rebal_dates[-1]

    # Run all 6 strategies
    w_baa = strategy_baa(prices, rebal_dates)
    w_faa = strategy_faa(prices, rebal_dates)
    w_raa = strategy_raa(prices, rebal_dates, unrate)
    w_paa = strategy_paa(prices, rebal_dates)
    w_laa = strategy_laa(prices, rebal_dates, unrate)
    w_haa = strategy_haa(prices, rebal_dates)

    # Combine
    w_caa = combine_caa([w_baa, w_faa, w_raa, w_paa, w_laa, w_haa])

    # Get latest weights
    latest_caa = w_caa.loc[latest_date]
    active_positions = latest_caa[latest_caa > 0.001].sort_values(ascending=False)

    signal = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'strategy_weights': {
            'BAA': w_baa.loc[latest_date].to_dict(),
            'FAA': w_faa.loc[latest_date].to_dict(),
            'RAA': w_raa.loc[latest_date].to_dict(),
            'PAA': w_paa.loc[latest_date].to_dict(),
            'LAA': w_laa.loc[latest_date].to_dict(),
            'HAA': w_haa.loc[latest_date].to_dict(),
        },
        'caa_weights': latest_caa.to_dict(),
        'active_positions': active_positions.to_dict(),
    }

    # Print signal
    print(f"\n{'='*50}")
    print(f"  CAA LIVE SIGNAL — {signal['date']}")
    print(f"{'='*50}")
    print(f"\n  Active Positions:")
    for asset, weight in active_positions.items():
        bar = '█' * int(weight * 40)
        print(f"    {asset:6s}  {weight:6.1%}  {bar}")
    print(f"\n  Individual Strategy Breakdown:")
    for strat_name in ['BAA', 'FAA', 'RAA', 'PAA', 'LAA', 'HAA']:
        sw = signal['strategy_weights'][strat_name]
        active = {k: v for k, v in sw.items() if v > 0.001}
        active_str = ', '.join([f"{k}:{v:.0%}" for k, v in active.items()])
        print(f"    {strat_name}: {active_str}")

    return signal


# =============================================================================
# 9. TELEGRAM BRIEFING FORMAT
# =============================================================================

def format_telegram_message(signal):
    """Format signal as Telegram-ready message."""
    msg = f"📊 *CAA Monthly Signal*\n"
    msg += f"📅 Date: {signal['date']}\n\n"
    msg += f"*Active Positions:*\n"

    for asset, weight in signal['active_positions'].items():
        emoji = '🟢' if asset in RISK_ASSETS else '🔵'
        msg += f"  {emoji} {asset}: {weight:.1%}\n"

    msg += f"\n*Strategy Breakdown:*\n"
    for strat_name in ['BAA', 'FAA', 'RAA', 'PAA', 'LAA', 'HAA']:
        sw = signal['strategy_weights'][strat_name]
        active = {k: v for k, v in sw.items() if v > 0.001}
        if active:
            active_str = ' | '.join([f"{k} {v:.0%}" for k, v in active.items()])
            msg += f"  `{strat_name}`: {active_str}\n"

    return msg


def test_download():
    """Quick diagnostic: test downloading a single ticker to verify yfinance works."""
    print("\n--- yfinance Download Test ---")
    print(f"yfinance version: {yf.__version__}")

    raw = yf.download('SPY', start='2024-01-01', period='1mo', progress=False, auto_adjust=True)
    print(f"Raw type: {type(raw)}")
    print(f"Raw columns: {raw.columns.tolist()}")
    print(f"MultiIndex? {isinstance(raw.columns, pd.MultiIndex)}")
    if isinstance(raw.columns, pd.MultiIndex):
        print(f"Level 0: {raw.columns.get_level_values(0).unique().tolist()}")
        print(f"Level 1: {raw.columns.get_level_values(1).unique().tolist()}")
    print(f"Shape: {raw.shape}")
    print(f"Head:\n{raw.head()}")

    series = _extract_close(raw)
    print(f"\nExtracted Series length: {len(series)}")
    print(f"Last 3 values:\n{series.tail(3)}")
    print("--- Test Complete ---\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Step 0: Verify yfinance download works
    test_download()

    results = run_caa(
        start_date='1996-01-01',
        backtest_start='1997-02-01',
        commission=0.0021,  # 0.21% per the report
    )

    # Step 2: Run leverage backtests
    results = run_leverage_backtest(results, commission=0.0021)

    # Optionally generate live signal
    # signal = generate_live_signal(results['prices'])
    # telegram_msg = format_telegram_message(signal)
    # print(telegram_msg)
