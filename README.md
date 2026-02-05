# BackTesting Framework

A comprehensive Python-based backtesting engine for evaluating factor-based investment strategies. This framework provides robust tools for analyzing factor performance, computing portfolio metrics, and visualizing results with professional-grade plots.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Parameters](#parameters)
- [Metrics](#metrics)
- [Mathematical Formulations](#mathematical-formulations)
- [Output](#output)
- [Examples](#examples)
- [Limitations and Assumptions](#limitations-and-assumptions)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The `BackTest` class is a sophisticated backtesting framework designed to evaluate factor-based investment strategies. It enables quantitative analysts and researchers to:

- **Test Factor Strategies**: Evaluate the predictive power of factors on asset returns
- **Compute Portfolio Metrics**: Calculate comprehensive performance indicators including Sharpe ratio, Sortino ratio, Calmar ratio, and maximum drawdown
- **Analyze Group Performance**: Partition assets into quantile groups based on factor values and compare their returns
- **Visualize Results**: Generate professional plots for cumulative returns, daily returns, and information coefficients
- **Assess Factor Quality**: Compute Information Coefficient (IC) and Rank IC to measure factor effectiveness

### Key Capabilities

1. **Factor-Based Portfolio Construction**: Assets are grouped into quantiles based on factor values
2. **Flexible Rebalancing**: Customizable rebalancing frequency (in days)
3. **Multiple Weighting Schemes**: Supports five weighting methods: equal, factor-based, inverse volatility, mean-variance optimization, and market capitalization
4. **Robust Preprocessing**: Automated data cleaning, outlier removal, and normalization
5. **Comprehensive Metrics**: Suite of risk-adjusted performance measures
6. **Professional Visualization**: High-quality plots for analysis and presentation

## Features

### Core Functionality

- **Data Preprocessing**: Automated handling of missing values, outliers, and data alignment
- **Factor Normalization**: Optional z-score normalization for factor values
- **Quantile Grouping**: Partition assets into specified number of groups based on factor quantiles
- **Return Calculation**: Daily and cumulative return computation with support for both multiplicative and additive approaches
- **Performance Metrics**: Comprehensive set of risk-adjusted performance indicators
- **Information Coefficient Analysis**: Time-series and histogram visualization of Rank IC
- **Group Analysis**: Detailed analysis of returns across different factor quantile groups

### Advanced Features

- **Forward-Looking Bias Prevention**: Proper alignment of factors with future returns
- **Extensible Architecture**: Designed to accommodate additional weighting methods and metrics
- **Automated Reporting**: JSON export of key metrics and PNG export of visualization plots
- **Robust Error Handling**: Protection against division by zero and NaN values

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Dependencies

The framework requires the following Python packages:

```bash
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Installation Steps

1. **Clone or download the framework**:
   ```bash
   # If using version control
   git clone <repository-url>
   cd 2.4\ -\ batcktest_framework
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

3. **Verify installation**:
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   print("All dependencies installed successfully!")
   ```

## Quick Start

### Basic Usage Example

```python
import pandas as pd
from flash_bt.bt._backtest import BackTest

# Load your data
price_df = pd.read_csv('price_data.csv', index_col='date')
factor_df = pd.read_csv('factor_data.csv', index_col='date')

# Ensure datetime index
price_df.index = pd.to_datetime(price_df.index)
factor_df.index = pd.to_datetime(factor_df.index)

# Initialize and run backtest
backtest = BackTest(
    factor_df=factor_df,
    price_df=price_df,
    fee=0.0003,          # Transaction fee (0.03%)
    rebalance_period=20,  # Rebalance every 20 trading days
    n_groups=5,           # Create 5 quantile groups
    weight_method='equal',
    auto_run=True         # Automatically run backtest
)

# Access results
print(f"Sharpe Ratio: {backtest.sharpe_ratio:.4f}")
print(f"Max Drawdown: {backtest.max_drawdown:.4f}")
print(f"Rank IC: {backtest.rank_ic:.4f}")
```

### Expected Output

The framework will create two directories:

- **`metrics_result/`**: Contains `backtest_metrics.json` with key performance metrics
- **`figures_result/`**: Contains visualization plots:
  - `returns.png`: Cumulative and daily returns
  - `ic.png`: Information coefficient analysis
  - `avg_group_returns.png`: Average returns by group

## Detailed Usage

### Step-by-Step Workflow

#### Step 1: Data Preparation

Ensure your data is in the correct format:

```python
# Price data format
price_df = pd.DataFrame(
    index=pd.date_range('2020-01-01', periods=252, freq='D'),
    columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
)

# Factor data format (must have same columns as price_df)
factor_df = pd.DataFrame(
    index=pd.date_range('2020-01-01', periods=252, freq='D'),
    columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
)
```

#### Step 2: Initialize BackTest

```python
backtest = BackTest(
    factor_df=factor_df,
    price_df=price_df,
    rebalance_period=20,
    n_groups=5,
    weight_method='equal',
    need_preprocess=True,
    need_normalize=True,
    price_threshold=1e6,
    factor_threshold=10,
    need_plot=True,
    metrics_path='./custom_metrics',
    figures_path='./custom_figures',
    cumprod=True,
    auto_run=False  # Run manually
)
```

#### Step 3: Run Backtest (if auto_run=False)

```python
metrics = backtest.run()
```

#### Step 4: Access Results

Please note that the results are only accessible after bt.run() is called. Otherwise, they will be None.

```python
# Portfolio returns
daily_returns = backtest.daily_ret
cumulative_returns = backtest.cum_ret

# Performance metrics
sharpe = backtest.sharpe_ratio
sortino = backtest.sortino_ratio
calmar = backtest.calmar_ratio
max_dd = backtest.max_drawdown
win_rate = backtest.win_rate

# Information coefficients
rank_ic = backtest.rank_ic
rank_ic_series = backtest.rank_ic_time_series
cumulative_rank_ic = backtest.cumulative_rank_ic

# Group analysis
avg_group_returns = backtest.avg_group_ret
avg_group_daily_returns = backtest.avg_group_daily_ret
avg_group_cum_returns = backtest.avg_group_cum_ret
```

#### Step 5: Custom Visualization

```python
# Plot returns
backtest.plot_ret()

# Plot IC (time series or histogram)
backtest.plot_ic(plot_type='time_series')
backtest.plot_ic(plot_type='histogram')

# Plot group returns
backtest.plot_avg_group_ret()
```

## Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `factor_df` | pd.DataFrame | Required | Factor values for each asset at each time point. Must have same columns as `price_df` |
| `price_df` | pd.DataFrame | Required | Close prices for each asset at each time point. Must have same columns as `factor_df` |
| `fee` | float | Required | Transaction fee per trade (as a fraction of trade amount). Default is 0.0003. Fee is applied based on portfolio turnover at each rebalancing: fee_amount = portfolio_value × fee × turnover, where turnover = Σ|weight_change| / 2 |
| `rebalance_period` | int | Required | Rebalancing frequency in days. Number of periods between portfolio shifts |
| `n_groups` | int | Required | Number of quantile groups for asset partitioning |
| `weight_method` | str | "equal" | Weighting scheme within groups. Options: "equal", "factor", "inv_vol", "mean_var", "market_cap". See [Weighting Methods](#weighting-methods) for details |
| `market_cap_df` | pd.DataFrame | None | Market capitalization data for each asset at each time point. Required if `weight_method="market_cap"`. Must have same columns as `price_df` |
| `need_preprocess` | bool | True | Whether to preprocess data before backtesting |
| `need_normalize` | bool | True | Whether to normalize factor values using z-score |
| `price_threshold` | float |1e6 | Upper bound for valid price values |
| `factor_threshold` | float | 10 | Maximum standard deviations from mean allowed for factor values |
| `need_plot` | bool | True | Whether to generate visualization plots |
| `metrics_path` | str | None | Path to save backtest metrics (default: `metrics_result/`) |
| `figures_path` | str | None | Path to save figures (default: `figures_result/`) |
| `cumprod` | bool | True | Whether to use multiplicative (True) or additive (False) cumulative returns |
| `auto_run` | bool | False | Whether to automatically run backtest on initialization |

### Method Parameters

#### `preprocess(price_df, factor_df)`

Preprocesses data for backtesting.

**Parameters:**
- `price_df` (pd.DataFrame): Close prices for each asset
- `factor_df` (pd.DataFrame): Factor values for each asset

**Preprocessing Steps:**
1. Remove missing values (NaNs)
2. Remove outliers:
   - Negative or zero prices
   - Prices exceeding `price_threshold`
   - Factor values outside `factor_threshold` standard deviations from mean
3. Align factor and price DataFrames by time index
4. Normalize factor values if `need_normalize=True`

#### `compute_metrics(price_df, factor_df, cumprod=True)`

Computes comprehensive backtest metrics.

**Parameters:**
- `price_df` (pd.DataFrame): Close prices for each asset
- `factor_df` (pd.DataFrame): Factor values for each asset
- `cumprod` (bool): Cumulative return calculation method

**Returns:**
- `dict`: Dictionary containing all computed metrics

#### `plot_ic(plot_type="time_series")`

Plots information coefficient analysis.

**Parameters:**
- `plot_type` (str): Type of plot. Options: "time_series" or "histogram"

## Metrics

### Performance Metrics

#### 1. Daily Returns ($r_t$)

The daily return for each asset is calculated as:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

where $P_t$ is the price at time $t$.

#### 2. Cumulative Returns ($R_t$)

**Multiplicative (cumprod=True):**

$$R_t = \prod_{i=1}^{t} (1 + r_i) - 1$$

**Additive (cumprod=False):**

$$R_t = \sum_{i=1}^{t} r_i$$

#### 3. Sharpe Ratio

The annualized Sharpe ratio measures risk-adjusted return:

$$\text{Sharpe} = \frac{\bar{r}}{\sigma_r} \times \sqrt{252}$$

where:
- $\bar{r}$ is the mean daily return
- $\sigma_r$ is the standard deviation of daily returns
- 252 is the number of trading days in a year

#### 4. Sortino Ratio

The Sortino ratio considers only downside risk:

$$\text{Sortino} = \frac{\bar{r}}{\sigma_{down}} \times \sqrt{252}$$

where $\sigma_{down}$ is the standard deviation of negative returns:

$$\sigma_{down} = \sqrt{\frac{1}{N_{down}} \sum_{r_t < 0} r_t^2}$$

#### 5. Calmar Ratio

The Calmar ratio relates annualized return to maximum drawdown:

$$\text{Calmar} = \frac{\bar{r} \times \sqrt{252}}{|\text{Max Drawdown}|}$$

#### 6. Maximum Drawdown

The maximum peak-to-trough decline:

$$\text{Max Drawdown} = \min_{t} \left( \frac{P_t}{\max_{0 \leq s \leq t} P_s} - 1 \right)$$

For cumulative returns:

$$\text{Drawdown}_t = \frac{1 + R_t}{\max_{0 \leq s \leq t} (1 + R_s)} - 1$$

#### 7. Win Rate

The fraction of positive-return periods:

$$\text{Win Rate} = \frac{N_{positive}}{N_{total}}$$

where $N_{positive}$ is the number of periods with $r_t > 0$.

### Information Coefficient Metrics

#### 8. Rank IC (Information Coefficient)

The Spearman rank correlation between factor values and forward returns:

$$\text{Rank IC}_t = \text{corr}(\text{rank}(F_t), \text{rank}(R_{t+1}))$$

where:
- $F_t$ is the factor value at time $t$
- $R_{t+1}$ is the return at time $t+1$
- $\text{rank}(\cdot)$ converts values to ranks
- $\text{corr}(\cdot, \cdot, \text{method='spearman'})$ computes Spearman correlation

#### 9. Cumulative Rank IC

The cumulative sum of Rank IC over time:

$$\text{Cumulative Rank IC}_t = \sum_{i=1}^{t} \text{Rank IC}_i$$

### Group Analysis Metrics

#### 10. Average Group Returns

Assets are partitioned into $n$ groups based on factor quantiles at each rebalance date. For group $g$:

$$\bar{r}_g = \frac{1}{T_g} \sum_{t \in \text{periods}} r_{g,t}$$

where $r_{g,t}$ is the return of group $g$ at time $t$ and $T_g$ is the number of periods.

## Mathematical Formulations

### Factor Normalization

When `need_normalize=True`, factor values are standardized using z-score normalization:

$$F'_{i,t} = \frac{F_{i,t} - \mu_t}{\sigma_t}$$

where:
- $F_{i,t}$ is the original factor value for asset $i$ at time $t$
- $\mu_t$ is the mean factor value across all assets at time $t$
- $\sigma_t$ is the standard deviation of factor values across all assets at time $t$

### Quantile Grouping

Assets are assigned to groups using quantile-based partitioning:

$$g_{i,t} = \text{qcut}(F_{i,t}, q=n)$$

where `qcut` assigns assets to $n$ groups based on factor quantiles.

### Portfolio Return Calculation

For equal-weighted portfolios within each group:

$$r_{g,t} = \frac{1}{N_g} \sum_{i \in G_g} r_{i,t}$$

where:
- $G_g$ is the set of assets in group $g$
- $N_g = |G_g|$ is the number of assets in group $g$
- $r_{i,t}$ is the return of asset $i$ at time $t$

### Overall Portfolio Return

The overall portfolio return is the average across all assets:

$$r_{portfolio,t} = \frac{1}{N} \sum_{i=1}^{N} r_{i,t}$$

where $N$ is the total number of assets.

### Weighting Methods

The framework supports five different weighting schemes for portfolio construction:

#### 1. Equal Weighting (`weight_method="equal"`)

All assets receive equal weights:

$$w_i = \frac{1}{N}$$

where $N$ is the number of assets in the portfolio.

**Characteristics:**
- Simple and transparent
- No bias toward any particular asset
- Suitable for testing factor effectiveness without confounding effects

#### 2. Factor-Based Weighting (`weight_method="factor"`)

Weights are proportional to factor values:

$$w_i = \frac{F_i}{\sum_{j=1}^{N} F_j}$$

where $F_i$ is the factor value for asset $i$. If factor values are negative, they are shifted to ensure positivity:

$$F'_i = F_i - \min(F) + \epsilon$$

where $\epsilon$ is a small constant (1e-6) to avoid zero weights.

**Characteristics:**
- Directly uses factor information
- Higher factor values receive higher weights
- Suitable when factor values represent investment signals

#### 3. Inverse Volatility Weighting (`weight_method="inv_vol"`)

Weights are inversely proportional to historical volatility:

$$w_i = \frac{1/\sigma_i}{\sum_{j=1}^{N} 1/\sigma_j}$$

where $\sigma_i$ is the historical volatility of asset $i$, calculated using a rolling window:

$$\sigma_i = \sqrt{\frac{1}{W-1}\sum_{t=T-W+1}^{T} (r_{i,t} - \bar{r}_i)^2}$$

where:
- $W$ is the window size (20 trading days)
- $r_{i,t}$ is the return of asset $i$ at time $t$
- $\bar{r}_i$ is the mean return of asset $i$ over the window

**Characteristics:**
- Risk-aware weighting
- Lower volatility assets receive higher weights
- Suitable for risk-averse strategies

#### 4. Mean-Variance Optimization (`weight_method="mean_var"`)

Markowitz optimal portfolio maximizing Sharpe ratio:

$$w^* = \frac{\Sigma^{-1}\mu}{1^T\Sigma^{-1}\mu}$$

where:
- $\Sigma$ is the covariance matrix of returns
- $\mu$ is the vector of expected returns
- $1$ is a vector of ones
- $\Sigma^{-1}$ is the inverse of the covariance matrix

The covariance matrix is estimated using historical returns (60-day window):

$$\Sigma_{i,j} = \frac{1}{T-1}\sum_{t=1}^{T} (r_{i,t} - \bar{r}_i)(r_{j,t} - \bar{r}_j)$$

For numerical stability, a small regularization term is added:

$$\Sigma' = \Sigma + \epsilon I$$

where $I$ is the identity matrix and $\epsilon = 1e-8$.

**Constraints:**
- Long-only: $w_i \geq 0$ for all $i$
- Full investment: $\sum_{i=1}^{N} w_i = 1$

**Characteristics:**
- Theoretically optimal for risk-return trade-off
- Maximizes Sharpe ratio
- Requires sufficient historical data (60+ days)

#### 5. Market Capitalization Weighting (`weight_method="market_cap"`)

Weights are proportional to market capitalization:

$$w_i = \frac{MC_i}{\sum_{j=1}^{N} MC_j}$$

where $MC_i$ is the market capitalization of asset $i$ at the rebalance date.

**Characteristics:**
- Mimics market index composition
- Larger companies receive higher weights
- Requires external market cap data

### Transaction Fee Calculation

Transaction fees are applied based on portfolio turnover at each rebalancing:

$$\text{Turnover}_t = \frac{1}{2}\sum_{i=1}^{N} |w_{i,t} - w_{i,t-1}|$$

$$\text{Fee}_t = V_t \times f \times \text{Turnover}_t$$

where:
- $w_{i,t}$ is the weight of asset $i$ at time $t$
- $V_t$ is the portfolio value at time $t$
- $f$ is the transaction fee rate (e.g., 0.0003 for 0.03%)

The portfolio value after fees:

$$V'_t = V_t - \text{Fee}_t$$

This approach properly reflects actual trading costs rather than assuming 100% turnover.

## Output

### Directory Structure

After running the backtest, the following structure is created:

```
project_directory/
├── metrics_result/
│   └── backtest_metrics.json
├── figures_result/
│   ├── returns.png
│   ├── ic.png
│   └── avg_group_returns.png
└── _backtest.py
```

### Metrics JSON Format

The `backtest_metrics.json` file contains:

```json
{
  "sharpe_ratio": 1.2345,
  "sortino_ratio": 1.6789,
  "calmar_ratio": 0.9876,
  "max_drawdown": -0.1234,
  "win_rate": 0.5678,
  "rank_ic": 0.0456
}
```

### Visualization Plots

#### 1. Returns Plot (`returns.png`)

- **Top subplot**: Cumulative returns over time
  - Overall portfolio return (bold red line)
  - Individual group returns (colored lines)
- **Bottom subplot**: Daily returns over time
  - Overall daily return (bold red line)
  - Individual group daily returns (colored lines)

#### 2. Information Coefficient Plot (`ic.png`)

**Time Series Mode**:
- Primary y-axis: Rank IC values (blue line)
- Secondary y-axis: Cumulative Rank IC (red line)
- Horizontal reference line at y=0

**Histogram Mode**:
- Distribution of Rank IC values
- Mean line (red dashed)
- Median line (green dashed)
- Frequency counts

#### 3. Average Group Returns Plot (`avg_group_returns.png`)

- Bar chart showing average return for each group
- Value labels on top of each bar
- Groups labeled 0 to n_groups-1

## Examples

### Example 1: Basic Momentum Factor Backtest

```python
import pandas as pd
import numpy as np
from flash_bt.bt._backtest import BackTest

# Generate synthetic momentum factor (12-month return)
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']

# Generate price data
price_df = pd.DataFrame(
    np.random.lognormal(mean=0.001, sigma=0.02, size=(len(dates), len(assets))),
    index=dates,
    columns=assets
).cumprod()

# Generate momentum factor (past 252-day return)
factor_df = price_df.pct_change(252)

# Run backtest
backtest = BackTest(
    factor_df=factor_df,
    price_df=price_df,
    fee=0.0003,
    rebalance_period=21,  # Monthly rebalancing
    n_groups=5,
    weight_method='equal',
    auto_run=True
)

print(f"Sharpe Ratio: {backtest.sharpe_ratio:.4f}")
print(f"Rank IC: {backtest.rank_ic:.4f}")
```

### Example 2: Value Factor with Custom Settings

```python
import pandas as pd
from flash_bt.bt._backtest import BackTest

# Load value factor data (e.g., P/E ratio)
price_df = pd.read_csv('stock_prices.csv', index_col='date')
factor_df = pd.read_csv('pe_ratios.csv', index_col='date')

# Convert index to datetime
price_df.index = pd.to_datetime(price_df.index)
factor_df.index = pd.to_datetime(factor_df.index)

# Run backtest with custom settings
backtest = BackTest(
    factor_df=factor_df,
    price_df=price_df,
    fee=0.0003,
    rebalance_period=63,  # Quarterly rebalancing
    n_groups=10,          # Decile analysis
    weight_method='equal',
    need_preprocess=True,
    need_normalize=True,
    price_threshold=1e6,
    factor_threshold=5,   # More conservative outlier removal
    need_plot=True,
    metrics_path='./value_factor_metrics',
    figures_path='./value_factor_figures',
    cumprod=True,
    auto_run=True
)

# Analyze group performance
print("Average Returns by Group:")
print(backtest.avg_group_ret)
```

### Example 3: Manual Backtest Execution

```python
from flash_bt.bt._backtest import BackTest

# Initialize without auto-running
backtest = BackTest(
    factor_df=factor_df,
    price_df=price_df,
    fee=0.0003,
    rebalance_period=20,
    n_groups=5,
    auto_run=False
)

# Perform custom analysis before running
print(f"Number of assets: {len(backtest.stocks)}")
print(f"Date range: {backtest.time_index[0]} to {backtest.time_index[-1]}")

# Run backtest
metrics = backtest.run()

# Access all metrics
for key, value in metrics.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.4f}")
```

### Example 4: Comparing Multiple Factors

```python
from flash_bt.bt._backtest import BackTest

factors = {
    'momentum': momentum_df,
    'value': value_df,
    'size': size_df
}

results = {}

for factor_name, factor_df in factors.items():
    backtest = BackTest(
        factor_df=factor_df,
        price_df=price_df,
        fee=0.0003,
        rebalance_period=21,
        n_groups=5,
        auto_run=True
    )
    
    results[factor_name] = {
        'sharpe': backtest.sharpe_ratio,
        'rank_ic': backtest.rank_ic,
        'max_dd': backtest.max_drawdown
    }

# Compare results
results_df = pd.DataFrame(results).T
print("Factor Comparison:")
print(results_df)
```

### Example 5: Custom Visualization

```python
from flash_bt.bt._backtest import BackTest
import matplotlib.pyplot as plt

# Run backtest
backtest = BackTest(
    factor_df=factor_df,
    price_df=price_df,
    fee=0.0003,
    rebalance_period=20,
    n_groups=5,
    auto_run=True
)

# Generate time series IC plot
backtest.plot_ic(plot_type='time_series')

# Generate histogram IC plot
backtest.plot_ic(plot_type='histogram')

# Custom analysis of Rank IC
rank_ic_series = backtest.rank_ic_time_series
print(f"Mean Rank IC: {rank_ic_series.mean():.4f}")
print(f"Std Rank IC: {rank_ic_series.std():.4f}")
print(f"ICIR (IC/Std): {rank_ic_series.mean() / rank_ic_series.std():.4f}")
```

### Example 6: Comparing Different Weighting Methods

```python
import pandas as pd
from flash_bt.bt._backtest import BackTest

# Load data
price_df = pd.read_csv('stock_prices.csv', index_col='date')
factor_df = pd.read_csv('factor_data.csv', index_col='date')
market_cap_df = pd.read_csv('market_cap_data.csv', index_col='date')

# Convert index to datetime
price_df.index = pd.to_datetime(price_df.index)
factor_df.index = pd.to_datetime(factor_df.index)
market_cap_df.index = pd.to_datetime(market_cap_df.index)

# Test different weighting methods
weight_methods = ['equal', 'factor', 'inv_vol', 'mean_var', 'market_cap']
results = {}

for method in weight_methods:
    if method == 'market_cap':
        # Market cap weighting requires market_cap_df
        backtest = BackTest(
            factor_df=factor_df,
            price_df=price_df,
            market_cap_df=market_cap_df,
            fee=0.0003,
            rebalance_period=20,
            n_groups=5,
            weight_method=method,
            auto_run=True
        )
    else:
        # Other methods don't require market_cap_df
        backtest = BackTest(
            factor_df=factor_df,
            price_df=price_df,
            fee=0.0003,
            rebalance_period=20,
            n_groups=5,
            weight_method=method,
            auto_run=True
        )
    
    results[method] = {
        'sharpe': backtest.sharpe_ratio,
        'sortino': backtest.sortino_ratio,
        'calmar': backtest.calmar_ratio,
        'max_dd': backtest.max_drawdown,
        'win_rate': backtest.win_rate
    }

# Compare results
results_df = pd.DataFrame(results).T
print("Weighting Method Comparison:")
print(results_df)
```

## Limitations and Assumptions

### Data Requirements

1. **Time Alignment**: The factor and price DataFrames must have overlapping time indices. Non-overlapping periods are automatically removed.

2. **Column Consistency**: Both DataFrames must have identical columns (asset identifiers).

3. **Minimum Data Length**: At least `rebalance_period + 1` observations are required for meaningful backtesting.

4. **Minimum Assets**: At least `n_groups` assets are required to form groups. If fewer assets are available, some groups may be empty.

### Methodological Assumptions

1. **Multiple Weighting Methods**: The framework supports five weighting schemes: equal, factor-based, inverse volatility, mean-variance optimization, and market capitalization. See [Weighting Methods](#weighting-methods) for details.

2. **Transaction Costs**: The framework accounts for transaction costs based on portfolio turnover. Fees are applied at each rebalancing: fee_amount = portfolio_value × fee × turnover, where turnover = Σ|weight_change| / 2. This properly reflects actual trading costs rather than assuming 100% turnover.

3. **No Liquidity Constraints**: The framework assumes infinite liquidity and does not consider trading volume or position limits.

4. **Perfect Execution**: Assumes trades are executed at specified prices without delay or execution risk.

5. **Daily Rebalancing Within Period**: While portfolios are rebalanced every `rebalance_period` days, daily returns are calculated assuming constant weights within each period.

### Forward-Looking Bias Prevention

The framework implements several safeguards to prevent forward-looking bias:

1. **Factor Alignment**: Factors at time $t$ are aligned with returns at time $t+1$:
   $$\text{Rank IC}_t = \text{corr}(\text{rank}(F_t), \text{rank}(R_{t+1}))$$

2. **Group Assignment**: Groups are formed based on factor values at rebalance dates, and returns are calculated for subsequent periods.

3. **No Future Data**: The framework does not use future information in any calculations.

### Statistical Limitations

1. **Small Sample Bias**: With limited data or few assets, statistical metrics may be unreliable.

2. **Non-Normality**: Financial returns often exhibit fat tails and skewness, which may affect ratio-based metrics like Sharpe ratio.

3. **Autocorrelation**: Returns may be autocorrelated, violating independence assumptions in some statistical tests.

4. **Survivorship Bias**: If the dataset does not include delisted assets, results may be upward biased.

### Edge Cases

1. **Zero Volatility**: If all returns are identical (zero volatility), Sharpe and Sortino ratios are set to 0.0 to avoid division by zero.

2. **No Negative Returns**: If all returns are positive, Sortino ratio defaults to Sharpe ratio.

3. **Empty Groups**: If insufficient assets are available, some groups may be empty and excluded from analysis.

4. **Missing Data**: The framework automatically removes rows or columns with missing data, which may reduce sample size.

5. **Outlier Removal**: Aggressive outlier removal (`factor_threshold` too low) may eliminate valid data points.

### Performance Considerations

1. **Memory Usage**: Large datasets (many assets over long periods) may require significant memory.

2. **Computation Time**: The framework computes metrics for all groups and time periods, which can be time-intensive for large datasets.

3. **Plot Generation**: Generating plots for large datasets may be slow.

## Best Practices

### Data Preparation

1. **Quality Check**: Verify data quality before running backtests:
   ```python
   # Check for missing values
   print(price_df.isnull().sum())
   print(factor_df.isnull().sum())
   
   # Check for outliers
   print(price_df.describe())
   print(factor_df.describe())
   ```

2. **Time Alignment**: Ensure factor and price data cover the same time period:
   ```python
   common_dates = price_df.index.intersection(factor_df.index)
   print(f"Common dates: {len(common_dates)}")
   ```

3. **Asset Consistency**: Verify that both DataFrames have the same assets:
   ```python
   assert set(price_df.columns) == set(factor_df.columns)
   ```

### Parameter Selection

1. **Rebalance Period**: Choose based on your strategy's intended turnover:
   - Daily trading: `rebalance_period=1`
   - Weekly: `rebalance_period=5`
   - Monthly: `rebalance_period=21`
   - Quarterly: `rebalance_period=63`

2. **Number of Groups**: Balance granularity with statistical power:
   - Few assets: Use fewer groups (3-5)
   - Many assets: Use more groups (10-20)

3. **Factor Threshold**: Adjust based on factor characteristics:
   - Volatile factors: Higher threshold (10-15)
   - Stable factors: Lower threshold (3-5)

### Interpretation Guidelines

1. **Sharpe Ratio**:
   - > 2.0: Excellent
   - 1.0 - 2.0: Good
   - 0.5 - 1.0: Moderate
   - < 0.5: Poor

2. **Rank IC**:
   - > 0.05: Strong predictive power
   - 0.02 - 0.05: Moderate predictive power
   - < 0.02: Weak predictive power

3. **ICIR (IC/Std)**:
   - > 0.5: Stable factor
   - 0.25 - 0.5: Moderately stable
   - < 0.25: Unstable factor

4. **Maximum Drawdown**:
   - < -10%: Low risk
   - -10% to -25%: Moderate risk
   - > -25%: High risk

### Validation Strategies

1. **Out-of-Sample Testing**: Split data into in-sample and out-of-sample periods:
   ```python
   split_date = '2021-12-31'
   train_factor = factor_df[:split_date]
   train_price = price_df[:split_date]
   test_factor = factor_df[split_date:]
   test_price = price_df[split_date:]
   
   # Train on in-sample data
   backtest_train = BackTest(train_factor, train_price, 21, 5, auto_run=True)
   
   # Test on out-of-sample data
   backtest_test = BackTest(test_factor, test_price, 21, 5, auto_run=True)
   ```

2. **Cross-Validation**: Use time-series cross-validation for robust results.

3. **Benchmark Comparison**: Compare against relevant benchmarks:
   ```python
   # Calculate benchmark returns (e.g., market index)
   benchmark_returns = benchmark_df.pct_change().dropna()
   
   # Compare Sharpe ratios
   print(f"Strategy Sharpe: {backtest.sharpe_ratio:.4f}")
   print(f"Benchmark Sharpe: {(benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)):.4f}")
   ```

## Troubleshooting

### Common Issues

#### 1. ValueError: "No valid data after removing nans and outliers."

**Cause**: Too much data removed during preprocessing.

**Solution**:
- Check data quality: `print(factor_df.isnull().sum())`
- Adjust thresholds: Increase `price_threshold` or `factor_threshold`
- Disable normalization: Set `need_normalize=False`

#### 2. ValueError: "No overlapping time index between valid factor and price data."

**Cause**: Factor and price data have no common dates.

**Solution**:
- Check time ranges: `print(factor_df.index[0], price_df.index[0])`
- Ensure both have DatetimeIndex: `factor_df.index = pd.to_datetime(factor_df.index)`
- Verify date formats are consistent

#### 3. AssertionError: "factor_df must have the same columns as price_df."

**Cause**: Column mismatch between DataFrames.

**Solution**:
- Find common assets: `common_assets = factor_df.columns.intersection(price_df.columns)`
- Subset both DataFrames: `factor_df = factor_df[common_assets]`

#### 4. Zero or NaN Sharpe/Sortino/Calmar Ratios

**Cause**: Zero volatility or division by zero.

**Solution**:
- Check if returns are constant: `print(backtest.daily_ret.std())`
- Verify data has sufficient variation
- This is expected behavior for constant returns

#### 5. Empty Groups in Analysis

**Cause**: Insufficient assets for specified number of groups.

**Solution**:
- Reduce `n_groups` parameter
- Increase number of assets in dataset
- Check if preprocessing removed too many assets

#### 6. Memory Error with Large Datasets

**Cause**: Dataset too large for available memory.

**Solution**:
- Reduce time period or number of assets
- Process data in chunks
- Increase available memory

#### 7. Slow Performance

**Cause**: Large dataset or complex calculations.

**Solution**:
- Disable plotting: Set `need_plot=False`
- Reduce `n_groups`
- Use subset of data for testing

### Debugging Tips

1. **Enable Verbose Output**: Add print statements to track progress:
   ```python
   print(f"Preprocessing: {len(factor_df)} rows")
   print(f"Running backtest...")
   print(f"Computing metrics...")
   ```

2. **Inspect Intermediate Results**:
   ```python
   # Check preprocessed data
   print(backtest.factor_df.head())
   print(backtest.price_df.head())
   
   # Check returns
   print(backtest.daily_ret.describe())
   ```

3. **Validate Metrics**:
   ```python
   # Manually calculate Sharpe ratio
   manual_sharpe = (backtest.daily_ret.mean() / backtest.daily_ret.std() * np.sqrt(252))
   print(f"Manual Sharpe: {manual_sharpe:.4f}")
   print(f"Computed Sharpe: {backtest.sharpe_ratio:.4f}")
   ```

4. **Check Group Assignments**:
   ```python
   # Verify group distribution
   print(backtest.avg_group_ret)
   ```

### Getting Help

If you encounter issues not covered here:

1. Check the code comments in `_backtest.py` for detailed explanations
2. Verify your data format matches the expected structure
3. Start with a simple example and gradually increase complexity
4. Ensure all dependencies are correctly installed

## License

This framework is provided as-is for educational and research purposes.

## Contributing

The framework is designed with extensibility in mind. Future enhancements may include:

- Additional weighting methods (market-cap, risk-parity, etc.)
- Transaction cost modeling
- Liquidity constraints
- Advanced risk metrics (VaR, CVaR)
- Multi-factor models
- Portfolio optimization
- Monte Carlo simulations

## Citation

If you use this framework in your research, please cite appropriately.

## Version History

- **Current Version**: Comprehensive backtesting framework with factor analysis
- **Key Features**: Equal-weighted portfolios, quantile grouping, comprehensive metrics, visualization

---

**Note**: This framework is a research tool and should not be used for live trading without thorough validation and additional risk management measures.
