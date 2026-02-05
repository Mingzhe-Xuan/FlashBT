import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'e:\\课业\\HKU-RA\\2.4 - batcktest_framework\\flash_bt')
from _backtest import BackTest

# Create sample data for testing
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=200, freq='D')
assets = [f'Asset_{i}' for i in range(10)]

# Factor data
factor_df = pd.DataFrame(
    np.random.randn(200, 10), 
    index=dates, 
    columns=assets
)

# Price data
price_df = pd.DataFrame(
    100 + np.random.randn(200, 10).cumsum(axis=0), 
    index=dates, 
    columns=assets
)

print("Testing mean-variance weight method...")
print("=" * 60)

# Test: Mean-variance weighting
print("\nTesting Mean-Variance Weighting...")
try:
    bt_mean_var = BackTest(
        factor_df=factor_df,
        price_df=price_df,
        fee=0.0003,
        rebalance_period=20,
        n_groups=3,
        weight_method='mean_var',
        need_plot=False,
        need_preprocess=False,
        cumprod=True,
        auto_run=True
    )
    print(f"✓ Mean-variance weighting initialized successfully")
    print(f"  Sharpe Ratio: {bt_mean_var.sharpe_ratio:.4f}")
    print(f"  Sortino Ratio: {bt_mean_var.sortino_ratio:.4f}")
    print(f"  Calmar Ratio: {bt_mean_var.calmar_ratio:.4f}")
    print(f"  Max Drawdown: {bt_mean_var.max_drawdown:.4f}")
    print(f"  Win Rate: {bt_mean_var.win_rate:.4f}")
    print(f"  Rank IC: {bt_mean_var.rank_ic:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed!")
