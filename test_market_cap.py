import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'e:\\课业\\HKU-RA\\2.4 - batcktest_framework\\flash_bt\\bt')
from _backtest import BackTest

# Create sample data for testing
np.random.seed(42)
dates = pd.date_range('2015-01-05', periods=200, freq='D')
assets = ['000001.SH', '000016.SH', '000300.SH', '000510.CSI', '000905.SH']

# Factor data
factor_df = pd.DataFrame(
    np.random.randn(200, 5), 
    index=dates, 
    columns=assets
)

# Price data
price_df = pd.DataFrame(
    100 + np.random.randn(200, 5).cumsum(axis=0), 
    index=dates, 
    columns=assets
)

# Market cap data (similar to the example file)
market_cap_df = pd.DataFrame(
    np.random.uniform(1000, 10000, (200, 5)), 
    index=dates, 
    columns=assets
)

print("Testing market_cap weight method...")
print("=" * 80)

# Test: Market cap weighting
print("\nTesting Market Cap Weighting...")
try:
    bt_market_cap = BackTest(
        factor_df=factor_df,
        price_df=price_df,
        market_cap_df=market_cap_df,
        fee=0.0003,
        rebalance_period=20,
        n_groups=3,
        weight_method='market_cap',
        need_plot=False,
        need_preprocess=False,
        cumprod=True,
        auto_run=True
    )
    print(f"✓ Market cap weighting initialized successfully")
    print(f"  Sharpe Ratio: {bt_market_cap.sharpe_ratio:.4f}")
    print(f"  Sortino Ratio: {bt_market_cap.sortino_ratio:.4f}")
    print(f"  Calmar Ratio: {bt_market_cap.calmar_ratio:.4f}")
    print(f"  Max Drawdown: {bt_market_cap.max_drawdown:.4f}")
    print(f"  Win Rate: {bt_market_cap.win_rate:.4f}")
    print(f"  Rank IC: {bt_market_cap.rank_ic:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test: Market cap weighting without market_cap_df (should raise error)
print("\nTesting Market Cap Weighting without market_cap_df (should raise error)...")
try:
    bt_no_mcap = BackTest(
        factor_df=factor_df,
        price_df=price_df,
        market_cap_df=None,
        fee=0.0003,
        rebalance_period=20,
        n_groups=3,
        weight_method='market_cap',
        need_plot=False,
        need_preprocess=False,
        cumprod=True,
        auto_run=True
    )
    print(f"✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

# Test: Equal weighting with market_cap_df (should work, market_cap_df ignored)
print("\nTesting Equal Weighting with market_cap_df (should work)...")
try:
    bt_equal_with_mcap = BackTest(
        factor_df=factor_df,
        price_df=price_df,
        market_cap_df=market_cap_df,
        fee=0.0003,
        rebalance_period=20,
        n_groups=3,
        weight_method='equal',
        need_plot=False,
        need_preprocess=False,
        cumprod=True,
        auto_run=True
    )
    print(f"✓ Equal weighting with market_cap_df initialized successfully")
    print(f"  Sharpe Ratio: {bt_equal_with_mcap.sharpe_ratio:.4f}")
    print(f"  Sortino Ratio: {bt_equal_with_mcap.sortino_ratio:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("All tests completed!")
