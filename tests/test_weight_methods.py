import pandas as pd
import numpy as np
from _backtest import BackTest

# Create sample data for testing
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
assets = [f'Asset_{i}' for i in range(10)]

# Factor data
factor_df = pd.DataFrame(
    np.random.randn(100, 10), 
    index=dates, 
    columns=assets
)

# Price data
price_df = pd.DataFrame(
    100 + np.random.randn(100, 10).cumsum(axis=0), 
    index=dates, 
    columns=assets
)

print("Testing weight methods...")
print("=" * 60)

# Test 1: Equal weighting
print("\n1. Testing Equal Weighting...")
try:
    bt_equal = BackTest(
        factor_df=factor_df,
        price_df=price_df,
        fee=0.0003,
        rebalance_period=10,
        n_groups=3,
        weight_method='equal',
        need_plot=False,
        need_preprocess=False,
        cumprod=True,
        auto_run=True
    )
    print(f"   ✓ Equal weighting initialized successfully")
    print(f"   Sharpe Ratio: {bt_equal.sharpe_ratio:.4f}")
    print(f"   Sortino Ratio: {bt_equal.sortino_ratio:.4f}")
    print(f"   Max Drawdown: {bt_equal.max_drawdown:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Factor-based weighting
print("\n2. Testing Factor-based Weighting...")
try:
    bt_factor = BackTest(
        factor_df=factor_df,
        price_df=price_df,
        fee=0.0003,
        rebalance_period=10,
        n_groups=3,
        weight_method='factor',
        need_plot=False,
        need_preprocess=False,
        cumprod=True,
        auto_run=True
    )
    print(f"   ✓ Factor-based weighting initialized successfully")
    print(f"   Sharpe Ratio: {bt_factor.sharpe_ratio:.4f}")
    print(f"   Sortino Ratio: {bt_factor.sortino_ratio:.4f}")
    print(f"   Max Drawdown: {bt_factor.max_drawdown:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Inverse volatility weighting
print("\n3. Testing Inverse Volatility Weighting...")
try:
    bt_inv_vol = BackTest(
        factor_df=factor_df,
        price_df=price_df,
        fee=0.0003,
        rebalance_period=10,
        n_groups=3,
        weight_method='inv_vol',
        need_plot=False,
        need_preprocess=False,
        cumprod=True,
        auto_run=True
    )
    print(f"   ✓ Inverse volatility weighting initialized successfully")
    print(f"   Sharpe Ratio: {bt_inv_vol.sharpe_ratio:.4f}")
    print(f"   Sortino Ratio: {bt_inv_vol.sortino_ratio:.4f}")
    print(f"   Max Drawdown: {bt_inv_vol.max_drawdown:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Invalid weight method
print("\n4. Testing Invalid Weight Method...")
try:
    bt_invalid = BackTest(
        factor_df=factor_df,
        price_df=price_df,
        fee=0.0003,
        rebalance_period=10,
        n_groups=3,
        weight_method='invalid_method',
        need_plot=False,
        need_preprocess=False,
        cumprod=True
    )
    print(f"   ✗ Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError: {e}")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
