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

print("Comparing all weight methods...")
print("=" * 80)

methods = ['equal', 'factor', 'inv_vol', 'mean_var']
results = {}

for method in methods:
    print(f"\nTesting {method.upper()} Weighting...")
    try:
        bt = BackTest(
            factor_df=factor_df,
            price_df=price_df,
            fee=0.0003,
            rebalance_period=20,
            n_groups=3,
            weight_method=method,
            need_plot=False,
            need_preprocess=False,
            cumprod=True,
            auto_run=True
        )
        results[method] = {
            'sharpe': bt.sharpe_ratio,
            'sortino': bt.sortino_ratio,
            'calmar': bt.calmar_ratio,
            'max_dd': bt.max_drawdown,
            'win_rate': bt.win_rate,
            'rank_ic': bt.rank_ic
        }
        print(f"  ✓ {method.upper()} initialized successfully")
        print(f"    Sharpe Ratio: {bt.sharpe_ratio:.4f}")
        print(f"    Sortino Ratio: {bt.sortino_ratio:.4f}")
        print(f"    Calmar Ratio: {bt.calmar_ratio:.4f}")
        print(f"    Max Drawdown: {bt.max_drawdown:.4f}")
        print(f"    Win Rate: {bt.win_rate:.4f}")
        print(f"    Rank IC: {bt.rank_ic:.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Summary Comparison:")
print("-" * 80)
print(f"{'Method':<15} {'Sharpe':<10} {'Sortino':<10} {'Calmar':<10} {'Max DD':<10} {'Win Rate':<10}")
print("-" * 80)

for method in methods:
    if method in results:
        r = results[method]
        print(f"{method.upper():<15} {r['sharpe']:>9.4f} {r['sortino']:>9.4f} {r['calmar']:>9.4f} {r['max_dd']:>9.4f} {r['win_rate']:>9.4f}")

print("-" * 80)
print("All tests completed!")
