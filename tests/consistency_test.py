import pandas as pd
import numpy as np
from flash_bt.backtest import Backtest

# Load the corrected data
price_df = pd.read_csv("example_price.csv", index_col=0, parse_dates=True).dropna()
factor_df = pd.read_csv("example_factors.csv", index_col=0, parse_dates=True).dropna()

print("Testing consistency across different rebalance periods...")

for rebalance_period in [1, 2, 5]:
    print(f"\n--- Testing rebalance_period = {rebalance_period} ---")
    try:
        bt = Backtest(
            factor_df=factor_df,
            price_df=price_df,
            fee=3 * 1e-4,
            rebalance_period=rebalance_period,
            n_groups=5,
        )
        metrics = bt.run()
        print(f"Success! Groups average returns: {bt.avg_group_ret.values}")
        print(f"Groups daily returns shape: {bt.avg_group_daily_ret.shape}")
    except Exception as e:
        print(f"Error with rebalance_period {rebalance_period}: {str(e)}")
        import traceback
        traceback.print_exc()