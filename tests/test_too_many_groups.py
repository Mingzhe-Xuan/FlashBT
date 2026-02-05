import pandas as pd
from flash_bt.backtest import Backtest

# Load the data
price_df = pd.read_csv("example_price.csv", index_col=0, parse_dates=True).dropna()
factor_df = pd.read_csv("example_factors.csv", index_col=0, parse_dates=True).dropna()

print(f"Number of assets: {len(price_df.columns)}")
print(f"Asset names: {list(price_df.columns)}")

# Test with n_groups greater than number of assets
n_assets = len(price_df.columns)
n_groups_too_many = n_assets + 5

print(f"\nTesting with n_groups = {n_groups_too_many} (more than {n_assets} assets)...")

try:
    bt = Backtest(
        factor_df=factor_df,
        price_df=price_df,
        fee=3 * 1e-4,
        rebalance_period=5,
        n_groups=n_groups_too_many,  # Too many groups
    )
    metrics = bt.run()
    print("Backtest completed.")
    print(f"Groups average returns: {bt.avg_group_ret}")
    print(f"Groups daily returns shape: {bt.avg_group_daily_ret.shape}")
    print(f"Available groups: {list(bt.avg_group_daily_ret.columns) if not bt.avg_group_daily_ret.empty else 'None'}")
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()