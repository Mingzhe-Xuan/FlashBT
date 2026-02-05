from flash_bt.backtest import Backtest
import pandas as pd

# Load the corrected data
price_df = pd.read_csv("example_price.csv", index_col=0, parse_dates=True).dropna()
factor_df = pd.read_csv("example_factors.csv", index_col=0, parse_dates=True).dropna()

print("Testing with rebalance_period=1...")
print(f"Factor DataFrame shape: {factor_df.shape}")
print(f"Price DataFrame shape: {price_df.shape}")

try:
    bt = Backtest(
        factor_df=factor_df,
        price_df=price_df,
        fee=3 * 1e-4,
        rebalance_period=1,  # Changed to 1
        n_groups=5,
    )
    metrics = bt.run()
    print("Backtest with rebalance_period=1 completed successfully!")
    print(f"Groups average returns: {bt.avg_group_ret}")
    print(f"Groups daily returns shape: {bt.avg_group_daily_ret.shape}")
    print(f"Sample of group daily returns:")
    print(bt.avg_group_daily_ret.head(10))
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()