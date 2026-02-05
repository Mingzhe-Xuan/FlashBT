from flash_bt.backtest import Backtest
from calculate_ICM import calculate_ICM
import pandas as pd

def main():
    price_df = pd.read_csv("example_price.csv", index_col=0, parse_dates=True).dropna()
    look_back = 20
    top_n = 5

    print("Starting ICM calculation...")
    factor_df = calculate_ICM(price_df, look_back=look_back, top_n=top_n)
    factor_df.to_csv("example_factors.csv")
    print("ICM factors have been saved to example_factors.csv")

    print("Starting backtest...")
    bt = Backtest(
        factor_df=factor_df,
        price_df=price_df,
        fee=3 * 1e-4,
        rebalance_period=1,
        n_groups=5,
    )
    bt.run()
    print("Backtest results have been saved to example_backtest.png")

if __name__ == "__main__":
    main()