import pandas as pd
import numpy as np


def calculate_ICM(
    price_df: pd.DataFrame, look_back: int = 20, top_n: int = 5
) -> pd.DataFrame:
    # Suppose the data is cleaned
    factor_df = pd.DataFrame(index=price_df.index, columns=price_df.columns)

    # ICM_{d, industry} = \sum_{i=1}^n w_i * Ret_{d_i, industry}
    # w_i = 2^{-\frac{i-1}{n-1}}
    # where d_1, \cdots, d_n are the top n days with the highest absolute value
    # of daily return within the look_back period,
    # and Ret_{d_i, industry} is the return of industry d_i.
    daily_ret = price_df.pct_change().dropna()
    for ind in price_df.columns:
        daily_ret_ind = daily_ret[ind]
        for j in range(look_back, daily_ret_ind.shape[0]):
            top_n_abs_ret_ji = (
                daily_ret_ind.iloc[j - look_back : j]
                .abs()
                .sort_values(ascending=False)[:top_n]
            )
            w_ji = 2 ** (-np.arange(top_n) / (top_n - 1))
            factor_df.loc[factor_df.index[j], ind] = (top_n_abs_ret_ji * w_ji).sum()

    return factor_df.dropna()


def main():
    price_df = pd.read_csv("example_price.csv", index_col=0, parse_dates=True).dropna()
    look_back = 20
    top_n = 5

    print("Starting ICM calculation...")
    factor_df = calculate_ICM(price_df, look_back=20, top_n=top_n)
    factor_df.to_csv("example_factors.csv")
    print("ICM factors have been saved to example_factors.csv")


if __name__ == "__main__":
    main()
