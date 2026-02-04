# Backtesting framework for 横截面因子

## Input
factor_value: dataframe
close: dataframe
(possibly include error and noise)

## Output

Perform backtest by group and then visualize the cumulative return for each group. Within each group, we provide several apis to set the weights.

Also, we provide some choices to set the period of position shifting.