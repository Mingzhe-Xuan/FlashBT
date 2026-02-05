import pandas as pd
import numpy as np

# Simulate the rebalance date selection logic
dates = pd.date_range('2020-01-01', periods=10, freq='D')
rebalance_period = 1
rebalance_dates = dates[:: rebalance_period]

print(f"All dates: {dates.tolist()}")
print(f"Rebalance dates (every {rebalance_period} day(s)): {rebalance_dates.tolist()}")

# Simulate what happens with my fix
for i, date in enumerate(rebalance_dates[:-1]):  # Exclude last to avoid index out of bounds
    date_idx = dates.get_loc(date)
    # My fix: end_idx = min(date_idx + rebalance_period + 1, len(dates))
    end_idx = min(date_idx + rebalance_period + 1, len(dates))
    print(f"Date {date}, date_idx={date_idx}, end_idx={end_idx}")
    print(f"  Range for returns: {list(range(date_idx + 1, end_idx))}")
    if range(date_idx + 1, end_idx):
        return_dates = [dates[idx] for idx in range(date_idx + 1, end_idx)]
        print(f"  Will calculate returns for: {return_dates}")
    print()