import pandas as pd

df1 = pd.read_csv('example_factors.csv', index_col="Date")
df2 = pd.read_csv('example_price.csv', index_col="time")
print(df1.columns.to_list())
print(df2.columns.to_list())
# df1.to_csv('example_factors.csv', index=False)
# df2.to_csv('example_price.csv', index=False)
