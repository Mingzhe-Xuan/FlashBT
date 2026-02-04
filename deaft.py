import pandas as pd

df = pd.read_csv('example_factors.csv')
print(df.head())
df = df.drop(columns=['a', 'b', 'c', 'd', 'e', 'f'])
df.to_csv('example_factors.csv')