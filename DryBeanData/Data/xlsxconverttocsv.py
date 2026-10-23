import pandas as pd

df = pd.read_excel("Dry_Bean_Dataset.xlsx")
df.to_csv("dry_bean.csv", index=False)

print(df.head())