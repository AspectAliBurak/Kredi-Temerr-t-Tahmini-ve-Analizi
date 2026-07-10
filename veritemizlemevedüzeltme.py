import pandas as pd

df = pd.read_csv("Loan_Default.csv")

cat_cols = df.select_dtypes(include="string").columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df.to_csv("clean_data.csv", index=False)