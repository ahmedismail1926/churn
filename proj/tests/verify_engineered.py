import pandas as pd

print("="*70)
print("ENGINEERED DATA VERIFICATION")
print("="*70)

df = pd.read_csv('data/processed/engineered_data.csv')

print(f"\nShape: {df.shape}")
print(f"Total features: {df.shape[1]}")
print(f"Total samples: {df.shape[0]}")

print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\nData types:")
print(df.dtypes.value_counts())

print(f"\nSample (first 5 rows, first 8 columns):")
print(df.iloc[:5, :8])

print(f"\nStatistics (first 5 numerical columns):")
print(df.iloc[:, :5].describe())

print("\n" + "="*70)
print("SUCCESS - Engineered data is ready!")
print("="*70)
