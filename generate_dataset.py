import pandas as pd
import numpy as np

np.random.seed(42)

N = 300_000          # total transactions
N_HALF = N // 2      # 150,000 each class

print(f"Generating {N:,} transactions ({N_HALF:,} legit + {N_HALF:,} fraud)...")

# ── LEGIT TRANSACTIONS (Class = 0) ───────────────────────────────────────────
legit = pd.DataFrame({
    'distance_from_home':  np.random.exponential(scale=10,  size=N_HALF).clip(0),
    'distance_from_last_transaction': np.random.exponential(scale=5, size=N_HALF).clip(0),
    'ratio_to_median_purchase_price': np.random.lognormal(mean=0.1, sigma=0.4, size=N_HALF).clip(0.1, 10),
    'repeat_retailer':    np.random.choice([0, 1], size=N_HALF, p=[0.2, 0.8]),
    'used_chip':          np.random.choice([0, 1], size=N_HALF, p=[0.25, 0.75]),
    'used_pin_number':    np.random.choice([0, 1], size=N_HALF, p=[0.4, 0.6]),
    'online_order':       np.random.choice([0, 1], size=N_HALF, p=[0.7, 0.3]),
    'email_domain_risk':  np.random.beta(a=2, b=8, size=N_HALF),          # mostly low risk
    'transaction_hour':   np.random.normal(loc=13, scale=4, size=N_HALF).clip(0, 23).astype(int),
    'amount':             np.random.lognormal(mean=3.5, sigma=1.0, size=N_HALF).clip(1, 5000),
    'Class': 0
})

# ── FRAUD TRANSACTIONS (Class = 1) ───────────────────────────────────────────
fraud = pd.DataFrame({
    'distance_from_home':  np.random.exponential(scale=80,  size=N_HALF).clip(0),   # far from home
    'distance_from_last_transaction': np.random.exponential(scale=50, size=N_HALF).clip(0),
    'ratio_to_median_purchase_price': np.random.lognormal(mean=1.2, sigma=0.8, size=N_HALF).clip(0.1, 50),
    'repeat_retailer':    np.random.choice([0, 1], size=N_HALF, p=[0.65, 0.35]),    # new retailers
    'used_chip':          np.random.choice([0, 1], size=N_HALF, p=[0.8, 0.2]),      # chip rarely used
    'used_pin_number':    np.random.choice([0, 1], size=N_HALF, p=[0.85, 0.15]),    # PIN rarely used
    'online_order':       np.random.choice([0, 1], size=N_HALF, p=[0.2, 0.8]),      # mostly online
    'email_domain_risk':  np.random.beta(a=8, b=2, size=N_HALF),                    # high risk domains
    'transaction_hour':   np.random.choice(list(range(0, 6)) + list(range(22, 24)),
                                           size=N_HALF),                             # odd hours
    'amount':             np.random.lognormal(mean=5.0, sigma=1.2, size=N_HALF).clip(1, 20000),
    'Class': 1
})

# ── COMBINE & SHUFFLE ─────────────────────────────────────────────────────────
df = pd.concat([legit, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts()}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
output_path = 'creditcard_300k.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Saved as '{output_path}' — {len(df):,} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")
