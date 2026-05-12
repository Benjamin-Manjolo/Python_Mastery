import pandas as pd
import numpy as np

# Reproducible random numbers
np.random.seed(42)

# ============================================
# CREATE MOCK DATA
# ============================================

n_samples = 1000

# Generate features
ages = np.random.randint(18, 65, n_samples)

salaries = np.random.randint(20000, 150000, n_samples)

cities = np.random.choice(
    ['Blantyre', 'Lilongwe', 'Mzuzu'],
    n_samples
)

# ============================================
# CREATE TARGET VARIABLE
# ============================================

# Simple logic for purchased:
# Higher salary + older age = more likely to purchase

purchase_score = (
    (ages * 0.03) +
    (salaries * 0.00002) +
    np.random.normal(0, 1, n_samples)
)

# Convert to binary classes
purchased = (purchase_score > 3.5).astype(int)

# ============================================
# CREATE DATAFRAME
# ============================================

df = pd.DataFrame({
    'age': ages,
    'salary': salaries,
    'city': cities,
    'purchased': purchased
})

# ============================================
# ADD SOME MISSING VALUES
# ============================================

df.loc[5, 'salary'] = np.nan
df.loc[20, 'city'] = np.nan

# ============================================
# SAVE TO CSV
# ============================================

df.to_csv('data.csv', index=False)

# ============================================
# DISPLAY SAMPLE DATA
# ============================================

print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df['purchased'].value_counts())