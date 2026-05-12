import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load from url
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
cols = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age', 'outcome']

df = pd.read_csv(url, names=cols)

# First inspection
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df['outcome'].value_counts())

# Distribution of each feature by class
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
features_to_plot = [col for col in df.columns if col not in ['outcome']]

axes = axes.flatten()

for i, feat in enumerate(features_to_plot):
    sns.histplot(data=df, x=feat, hue='outcome', kde=True, ax=axes[i], palette='viridis', common_norm=False)
    axes[i].set_title(f'Distribution of {feat} by Outcome', fontsize=12)
    axes[i].set_xlabel(feat, fontsize=10)
    axes[i].set_ylabel('Density', fontsize=10)

plt.suptitle('Feature Distributions by Diabetes Outcome', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

# Correlation heatmap
# BUG 1 FIX: fmt='.2\nf' had an accidental line break in the string — fixed to '.2f'
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Pima Indians Diabetes Dataset', fontsize=16)
plt.show()

print(df.corr()['outcome'].sort_values(ascending=False))

# Clean and prepare the data
cols_with_zero = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)

for col in cols_with_zero:
    df[col].fillna(df[col].median(), inplace=True)

# BUG 2 FIX: 'zero_cols' was never defined — it should be 'cols_with_zero'
# BUG 3 FIX: also redundant since we already filled NaNs above — removed the duplicate fillna block

print(df.isnull().sum())

# Split features & target
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

X = df.drop(columns=['outcome']).values
y = df['outcome'].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a simple logistic regression classifier
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print('\nClassification report:')
print(classification_report(y_test, y_pred))
print(f'AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}')