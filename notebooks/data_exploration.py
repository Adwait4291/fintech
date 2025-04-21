import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Load the dataset
try:
    df = pd.read_csv("fintech_application_data.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: fintech_application_data.csv not found. Please ensure the file is uploaded.")
    # Exit or handle appropriately if the file isn't loaded
    exit() # Or raise an exception

# --- Basic Data Examination ---
print("\n--- First 5 Rows ---")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Basic Statistics (Numerical Features) ---")
# Include datetime columns if appropriate after conversion, exclude IDs
print(df.describe().to_markdown(numalign="left", stralign="left"))

print("\n--- Basic Statistics (Categorical Features) ---")
print(df.describe(include='object').to_markdown(numalign="left", stralign="left"))

# --- Data Quality Checks ---
print("\n--- Missing Values ---")
print(df.isnull().sum().to_markdown(numalign="left", stralign="left"))
# Note: Missing subscription_date is expected for users who didn't subscribe.

# --- Convert Date/Time Columns ---
# Convert relevant columns to datetime objects, handling potential errors
# Combine initial_access and access_time for a full timestamp if needed,
# but for trial start, initial_access seems sufficient.
df['initial_access'] = pd.to_datetime(df['initial_access'], errors='coerce')
df['subscription_date'] = pd.to_datetime(df['subscription_date'], errors='coerce')

# Check for conversion errors (NaT values created by coerce)
print("\n--- Date Conversion Issues ---")
print(f"Issues in 'initial_access': {df['initial_access'].isnull().sum()}")
# We expect NaNs in subscription_date, so we check how many non-NaNs became NaT
# This check is slightly complex, maybe just report total NaNs after conversion
print(f"NaNs in 'subscription_date' after conversion: {df['subscription_date'].isnull().sum()}")


# --- Identify Key Variables (Confirmation) ---
print("\n--- Key Variables Identified ---")
print("Initial Access Date:", 'initial_access' if 'initial_access' in df.columns else 'Not Found')
print("Subscription Date:", 'subscription_date' if 'subscription_date' in df.columns else 'Not Found')
print("Subscription Status:", 'subscription_status' if 'subscription_status' in df.columns else 'Not Found')

# --- Visualizations for User Behavior ---

# Distribution of Numerical Features
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
# Remove ID and status flags if they are not meaningful for distribution plots
numerical_features = [col for col in numerical_features if col not in ['user_id', 'weekday', 'played_game', 'premium_features_used', 'subscription_status']]

print("\n--- Plotting Numerical Feature Distributions ---")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1) # Adjust grid size as needed
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Count Plots for Categorical Features
categorical_features = df.select_dtypes(include='object').columns.tolist()
# Exclude high-cardinality or less relevant object columns for general plots
plot_cats = [col for col in categorical_features if col not in ['viewed_screens', 'access_time']] # access_time might be too granular

print("\n--- Plotting Categorical Feature Counts ---")
plt.figure(figsize=(15, 12)) # Adjust figure size
num_plots = len(plot_cats)
cols = 3 # Number of columns in subplot grid
rows = (num_plots + cols - 1) // cols # Calculate rows needed

for i, col in enumerate(plot_cats):
    plt.subplot(rows, cols, i + 1)
    # For high cardinality features, plot top N categories
    if df[col].nunique() > 20:
        top_n = df[col].value_counts().nlargest(15).index
        sns.countplot(y=df[df[col].isin(top_n)][col], order=top_n)
        plt.title(f'Top 15 {col}')
    else:
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Count of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Example: Relationship between a feature and subscription status
print("\n--- Example: Age vs Subscription Status ---")
plt.figure(figsize=(8, 6))
sns.boxplot(x='subscription_status', y='user_age', data=df)
plt.title('User Age Distribution by Subscription Status')
plt.xlabel('Subscription Status (0: No, 1: Yes)')
plt.ylabel('User Age')
plt.show()

print("\n--- Data Understanding & Exploration Complete ---")

# Keep the dataframe for the next steps
df_eda = df.copy()
```
> * **Code Description:** This script loads the data, performs initial checks (head, info, describe), identifies missing values, converts date columns, and generates histograms for numerical features and bar plots for categorical features to understand their distributions. It also shows an example boxplot comparing user age based on their overall subscription status.
> * **Next Steps:** The next block will focus on Feature Engineering, specifically creating the target variable based on the 7-day conversion window and deriving other potentially useful featur