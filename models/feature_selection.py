import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression # For RFE
from sklearn.ensemble import RandomForestClassifier # For feature importance

# Assuming df_processed is the dataframe from the previous step
df_select = df_processed.copy()

# --- Define Feature Types ---
# Drop original date columns, intermediate dates, status, and text columns not used directly
# Also drop user_id as it's an identifier
cols_to_drop = ['user_id', 'initial_access', 'access_time', 'subscription_date',
                'trial_end_date', 'conversion_window_end_date',
                'subscription_status', 'viewed_screens']
df_select = df_select.drop(columns=cols_to_drop)

# Identify numerical and categorical features for preprocessing
numerical_features = df_select.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove('converted_within_7_days') # Remove target variable

categorical_features = df_select.select_dtypes(include='object').columns.tolist()

print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# --- Define Target and Features ---
X = df_select.drop('converted_within_7_days', axis=1)
y = df_select['converted_within_7_days']

# --- Train-Test Split (Stratified) ---
# Split before imputation/scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Data Split ---")
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training target distribution:\n", y_train.value_counts(normalize=True))
print("Test target distribution:\n", y_test.value_counts(normalize=True))


# --- Preprocessing Pipeline ---
# Impute missing numerical values with median
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Scale after imputation
])

# Impute missing categorical values with 'Missing' and then OneHotEncode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Use sparse_output=False for easier handling later
])

# Create the preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any) - should be none here
)

# Fit the preprocessor on the training data and transform both train and test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after OneHotEncoding
ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# Convert processed arrays back to DataFrames (optional, but good for inspection)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)

print("\n--- Preprocessing Complete ---")
print("Processed training data shape:", X_train_processed_df.shape)


# --- Feature Selection ---

# 1. Correlation Analysis (on processed numerical features)
# Note: Correlation is less direct after OHE. We focus on original numericals or use MI.
print("\n--- Correlation Analysis (Numerical Features Pre-Scaling) ---")
plt.figure(figsize=(12, 8))
sns.heatmap(X_train[numerical_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
# Identify highly correlated features (e.g., |corr| > 0.8) - decide whether to drop one.

# 2. Mutual Information
print("\n--- Mutual Information ---")
mi_scores = mutual_info_classif(X_train_processed_df, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=all_feature_names).sort_values(ascending=False)
print("Top 20 features by Mutual Information:")
print(mi_series.head(20).to_markdown())

plt.figure(figsize=(10, 8))
mi_series.head(20).plot(kind='barh')
plt.title('Top 20 Features by Mutual Information')
plt.xlabel('Mutual Information Score')
plt.gca().invert_yaxis()
plt.show()

# 3. Recursive Feature Elimination (RFE) with Logistic Regression
print("\n--- Recursive Feature Elimination (RFE) ---")
# Select top N features (e.g., 20)
n_features_to_select = 25 # Adjust as needed
estimator = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced') # Use balanced weights due to potential imbalance
selector_rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
selector_rfe.fit(X_train_processed_df, y_train)

rfe_selected_features = X_train_processed_df.columns[selector_rfe.support_]
print(f"Top {n_features_to_select} features selected by RFE:")
print(list(rfe_selected_features))

# 4. Feature Importance from Tree-based Models (Random Forest)
print("\n--- Feature Importance (Random Forest) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_processed_df, y_train)
importances = rf.feature_importances_
importance_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

print("Top 20 features by Random Forest Importance:")
print(importance_series.head(20).to_markdown())

plt.figure(figsize=(10, 8))
importance_series.head(20).plot(kind='barh')
plt.title('Top 20 Features by Random Forest Importance')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.show()

# --- Select Final Feature Set ---
# Combine insights: Choose features consistently ranked high by MI and RF, possibly guided by RFE.
# For this example, let's select the top N features from RF importance.
N_FINAL_FEATURES = 20 # Choose the desired number
final_features = importance_series.head(N_FINAL_FEATURES).index.tolist()
print(f"\n--- Final Selected Features (Top {N_FINAL_FEATURES} from RF) ---")
print(final_features)

# Filter the processed dataframes to keep only selected features
X_train_selected = X_train_processed_df[final_features]
X_test_selected = X_test_processed_df[final_features]

print("\nShape after feature selection (Train):", X_train_selected.shape)
print("Shape after feature selection (Test):", X_test_selected.shape)

# Store selected features list for modeling phase
selected_feature_names = final_features
```
> * **Code Description:** This script first separates features (X) and the target variable (y). It splits the data into training and testing sets. Then, it sets up preprocessing pipelines using `ColumnTransformer` to impute missing values (median for numerical, 'Missing' for categorical) and apply scaling (StandardScaler) and encoding (OneHotEncoder). The script then performs feature selection using Mutual Information, Recursive Feature Elimination (RFE) with Logistic Regression, and feature importance from a Random Forest model. Finally, it selects a subset of features based on the Random Forest importance rankings and filters the training and testing sets accordingly.
> * **Next Steps:** With the data preprocessed and a feature set selected, the next step is to train and evaluate various classification models. We will also address the class imbalance identified earli