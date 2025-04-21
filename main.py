import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Keep seaborn as it might be used implicitly or desired for EDA notebooks
from datetime import timedelta
import joblib # For saving/loading models and preprocessors
import pathlib # For path handling

# --- Sklearn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline # Renamed to avoid conflict
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc)

# --- Imblearn Imports ---
from imblearn.over_sampling import SMOTE

# --- Configuration ---
DATA_FILEPATH = "fintech_application_data.csv" # Make sure this file is accessible
MODEL_OUTPUT_DIR = "models" # Directory to save outputs
N_FEATURES_TO_SELECT = 20 # Number of features to select based on importance
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Create Output Directory ---
pathlib.Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- Function Definitions ---

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    print(f"Loading data from: {filepath}")
    try:
        # Explicitly state dtype for potentially mixed-type columns if known, otherwise rely on pandas inference
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        # Basic check
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        # Consider specific error handling, e.g., for mixed types:
        # print("Consider specifying dtype='unicode' or handling specific columns.")
        return None


def engineer_features(df):
    """Engineers features, including the target variable."""
    print("Engineering features...")
    df_fe = df.copy()

    # Convert date columns
    df_fe['initial_access'] = pd.to_datetime(df_fe['initial_access'], errors='coerce')
    df_fe['subscription_date'] = pd.to_datetime(df_fe['subscription_date'], errors='coerce')

    # Drop rows where date conversion failed for initial_access (shouldn't happen if data is clean)
    df_fe.dropna(subset=['initial_access'], inplace=True)

    # --- Target Variable ---
    df_fe['trial_end_date'] = df_fe['initial_access'] + timedelta(days=15)
    df_fe['conversion_window_end_date'] = df_fe['initial_access'] + timedelta(days=22)
    df_fe['converted_within_7_days'] = np.where(
        (df_fe['subscription_status'] == 1) &
        (df_fe['subscription_date'].notna()) &
        (df_fe['subscription_date'] > df_fe['trial_end_date']) &
        (df_fe['subscription_date'] <= df_fe['conversion_window_end_date']),
        1, 0
    )

    # --- Other Features ---
    df_fe['initial_access_dayofweek'] = df_fe['initial_access'].dt.dayofweek
    df_fe['initial_access_month'] = df_fe['initial_access'].dt.month
    # Ensure 'viewed_screens' is treated as string before splitting
    df_fe['num_screens_viewed'] = df_fe['viewed_screens'].astype(str).apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)
    df_fe['avg_session_duration'] = df_fe['session_duration'] # Assuming representative
    df_fe['age_x_screen_count'] = df_fe['user_age'] * df_fe['screen_count']
    # Ensure screen_count is not zero before division
    df_fe['premium_use_ratio'] = (df_fe['premium_features_used'] / df_fe['screen_count'].replace(0, np.nan)).fillna(0)


    # --- Drop Unnecessary Columns ---
    # Define columns to keep explicitly might be safer if input changes
    cols_to_drop = ['user_id', 'initial_access', 'access_time', 'subscription_date',
                    'trial_end_date', 'conversion_window_end_date',
                    'subscription_status', 'viewed_screens']
    # Only drop columns that actually exist in the dataframe
    cols_to_drop_existing = [col for col in cols_to_drop if col in df_fe.columns]
    df_fe = df_fe.drop(columns=cols_to_drop_existing)


    print("Feature engineering complete.")
    print(f"Shape after FE: {df_fe.shape}")
    print("Target variable distribution:")
    # Check if target variable exists before printing value counts
    if 'converted_within_7_days' in df_fe.columns:
        print(df_fe['converted_within_7_days'].value_counts(normalize=True))
    else:
        print("Target variable 'converted_within_7_days' not found after feature engineering.")


    return df_fe

def define_preprocessor(numerical_features, categorical_features):
    """Defines the preprocessing pipeline for numerical and categorical features."""
    print("Defining preprocessor...")
    # Check if feature lists are empty
    if not numerical_features and not categorical_features:
        print("Warning: Both numerical and categorical feature lists are empty.")
        # Return an empty transformer or handle as appropriate
        return ColumnTransformer(transformers=[], remainder='passthrough') # Example: return empty transformer

    transformers_list = []
    if numerical_features:
        numerical_transformer = SklearnPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers_list.append(('num', numerical_transformer, numerical_features))

    if categorical_features:
        categorical_transformer = SklearnPipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Consider sparse_output=True for large datasets
        ])
        transformers_list.append(('cat', categorical_transformer, categorical_features))


    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='passthrough' # Keep other columns if any - should ideally be none
    )
    print("Preprocessor defined.")
    return preprocessor

def select_features(X_train_processed, y_train, preprocessor, numerical_features, categorical_features, n_features):
    """Selects top features based on Random Forest importance."""
    print(f"Selecting top {n_features} features using Random Forest importance...")

    # Check if X_train_processed is empty
    if X_train_processed.shape[1] == 0:
        print("Warning: Processed training data has no features. Skipping feature selection.")
        return [], []

    # Get feature names after preprocessing
    all_feature_names = []
    try:
        # Get numerical feature names (they remain the same)
        if 'num' in preprocessor.named_transformers_ and numerical_features:
             all_feature_names.extend(numerical_features)

        # Get categorical feature names after OHE
        if 'cat' in preprocessor.named_transformers_:
            cat_pipeline = preprocessor.named_transformers_['cat']
            if 'onehot' in cat_pipeline.named_steps:
                 ohe_feature_names = cat_pipeline.named_steps['onehot'].get_feature_names_out(categorical_features)
                 all_feature_names.extend(list(ohe_feature_names))

        # Handle remainder columns if necessary (though ideally should be none)
        if preprocessor.remainder == 'passthrough' and preprocessor.transformers_:
             # This part requires knowing which columns were *not* transformed
             # It's complex to get remainder names reliably without tracking original columns
             pass # Simplification: Assume no important remainder columns for selection


        # If feature names couldn't be retrieved, use generic names
        if not all_feature_names or len(all_feature_names) != X_train_processed.shape[1]:
             print("Warning: Could not reliably retrieve all feature names. Using generic names.")
             all_feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]


    except Exception as e:
        print(f"Warning: Error retrieving feature names ({e}). Using generic names.")
        all_feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]


    # Train a temporary RF model to get feature importances
    # Use try-except for model training
    try:
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        rf_selector.fit(X_train_processed, y_train)
        importances = rf_selector.feature_importances_
    except Exception as e:
        print(f"Error training RandomForestClassifier for feature selection: {e}")
        return [], [] # Return empty lists if selection fails


    # Ensure lengths match before creating DataFrame
    if len(all_feature_names) != len(importances):
         print(f"Warning: Mismatch between number of feature names ({len(all_feature_names)}) and importances ({len(importances)}). Using available names.")
         # Adjust lengths if possible, or fallback
         min_len = min(len(all_feature_names), len(importances))
         all_feature_names = all_feature_names[:min_len]
         importances = importances[:min_len]
         if not all_feature_names: # If still empty, cannot proceed
              print("Error: Cannot perform feature selection due to name/importance mismatch.")
              return [], []


    importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Adjust n_features if it's larger than the available features
    n_features = min(n_features, len(all_feature_names))
    if n_features == 0:
        print("Warning: No features available to select.")
        return [], []


    # Select top N features
    selected_feature_names = importance_df['feature'].head(n_features).tolist()
    print(f"Selected features: {selected_feature_names}")

    # Find indices of selected features in the original processed array
    # This assumes the order of columns in X_train_processed matches all_feature_names
    try:
        # Recalculate indices based on potentially adjusted all_feature_names
        current_feature_map = {name: idx for idx, name in enumerate(all_feature_names)}
        selected_indices = [current_feature_map[name] for name in selected_feature_names]
    except KeyError as e:
        print(f"Error finding index for feature: {e}. Check feature name consistency.")
        return [], []


    return selected_feature_names, selected_indices

def train_final_model(X_train_selected, y_train):
    """Applies SMOTE and trains the final LightGBM model."""
    print("Applying SMOTE and training final model (LightGBM)...")

    # Check if training data is empty
    if X_train_selected.shape[0] == 0 or X_train_selected.shape[1] == 0:
        print("Error: Training data is empty after feature selection. Cannot train model.")
        return None

    # Apply SMOTE
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
        print("SMOTE applied.")
        print("Class distribution after SMOTE:")
        print(pd.Series(y_train_resampled).value_counts(normalize=True))
    except Exception as e:
        print(f"Error during SMOTE: {e}. Training on original selected data.")
        X_train_resampled, y_train_resampled = X_train_selected, y_train # Fallback


    # Train LightGBM
    try:
        lgbm = LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1) # Using balanced class weight as well
        lgbm.fit(X_train_resampled, y_train_resampled)
        print("LightGBM model trained successfully.")
        return lgbm
    except Exception as e:
        print(f"Error training LightGBM model: {e}")
        return None


def evaluate_model(model, X_test_selected, y_test, model_name="LightGBM"):
    """Evaluates the trained model on the test set."""
    print(f"\n--- Evaluating {model_name} Model ---")

    # Check if model exists or test data is valid
    if model is None:
        print("Error: Model object is None. Cannot evaluate.")
        return
    if X_test_selected.shape[0] == 0 or X_test_selected.shape[1] == 0:
        print("Error: Test data is empty after feature selection. Cannot evaluate.")
        return

    try:
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    except Exception as e:
        print(f"Error during prediction/probability calculation: {e}")
        return


    # Calculate Metrics safely
    try:
        accuracy = accuracy_score(y_test, y_pred)
        # Use zero_division=0 for precision, recall, f1
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return


    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


    # Plotting (optional, can fail in non-GUI environments)
    try:
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()


        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Warning: Plotting failed. Maybe running in a non-GUI environment? Error: {e}")


def save_pipeline(preprocessor, selected_feature_names, selected_indices, model, output_dir):
    """Saves the preprocessor, selected feature names/indices, and the trained model."""
    # Check if components exist before saving
    if preprocessor is None or model is None or selected_indices is None:
        print("Error: One or more pipeline components are missing. Cannot save.")
        return

    print(f"Saving pipeline components to {output_dir}...")
    try:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # Ensure dir exists

        joblib.dump(preprocessor, output_path / 'preprocessor.joblib')
        joblib.dump(selected_feature_names, output_path / 'selected_feature_names.joblib')
        joblib.dump(selected_indices, output_path / 'selected_indices.joblib') # Save indices for filtering
        joblib.dump(model, output_path / 'lgbm_model.joblib')
        print("Pipeline components saved.")
    except Exception as e:
        print(f"Error saving pipeline components: {e}")


def load_prediction_pipeline(input_dir):
    """Loads the saved pipeline components."""
    print(f"Loading pipeline components from {input_dir}...")
    try:
        input_path = pathlib.Path(input_dir)
        preprocessor = joblib.load(input_path / 'preprocessor.joblib')
        # selected_feature_names = joblib.load(input_path / 'selected_feature_names.joblib') # Names useful for understanding
        selected_indices = joblib.load(input_path / 'selected_indices.joblib') # Indices needed for filtering
        model = joblib.load(input_path / 'lgbm_model.joblib')
        print("Pipeline components loaded.")
        return preprocessor, selected_indices, model
    except FileNotFoundError:
        print(f"Error: One or more pipeline files not found in {input_dir}.")
        return None, None, None
    except Exception as e:
        print(f"Error loading pipeline components: {e}")
        return None, None, None


def predict_pipeline(raw_df, preprocessor, selected_indices, model):
    """Applies the full pipeline (preprocess, select features, predict) to new raw data."""
    print("Applying prediction pipeline to new data...")

    # Check if components are valid
    if preprocessor is None or selected_indices is None or model is None:
        print("Error: Pipeline components not loaded correctly. Cannot predict.")
        return None, None
    if raw_df is None or raw_df.empty:
        print("Error: Input data for prediction is empty or None.")
        return None, None

    # Ensure raw_df has the same columns expected by the preprocessor
    # This requires careful handling of feature engineering consistency.
    # Assuming raw_df is already feature-engineered appropriately.
    # Drop target if present (common case for prediction data)
    if 'converted_within_7_days' in raw_df.columns:
         raw_df_predict = raw_df.drop(columns=['converted_within_7_days'])
    else:
         raw_df_predict = raw_df.copy()


    try:
        # 2. Preprocess data using the loaded preprocessor
        processed_data = preprocessor.transform(raw_df_predict)

        # Check if selected_indices are valid for processed_data shape
        max_index = max(selected_indices) if selected_indices else -1
        if max_index >= processed_data.shape[1]:
             print(f"Error: Max selected index ({max_index}) is out of bounds for processed data columns ({processed_data.shape[1]}).")
             return None, None
        if not selected_indices: # Handle case where no features were selected
             print("Warning: No features were selected by the pipeline. Cannot make predictions.")
             return None, None


        # 3. Select features using the loaded indices
        selected_data = processed_data[:, selected_indices]

        # 4. Predict using the loaded model
        predictions = model.predict(selected_data)
        probabilities = model.predict_proba(selected_data)[:, 1]
        print("Prediction complete.")
        return predictions, probabilities
    except Exception as e:
        print(f"Error during prediction pipeline execution: {e}")
        # Potentially log the raw_df_predict.info() or processed_data shape here for debugging
        return None, None


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Fintech Conversion Prediction Pipeline ---")

    # 1. Load Data
    df_raw = load_data(DATA_FILEPATH)

    if df_raw is not None:
        # 2. Engineer Features
        df_featured = engineer_features(df_raw)

        # Check if feature engineering produced a result
        if df_featured is None or df_featured.empty:
             print("Exiting: Feature engineering did not produce data.")
             exit()
        if 'converted_within_7_days' not in df_featured.columns:
             print("Exiting: Target variable not found after feature engineering.")
             exit()


        # 3. Prepare Data for Modeling
        X = df_featured.drop('converted_within_7_days', axis=1)
        y = df_featured['converted_within_7_days']

        # Identify feature types *before* splitting
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='object').columns.tolist()
        print(f"Identified {len(numerical_features)} numerical features.")
        print(f"Identified {len(categorical_features)} categorical features.")

        # Check if features exist
        if X.empty:
            print("Exiting: No features available for modeling after dropping target.")
            exit()

        # 4. Split Data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            print(f"Data split into Train ({X_train.shape}) and Test ({X_test.shape}) sets.")
        except ValueError as e:
            print(f"Error during train/test split (potentially too few samples for stratification): {e}")
            # Decide how to proceed: exit, use non-stratified split, etc.
            print("Exiting due to split error.")
            exit()


        # 5. Define and Fit Preprocessor (on Training Data Only)
        preprocessor = define_preprocessor(numerical_features, categorical_features)
        try:
            preprocessor.fit(X_train) # Fit only on training data
        except Exception as e:
            print(f"Error fitting preprocessor: {e}")
            print("Exiting pipeline.")
            exit()


        # 6. Transform Data
        try:
            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            print("Data transformed using preprocessor.")
            print(f"Processed training data shape: {X_train_processed.shape}")
        except Exception as e:
            print(f"Error transforming data with preprocessor: {e}")
            print("Exiting pipeline.")
            exit()


        # 7. Select Features (Based on Training Data)
        selected_feature_names, selected_indices = select_features(
            X_train_processed, y_train, preprocessor, numerical_features, categorical_features, N_FEATURES_TO_SELECT
        )

        # Check if feature selection was successful
        if not selected_indices:
             print("Warning: Feature selection did not return any indices. Proceeding with all processed features.")
             # Option 1: Use all processed features
             # selected_indices = list(range(X_train_processed.shape[1]))
             # Option 2: Exit if features are crucial
             print("Exiting: Feature selection failed.")
             exit()


        # 8. Filter Data based on Selected Features
        try:
            X_train_selected = X_train_processed[:, selected_indices]
            X_test_selected = X_test_processed[:, selected_indices]
            print(f"Data filtered to {X_train_selected.shape[1]} selected features.")
        except IndexError as e:
             print(f"Error filtering data with selected indices: {e}")
             print("Selected Indices:", selected_indices)
             print("Processed Data Shape:", X_train_processed.shape)
             print("Exiting pipeline.")
             exit()


        # 9. Train Final Model (with SMOTE)
        final_model = train_final_model(X_train_selected, y_train)

        # Check if model training was successful
        if final_model is None:
            print("Exiting: Model training failed.")
            exit()


        # 10. Evaluate Model
        evaluate_model(final_model, X_test_selected, y_test)

        # 11. Save Pipeline Components
        save_pipeline(preprocessor, selected_feature_names, selected_indices, final_model, MODEL_OUTPUT_DIR)

        # --- Example Prediction on New Data (using Test set as proxy) ---
        print("\n--- Example: Predicting on New Data ---")
        # Load the pipeline components back (simulating a separate prediction script)
        loaded_preprocessor, loaded_indices, loaded_model = load_prediction_pipeline(MODEL_OUTPUT_DIR)

        # Check if loading was successful
        if loaded_preprocessor and loaded_indices is not None and loaded_model:
            # Take a sample of the original *unprocessed* test data (X_test)
            # Important: The predict_pipeline expects data *before* preprocessing
            if not X_test.empty:
                 sample_raw_data = X_test.head(5).copy()
                 print(f"Predicting on {len(sample_raw_data)} sample raw records...")

                 # Use the prediction pipeline function
                 predictions, probabilities = predict_pipeline(
                     sample_raw_data, loaded_preprocessor, loaded_indices, loaded_model
                 )

                 # Check if prediction was successful
                 if predictions is not None and probabilities is not None:
                     print("\nSample Predictions:")
                     print(pd.DataFrame({'Prediction': predictions, 'Probability_Convert': probabilities}))
                 else:
                     print("Prediction on sample data failed.")
            else:
                 print("Test set is empty, cannot run prediction example.")
        else:
            print("Could not load pipeline components, skipping prediction example.")


        print("\n--- Pipeline Execution Complete ---")
else:
        print("Exiting: Data loading failed.")

# Code Description:
#   * This script defines functions for each major stage: loading data, engineering features, defining a preprocessor, selecting features (using RF importance), training the final model (LightGBM with SMOTE), evaluating the model, and saving/loading pipeline components.
#   * The main execution block (`if __name__ == "__main__":`) orchestrates these steps.
#   * It loads data, engineers features, splits into train/test sets.
#   * It fits the preprocessor *only* on the training data and then transforms both train and test sets.
#   * Feature selection is performed based on the processed *training* data.
#   * The selected feature indices are used to filter both the processed train and test sets.
#   * SMOTE is applied to the selected training data before training the final LightGBM model.
#   * The model is evaluated on the selected test data.
#   * Key components (preprocessor, selected feature indices, model) are saved using `joblib`.
#   * An example demonstrates how to load these components and use a `predict_pipeline` function to make predictions on new, raw data (using a sample from the original test set for demonstration).
# To Run:
#   1.  Save the code as `main.py`.
#   2.  Make sure the `fintech_application_data.csv` file is in the same directory or provide the correct path in `DATA_FILEPATH`.
#   3.  Ensure you have the necessary libraries installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `lightgbm`, `imblearn`, `joblib`). You can usually install them via pip or conda (e.g., `pip install pandas numpy scikit-learn matplotlib seaborn lightgbm imbalanced-learn joblib`).
#   4.  Run the script from your terminal: `python main.py`
#   5.  A `models` directory will be created containing the saved preprocessor, feature indices, and the trained model