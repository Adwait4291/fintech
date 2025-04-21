import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary model classes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC # Support Vector Classifier

# Import metrics and other utilities
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, auc,
                             classification_report, confusion_matrix)

# Import SMOTE for handling imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Use imblearn pipeline for SMOTE


# Assuming X_train_selected, X_test_selected, y_train, y_test are from the previous step
# And selected_feature_names contains the list of features used

print("--- Model Training & Selection ---")
print("Training data shape:", X_train_selected.shape)
print("Test data shape:", X_test_selected.shape)
print("Class distribution in y_train:\n", y_train.value_counts(normalize=True))

# --- Address Class Imbalance using SMOTE ---
# Apply SMOTE only to the training data within the pipeline or cross-validation later
# For simplicity here, we demonstrate applying it before training individual models
# Note: Best practice is to include SMOTE in a pipeline with the model, especially during CV/tuning.

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

print("\n--- After SMOTE Resampling (Training Data) ---")
print("Resampled training data shape:", X_train_resampled.shape)
print("Class distribution in y_train_resampled:\n", pd.Series(y_train_resampled).value_counts(normalize=True))


# --- Initialize Models ---
# Using default hyperparameters for initial comparison
# Added class_weight='balanced' where applicable as an alternative/complement to SMOTE
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=np.sum(y_train==0)/np.sum(y_train==1)), # Handle imbalance internally
    "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced'),
    # "SVM": SVC(probability=True, random_state=42, class_weight='balanced') # SVM can be slow on larger datasets
}

# --- Train and Evaluate Models ---
results = {}
trained_models = {}

print("\n--- Training and Evaluating Models ---")

for name, model in models.items():
    print(f"Training {name}...")

    # Create a pipeline with SMOTE (Optional but good practice)
    # pipeline = ImbPipeline(steps=[('smote', SMOTE(random_state=42)),
    #                              ('classifier', model)])
    # pipeline.fit(X_train_selected, y_train)
    # y_pred = pipeline.predict(X_test_selected)
    # y_pred_proba = pipeline.predict_proba(X_test_selected)[:, 1]

    # --- Alternative: Train on explicitly resampled data ---
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    # --- End Alternative ---

    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc
    }
    trained_models[name] = model # Store the trained model

    print(f"\n--- Results for {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 30)

# --- Compare Models ---
results_df = pd.DataFrame(results).T.sort_values(by='F1-Score', ascending=False)
print("\n--- Model Comparison ---")
print(results_df.to_markdown())

# --- Cross-Validation (Example for the best model based on F1) ---
best_model_name = results_df.index[0]
best_model = models[best_model_name] # Get the un-trained version for CV pipeline

print(f"\n--- Cross-Validating Best Model ({best_model_name}) ---")
# Use imblearn pipeline for CV with SMOTE
cv_pipeline = ImbPipeline(steps=[('smote', SMOTE(random_state=42)),
                                 ('classifier', best_model)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Use the original training data (X_train_selected, y_train) for CV
cv_scores_f1 = cross_val_score(cv_pipeline, X_train_selected, y_train, cv=cv, scoring='f1')
cv_scores_roc_auc = cross_val_score(cv_pipeline, X_train_selected, y_train, cv=cv, scoring='roc_auc')

print(f"Cross-Validation F1 Scores: {cv_scores_f1}")
print(f"Mean F1 Score: {np.mean(cv_scores_f1):.4f} (+/- {np.std(cv_scores_f1):.4f})")
print(f"Cross-Validation ROC AUC Scores: {cv_scores_roc_auc}")
print(f"Mean ROC AUC Score: {np.mean(cv_scores_roc_auc):.4f} (+/- {np.std(cv_scores_roc_auc):.4f})")

# --- Hyperparameter Tuning (Placeholder) ---
# Example using GridSearchCV (would replace the simple fit above)
# from sklearn.model_selection import GridSearchCV
# param_grid = { 'n_estimators': [100, 200], 'max_depth': [5, 10, None] } # Example for RF
# grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
#                            param_grid, cv=cv, scoring='f1')
# grid_search.fit(X_train_resampled, y_train_resampled)
# print("Best parameters found:", grid_search.best_params_)
# best_model_tuned = grid_search.best_estimator_
# Re-evaluate best_model_tuned on test set...

print("\n--- Model Training and Selection Complete ---")
# Keep results and trained models for the next step
final_results_df = results_df
final_trained_models = trained_models
```
> * **Code Description:** This script addresses class imbalance in the training data using SMOTE. It then initializes several common classification models (Logistic Regression, Random Forest, XGBoost, LightGBM). Each model is trained on the resampled training data, and predictions are made on the original (unseen) test set. Key performance metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC) are calculated and printed for each model, along with a classification report and confusion matrix. The models are compared in a summary table. Finally, it demonstrates how to perform cross-validation using `StratifiedKFold` and an `imblearn` pipeline (incorporating SMOTE within each fold) for the best-performing model based on the initial F1-score. A placeholder comment indicates where hyperparameter tuning would typically occur.
> * **Next Steps:** The final step involves deeper evaluation of the selected model(s), including plotting ROC and Precision-Recall curves, analyzing feature importance, and interpreting the results to provide actionable insigh