import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import metrics and plotting functions
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Assuming final_results_df, final_trained_models, X_test_selected, y_test, selected_feature_names exist

print("--- Model Evaluation & Interpretation ---")

# Identify the best model based on F1-score (or ROC AUC) from the results
best_model_name = final_results_df.index[0] # Assumes sorted by F1 descending
best_model = final_trained_models[best_model_name]
print(f"Selected best model: {best_model_name}")

# --- Generate Predictions with the Best Model ---
y_pred_best = best_model.predict(X_test_selected)
y_pred_proba_best = best_model.predict_proba(X_test_selected)[:, 1] # Probability of class 1

# --- Plot ROC Curve ---
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) - {best_model_name}')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- Plot Precision-Recall Curve ---
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_best)
pr_auc = auc(recall, precision) # Note: order is recall, precision for auc

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
# Calculate F1 scores for different thresholds to find the optimal point maybe
# f1_scores = 2 * (precision * recall) / (precision + recall)
# best_threshold_idx = np.argmax(f1_scores[:-1]) # Exclude last value if necessary
# plt.scatter(recall[best_threshold_idx], precision[best_threshold_idx], marker='o', color='red', label='Best F1 Threshold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall Curve - {best_model_name}')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# --- Confusion Matrix Visualization ---
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.show()

# --- Feature Importance Interpretation ---
# Works directly for tree-based models (RF, XGB, LGBM)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n--- Feature Importances ({best_model_name}) ---")
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': selected_feature_names, # Use the names of the selected features
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance_df.head(15).to_markdown(index=False))

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.show()

# For Logistic Regression, look at coefficients
elif hasattr(best_model, 'coef_'):
    print(f"\n--- Feature Coefficients ({best_model_name}) ---")
    coefficients = best_model.coef_[0] # Get coefficients for the single class
    feature_coeffs_df = pd.DataFrame({
        'Feature': selected_feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', key=abs, ascending=False) # Sort by absolute value

    print(feature_coeffs_df.head(15).to_markdown(index=False))

    # Plotting coefficients requires care due to positive/negative values
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=feature_coeffs_df.head(15))
    plt.title(f'Top 15 Feature Coefficients (Absolute Value) - {best_model_name}')
    plt.tight_layout()
    plt.show()

else:
    print(f"Feature importance/coefficients not directly available for {best_model_name}.")

# --- Error Analysis (Conceptual) ---
# - Examine instances where the model made incorrect predictions (False Positives/Negatives).
# - Look for patterns: Are errors concentrated in specific user segments (e.g., age groups, locations)?
# - Compare feature values for misclassified vs correctly classified instances.
# Example: Create a dataframe of misclassified samples
# misclassified_indices = X_test_selected.index[y_test != y_pred_best]
# misclassified_samples = df.loc[misclassified_indices] # Get original data
# print("\nExample Misclassified Samples:")
# print(misclassified_samples.head())
# Further analysis would involve comparing distributions of features for these samples.

# --- Business Impact Metrics (Conceptual) ---
# - True Positives (TP): Correctly identified users who will convert. -> Target these users effectively.
# - False Positives (FP): Users predicted to convert but didn't. -> Wasted marketing/retention effort?
# - False Negatives (FN): Users predicted not to convert but did. -> Missed opportunity.
# - True Negatives (TN): Correctly identified users who won't convert. -> Avoid unnecessary effort.
# Calculate potential ROI based on cost of intervention vs. value of a converted user.
# Example: value_of_conversion = $50; cost_of_intervention = $5
# Benefit = (TP * value_of_conversion) - ((TP + FP) * cost_of_intervention)
# Cost of Missed Opp = FN * value_of_conversion

# --- Model Fairness Assessment (Conceptual) ---
# - Check if model performance (e.g., precision, recall) is consistent across different demographic groups
#   (e.g., location, device_type, potentially age groups if sensitive).
# - Requires analyzing model metrics on subsets of the test data defined by these sensitive attributes.

print("\n--- Model Evaluation and Interpretation Complete ---")

```
> * **Code Description:** This script focuses on the best-performing model identified in the previous step. It generates and plots the ROC curve and the Precision-Recall curve to visualize the trade-offs between different classification thresholds. It also displays the confusion matrix for a clearer view of true/false positives and negatives. Crucially, it extracts and plots feature importances (for tree-based models) or coefficients (for linear models) to understand which factors drive the model's predictions. Conceptual points for error analysis, business impact calculation, and fairness assessment are included as comments.
> * **Next Steps:** Based on this analysis, the final step involves summarizing the results, drawing conclusions, and providing recommendatio