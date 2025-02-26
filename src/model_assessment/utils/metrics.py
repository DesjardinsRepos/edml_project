from fairlearn.metrics import (
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_negative_rate,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

BINARY_CLASSIFICATION_METRICS = {
    "count": count,
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "sensitivity/recall": recall_score,
    "selection rate": selection_rate,
    "specificity/selectivity": true_negative_rate,
    "false positive rate": false_positive_rate,
    "false negative rate": false_negative_rate,
    "MCC": matthews_corrcoef,
}
MULTICLASS_CLASSIFICATION_METRICS = {
    "count": count,
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "sensitivity/recall": recall_score,
    "MCC": matthews_corrcoef,
}

GLOBAL_CLASSIFICATION_METRICS = {
    "demographic parity difference": demographic_parity_difference,
    "demographic parity ratio": demographic_parity_ratio,
    "equalized odds difference": equalized_odds_difference,
    "equalized odds ratio": equalized_odds_ratio,
    "equal opportunity difference": equal_opportunity_difference,
    "equal opportunity ratio": equal_opportunity_ratio,
}

REGRESSION_METRICS = {
    "count": count,
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "RMSE": root_mean_squared_error,
    "MdAE": median_absolute_error,
    "MAPE": mean_absolute_percentage_error,
    "R2": r2_score,
}
