import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl
from scipy.stats import entropy

sys.path.append(str(Path(__file__).resolve().parents[2]))


def evaluate_data_balance(
    train: pl.DataFrame, 
    val: Optional[pl.DataFrame], 
    test: Optional[pl.DataFrame], 
    train_stats: dict, 
    val_stats: Optional[dict], 
    test_stats: Optional[dict], 
    feature_types: dict,
    skewness_threshold: float, 
    kurtosis_threshold: float, 
    imbalance_threshold_absolute: float, 
    imbalance_threshold_relative: float
) -> Dict:
    """Evaluate the data balance of the dataset by comparing the distributions of the splits and identifying imbalanced and highly skewed features.

    Args:
        train (pl.DataFrame): Training data
        val (Optional[pl.DataFrame]): Validation data
        test (Optional[pl.DataFrame]): Test data
        train_stats (dict): Statistics of the dataprofiling for the training data
        val_stats (Optional[dict]): Statistics of the dataprofiling for the validation data
        test_stats (Optional[dict]): Statistics of the dataprofiling for the test data
        feature_types (dict): Dictionary containing the types of the features
        skewness_threshold (float): Threshold for skewness
        kurtosis_threshold (float): Threshold for kurtosis
        imbalance_threshold_absolute (float): Absolute threshold for data imbalance
        imbalance_threshold_relative (float): Relative threshold for data imbalance

    Returns:
        Dict: Dictionary containing the evaluation results
    """
    results:Dict = {}

    splits = {"train": (train, train_stats), "val": (val, val_stats), "test": (test, test_stats)}

    for split_name, (split, stats) in splits.items():
        if split is None:
            continue

        for column in split.columns:
            col_type = feature_types.get(column, None)
            col_stats = stats.get(column, None) if stats else None

            if col_type == "num":
                skewness, kurtosis = col_stats["skewness"], col_stats["kurtosis"]
                highly_skewed = abs(skewness) > skewness_threshold
                extreme_kurtosis = abs(kurtosis) > kurtosis_threshold
                
                results.setdefault(column, {}).update({
                    f"{split_name}_highly_skewed": highly_skewed,
                    f"{split_name}_extreme_kurtosis": extreme_kurtosis,
                })

            elif col_type == "cat":
                n_unique, value_counts = col_stats["n_unique"], col_stats["value_counts"]

                total_values = sum(value_counts.values())
                expected_count = total_values / n_unique if n_unique > 0 else 0

                imbalanced_list = []
                for value, count in value_counts.items():
                    if abs(count - expected_count) > imbalance_threshold_absolute * total_values or abs(count - expected_count) / (expected_count + 1e-8) > imbalance_threshold_relative:
                        imbalanced_list.append(value)
                
                results.setdefault(column, {}).update({
                    f"{split_name}_imbalanced_list": imbalanced_list,
                })

    # Compare distributions of splits
    for split_name, (split, stats) in splits.items():
        if split is None or split_name == "train":
            continue

        for column in split.columns:
            col_type = feature_types.get(column, None)
            col_stats = stats.get(column, None) if stats else None

            if col_type == "num":
                ref_mean, ref_std = train_stats[column]["mean"], train_stats[column]["std"]
                mean_diff = abs(col_stats["mean"] - ref_mean) / (ref_std + 1e-8)

                results.setdefault(column, {}).update({
                    f"{split_name}_mean_diff_zscore": mean_diff
                })

            elif col_type == "cat":
                ref_value_counts = train_stats[column]["value_counts"]
                ref_total_values = sum(ref_value_counts.values())
                ref_prob_dist = np.array([count / ref_total_values for count in ref_value_counts.values()])

                value_counts = stats[column]["value_counts"]
                total_values = sum(value_counts.values())
                prob_dist = np.array([count / total_values for count in value_counts.values()])

                # Align distributions for comparison
                all_keys = set(ref_value_counts.keys()).union(value_counts.keys())
                ref_prob_dist = np.array([ref_value_counts.get(k, 0) / ref_total_values for k in all_keys])
                prob_dist = np.array([value_counts.get(k, 0) / total_values for k in all_keys])

                # Compute Jensen-Shannon divergence
                js_divergence = 0.5 * (entropy(prob_dist, 0.5 * (prob_dist + ref_prob_dist)) +
                                       entropy(ref_prob_dist, 0.5 * (prob_dist + ref_prob_dist)))

                results.setdefault(column, {}).update({
                    f"{split_name}_js_divergence": js_divergence
                })

    return results