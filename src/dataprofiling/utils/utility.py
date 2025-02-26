import polars as pl
import random
from typing import Any, Dict

def _sample_columns(
    df: pl.DataFrame,
    feature_types: Dict[str, str], 
    statistics: Dict[str, Dict],
    sample_size: int,
) -> Dict[str, Any]:
    """
    Samples data from a Polars DataFrame based on feature types.

    Args:
        df (pl.DataFrame): The input Polars DataFrame containing the data.
        feature_types (Dict[str, str]): A dictionary mapping column names to their detected feature type ("num", "cat", "txt").
        sample_size (int): The maximum number of samples to retrieve per column.
        statistics (Dict[str, Dict[str, Any]]): A dictionary containing computed statistics for each column.
            This should be generated using `profile_statistics` from `static_profiling.py`.

            Expected keys per column:
            - "min" (float): Minimum value for "num" columns.
            - "max" (float): Maximum value for "num" columns.
            - "25%", "50%", "75%", "95%" (float): Percentile values for "num" columns.
            - "n_unique" (int): Number of unique values for "num" and "cat" columns.
            - "value_counts" (Dict[Any, int]): Dictionary mapping unique values to their occurrence count for "cat" columns.


    Returns:
        Dict[str, Any]: A dictionary containing sampled data for each column.
        - For numerical columns ("num"): A sorted list of representative quantiles and additional random values.
        - For categorical columns ("cat"): A dictionary of the most frequent categories (if more than `sample_size` unique categories exist).
        - For text columns ("txt"): A random subset of the available non-null text values.
    """
    sampled_data: Dict[str, Any] = {}
    for col in df.columns:
        values = df[col].drop_nulls().to_list()
        
        if feature_types[col] == "num":
            quantiles = [
                statistics[col]["min"],
                statistics[col]["25%"],
                statistics[col]["50%"],
                statistics[col]["75%"],
                statistics[col]["max"]
            ]
            
            remaining_sample_size = max(sample_size - len(quantiles), 0)
            additional_samples = random.sample(values, remaining_sample_size)
            sampled_data[col] = sorted(list(quantiles + additional_samples))
        
        elif feature_types[col] == "cat":
            num_categories = statistics[col]["n_unique"]
            category_counts = statistics[col]["value_counts"]            
            if num_categories > sample_size:
                sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)  
                top_categories = {cat:val for cat, val in sorted_cats[:sample_size]}
                sampled_data[col] = top_categories
            else:
                sampled_data[col] = category_counts

        elif feature_types[col] == "txt":
            sampled_data[col] = random.sample(values, min(len(values), sample_size))

    return sampled_data
