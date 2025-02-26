import re
import statistics as statistics_modul
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from polars.exceptions import InvalidOperationError

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataprofiling.utils.feature_types import detect_feature_types, is_numeric


def profile_statistics(
    df: pl.DataFrame,
    feature_types: Optional[Dict[str, str]] = None,
    cat_thres: float = 0.02
) -> Dict[str, Dict[str, Any]]:
    """Generate statistical summaries for numerical, categorical, and text features in a given dataset.

    This function analyzes the given Polars DataFrame and computes various statistics based on the detected or provided feature types.

    Args:
        df (pl.DataFrame): Input dataset as a Polars DataFrame.
        feature_types (Optional[Dict[str, str]]): Dictionary mapping column names to feature types 
            (e.g., `"num"` for numerical, `"cat"` for categorical, `"txt"` for text). 
            If None, feature types are automatically detected.
        cat_thres (float, optional): 
            The threshold used to classify a numerical feature as categorical if feature_types must be
            calculated. A feature is considered categorical if `num_unique / num_total < cat_thres`.
            Default is `0.02` (i.e., features where unique values are less than 2% of the total count).
    Returns:
        
        statistics: A dictionary where each column name maps to a dictionary of computed statistics.
        
        - **Structure:**  
          `{ column_name: { statistic_name: value } }`  
              
        - **For Numerical Features (`statistic_name -> "dformat": "num"`)**:
            - `"mean"` (float): Mean value
            - `"std"` (float): Standard deviation
            - `"min"` (float): Minimum value
            - `"max"` (float): Maximum value
            - `"25%"`, `"50%"`, `"75%"` (float): Percentiles
            - `"n_unique"` (int): Number of unique values
            - `"null_count"` (int): Count of null values

        - **For Categorical Features (`statistic_name -> "dformat": "cat"`)**:
            - `"n_unique"` (int): Number of unique categories
            - `"value_counts"` (dict[str, int]): Dictionary of unique values and their frequencies
            - `"null_count"` (int): Count of null values

        - **For Text Features (`statistic_name -> "dformat": "txt"`)**:
            - `"n_unique"` (int): Number of unique text entries
            - `"null_count"` (int): Count of null values
            - `"mean_char_length"` (float): Average length of text in characters
            - `"std_char_length"` (float): Standard deviation of character lengths
            - `"min_char_length"` (int): Minimum character length
            - `"max_char_length"` (int): Maximum character length
            - `"mean_word_count"` (float): Average word count per entry
            - `"most_common_words"` (dict[str, int]): Dictionary of 10 most common words and their frequencies
    """
    
    if feature_types is None:
        feature_types = detect_feature_types(df, cat_thres=cat_thres)
    if not set(df.columns).issubset(feature_types.keys()):
        miss_ft_type_columns = set(df.columns) - set(feature_types.keys())
        missing_df = df.select(list(miss_ft_type_columns))
        feature_types |= detect_feature_types(missing_df, cat_thres=cat_thres)
    
    num_stats = get_numerical_stats(
        df = df,
        cols = [key for key, val in feature_types.items() if val == "num"]
    )
    cat_stats = get_categorical_stats(
        df = df,
        cols = [key for key, val in feature_types.items() if val == "cat"]
    )
    txt_stats = get_text_stats(
        df = df,
        cols = [key for key, val in feature_types.items() if val == "txt"]
    )
    return num_stats | cat_stats | txt_stats
     
def get_numerical_stats(
    df: pl.DataFrame,
    cols: List[str]
) -> Dict[str, Dict[str, int | float | str]]:
    """Compute numerical statistics for the specified columns in a Polars DataFrame.

    Args:
        df (pl.DataFrame): Input dataset.
        cols (List[str]): List of numerical columns to analyze.
        
    Returns:
        statistics (Dict): Dictionary where each column name maps to statistics.
        - "dformat" (str): Feature type ("num").
        - "mean" (float): Mean value.
        - "std" (float): Standard deviation.
        - "kurtosis" (float): Measure of tail heaviness.
        - "skewness" (float): Measure of asymmetry.
        - "min" / "max" (float): Minimum and maximum values.
        - "25%", "50%", "75%", "95%" (float): Percentile values.
        - "n_unique" (int): Number of unique values.
        - "null_count" (int): Count of missing values.
        
    """
    ldf = df.lazy()
    statistics = {}
    
    for col in cols:
        if not is_numeric(df[col]):
            warnings.warn(f"Column '{col}' isn't numeric so all statistical values will be set to 0.", UserWarning)
            stats: Dict[str, int | float | str] = {
                "mean": 0,
                "std": 0,
                "kurtosis": 0,
                "skewness": 0,
                "min": 0,
                "max": 0,
                "25%": 0,
                "50%": 0,
                "75%": 0,
                "95%": 0,
                "n_unique": df[col].n_unique(),
                "null_count": df[col].is_null().sum(),
                "dformat": "num"
            }
            statistics[col] = stats
            continue
         
        stats = (
            ldf.select(
                [
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                    pl.col(col).kurtosis().alias("kurtosis"),
                    pl.col(col).skew().alias("skewness"),
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max"),
                    pl.col(col).quantile(0.25).alias("25%"),
                    pl.col(col).quantile(0.5).alias("50%"),
                    pl.col(col).quantile(0.75).alias("75%"),
                    pl.col(col).n_unique().alias("n_unique"),
                    pl.col(col).is_null().sum().alias("null_count"),
                ]
            )
            .collect()
            .to_dicts()[0]
        )
        stats["dformat"] = "num"    
        statistics[col] = stats
    
    return statistics
    
def get_categorical_stats(
    df: pl.DataFrame,
    cols: List[str]
) -> Dict[str, Dict[str, int | float | str | Dict[str, int]]]:
    """Compute categorical statistics for the specified columns in a Polars DataFrame.

    Args:
        df (pl.DataFrame): Input dataset.
        cols (List[str]): List of categorical columns to analyze.

    Returns:
        statistics (Dict): Dictionary where each column name maps to its computed statistics.
        - "dformat" (str): Feature type ("cat").
        - "n_unique" (int): Number of unique categories.
        - "value_counts" (Dict[str, int]): Dictionary of unique values and their frequencies.
        - "null_count" (int): Count of missing values.
    """
    statistics = {}
    for col in cols:
        non_null_col = df[col].drop_nulls()
        value_counts_series = non_null_col.value_counts()
        value_counts_dict: Dict[str, int] = {
            str(k): int(v) for k, v in zip(
                value_counts_series[col].cast(pl.Utf8).to_list(),
                value_counts_series["count"].to_list()
            )
        }

        stats: Dict[str, int | float | str | Dict[str, int]] = {
            "dformat": "cat",
            "n_unique": non_null_col.n_unique(),
            "value_counts": value_counts_dict, 
            "null_count": df[col].is_null().sum()
        }
        statistics[col] = stats
        
    return statistics

def get_text_stats(
    df: pl.DataFrame,
    cols: List[str]
) -> Dict[str, Dict[str, int | float | str | Dict[str, int]]]:
    """Compute text statistics for the specified columns in a Polars DataFrame.

    Args:
        df (pl.DataFrame): Input dataset.
        cols (List[str]): List of text columns to analyze.

    Returns:
        statistics (Dict): Dictionary where each column name maps to its computed statistics.

        - "dformat" (str): Feature type ("txt").
        - "n_unique" (int): Number of unique text entries.
        - "null_count" (int): Count of missing values.
        - "mean_char_length" (float): Average character length of text entries.
        - "std_char_length" (float): Standard deviation of character lengths.
        - "min_char_length" / "max_char_length" (int): Minimum and maximum character lengths.
        - "mean_word_count" (float): Average word count per entry.
        - "most_common_words" (Dict[str, int]): Dictionary of the 10 most common words and their frequencies.
    """
    statistics = {}
    
    for col in cols:
        if df[col].dtype != pl.Utf8:
            try:
                df = df.with_columns(df[col].cast(pl.Utf8))
            except InvalidOperationError:
                warnings.warn(f"Column '{col}' could not be cast to string; check if column '{col}' is a text column.", UserWarning)
                continue

        non_null_col = df[col].drop_nulls()
        text_list = non_null_col.to_list()
        char_counts = [len(text) for text in text_list if text]
        
        word_counts = [len(_clean_text(text).split()) for text in text_list if text]
        all_texts = " ".join(text_list)
        all_texts_cleaned = _clean_text(all_texts)
        most_common_words = dict(Counter(all_texts_cleaned.split()).most_common(10))
        
        stats: Dict[str, int | float | str | Dict[str, int]] = {
            "dformat": "txt",
            "n_unique": non_null_col.n_unique(),
            "null_count": df[col].null_count(),
            "mean_char_length": sum(char_counts) / len(char_counts) if char_counts else 0,
            "std_char_length": statistics_modul.stdev(char_counts) if len(char_counts) > 1 else 0.0,
            "min_char_length": min(char_counts) if char_counts else 0,
            "max_char_length": max(char_counts) if char_counts else 0,
            "mean_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
            "most_common_words": most_common_words,
        }
    
        statistics[col] = stats
    
    return statistics
  
def _clean_text(
    text: str
) -> str:
    """Preprocess and clean a text string by removing numbers, punctuation, and extra spaces.
    Args:
        text (str): Input text string.
    Returns:
        cleaned_text (str): Cleaned text with numbers and special characters removed, spaces normalized, and converted to lowercase.
    """
    if not text:
        return ""
    text = re.sub(r"\d+", " ", text)  
    text = re.sub(r"[!?(),:;+#-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip() 
    text = text.lower()
    
    return text
