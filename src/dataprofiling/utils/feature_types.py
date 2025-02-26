import polars as pl
from typing import Dict

def detect_feature_type(series: pl.Series, cat_thres: float = 0.02) -> str:
    """
    Determines the feature type of a given Polars Series.

    This function classifies the given series into one of the following types:
    - "cat" (categorical): If the ratio of unique values to total values is below `cat_thres`.
    - "num" (numerical): If the series contains numeric data types.
    - "txt" (text): If neither categorical nor numerical, the series is considered text.

    Args:
        series (pl.Series): The Polars Series to analyze.
        cat_thres (float, optional): The threshold for determining categorical features.
            If the proportion of unique values in the series is below this threshold,
            it is classified as categorical. Defaults to 0.02 (2%).

    Returns:
        str: The detected feature type, one of:
            - "cat" for categorical data.
            - "num" for numerical data.
            - "txt" for text data.
    """
    if is_categorical(series, cat_thres=cat_thres):
        return "cat"
    elif is_numeric(series):
        return "num"
    else:
        return "txt"

def detect_feature_types(df: pl.DataFrame, cat_thres: float = 0.02) -> Dict[str, str]:
    """
    Identifies the feature types of all columns in a Polars DataFrame.

    This function classifies each column in the DataFrame into one of three types:
    - "num" (numerical): Columns containing numerical values.
    - "cat" (categorical): Columns with a unique value ratio below `cat_thres`.
    - "txt" (text): Columns identified as text-based features.

    The classification is performed using the `detect_feature_type` function.

    Args:
        df (pl.DataFrame): A Polars DataFrame containing the dataset.
        cat_thres (float, optional): The threshold for determining categorical features.
            If the ratio of unique values to total values in a column is below this threshold, 
            it is classified as categorical. Defaults to 0.02 (2%).

    Returns:
        Dict: A dictionary where keys are column names and values are detected feature types 
            ("num" for numerical, "cat" for categorical, "txt" for text).
    """
    feature_types = {}
    for col in df.columns:
        feature_types[col] = detect_feature_type(df[col], cat_thres=cat_thres)
    return feature_types

def is_numeric(series: pl.Series) -> bool:
    """
    Checks if a Polars Series contains numerical values.

    Args:
        series (pl.Series): The Polars Series to analyze.

    Returns:
        bool: True if the series contains numerical values, otherwise False.
    """
    numeric_dtypes = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
        pl.Float32, pl.Float64, pl.Decimal
    }
    
    if series.dtype in numeric_dtypes:
        return True
    if isinstance(series.dtype, pl.Datetime) or series.dtype == pl.Date:
        return False
    return False

def is_categorical(series: pl.Series, cat_thres: float = 0.02) -> bool:
    """
    Determines if a Polars Series contains categorical data.

    A series is considered categorical if the proportion of unique values 
    relative to the total number of values is below a specified threshold (`cat_thres`).
    This method helps distinguish between categorical and high-cardinality numeric or text features.

    Args:
        series (pl.Series): The Polars Series to analyze.
        cat_thres (float, optional): The threshold for determining categorical data. If the ratio of 
            unique values to total values is below this threshold, the series is classified as categorical.
            Defaults to 0.02 (i.e., 2% unique values or less).

    Returns:
        bool: True if the series is classified as categorical, otherwise False.
    """
    unique_count = series.n_unique()
    total_count = len(series)
    
    if unique_count / total_count < cat_thres:
        return True

    return False

def is_text(series: pl.Series) -> bool:
    """
    Checks if a Polars Series contains text data.

    Args:
        series (pl.Series): The Polars Series to analyze.

    Returns:
        bool: True if the series contains text, otherwise False.
    """
    
    if not pl.String in [series.dtype, series.dtype.base_type()]:
        return False

    avg_length = series.str.len_chars().drop_nulls().mean()
    if isinstance(avg_length, (int, float)):
        avg_length = float(avg_length) if avg_length is not None else 0.0
    else:
        return False
    
    return avg_length > 0.0
