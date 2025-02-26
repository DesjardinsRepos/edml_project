import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Set

import openfe
import polars as pl
import psutil
from sklearn.model_selection import train_test_split

from src.data_collection.prompting import prompt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from decimal import InvalidOperation

from babel.numbers import parse_decimal

from src.custom_data_types import DownloadedData, ProcessedData, ProfiledData

NUMERIC_TYPES: Set[str] = {"Int8", "Int16", "Int32", "Int64", "Float32", "Float64"}


def _get_physical_cores():
    try:
        return psutil.cpu_count(logical=False)
    except Exception:
        import multiprocessing

        return multiprocessing.cpu_count()


def _set_target_type(target: pl.Series) -> str:
    """Sets the target type based on the unique values in the target column
    Args:
        target (pl.Series): The target column
    Returns:
        str: The target type
    """
    cat_cols = set([pl.String, pl.Categorical, pl.Object, pl.Utf8, pl.Enum, pl.Boolean, pl.Date, pl.Datetime, pl.Time])
    if target.n_unique() == 2:
        return "binary"
    elif target.dtype in cat_cols and target.n_unique() < 20:
        return "multiclass"
    else:
        return "regression"


def _safe_cast_to_float(s: pl.Series) -> pl.Series:
    """Casts a series to float, handling various international number formats

    Args:
        s (pl.Series): series to be casted

    Returns:
        pl.Series: casted series
    """
    if s.dtype not in [pl.Utf8, pl.Object, pl.String]:
        try:
            return s.cast(pl.Float64)
        except:
            return s
    locales = ["en_US", "de_DE", "fr_FR", "es_ES", "en_GB"]
    try:
        return s.cast(pl.Float64)
    except:
        pass

    def parse_number(val_str):
        if val_str is None or val_str == "":
            return None

        for loc in locales:
            try:
                parsed = parse_decimal(val_str, locale=loc)
                return float(parsed)
            except (ValueError, InvalidOperation):
                continue
        try:
            # Remove currency symbols
            for symbol in ["$", "€", "£", "¥", "₹", "%"]:
                val_str = val_str.replace(symbol, "")

            # Handle decimal separators
            if "," in val_str and "." not in val_str:
                val_str = val_str.replace(",", ".")
            elif "," in val_str and "." in val_str:
                val_str = val_str.replace(",", "")

            return float(val_str)
        except ValueError:
            return None

    values = s.to_list()
    parsed_values = [parse_number(str(val).strip()) if val is not None else None for val in values]
    return pl.Series(parsed_values, dtype=pl.Float64)


def autoML_prep(downloader_data: DownloadedData, verbose_conversion: bool = False) -> ProcessedData:
    """Preprocesses the data for AutoML

    Args:
        downloader_data (DownloadedData): Dict containing the data to be preprocessed
        verbose_conversion (bool, optional): If True, prints conversion messages for numeric columns. Defaults to False.

    Raises:
        ValueError: If downloader_data is empty

    Returns:
        ProcessedData: Dict containing the preprocessed data
    """
    if not downloader_data:
        raise ValueError("Invalid input: downloader_data cannot be empty")
    processed_data = ProcessedData(
        train=downloader_data["train"],
        test=downloader_data.get("test", None),
        val=downloader_data.get("val", None),
        target=downloader_data["target"],
        columns_of_interest=downloader_data.get("columns_of_interest"),
        sensitive_features=downloader_data.get("sensitive_features", None),
        description=downloader_data.get("description", None),
        title=downloader_data.get("title"),
        seed=downloader_data.get("seed"),
        path=downloader_data.get("path"),
        eval_metric=None,
        problem_type=None,
        model_type=None,
        pos_label=None,
    )
    # Autogluon doesn't want you to define your own validation set
    if processed_data["val"] is not None:
        processed_data["train"] = pl.concat([processed_data["train"], processed_data["val"]], how="vertical")
        processed_data["val"] = None
    # Create test split if none exists
    if processed_data["test"] is None:
        try:
            processed_data["train"], processed_data["test"] = train_test_split(
                processed_data["train"],
                test_size=0.2,
                stratify=processed_data["train"].select(processed_data["target"]),
                random_state=processed_data["seed"],
            )
        except:
            processed_data["train"], processed_data["test"] = train_test_split(
                processed_data["train"],
                test_size=0.2,
                stratify=None,
                random_state=processed_data["seed"],
            )
    # Determine problem type
    target_combined = pl.concat(
        [processed_data["train"][[processed_data["target"]]], processed_data["test"][[processed_data["target"]]]],
        how="vertical",
    ).to_series()
    problem_type = _set_target_type(target_combined)
    # Convert target to float for regression problems
    if problem_type == "regression":
        processed_data["train"] = processed_data["train"].with_columns(
            _safe_cast_to_float(processed_data["train"][processed_data["target"]]).alias(processed_data["target"])
        )
        processed_data["test"] = processed_data["test"].with_columns(
            _safe_cast_to_float(processed_data["test"][processed_data["target"]]).alias(processed_data["target"])
        )
    # Try to convert potential numeric columns to float
    for split in ["train", "test"]:
        if processed_data[split] is None:
            continue

        # Get all string-like columns excluding the target (we'll handle that separately)
        string_cols = [
            col
            for col in processed_data[split].columns
            if processed_data[split][col].dtype in [pl.Utf8, pl.Object, pl.String] and col != processed_data["target"]
        ]

        # Try to convert each string column that might contain numeric values
        converted_columns = []
        for col in string_cols:
            try:
                # First check if the column might contain numeric values
                sample = processed_data[split][col].drop_nulls().head(10).to_list()

                # Skip if empty or all values seem non-numeric (contains letters)
                if not sample or all(any(c.isalpha() for c in str(val)) for val in sample):
                    continue

                # The column is potentially numeric if we reach here
                # Get *total* row count (including nulls) for comparison
                total_count = processed_data[split][col].len()

                if total_count == 0:
                    continue

                # Attempt conversion
                converted = _safe_cast_to_float(processed_data[split][col])
                converted_non_null = converted.drop_nulls()
                converted_count = converted_non_null.len()

                # Allow conversion if at least 90% of the total values (including original nulls)
                # were successfully converted to non-null values
                if converted_count >= 0.9 * total_count:
                    converted_columns.append(converted.alias(col))
                elif verbose_conversion:
                    # Only print failed conversions if verbose mode is on
                    print(
                        f"Skipping conversion for column '{col}' in {split} split - only "
                        f"{converted_count}/{total_count} ({converted_count / total_count:.1%}) "
                        f"values could be converted to numeric (minimum 90% required)."
                    )
            except Exception as e:
                # Always show unexpected exceptions for debugging
                print(f"Warning: Could not convert column '{col}' in {split} split to float: {e}")

        # Apply all conversions at once if any were successful
        if converted_columns:
            processed_data[split] = processed_data[split].with_columns(converted_columns)
            if verbose_conversion:
                print(f"Successfully converted {len(converted_columns)} columns to numeric in {split} split.")

    # Filter out rows with null or invalid target values
    for split in ["train", "test"]:
        target_col = processed_data["target"]
        processed_data[split] = processed_data[split].filter(
            processed_data[split][target_col].is_not_null()
            & (
                not processed_data[split][target_col].dtype.is_float()
                or (processed_data[split][target_col].is_finite() & processed_data[split][target_col].is_not_nan())
            )
        )
    # Determine evaluation metric and positive label
    if problem_type in ["binary", "multiclass"]:
        eval_metric = "f1_weighted"
        unique_values = processed_data["train"][processed_data["target"]].unique().to_list()
        prompt_text = f"""
        This is the list of possible values for the target column.
        {unique_values}
        
        Identify the positive class from the list above and provide the name of the column.
        Answer in the following format and do not provide reasoning:
        {{
            "positive_column": <column_name>
        }}
        """.replace("'", "")

        answer = json.loads(prompt(prompt_text).strip("```json").strip("```").strip())
        pos_label = answer["positive_column"]
    else:
        eval_metric = "rmse"
        pos_label = None

    return ProcessedData(
        train=processed_data["train"],
        test=processed_data["test"],
        val=processed_data["val"],
        target=processed_data["target"],
        columns_of_interest=processed_data.get("columns_of_interest", None),
        sensitive_features=processed_data.get("sensitive_features", None),
        description=processed_data.get("description", None),
        title=processed_data.get("title"),
        seed=processed_data.get("seed"),
        path=processed_data.get("path"),
        eval_metric=eval_metric,
        problem_type=problem_type,
        model_type="TabularPredictor",
        pos_label=pos_label,
    )


def feature_generation(
    processeddata: ProcessedData,
    profiledata: ProfiledData,
    n_features: int = 10,
    verbose: bool = False,
    feature_boosting: bool = False,
) -> ProcessedData:
    cat_cols = [
        col
        for col in profiledata["train_stats"].keys()
        if col != processeddata["target"] and profiledata["train_stats"][col].get("dformat", "not_cat") == "cat"
    ]
    train = processeddata["train"].select(pl.exclude([processeddata["target"]])).to_pandas()
    test = processeddata["test"].select(pl.exclude([processeddata["target"]])).to_pandas()
    object_cols = [col for col in train.columns if col != processeddata["target"] and train[col].dtype == "object"]
    cat_cols = list(set(cat_cols + object_cols))
    ofe = openfe.OpenFE()
    (Path(processeddata["path"]) / "FE").mkdir(parents=True, exist_ok=True)
    fit_params = dict(
        data=train,
        label=processeddata["train"].select([processeddata["target"]]).to_pandas(),
        feature_boosting=feature_boosting,
        tmp_save_path=processeddata["path"] / Path("FE/temp_openfe.feather"),
        n_jobs=_get_physical_cores(),
        verbose=verbose,
    )
    if cat_cols:
        fit_params["categorical_features"] = cat_cols
    with open(os.devnull, "w") as f, redirect_stdout(f):
        ofe.fit(**fit_params)
        train_x, test_x = openfe.transform(
            X_train=train,
            X_test=test,
            new_features_list=ofe.new_features_list[:n_features],
            n_jobs=_get_physical_cores(),
        )
    data = processeddata.copy()
    data["train"] = pl.from_pandas(train_x).with_columns(processeddata["train"].select([processeddata["target"]]))
    data["test"] = pl.from_pandas(test_x).with_columns(processeddata["test"].select([processeddata["target"]]))
    data["path"] = processeddata["path"] / Path("FE")
    return data
