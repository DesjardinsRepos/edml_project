import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import polars as pl

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_collection.prompting import prompt
from src.dataprofiling.static_profiling import (
    get_categorical_stats,
    get_numerical_stats,
    get_text_stats,
)
from src.dataprofiling.utils.feature_types import detect_feature_type
from src.dataprofiling.utils.prompts import SEMENTIC_COLUMNSNAME_PROMPT
from src.dataprofiling.utils.utility import _sample_columns

def detect_semantic_types(
    df: pl.DataFrame,
    semantic_acc: Literal["low", "mid", "high"] = "low",
    feature_types: Optional[Dict[str, str]] = None, 
    target_cols: Optional[List[str]] = None,
    meta_information: Optional[str] = None,
    dataset_name: Optional[str] = None,
    cat_thres: float = 0.02
) -> Dict[str, str]:
    """Detects semantic types of dataset columns using an LLM-based approach.

    This function analyzes column names, sample values, and statistical properties to determine their 
    semantic meaning. A language model is used to enhance the detection process.

    Args:
        df (pl.DataFrame): The dataset containing the columns to analyze.
        semantic_acc (Literal["low", "mid", "high"], optional): Accuracy level of the analysis.
            Higher accuracy uses more samples and a more powerful LLM. Default: "low".
        feature_types (Optional[Dict[str, str]], optional): Predefined feature types for specific columns.
            If None, they will be detected automatically. Default: None.
        target_cols (Optional[List[str]], optional): List of column names to analyze.
            If None, all columns are analyzed. Default: None.
        meta_information (Optional[str], optional): Additional metadata about the dataset to improve context for the model. Default: None.
        dataset_name (Optional[str], optional): Name of the dataset for contextual information. Default: None.
        cat_thres (float, optional): Threshold for classifying a column as categorical,
            based on the ratio of unique values to the total number of entries. Default: 0.02.

    Returns:
        Dict: A dictionary mapping column names to their detected semantic types.
    """

    config = _get_accuracy_config(accuracy=semantic_acc)
    
    if target_cols is None:
        target_cols = df.columns 


    df, feature_types, statistics = _get_target_information(
        df=df,
        feature_types=feature_types,
        target_cols=target_cols,
        cat_thres=cat_thres
    )
    
    sampled_data = _sample_columns(
        df = df,
        sample_size = config["sample_size"],
        statistics = statistics,
        feature_types = feature_types)

    prompt = _prepare_prompt(
        target_cols,
        sampled_data,
        feature_types,
        statistics,
        meta_information,
        dataset_name
    )
    
    return _run_llm(prompt, config["llm"])

def _get_target_information(
    df: pl.DataFrame,
    feature_types: Optional[Dict[str, str]],
    target_cols: List[str],
    cat_thres: float = 0.02
) -> Tuple[pl.DataFrame, dict, dict]:
    """Extracts feature types and statistical properties of selected dataset columns.

    This function determines the data type (numerical, categorical, text) of each selected column 
    and calculates relevant statistics based on its type.

    Args:
        df (pl.DataFrame): The dataset containing the columns to analyze.
        feature_types (Optional[Dict[str, str]]): A dictionary mapping column names to their feature types.
            If None, feature types are inferred automatically.
        target_cols (List[str]): List of columns to analyze.
        cat_thres (float, optional): Threshold for classifying a column as categorical,
            based on the ratio of unique values to the total number of entries. Default: 0.02.

    Returns:
        Tuple:
            - The filtered DataFrame containing only the target columns.
            - A dictionary mapping column names to their detected feature types.
            - A dictionary containing computed statistics for each column.
    """
    df = df.select(target_cols)
    
    target_feature_types: Dict[str, str] = {}
    target_statistics: Dict[str, Dict[str, Any]] = {}
    for col in target_cols:
        if feature_types is not None:
            target_feature_types[col] = feature_types.get(col, detect_feature_type(df[col], cat_thres=cat_thres))
        else:
            target_feature_types[col] = detect_feature_type(df[col], cat_thres=cat_thres)
        
        if target_feature_types[col] == "num":
            target_statistics[col] = get_numerical_stats(df, [col])[col]
        elif target_feature_types[col] == "cat":
            target_statistics[col] = get_categorical_stats(df, [col])[col]
        elif target_feature_types[col] == "txt":
            target_statistics[col] = get_text_stats(df, [col])[col]
        else:
            raise ValueError(f"No such type {target_feature_types[col]} found.")
        
    return df, target_feature_types, target_statistics

def _get_accuracy_config(accuracy: str) -> Dict:
    """Returns configuration settings based on the specified accuracy level.

    Different accuracy levels determine the sample size and the LLM model used for semantic detection.

    Args:
        accuracy (str): Accuracy level ("low", "mid", "high"). 
            - "low": Uses fewer samples and a lightweight LLM.
            - "mid": Uses more samples but a lightweight LLM.
            - "high": Uses more samples and a more capable LLM.

    Returns:
        dict: A dictionary containing:
            - "llm" (str): The name of the language model to use.
            - "sample_size" (int): The number of samples per column.
    """
    configs = {
        "low":  {"llm": "gpt-4o-mini", "sample_size": 10},
        "mid":  {"llm": "gpt-4o-mini", "sample_size": 20},
        "high": {"llm": "gpt-4o"     , "sample_size": 20}
    }
    return configs[accuracy]

def _prepare_prompt(
    target_cols: list, 
    sampled_data: dict, 
    feature_types: dict,
    statistics: dict,
    meta_information: Optional[str],
    dataset_name: Optional[str]
) -> str:
    """Generates a structured prompt for the LLM to infer semantic types of dataset columns.

    The prompt includes:
    - Column names and their detected feature types.
    - Sample values extracted from each column.
    - Basic statistical properties for numerical, categorical, and text-based columns.
    - Optional metadata and dataset name for additional context.

    Args:
        target_cols (list): List of column names to analyze.
        sampled_data (dict): Sampled data points from each column.
        feature_types (dict): A dictionary mapping columns to detected feature types.
        statistics (dict): Column statistics such as mean, variance, or unique values.
        meta_information (Optional[str]): Additional metadata of dataset for better LLM understanding. Defaults to None.
        dataset_name (Optional[str]): The dataset's name for contextual understanding. Defaults to None.

    Returns:
        str: A formatted prompt to be used with the LLM.
    """
    
    prompt = SEMENTIC_COLUMNSNAME_PROMPT + "\n\nActual Input: \n\n"
    
    if dataset_name is not None:
        prompt += f"Dataset name: {dataset_name}"
    if meta_information is not None:
        prompt += f"Meta: {meta_information}\n" 

    for col_name in target_cols:
        prompt += f"Column: {col_name}\n"
        prompt += f"Type:   {feature_types[col_name]}\n"
        prompt += f"Values: {sampled_data[col_name]}\n" 

        if feature_types[col_name] == "num":
            stats_str = ", ".join(f"{key.capitalize()}={value}" for key, value in statistics[col_name].items())
        elif feature_types[col_name] == "cat":
            stats_str = f"n_unique: {statistics[col_name]['n_unique']}"
        elif feature_types[col_name] == "txt":
            stats_str = f"text_type: mean_char_length: {statistics[col_name]['mean_char_length']}"
        else:
            raise ValueError(f"Featuretype {feature_types[col_name]} is not valide.")
        
        prompt += f"Statistics: {stats_str}\n"
        prompt += "\n"

    return prompt

def _parse_llm_result(
    response: str
) -> Dict:
    """Parses the response from the LLM into a structured dictionary.

    The function expects the response to be formatted as a JSON-like structure, where
    column names are mapped to their inferred semantic types.

    Args:
        response (str): The raw LLM response as a string.

    Returns:
        Dict: A dictionary mapping column names to their detected semantic types.
    """
    processed_response = response.strip('```json').strip('```').strip().split("\n")
    
    column_mapping = {}
    for line in processed_response:
        key, value = line.split(":", 1)
        column_mapping[key.strip()] = value.strip()
    
    return column_mapping

def _run_llm(
    detection_prompt: str, 
    llm: str
) -> dict:
    """Executes an LLM request to detect semantic types based on the given prompt.

    This function sends the structured prompt to the specified LLM via an API or direct call
    and processes the result.

    Args:
        detection_prompt (str): The formatted prompt describing dataset features.
        llm (str): The LLM model to use for processing (e.g., "gpt-4o", "gpt-4o-mini").

    Returns:
        dict: A dictionary mapping column names to their inferred semantic types.
    """
    
    response = prompt(detection_prompt, type="API", model=llm)
    return _parse_llm_result(response=response)
