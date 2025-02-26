from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import polars as pl


class DownloadedData(TypedDict):
    train: pl.DataFrame
    test: Optional[pl.DataFrame]
    val: Optional[pl.DataFrame]
    target: str
    columns_of_interest: List[str]
    sensitive_features: Optional[List[str]]
    description: str
    title: str
    seed: int
    path: Path


class ProcessedData(TypedDict):
    train: pl.DataFrame
    test: pl.DataFrame
    val: Optional[pl.DataFrame]
    target: str
    columns_of_interest: List[str]
    sensitive_features: Optional[List[str]]
    description: str
    title: str
    seed: int
    path: Path
    eval_metric: str
    problem_type: str
    model_type: str
    pos_label: Optional[str]


class AutoMLData(TypedDict):
    train: pl.DataFrame
    tranformed_train: pl.DataFrame
    test: pl.DataFrame
    transformed_test: pl.DataFrame
    val: Optional[pl.DataFrame]
    target: str
    columns_of_interest: List[str]
    sensitive_features: Optional[List[str]]
    description: str
    title: str
    seed: int
    path: Path
    eval_metric: str
    problem_type: str
    pos_label: Optional[str]
    predictions: Optional[pl.Series]
    prediction_probs: Optional[pl.DataFrame]


class ModelAssessmentData(TypedDict):
    train: pl.DataFrame
    tranformed_train: pl.DataFrame
    test: pl.DataFrame
    transformed_test: pl.DataFrame
    val: Optional[pl.DataFrame]
    target: str
    columns_of_interest: List[str]
    sensitive_features: Optional[List[str]]
    description: str
    title: str
    seed: int
    path: Path
    eval_metric: str
    problem_type: str
    pos_label: Optional[str]
    predictions: Optional[pl.Series]
    prediction_probs: Optional[pl.DataFrame]
    binned_test: Optional[pl.DataFrame]
    binned_train: Optional[pl.DataFrame]


class ProfiledData(TypedDict):
    train_stats: Dict[str, Dict[str, Any]] | None
    val_stats: Optional[Dict[str, Dict[str, Any]]]
    test_stats: Optional[Dict[str, Dict[str, Any]]]
    train_solved_issues: pl.DataFrame | None
    val_solved_issues: Optional[pl.DataFrame]
    test_solved_issues: Optional[pl.DataFrame]
    train_outlier: pl.DataFrame | None
    val_outlier: Optional[pl.DataFrame]
    test_outlier: Optional[pl.DataFrame]
    train_misslabeling: pl.DataFrame | None
    val_misslabeling: Optional[pl.DataFrame]
    test_misslabeling: Optional[pl.DataFrame]
    data_leakage_split_based: bool
    near_duplicated_issues: pl.DataFrame | None
    boolean_near_duplicates_removal_masks: Dict[str, List[bool] | None] | None
    data_leakage_correlation_based: bool
    correlation_matrix: pl.DataFrame | None
    highly_correlated_features: List[tuple] | None
    semantic_types: Dict[str, str] | None
    data_balancing: Dict | None
