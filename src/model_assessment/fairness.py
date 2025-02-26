import inspect
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import polars as pl
from fairlearn.metrics import MetricFrame

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.custom_data_types import AutoMLData
from src.dataprofiling.static_profiling import profile_statistics
from src.model_assessment.binning import Binning
from src.model_assessment.utils.metrics import (
    BINARY_CLASSIFICATION_METRICS,
    GLOBAL_CLASSIFICATION_METRICS,
    MULTICLASS_CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"^sklearn\.")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"^sklearn\.")
warnings.filterwarnings("ignore", category=UserWarning, module=r"^sklearn\.")


@dataclass(frozen=True, slots=True, eq=False)
class FairnessMetrics:
    """Dataclass for storing fairness metrics for a feature"""

    name: str
    base_frame: MetricFrame
    global_frame: Optional[MetricFrame] = None

    @staticmethod
    def _build_error_frame(error: Exception) -> pl.DataFrame:
        """Builds a DataFrame containing an error message. So that the program doesn't crash when an error occurs.

        Args:
            error (Exception): Exception to include in the DataFrame

        Returns:
            DataFrame: DataFrame containing the error message
        """
        return pl.DataFrame({"metric": "error", "value": str(error)})

    def _convert_to_polars(self, data: MetricFrame, include_index: bool = True, rename: bool = True) -> pl.DataFrame:
        """Converts a MetricFrame to a Polars DataFrame

        Args:
            data (MetricFrame): MetricFrame to convert
            include_index (bool): Decides whether to include the index. Defaults to True.
            rename (bool): Decides whether to rename the columns. Defaults to True.

        Returns:
            DataFrame: Polars DataFrame containing the data
        """
        try:
            if rename:
                return pl.DataFrame(pl.from_pandas(data, include_index=include_index)).rename(
                    {"index": "metric", "0": "value"}
                )
            else:
                return pl.DataFrame(pl.from_pandas(data, include_index=include_index))
        except Exception as e:
            return self._build_error_frame(e)

    def get_metrics(self) -> Dict[str, pl.DataFrame]:
        """Returns all metrics for the feature

        Returns:
            Dict containing:
                - overall: Overall metrics for the feature
                - by_group: Metrics by group for the feature for applicable metrics
                - difference: Difference in metrics between groups for applicable metrics
                - ratio: Ratio of metrics between groups for applicable metrics
        """
        try:
            if self.global_frame:
                overall_metrics = (
                    pl.DataFrame({**self.base_frame.overall, **self.global_frame})
                    .transpose(include_header=True)
                    .rename({"column": "metric", "column_0": "value"})
                )
            else:
                overall_metrics = self._convert_to_polars(self.base_frame.overall)
            return {
                "overall": overall_metrics,
                "by_group": self._convert_to_polars(self.base_frame.by_group, rename=False),
                "difference": self._convert_to_polars(self.base_frame.difference()),
                "ratio": self._convert_to_polars(self.base_frame.ratio()),
            }
        except Exception as e:
            return {
                "overall": self._build_error_frame(e),
                "by_group": self._build_error_frame(e),
                "difference": self._build_error_frame(e),
                "ratio": self._build_error_frame(e),
            }


class FairnessAssessor:
    """A class to handle the analysis of model performance and fairness."""

    DEFAULT_COMBINATION_THRESHOLDS = {
        5: 5,
        7: 4,
        9: 3,
        12: 2,
        15: None,
    }

    def __init__(
        self,
        automl_data: AutoMLData,
        max_unique_values: int = 5,
        combination_thresholds: Optional[Dict[int, Optional[int]]] = None,
    ):
        self.train_stats = profile_statistics(automl_data["train"])
        self.test_stats = profile_statistics(automl_data["test"])
        self.automl_input_data = automl_data
        self.test_data = automl_data["test"]
        self.metrics: Dict[str, FairnessMetrics] = {}
        self.max_unique_values = max_unique_values
        self._binning: Optional[Binning] = None
        self.combination_thresholds = combination_thresholds or self.DEFAULT_COMBINATION_THRESHOLDS

    @property
    def binning(self) -> Binning:
        """Lazy initialization of binning"""
        if self._binning is None:
            self._binning = Binning(automl_data=self.automl_input_data, columns_to_bin=[])
        return self._binning

    def _maybe_bin_feature(self, feature_name: str) -> pl.Series:
        """Bins a feature if it has too many unique values.

        Args:
            feature_name (str): Name of the feature to potentially bin

        Returns:
            pl.Series: Original or binned feature
        """
        feature = self.test_data[feature_name]
        if feature.n_unique() > self.max_unique_values:
            self.binning.columns_to_bin = [feature_name]
            binned_data = self.binning.build_binning(data_selection="test")
            return binned_data[feature_name]
        return feature

    def _create_combined_feature(self, features: List[str]) -> pl.Series:
        """Creates a combined feature from multiple features

        Args:
            features (List[str]): List of features to combine

        Returns:
            pl.Series: Series containing the combined feature
        """
        feature_series = [self._maybe_bin_feature(feature).cast(pl.Utf8).fill_null("missing") for feature in features]
        return self.test_data.select(pl.concat_str(feature_series, separator="_")).to_series()

    def _pass_params(self, func, pos_label: Optional[str] = None):
        """Creates a wrapped metric function with correct parameters based on its signature.

        Args:
            func: The metric function to analyze and wrap
            pos_label (Optional[str], optional): Positive label for classification. Defaults to None.
        """

        def wrapped(y_true, y_pred, **kwargs):
            sig = inspect.signature(func)
            params = {}

            if "adjusted" in sig.parameters:
                params["adjusted"] = True
            if "average" in sig.parameters:
                params["average"] = "macro"

            # First try without pos_label if possible
            if "pos_label" not in sig.parameters:
                params.update(kwargs)
                return func(y_true, y_pred, **params)

            # For metrics requiring pos_label, try adaptive approach
            return self._adaptive_metric_execution(func, y_true, y_pred, pos_label, params, kwargs)

        return wrapped

    def _adaptive_metric_execution(
        self, func: Callable, y_true: Any, y_pred: Any, pos_label: Optional[str], base_params: Dict, extra_kwargs: Dict
    ) -> Any:
        """Attempts to execute a metric function using different approaches for pos_label.

        Args:
            func: The metric function to execute
            y_true: True labels
            y_pred: Predicted labels
            pos_label: Explicitly provided positive label
            base_params: Base parameters for the function
            extra_kwargs: Additional keyword arguments

        Returns:
            The metric result or np.nan if all approaches fail
        """
        import numpy as np

        params = base_params.copy()
        params.update(extra_kwargs)

        # List of strategies to try in order:
        # 1. Try without pos_label first
        # 2. Try with explicitly provided pos_label
        # 3. Try with pos_label from automl_data
        # 4. Try with first unique value
        # 5. Try with the less frequent class as pos_label

        strategies = []

        # First strategy: Try without pos_label
        strategies.append({})

        # Second strategy: Try with explicitly provided pos_label
        if pos_label is not None:
            strategies.append({"pos_label": pos_label})

        # Third strategy: Try with pos_label from automl_data
        if self.automl_input_data["pos_label"] is not None:
            strategies.append({"pos_label": self.automl_input_data["pos_label"]})

        # Fourth strategy: Try with first unique value
        unique_values = pl.Series(y_true).unique()
        if len(unique_values) >= 1:
            strategies.append({"pos_label": unique_values[0]})

        # Fifth strategy: Try with less frequent class
        try:
            from collections import Counter

            counts = Counter(y_true)
            if len(counts) >= 2:
                less_frequent = min(counts.items(), key=lambda x: x[1])[0]
                strategies.append({"pos_label": less_frequent})
        except Exception:
            pass

        # Try each strategy until one works
        for strategy_params in strategies:
            try:
                strategy_copy = params.copy()
                strategy_copy.update(strategy_params)
                return func(y_true, y_pred, **strategy_copy)
            except Exception as e:
                last_error = e
                continue

        # If all strategies fail, return NaN and warn
        warnings.warn(f"All pos_label strategies failed for {func.__name__}: {str(last_error)}", UserWarning)
        return np.nan

    def _is_valid_binary_labels(self, y_true: pl.Series, y_pred: pl.Series) -> bool:
        """Check if labels are valid for binary fairness metrics without pos_label.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            bool: True if labels are valid binary values {0,1} or {-1,1}
        """
        # Get unique values from both series
        unique_values = set(y_true.unique().to_list() + y_pred.unique().to_list())

        # Check if values match either {0,1} or {-1,1}
        valid_sets = [{0, 1}, {-1, 1}]
        return unique_values in valid_sets

    # Then modify the _build_metric_frames method to use this check:
    def _build_metric_frames(
        self, sensitive_feature: pl.Series, feature_name: str, pos_label: Optional[str] = None
    ) -> tuple[Optional[MetricFrame], Optional[MetricFrame]]:
        """Builds MetricFrames for a single feature

        Args:
            sensitive_feature (pl.Series): Series containing the sensitive feature
            feature_name (str): Name of the feature
            pos_label (Optional[str]): Positive label for multiclass classification. Defaults to None.

        Returns:
            tuple[Optional[MetricFrame], Optional[MetricFrame]]: Tuple containing the base and global MetricFrames
        """
        METRICS_MAP = {
            "binary": BINARY_CLASSIFICATION_METRICS,
            "multiclass": MULTICLASS_CLASSIFICATION_METRICS,
            "regression": REGRESSION_METRICS,
        }

        problem_type = self.automl_input_data["problem_type"]
        y_true = self.test_data[self.automl_input_data["target"]]
        y_pred = self.automl_input_data["predictions"]
        universal_params = {
            "y_true": y_true,
            "y_pred": y_pred,
            "sensitive_features": {feature_name: sensitive_feature},
            "random_state": self.automl_input_data["seed"],
        }
        if problem_type == "binary":
            global_frame = {}
            binary_metrics = {k: self._pass_params(v, pos_label) for k, v in METRICS_MAP[problem_type].items()}

            base_frame = MetricFrame(
                metrics=binary_metrics,
                **universal_params,
            )
            if self._is_valid_binary_labels(y_true, y_pred):
                for key, metric_fn in GLOBAL_CLASSIFICATION_METRICS.items():
                    try:
                        result = metric_fn(
                            universal_params["y_true"], universal_params["y_pred"], sensitive_features=sensitive_feature
                        )
                    except Exception as e:
                        result = str(e)
                    global_frame[key] = result
            else:
                for key, metric_fn in GLOBAL_CLASSIFICATION_METRICS.items():
                    global_frame[key] = float("nan")
                warnings.warn("Fairlearn fairness metrics require binary labels to be in {0,1} or {-1,1}", UserWarning)

        elif problem_type == "multiclass":
            global_frame = {}
            multiclass_metrics = {k: self._pass_params(v, pos_label) for k, v in METRICS_MAP[problem_type].items()}
            base_frame = MetricFrame(
                metrics=multiclass_metrics,
                **universal_params,
            )
            
            if self._is_valid_binary_labels(y_true, y_pred):
                for key, metric_fn in GLOBAL_CLASSIFICATION_METRICS.items():
                    result = metric_fn(
                        universal_params["y_true"],
                        universal_params["y_pred"],
                        sensitive_features=sensitive_feature,
                    )
                    global_frame[key] = result
            else:
                for key, metric_fn in GLOBAL_CLASSIFICATION_METRICS.items():
                    global_frame[key] = float("nan")
                warnings.warn("Fairlearn fairness metrics require binary labels to be in {0,1} or {-1,1}", UserWarning)
        elif problem_type == "regression":
            base_frame = MetricFrame(
                metrics=METRICS_MAP["regression"],
                **universal_params,
            )
            global_frame = None
        else:
            warnings.warn(f"Problem type '{problem_type}' not supported", UserWarning)
            return None, None

        return base_frame, global_frame

    def analyze_feature(
        self, feature: str | List[str], pos_label: Optional[str] = None
    ) -> FairnessMetrics | Dict[str, FairnessMetrics]:
        """Analyzes model performance and fairness for a single or multiple features

        Args:
            sensitive_feature (str | List[str]): Name or list of names of the features to analyze
            pos_label (Optional[str]): Positive label for multiclass classification. Defaults to None.

        Returns:
            FairnessMetrics | Dict[str,FairnessMetrics]: Object containing all metrics for the feature or a dictionary containing all metrics for all features
        """
        if pos_label is None:
            pos_label = self.automl_input_data["pos_label"]
        if isinstance(feature, list):
            results = {}
            for feature in feature:
                normalized_feature_data = self._maybe_bin_feature(feature)
                base_frame, global_frame = self._build_metric_frames(normalized_feature_data, feature, pos_label)
                self.metrics[feature] = FairnessMetrics(feature, base_frame, global_frame)
                results[feature] = self.metrics[feature]
            return results
        else:
            normalized_feature_data = self._maybe_bin_feature(feature)
            base_frame, global_frame = self._build_metric_frames(normalized_feature_data, feature, pos_label)
            self.metrics[feature] = FairnessMetrics(feature, base_frame, global_frame)
            return self.metrics[feature]

    def analyze_intersectional(self, sensitive_features: List[str], pos_label: Optional[str] = None) -> FairnessMetrics:
        """Analyzes the intersection of multiple sensitive features

        Args:
            sensitive_features (List[str]): List of sensitive features to analyze
            pos_label (Optional[str]): Positive label for multiclass classification. Defaults to None.

        Returns:
            FairnessMetrics: Object containing all metrics for the intersection of the features
        """
        if pos_label is None:
            pos_label = self.automl_input_data["pos_label"]
        feature_name = "+".join(sensitive_features)
        sensitive_series = self._create_combined_feature(sensitive_features)

        base_frame, global_frame = self._build_metric_frames(sensitive_series, feature_name, pos_label)
        self.metrics[feature_name] = FairnessMetrics(feature_name, base_frame, global_frame)
        return self.metrics[feature_name]

    def _calculate_combination_threshold(self, num_features: int) -> Optional[int]:
        """Calculates the maximum number of features to combine based on total feature count

        Args:
            num_features (int): Total number of features to analyze

        Returns:
            Optional[int]: Maximum number of features to combine, None if no combinations should be made
        """
        for threshold, max_combinations in sorted(self.combination_thresholds.items()):
            if num_features <= threshold:
                return max_combinations
        return None

    def _analyze_feature_and_intersectional(self, features: List[str], pos_label: Optional[str] = None):
        """Analyzes features and their intersections

        Args:
            features (List[str]): List of features to analyze
            pos_label (Optional[str]): Positive label for multiclass classification. Defaults to None.
        """
        if pos_label is None:
            pos_label = self.automl_input_data["pos_label"]
        self.analyze_feature(features, pos_label)
        if len(features) > 1:
            max_combinations = self._calculate_combination_threshold(len(features))
            if max_combinations is not None:
                for i in range(2, min(len(features) + 1, max_combinations + 1)):
                    for combination in combinations(features, i):
                        self.analyze_intersectional(list(combination), pos_label)

    def analyze_all(
        self,
        features: Optional[List[str]] = None,
        intersections: bool = True,
        feature_type: str = "sens",
        pos_label: Optional[str] = None,
    ):
        """Automatically analyzes features and if desired their intersections

        Args:
            features (List[str], optional): You can provide a list of features to analyze othwerwise selects all features based on the feature_type. Defaults to None.
            intersections (bool, optional): Decides whether to analyze intersections. Defaults to True.
            feature_type (str, optional): Decides which features to analyze. Defaults to "sens". Options are: "sens", "cat", "all", "intrest".
            \n"sens" analyzes all sensitive features,
            \n"cat" analyzes all categorical features,
            \n"all" analyzes all features,
            \n"intrest" analyzes all columns of interest.
        """
        if pos_label is None:
            pos_label = self.automl_input_data["pos_label"]

        # Check if we're dealing with multiclass - only use numerical features if so
        is_multiclass = self.automl_input_data["problem_type"] == "multiclass"
        numerical_dtypes = {
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.Int128,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        }

        # If specific features are provided, filter for numerical ones if multiclass
        if features:
            # For multiclass problems, filter to only numerical features
            if is_multiclass:
                numerical_features = [
                    col
                    for col in features
                    if col in self.test_data.schema and self.test_data.schema[col] in numerical_dtypes
                ]
                if len(numerical_features) < len(features):
                    excluded = set(features) - set(numerical_features)
                    warnings.warn(
                        f"Warning: {len(excluded)} non-numerical features excluded for multiclass problem: {list(excluded)}",
                        UserWarning,
                    )
                features = numerical_features

            if not features:  # If no suitable features remain
                warnings.warn("No suitable features for analysis with current problem type", UserWarning)
                return

            if intersections:
                self._analyze_feature_and_intersectional(features, pos_label)
            else:
                self.analyze_feature(features, pos_label)
            return

        match feature_type:
            case "sens":
                # For multiclass, only use numerical sensitive features
                if is_multiclass and self.automl_input_data["sensitive_features"]:
                    numerical_sens = [
                        col
                        for col in self.automl_input_data["sensitive_features"]
                        if col in self.test_data.schema and self.test_data.schema[col] in numerical_dtypes
                    ]
                    if len(numerical_sens) < len(self.automl_input_data["sensitive_features"]):
                        excluded = set(self.automl_input_data["sensitive_features"]) - set(numerical_sens)
                        warnings.warn(
                            f"Warning: {len(excluded)} non-numerical sensitive features excluded for multiclass: {list(excluded)}",
                            UserWarning,
                        )
                    self.automl_input_data["sensitive_features"] = numerical_sens

                self.analyze_all_sens(intersections, pos_label)

            case "cat":
                cat_features = [
                    col
                    for col in self.train_stats.keys()
                    if col != self.automl_input_data["target"]
                    and self.train_stats[col].get("dformat", "not_cat") == "cat"
                ]

                if is_multiclass:
                    numerical_cats = [
                        col
                        for col in cat_features
                        if col in self.test_data.schema and self.test_data.schema[col] in numerical_dtypes
                    ]
                    if len(numerical_cats) < len(cat_features):
                        excluded = set(cat_features) - set(numerical_cats)
                        warnings.warn(
                            f"Warning: {len(excluded)} non-numerical categorical features excluded for multiclass: {list(excluded)}",
                            UserWarning,
                        )
                    cat_features = numerical_cats

                if not cat_features:
                    warnings.warn("No suitable categorical features for the current problem type", UserWarning)
                    return

                if intersections:
                    self._analyze_feature_and_intersectional(cat_features, pos_label)
                else:
                    self.analyze_feature(cat_features, pos_label)

            case "all":
                all_features = (
                    self.automl_input_data["train"].select(pl.exclude(self.automl_input_data["target"])).columns
                )
                if is_multiclass:
                    numerical_features = [
                        col
                        for col in all_features
                        if col in self.test_data.schema and self.test_data.schema[col] in numerical_dtypes
                    ]
                    if len(numerical_features) < len(all_features):
                        excluded = set(all_features) - set(numerical_features)
                        warnings.warn(
                            f"Warning: {len(excluded)} non-numerical features excluded for multiclass: {list(excluded)}",
                            UserWarning,
                        )
                    all_features = numerical_features

                if not all_features:
                    warnings.warn("No suitable features for the current problem type", UserWarning)
                    return

                if intersections:
                    self._analyze_feature_and_intersectional(all_features, pos_label)
                else:
                    self.analyze_feature(all_features, pos_label)

            case "intrest":
                features_of_intrest: Optional[List[str]] = [
                    col
                    for col in self.automl_input_data["columns_of_interest"]
                    if col != self.automl_input_data["target"] and col in self.automl_input_data["train"].columns
                ]

                if is_multiclass and features_of_intrest:
                    numerical_features = [
                        col
                        for col in features_of_intrest
                        if col in self.test_data.schema and self.test_data.schema[col] in numerical_dtypes
                    ]
                    if len(numerical_features) < len(features_of_intrest):
                        excluded = set(features_of_intrest) - set(numerical_features)
                        warnings.warn(
                            f"Warning: {len(excluded)} non-numerical features of interest excluded for multiclass: {list(excluded)}",
                            UserWarning,
                        )
                    features_of_intrest = numerical_features

                if features_of_intrest:
                    if intersections:
                        self._analyze_feature_and_intersectional(features_of_intrest, pos_label)
                    else:
                        self.analyze_feature(features_of_intrest, pos_label)
                else:
                    warnings.warn("No suitable features of interest for the current problem type", UserWarning)

            case _:
                warnings.warn(
                    "No features provided for analysis, proceeding with first available suitable feature.", UserWarning
                )

                # Find a suitable feature based on problem type
                if is_multiclass:
                    # For multiclass, use first available numerical feature
                    numerical_cols = [
                        col
                        for col in self.test_data.columns
                        if col != self.automl_input_data["target"]
                        and col in self.test_data.schema
                        and self.test_data.schema[col] in numerical_dtypes
                    ]

                    if not numerical_cols:
                        warnings.warn("No numerical features available for multiclass problem", UserWarning)
                        return

                    chosen_feature = numerical_cols[0]

                else:
                    # For binary/regression, try to use categorical feature first
                    cat_features = [
                        col
                        for col in self.train_stats.keys()
                        if col != self.automl_input_data["target"]
                        and self.train_stats[col].get("dformat", "not_cat") == "cat"
                    ]

                    if cat_features:
                        chosen_feature = cat_features[0]
                    else:
                        # If no categorical features, use first available feature
                        chosen_feature = (
                            self.automl_input_data["train"]
                            .select(pl.exclude(self.automl_input_data["target"]))
                            .columns[0]
                        )

                self.analyze_feature(chosen_feature, pos_label)

    def analyze_all_categorical(self, intersections: bool = True, pos_label: Optional[str] = None):
        """Analyzes all categorical features and if desired their intersections

        Args:
            intersections (bool, optional): Decides whether to analyze intersections. Defaults to True.
            pos_label (Optional[str]): Positive label for multiclass classification. Defaults to None.
        """
        if pos_label is None:
            pos_label = self.automl_input_data["pos_label"]
        cat_features = [
            col
            for col in self.train_stats.keys()
            if col != self.automl_input_data["target"] and self.train_stats[col].get("dformat", "not_cat") == "cat"
        ]
        if not cat_features:
            warnings.warn("No categorical features provided", UserWarning)
            return
        if intersections:
            self._analyze_feature_and_intersectional(cat_features, pos_label)
        else:
            self.analyze_feature(cat_features, pos_label)

    def analyze_all_sens(self, intersections: bool = True, pos_label: Optional[str] = None):
        """Analyzes all sensitive features and if desired their intersections

        Args:
            intersections (bool, optional): Decides whether to analyze intersections. Defaults to True.
            pos_label (Optional[str]): Positive label for multiclass classification. Defaults to None.
        """
        if pos_label is None:
            pos_label = self.automl_input_data["pos_label"]
        if not self.automl_input_data["sensitive_features"]:
            warnings.warn("No sensitive features provided", UserWarning)
            return
        if intersections:
            self._analyze_feature_and_intersectional(self.automl_input_data["sensitive_features"], pos_label)
        else:
            self.analyze_feature(self.automl_input_data["sensitive_features"], pos_label)

    def get_all_metrics(self) -> Dict[str, Dict[str, pl.DataFrame]]:
        """Returns all metrics for all features

        Returns:
            dict[str, dict[str, pl.DataFrame]]: Dictionary containing all metrics for all features
            - overall: Overall metrics for the feature if applicable includes special fairness metrics
            - by_group: Metrics by group for the feature for applicable metrics
            - difference: Difference in metrics between groups for applicable metrics
            - ratio: Ratio of metrics between groups for applicable metrics
        """
        return {name: metrics.get_metrics() for name, metrics in self.metrics.items()}


if __name__ == "__main__":
    from src.auto_ml.model import AutoMLModel
    from src.data_collection.column_selection import get_dataset
    from src.data_preprocessing.preprocessing import autoML_prep, feature_generation
    from src.dataprofiling.data_summary import DataSummary
    from src.model_assessment.binning import Binning
    from src.model_assessment.fairness import FairnessAssessor

    d = "https://huggingface.co/datasets/naabiil/Obesity_Levels_Estimation"

    data = get_dataset(d)
    print(f"Target: {data['target']}")
    print(f"Sensitive: {data['sensitive_features']}")

    data_summary = DataSummary(data)
    data_summary.create_summary()
    profiled_data = data_summary.export()
    proccessed_data = autoML_prep(data)
    predictor = AutoMLModel(proccessed_data, time_limit=120, preset="medium", load=True, verbosity=0)
    predictor.run_auto_ml()
    automl_data = predictor.auto_ml_data
    cleaned_dict = data_summary.solve_issues(return_results=True)
    if cleaned_dict["train"] is not None:
        data["train"] = cleaned_dict["train"]
    if cleaned_dict["val"] is not None:
        data["val"] = cleaned_dict["val"]
    if cleaned_dict["test"] is not None:
        data["test"] = cleaned_dict["test"]
    cleaned_proccessed_data = autoML_prep(data)
    proccessed_data_engineered = feature_generation(
        processeddata=cleaned_proccessed_data, profiledata=profiled_data, n_features=5
    )
    predictor_engineered = AutoMLModel(
        proccessed_data_engineered, time_limit=120, preset="medium", load=True, verbosity=0
    )
    predictor_engineered.run_auto_ml()
    engineered_automl_data = predictor_engineered.auto_ml_data
    fas = FairnessAssessor(automl_data, profiled_data)
    fas.analyze_all_sens()
    test_ = fas.get_all_metrics()
    print(test_)
