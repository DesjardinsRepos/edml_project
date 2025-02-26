import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Tuple
from warnings import warn

import numpy as np
import polars as pl
import plotly.express as px
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.custom_data_types import DownloadedData, ProcessedData, ProfiledData
from src.dataprofiling.data_balance import evaluate_data_balance
from src.dataprofiling.quality_issue_detection import (
    detect_correlation_dataleakage,
    detect_quality_issues,
)
from src.dataprofiling.semantic_detection import detect_semantic_types
from src.dataprofiling.static_profiling import profile_statistics
from src.dataprofiling.utils.feature_types import detect_feature_type, detect_feature_types

PRETTY_TRANSLATE = str.maketrans("", "", "[]'")


class DataSummary:
    """
    A class for analyzing and profiling datasets. Most of the statistic and issue related attributes get set after
    calling the `create_summary()` method.

    This class provides various methods to identify data-related issues such as:
    - Missing values
    - Data imbalance
    - Feature correlations
    - Outliers
    - Label inconsistencies
    - Near-duplicate records

    Key Features:
    - Computes general statistics and quality issues of the given dataset in one function (`create_summary()`)
    - Detects data quality issues (`quality_issue_detection()`)
    - Identifies potential data leakage (`detect_corr_data_leakage()`)
    - Resolves detected issues (`solve_issues()`)
    - Provides visualization options (`plot_correlation_matrix()`, `plot_column_statistics()`)

    Attributes:
        train (pl.DataFrame | None): 
            Training dataset split. This split **must be provided**.

        val (pl.DataFrame | None): 
            Validation dataset split. This split is **optional**.

        test (pl.DataFrame | None): 
            Test dataset split. This split is **optional**.

        target (str | None): 
            The name of the target (label) column.

        feature_types (dict): 
            Dictionary mapping feature names to their detected or predefined types:
            - `"num"`: Numerical features.
            - `"cat"`: Categorical features.
            - `"txt"`: Text-based features.

        data_balance (dict | None): 
            Stores information about data distribution across splits, including imbalance metrics.

        correlation_matrix (pl.DataFrame | None): 
            The correlation matrix of numerical and categorical features (textfeatures are not supported now). 
            Used to detect highly correlated variables.

        highly_correlated_features (List[Tuple[str, str]] | None): 
            A list of feature/column pairs that are highly correlated, indicating potential data leakage.

        data_leakage_correlation_based (bool): 
            **True** if strong feature correlations suggest data leakage, otherwise **False**.

        near_duplicates_issues (pl.DataFrame | None): 
            DataFrame containing detected **near-duplicate** records.

        boolean_near_duplicates_removal_masks (Dict[str, List[bool] | None] | None): 
            Dictionary containing **boolean masks** for removing near-duplicates in `train`, `val`, and `test`.

        data_leakage_split_based (bool): 
            **True** if near-duplicates between dataset splits suggest data leakage, otherwise **False**.

        train_outliers (pl.DataFrame | None): 
            DataFrame containing detected **outliers** in the **training split**, including an `outlier_mask` column in 
            which `True` incidicates that the corresponding entry in the **train split**  is an outlier.

        val_outliers (pl.DataFrame | None): 
            DataFrame containing detected **outliers** in the **validation split**, including an `outlier_mask` column in 
            which `True` incidicates that the corresponding entry in the **val split**  is an outlier.

        test_outliers (pl.DataFrame | None): 
            DataFrame containing detected **outliers** in the **test split**, including an `outlier_mask` column in 
            which `True` incidicates that the corresponding entry in the **test split**  is an outlier.

        train_misslabelings (pl.DataFrame | None): 
            DataFrame containing detected **label issues** in the **training split**, including:
            - `label_issue_mask`: **True** if a label is likely incorrect.
            - `original_label`: The original label.
            - `predicted_label`: The corrected label suggested by **CleanLab**.

        val_misslabelings (pl.DataFrame | None): 
            DataFrame containing detected **label issues** in the **validation split** (same structure as `train_misslabelings`).

        test_misslabelings (pl.DataFrame | None): 
            DataFrame containing detected **label issues** in the **test split** (same structure as `train_misslabelings`).

        train_solved_issues (pl.DataFrame | None): 
            Training dataset with resolved data quality issues (outliers, mislabelings, near-duplicates removed).

        val_solved_issues (pl.DataFrame | None): 
            Validation dataset with resolved data quality issues (outliers, mislabelings, near-duplicates removed).

        test_solved_issues (pl.DataFrame | None): 
            Test dataset with resolved data quality issues (outliers, mislabelings, near-duplicates removed).

        semantic_types (Dict[str, str] | None): 
            Dictionary mapping feature names to **semantic types** detected using an **LLM-based model**.
        
        ...
    """

    def __init__(
        self,
        data: DownloadedData | ProcessedData,
        feature_types: Optional[Dict[str, str]] = None,
        pred_probs: Optional[np.ndarray] = None,
        cat_thres: float = 0.02,
        pred_prob_model: ClassifierMixin = HistGradientBoostingClassifier(),
        cv_folds: int = 5, 
        txt_emb_model: str = "all-MiniLM-L6-v2",
        max_txt_emb_dim_per_col: int = 50, 
        knn_metric: Literal["euclidean", "cosine"] = "euclidean", 
        knn_neighbors: int = 10,
        high_corr_thr: float = 0.95,
        semantic_acc: Literal["low", "mid", "high"] = "low",
        skewness_threshold: int = 1,
        kurtosis_threshold: int = 3,
        imbalance_threshold_absolute: float = 0.2,
        imbalance_threshold_relative: float = 1,
        mean_zdiff_threshold: float = 1.0,
        js_divergence_threshold: float = 0.2
    ):
        """
        Initializes the `DataSummary` object with the provided dataset and configuration parameters.

        Args:
            data (DownloadedData | ProcessedData): 
                A dictionary-like object containing the dataset splits (`train`, `val`, `test`) 
                and the target variable, dataset title, and description.
            
            feature_types (dict, optional): 
                A dictionary mapping feature names to their data types (`"num"`, `"cat"`, `"txt"`). 
                If `None`, feature types will be automatically detected.
            
            pred_probs (np.ndarray, optional): 
                An array of prediction probabilities used for label issue detection. 
                If provided, it should match the order of the dataset splits (`train`, `val`, `test`), 
                meaning that the np.ndarray should concat the pred_probs of train, val and test split 
                or at least of the splits that are provided by the DownloadedData object.
                If provided no pred_probs are calculated via pred_prob_model and cv_folds.
            
            cat_thres (float, optional): 
                The threshold used to classify a numerical feature as categorical.
                A feature is considered categorical if `num_unique / num_total < cat_thres`.
                Default is `0.02` (i.e., features where unique values are less than 2% of the total count).
            
            pred_prob_model (ClassifierMixin, optional): 
                A scikit-learn compatible classification model used to generate prediction probabilities 
                for label issue detection. If a regression target is detected, this model is ignored.
                Default: `HistGradientBoostingClassifier()`.
            
            cv_folds (int, optional): 
                Number of cross-validation folds used with the pred_prob_model for to create own pred_probs for
                the cleanlab issues detection.
                Default is `5`.
            
            txt_emb_model (str, optional): 
                Name of the SentenceTransformer model from sentence_transformers lib used for text column embeddings.
                This model is used for encoding text features before analysis.
                Default is `"all-MiniLM-L6-v2"` (a lightweight transformer model).
            
            max_txt_emb_dim_per_col (int, optional): 
                Maximum number of PCA dimensions retained per text column after text encoding.
                This is used to limit the dimensionality of transformed text features.
                Default is `50`.

            knn_metric Literal["euclidean", "cosine"]: 
                Distance metric used for K-Nearest Neighbors graph calculation in issue detection.
                - `"euclidean"`: Better if numerical features/ columns should have higher weight for 
                example for issue detection (e.g. outlier detection) than categorical
                - `"cosine"`: Better for categorical and text-heavy datasets where those columns should
                have higher weight for example for issue detection (e.g. outlier detection)
                Default is `"euclidean"`.
            
            knn_neighbors (int, optional): 
                Number of nearest neighbors considered when constructing the KNN graph 
                for issue detection (e.g., CleanLab label noise detection).
                Default is `10`.
            
            high_corr_thr (float): Threshold for detecting high correlation between features.  
                This value defines the correlation level at which potential data leakage is identified,  
                meaning that one feature explains another to an excessive degree. Correlation is considered  
                between numerical-numerical, numerical-categorical, and categorical-categorical feature pairs.
            
            semantic_acc (str, optional): 
                Accuracy level for semantic type detection.
                - `"low"`: Uses fewer samples (10) and a lightweight language model (gpt-4o-mini).
                - `"mid"`: Uses more samples (20) but a lightweight language model (gpt-4o-mini).
                - `"high"`: Uses more samples (20) and a more capable language model (gpt-4o).
                Default is `"low"`.
            
            skewness_threshold (int, optional): 
                Threshold for detecting skewness in numerical feature distributions.
                A higher value allows for more skewed distributions before being flagged.
                Default is `1` (values above 1 indicate right/left-skewed distributions).
            
            kurtosis_threshold (int, optional): 
                Threshold for detecting excessive kurtosis (peakedness) in distributions.
                A higher value indicates heavier tails than a normal distribution.
                Default is `3` (the standard kurtosis of a normal distribution).
            
            imbalance_threshold_absolute (float, optional): 
                Absolute imbalance threshold for categorical features.
                A feature is considered imbalanced if the proportion of the least frequent category 
                is below this threshold.
                Default is `0.2`.
            
            imbalance_threshold_relative (float, optional): 
                Relative imbalance threshold for categorical features.
                Imbalance is detected if the ratio between the most and least frequent categories exceeds this value.
                Default is `1.0`.
            
            mean_zdiff_threshold (float, optional): 
                Threshold for detecting distributional shifts using the Z-score difference.
                Features exceeding this threshold show a significant mean difference across dataset splits.
                Default is `1.0`.
            
            js_divergence_threshold (float, optional): 
                Threshold for detecting distribution shifts using Jensen-Shannon divergence.
                Features with JS divergence above this value are flagged.
                Default is `0.2`.

        Raises:
            ValueError: If the `target` variable is missing.
            ValueError: If the `train` dataset split is missing.
        """
        self.skewness_threshold = skewness_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.imbalance_threshold_absolute = imbalance_threshold_absolute
        self.imbalance_threshold_relative = imbalance_threshold_relative
        self.mean_zdiff_threshold = mean_zdiff_threshold
        self.js_divergence_threshold = js_divergence_threshold
        self.pred_probs = pred_probs
        self.pred_prob_model = pred_prob_model
        self.cv_folds = cv_folds
        self.txt_emb_model = txt_emb_model
        self.max_txt_emb_dim_per_col = max_txt_emb_dim_per_col
        self.knn_metric: Literal["euclidean", "cosine"] = knn_metric
        self.knn_neighbors = knn_neighbors
        self.cat_thres = cat_thres 
        
        self.high_corr_thr: float = high_corr_thr
        self.semantic_acc: Literal["low", "mid", "high"] = semantic_acc
        
        self.train: pl.DataFrame | None = data.get("train")
        self.val: pl.DataFrame | None  = data.get("val")
        self.test: pl.DataFrame | None = data.get("test")
        self.target: str | None= data.get("target")
        self.columns_of_interest: List[str] | None = data.get("columns_of_interest")
        self.dataset_describtion: str | None = data.get("description")
        self.dataset_title: str | None = data.get("title")
        self.random_seed: int | None = data.get("seed")
        self.save_path: Path | None = data.get("path")

        if self.target is None:
            raise ValueError("The target variable is missing (None). Please ensure 'target' is provided.")
        if self.train is None:
            raise ValueError("The train split must be avaible. Please ensure 'train' is provided.")
        self.check_data_integrity()
        self.dataset: pl.DataFrame = self._merge_dfs()
        self.column_names: List[str] = self.dataset.columns
        self.feature_types: Dict[str, str] = self._detect_feature_types(feature_types)

        self.train_stats: Dict[str, Dict[str, Any]] | None = None
        self.val_stats: Optional[Dict[str, Dict[str, Any]]] = None
        self.test_stats: Optional[Dict[str, Dict[str, Any]]] = None

        self.train_solved_issues: pl.DataFrame | None = None
        self.val_solved_issues: Optional[pl.DataFrame] = None
        self.test_solved_issues: Optional[pl.DataFrame] = None
        
        self.train_outliers: pl.DataFrame | None = None
        self.val_outliers: Optional[pl.DataFrame] = None
        self.test_outliers: Optional[pl.DataFrame] = None
        
        self.train_misslabelings: pl.DataFrame | None = None
        self.val_misslabelings: Optional[pl.DataFrame] = None
        self.test_misslabelings: Optional[pl.DataFrame] = None
        
        self.data_leakage_split_based: bool = False
        self.near_duplicates_issues: pl.DataFrame | None = None
        self.boolean_near_duplicates_removal_masks: Dict[str, List[bool] | None] | None = None
        
        self.data_leakage_correlation_based: bool = False
        self.correlation_matrix: pl.DataFrame | None = None
        self.highly_correlated_features: List[tuple] | None = None
        
        self.semantic_types: Dict[str, str] | None = None
        self.data_balance: Dict | None = None

    def quick_summary(self) -> dict:
        """A quick summary of the data issues detected.

        Returns:
            A dictionary of the following format:
            {
                "string_output": str, A human-readable string summarizing the issues
                "outliers": {
                    "train": int,
                    "val": int,
                    "test": int
                },
                "label_issues": {
                    "train": int,
                    "val": int,
                    "test": int
                },
                "near_duplicates": {
                    "tvc": int, train-validation near duplicates
                    "ttc": int, train-test near duplicates
                    "vtc": int, validation-test near duplicates
                    "genc": int near duplicates in the same split
                },
                "leakage_correlation": int,
                "data_balance": int
            }
        """
        
        # 1. Outliers

        outliers_text = ""

        has_outliers = any([
            self.train_outliers is not None and self.train_outliers["outlier_mask"].sum() > 0,
            self.val_outliers is not None and self.val_outliers["outlier_mask"].sum() > 0,
            self.test_outliers is not None and self.test_outliers["outlier_mask"].sum() > 0
        ])

        if has_outliers:
            troc = self.train_outliers["outlier_mask"].sum() if self.train_outliers is not None else 0
            voc = self.val_outliers["outlier_mask"].sum() if self.val_outliers is not None else 0
            toc = self.test_outliers["outlier_mask"].sum() if self.test_outliers is not None else 0
            tr_outliers = f"{troc} in train, " if self.train_outliers is not None else ""
            v_outliers = f"{voc} in val, " if self.val_outliers is not None else ""
            te_outliers = f"{toc} in test, " if self.test_outliers is not None else ""
            
            outliers_text = f"⚠️  Outliers detected: {tr_outliers}{v_outliers}{te_outliers}"[0:-2]

        # 2. Label issues

        label_issues_text = ""

        has_mislabelings = any([
            self.train_misslabelings is not None and self.train_misslabelings["label_issue_mask"].sum() > 0,
            self.val_misslabelings is not None and self.val_misslabelings["label_issue_mask"].sum() > 0,
            self.test_misslabelings is not None and self.test_misslabelings["label_issue_mask"].sum() > 0
        ])

        if has_mislabelings:
            trm = self.train_misslabelings["label_issue_mask"].sum() if self.train_misslabelings is not None else 0
            vm = self.val_misslabelings["label_issue_mask"].sum() if self.val_misslabelings is not None else 0
            tem = self.test_misslabelings["label_issue_mask"].sum() if self.test_misslabelings is not None else 0
            tr_misslabelings = f"{trm} in train, " if self.train_misslabelings is not None else ""
            v_misslabelings = f"{vm} in val, " if self.val_misslabelings is not None else ""
            te_misslabelings = f"{tem} in test, " if self.test_misslabelings is not None else ""
            
            label_issues_text = f"⚠️  Mislabelings detected: {tr_misslabelings}{v_misslabelings}{te_misslabelings}"[0:-2]

        # 3. Data leakage split based

        data_leakage_text = ""
        if self.data_leakage_split_based and self.near_duplicates_issues is not None:
            group_sets: DefaultDict[str, Set[int]] = defaultdict(set)
            cross_split_sets: DefaultDict[str, Set[int]] = defaultdict(set)
            
            for row in self.near_duplicates_issues.iter_rows(named=True):
                if row["dataset"] == row["near_duplicate_dataset"]:
                    group_sets[f"{row['dataset']}-{row['dataset']}"] |= {row["new_id"], row["near_duplicate_id"]}
                else:
                    cross_split_sets[f"{row['dataset']}-{row['near_duplicate_dataset']}"] |= {row["near_duplicate_id"]}

            duplicates_summary = {
                "train-train": len(group_sets.get("train-train", set())),
                "val-val": len(group_sets.get("val-val", set())),
                "test-test": len(group_sets.get("test-test", set())),
                "train-val": len(cross_split_sets.get("train-val", set())),
                "train-test": len(cross_split_sets.get("train-test", set())),
                "val-test": len(cross_split_sets.get("val-test", set())),
            }

            same_split_count = duplicates_summary["train-train"] + duplicates_summary["val-val"] + duplicates_summary["test-test"] 
            cross_split_count = duplicates_summary["train-val"] + duplicates_summary["val-test"] + duplicates_summary["train-test"] 
            
            split_dup = ""
            if duplicates_summary["train-val"] > 0:
                split_dup = "train and validation datasets"
            if duplicates_summary["train-test"] > 0:
                split_dup = "train and test datasets" if split_dup == "" else "all datasets"
            if duplicates_summary["val-test"] > 0:
                split_dup = "validation and test datasets" if split_dup == "" else "all datasets"
            
            if same_split_count > 0:
                data_leakage_text = f"\n    ⚠️  {same_split_count} Near duplicates in the same split detected"
            data_leakage_text += f"\n    ⚠️  Data leakage detected: {cross_split_count} Near duplicates across {split_dup}"


        # 4. Data leakage correlation based

        if self.data_leakage_correlation_based:
            if len(self.highly_correlated_features) > 3:
                aff_cols = str(self.highly_correlated_features[:3])[:-1] + ", ..."
            else:
                aff_cols = str(self.highly_correlated_features)
            aff_cols = aff_cols.translate(PRETTY_TRANSLATE)
            data_leakage_text += "\n    ⚠️  Data leakage detected: Feature correlation above threshold"
            data_leakage_text += f"\n       -> {len(self.highly_correlated_features)} Instances: {aff_cols}"

        # 5. Data balance

        data_balance_text = ""
        if self.data_balance is not None:
            actual_balance_issues = []
            for k, v in self.data_balance.items():
                if "train_highly_skewed" in v:
                    is_skewed = v.get("train_highly_skewed", False) or v.get("val_highly_skewed", False) or v.get("test_highly_skewed", False)
                    has_kurtosis = v.get("train_kurtosis", None) or v.get("val_kurtosis", None) or v.get("test_kurtosis", None)
                    high_zscore = v.get("val_mean_diff_zscore", 0.0) > self.mean_zdiff_threshold or v.get("test_mean_diff_zscore", 0.0) > self.mean_zdiff_threshold
                    if is_skewed or has_kurtosis or high_zscore:
                        actual_balance_issues.append(k)
                else:
                    imbalanced_categories = v.get("train_imbalanced_list", []) + v.get("val_imbalanced_list", []) + v.get("test_imbalanced_list", [])
                    if len(imbalanced_categories) > 0:
                        actual_balance_issues.append(k)
                    high_js_div = v.get("val_js_divergence", 0.0) > self.js_divergence_threshold or v.get("test_js_divergence", 0.0) > self.js_divergence_threshold
                    if high_js_div:
                        actual_balance_issues.append(k)

            if len(actual_balance_issues) > 5:
                unbalanced_cols = str(actual_balance_issues[:5])[:-1] + ", ..."
            else:
                unbalanced_cols = str(actual_balance_issues)
            
            if len(actual_balance_issues) > 0:
                unbalanced_cols = unbalanced_cols.translate(PRETTY_TRANSLATE)
                data_balance_text = f"⚠️  {len(actual_balance_issues)} Unbalanced columns detected: {unbalanced_cols}"

        string_output = f"""\n
                    ----- Short Data Issue Summary -----
To see more detailed data you can call other methods on the DataSummary object.
    {outliers_text}
    {label_issues_text}{data_leakage_text}
    {data_balance_text}
        """

        return {
            "string_output": string_output,
            "outliers": {
                "train": troc if has_outliers else 0,
                "val": voc if has_outliers else 0,
                "test": toc if has_outliers else 0
            },
            "label_issues": {
                "train": trm if has_mislabelings else 0,
                "val": vm if has_mislabelings else 0,
                "test": tem if has_mislabelings else 0
            },
            "near_duplicates": {
                "tvc": duplicates_summary["train-val"] if self.data_leakage_split_based else 0,
                "ttc": duplicates_summary["train-test"] if self.data_leakage_split_based else 0,
                "vtc": duplicates_summary["val-test"] if self.data_leakage_split_based else 0,
                "genc": same_split_count if self.data_leakage_split_based else 0            
            },
            "leakage_correlation": len(self.highly_correlated_features) if self.data_leakage_correlation_based else 0,
            "data_balance": len(actual_balance_issues) if self.data_balance is not None else 0
        }

    def create_summary(self):
        """Generates a detailed summary of the dataset and updates instance variables.

        This function performs multiple data profiling tasks and updates corresponding instance 
        variables with the computed results:

        - Computes general statistics (`train_stats`, `val_stats`, `test_stats`).
        - Detects correlation-based data leakage and sets (`data_leakage_correlation_based`, 
        `correlation_matrix`, `highly_correlated_features`).
        - Identifies and marks outliers (`train_outliers`, `val_outliers`, `test_outliers`).
        - Detects potential label issues (`train_misslabelings`, `val_misslabelings`, `test_misslabelings`).
        - Finds near-duplicate records within and across splits (`near_duplicates_issues`, 
        `boolean_near_duplicates_removal_masks`, `data_leakage_split_based`).
        - Determines the semantic types of features (`semantic_types`).
        - Evaluates data balance and skewness (`data_balance`).

        If a computation fails, the respective instance variable remains `None` or is set to its default value.
        """
        print(">>> Collecting statistical information of the data...")
        self.static_profiling()

        try:
            print(">>> Checking data leakage due to highly correlation of features...")
            self.detect_corr_data_leakage()
        except Exception as e:
            warn(f"⚠️  Data leakage detection failed: {e}")

        print(">>> Checking for problems such as labelling issues, outliers and data leakage using near duplicate detection...")
        self.quality_issue_detection()
        
        print(">>> Checking the semantic types of the columns...")
        self.semantic_type_detection()
        
        print(">>> Checking data balance...")
        self.data_balance_eval()
        
        self.created_summary = True

    def solve_issues(
        self, 
        consider_near_duplicates: bool = True, 
        return_results: bool = False
    ) -> Dict[str, pl.DataFrame | None] | None:
        """Solves identified data issues, including outliers, label issues, and near duplicates.

        Args:
            consider_near_duplicates (bool): Whether to remove near duplicates. Default to True
            return_results (bool, optional): Whether to return the cleaned datasplits. Defaults to False.

        Returns:
            Optional[Dict]: Dictionary containing cleaned train, val, and test datasets if return_results is True.
        """
        label_solved: Dict[str, pl.DataFrame | None] = {}
        for s, split, split_misslabeling in zip(
            ["train", "val", "test"], 
            [self.train, self.val, self.test], 
            [self.train_misslabelings, self.val_misslabelings, self.test_misslabelings]
        ):
            if split is not None and split_misslabeling is not None:
                solved_split = split.with_columns(
                    pl.when(split_misslabeling["label_issue_mask"])
                    .then(split_misslabeling["predicted_label"])
                    .otherwise(split_misslabeling["original_label"])
                    .alias(self.target)
                )
                label_solved[s] = solved_split
            else:
                label_solved[s] = None

        filtered_splits: Dict[str, pl.DataFrame | None] = {}
        combined_boolean = self.merge_boolean_masks(consider_near_duplicates=consider_near_duplicates)
        for s, split in zip(
            ["train", "val", "test"], 
            [self.train, self.val, self.test], 
        ):
            if label_solved[s] is not None:
                dataset_split = label_solved[s]
            elif split is not None:
                dataset_split = split
            else:
                filtered_splits[s] = None
                continue
            assert dataset_split is not None 
            
            if combined_boolean.get(s, None) is None:
                filtered_splits[s] = dataset_split
            else:
                boolean_list = np.array(combined_boolean[s], dtype=bool)
                filtered_splits[s] = dataset_split.filter(boolean_list)

        for s in ["train", "val", "test"]:
            if s not in filtered_splits:
                filtered_splits[s] = None

        for s, df in filtered_splits.items():
            setattr(self, f"{s}_solved_issues", df)

        return filtered_splits if return_results else None

    def export(self) -> ProfiledData:
        """Exports all computed profiling results into a structured `ProfiledData` object.
        Calling create_summary first is highly recommended since else most of the corresponding
        instance attributes are `None`

        The exported object contains:
        - Statistical summaries for train, val, and test datasets.
        - Detected data quality issues, including outliers, mislabelings, and near duplicates.
        - Data leakage indicators, both correlation-based and split-based.
        - Feature-related insights, such as semantic types and data balance metrics.

        If certain analyses were not performed or yielded no results, the corresponding attributes remain `None`.

        Returns:
            ProfiledData: A structured object containing all available profiling results.
        """
        profiled_data = ProfiledData(
            train_stats = self.train_stats,
            val_stats = self.val_stats,
            test_stats = self.test_stats,
            train_solved_issues = self.train_solved_issues,
            val_solved_issues = self.val_solved_issues,
            test_solved_issues = self.test_solved_issues,
            train_outlier = self.train_outliers,
            val_outlier = self.val_outliers,
            test_outlier = self.test_outliers,
            train_misslabeling = self.train_misslabelings,
            val_misslabeling = self.val_misslabelings,
            test_misslabeling = self.test_misslabelings,
            data_leakage_split_based = self.data_leakage_split_based,
            near_duplicated_issues = self.near_duplicates_issues,
            data_leakage_correlation_based = self.data_leakage_correlation_based,
            correlation_matrix = self.correlation_matrix,
            highly_correlated_features = self.highly_correlated_features,
            semantic_types = self.semantic_types,
            data_balancing = self.data_balance,
            boolean_near_duplicates_removal_masks = self.boolean_near_duplicates_removal_masks
        )
        return profiled_data

    def save(self) -> None:
        """Saves the profiling results to disk.

        Saves the `DataSummary` object and `ProfiledData` to pickle files inside the dataset's profiling directory
        which is set by the attribute `self.save_path`.
        """
        assert self.save_path is not None, "Parameter `self.save_path` must be set in order to save the results"
        save_path = self.save_path / "profiling"
        if not save_path.exists(): 
            save_path.mkdir()
        
        data_summary_save_path = save_path / "data_summary.pkl"
        if data_summary_save_path.exists():
            data_summary_save_path.unlink()
        with open(data_summary_save_path, "wb") as f:
            pickle.dump(self, f)

        profiled_data_save_path = save_path / "profiled_data.pkl"
        if profiled_data_save_path.exists():
            profiled_data_save_path.unlink()
        with open(profiled_data_save_path, "wb") as f:
            pickle.dump(self.export(), f)

    def quality_issue_detection(self, return_results: bool = False) -> Optional[List[pl.DataFrame]]:
        """Performs quality issue detection for outliers, mislabelings, and near duplicates.

        Args:
            return_results (bool, optional): Whether to return the detected issues. Defaults to False.

        Returns:
            Optional[List[pl.DataFrame]]: A list of DataFrames containing detected quality issues, if return_results is True.
        """
        train_df = self.train.with_columns(pl.lit("train").alias("dataset"))
        val_df = self.val.with_columns(pl.lit("val").alias("dataset")) if self.val is not None else None
        test_df = self.test.with_columns(pl.lit("test").alias("dataset")) if self.test is not None else None
        
        dfs = [df for df in [train_df, val_df, test_df] if df is not None]
        combined_df = pl.concat(dfs, how="diagonal")  
        combined_df = combined_df.with_row_index("ensemble_id_col")
        
        quality_issues = detect_quality_issues(
            df = combined_df, 
            feature_cols = list(set(combined_df.columns) - set([self.target, "dataset", "ensemble_id_col"])),
            target_col = self.target,
            feature_types = self.feature_types,
            pred_probs = self.pred_probs,
            pred_prob_model = self.pred_prob_model,
            cv_folds = self.cv_folds,
            txt_emb_model = self.txt_emb_model,
            max_txt_emb_dim_per_col = self.max_txt_emb_dim_per_col,
            knn_metric = self.knn_metric,
            knn_neighbors = self.knn_neighbors,
            cat_thres = self.cat_thres,
        )       
        
        results = []
        for split in ["train", "val", "test"]:
            if split in combined_df["dataset"].unique().to_list():
                subset = (
                    quality_issues
                    .filter(pl.col("dataset") == split)
                    .sort("ensemble_id_col")
                    .with_row_index(name="new_id")
                )

                mask_df = subset.select(["new_id", "is_outlier_issue", "outlier_score", "was_imputed"])
                mask_df = mask_df.rename({"is_outlier_issue": "outlier_mask"})
                setattr(self, f"{split}_outliers", mask_df)
                results.append(mask_df)
   
                mask_df = subset.select(["new_id", "is_label_issue", "label_score", self.target, "predicted_label", "was_imputed"])
                mask_df = mask_df.rename({"is_label_issue": "label_issue_mask"})
                mask_df = mask_df.rename({self.target: "original_label"})
                setattr(self, f"{split}_misslabelings", mask_df)
                results.append(mask_df)

            else:
                setattr(self, f"{split}_outliers", None)
                setattr(self, f"{split}_misslabelings", None)


        id_mappings = {}

        for split in ["train", "val", "test"]:
            if split in quality_issues["dataset"].unique().to_list():
                subset = (
                    quality_issues.filter(pl.col("dataset") == split)
                    .sort("ensemble_id_col")
                    .with_row_index(name="new_id")
                )
                id_mappings[split] = dict(zip(subset["ensemble_id_col"].to_list(), subset["new_id"].to_list()))

        seen_duplicates = set()
        new_near_duplicate_records = []

        for idx, duplicate_set, original_dataset in zip(
            quality_issues["ensemble_id_col"].to_list(),
            quality_issues["near_duplicate_sets"].to_list(),
            quality_issues["dataset"].to_list()
        ):
            for near_dup in duplicate_set:
                if (near_dup, idx) in seen_duplicates or (idx, near_dup) in seen_duplicates:
                    continue
                near_duplicate_dataset = quality_issues.filter(pl.col("ensemble_id_col") == near_dup)["dataset"].item()
                new_idx = id_mappings.get(original_dataset, {}).get(idx, None)
                new_near_dup = id_mappings.get(near_duplicate_dataset, {}).get(near_dup, None)

                if new_idx is not None and new_near_dup is not None:
                    new_near_duplicate_records.append((new_idx, original_dataset, new_near_dup, near_duplicate_dataset))
                    seen_duplicates.add((idx, near_dup))
                    if original_dataset != near_duplicate_dataset:
                        self.data_leakage_split_based = True

        if new_near_duplicate_records != []:
            self.near_duplicates_issues = pl.DataFrame(
                new_near_duplicate_records,
                schema=["new_id", "dataset", "near_duplicate_id", "near_duplicate_dataset"],
                orient="row"
            )
            train_df = train_df.with_row_index(name="new_id")
            if val_df is not None:
                val_df = val_df.with_row_index(name="new_id")
            if test_df is not None:
                test_df = test_df.with_row_index(name="new_id")
            
            self.boolean_near_duplicates_removal_masks = self.remove_cyclic_duplicates(
                df = self.near_duplicates_issues,
                train_df = train_df,
                val_df = val_df,
                test_df = test_df
            ) 

        else:
            self.data_leakage_split_based = False
            self.near_duplicates_issues = None
            self.boolean_near_duplicates_removal_masks = None

        if return_results:
            results = [df for df in [
                self.train_outliers, 
                self.val_outliers, 
                self.test_outliers, 
                self.train_misslabelings, 
                self.val_misslabelings, 
                self.test_misslabelings, 
                self.near_duplicates_issues
            ] if df is not None]
            return results
        else:
            return None

    def detect_corr_data_leakage(self, return_results: bool = False) -> Optional[Tuple[pl.DataFrame, List[tuple], bool]]:
        """Calculate correlation between all numeric and categorical features and detects 
        data leakage based on these feature correlation.

        Args:
            return_results (bool, optional): Whether to return correlation matrix and results. Defaults to False.

        Returns:
            Optional[Tuple]: Correlation matrix, highly correlated features, 
            and boolean flag for data leakage detection due to high correlation between the features
            if return_results is True.
        """
        
        corr_matrix, highly_correlated_features = detect_correlation_dataleakage(
            df = self.dataset,
            feature_types = self.feature_types,
            high_corr_thr = self.high_corr_thr
        )
        
        self.correlation_matrix = corr_matrix
        self.highly_correlated_features = highly_correlated_features
        
        if highly_correlated_features != []:
            self.data_leakage_correlation_based = True
        
        if return_results:
            return corr_matrix, highly_correlated_features, self.data_leakage_correlation_based
        else:
            return None

    def semantic_type_detection(self, return_results: bool = False) -> Dict[str, str] | None:
        """Detects semantic types of dataset features (columns).

        Args:
            return_results (bool, optional): Whether to return detected semantic types. Defaults to False.

        Returns:
        
            Dict | None: Dictionary mapping feature names to detected semantic types if return_results is True.
        """
        self.semantic_types = detect_semantic_types(
            df = self.dataset,
            semantic_acc = self.semantic_acc,
            feature_types = self.feature_types,
            meta_information = self.dataset_describtion,
            dataset_name = self.dataset_title,
            cat_thres=self.cat_thres
        )
        if return_results:
            return self.semantic_types
        else:
            return None

    def data_balance_eval(self, return_results: bool = False) -> Dict[str, float] | None:
        """Evaluates data balance for columns in train, val, and test dataset splits.

        Args:
            return_results (bool, optional): Whether to return data balance evaluation results. Defaults to False.

        Returns:
        
            Dict | None: Dictionary containing data balance metrics if return_results is True.
        """
        
        assert self.train is not None
        assert self.train_stats is not None
        self.data_balance = evaluate_data_balance(
            train = self.train,
            val = self.val,
            test = self.test,
            train_stats = self.train_stats,
            val_stats = self.val_stats,
            test_stats = self.test_stats,
            feature_types = self.feature_types,
            skewness_threshold=self.skewness_threshold,
            kurtosis_threshold=self.kurtosis_threshold,
            imbalance_threshold_absolute=self.imbalance_threshold_absolute,
            imbalance_threshold_relative=self.imbalance_threshold_relative
        )
        if return_results:
            return self.data_balance
        else:
            return None

    def static_profiling(self, return_results: bool = False) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, Dict[str, Any] | None] | None:
        """Performs static profiling to collect basic statistics of the columns of each datasplit.

        Args:
            return_results (bool, optional): Whether to return profiling results. Defaults to False.

        Returns:
            
            Tuple | None: 
            Train, val, and test statistics if return_results is True.
        """
        
        if self.train is not None:
            self.train_stats = profile_statistics(self.train, self.feature_types, cat_thres=self.cat_thres)
        if self.val is not None:
            self.val_stats = profile_statistics(self.val, self.feature_types, cat_thres=self.cat_thres)
        if self.test is not None:
            self.test_stats = profile_statistics(self.test, self.feature_types, cat_thres=self.cat_thres)
    
        if return_results:
            return self.train_stats, self.val_stats, self.train_stats
        else:
            return None

    def remove_cyclic_duplicates(
        self, 
        df: pl.DataFrame, 
        train_df: pl.DataFrame, 
        val_df: Optional[pl.DataFrame] = None, 
        test_df: Optional[pl.DataFrame] = None
    ) -> Dict[str, List[bool] | None]:
        """Removes cyclic near-duplicates from the dataset based on the `id` and `dup_id` column.

        Args:
            df (pl.DataFrame): DataFrame (`self.near_duplicated_issues`) containing detected near duplicates.
            train_df (pl.DataFrame): Training dataset.
            val_df (Optional[pl.DataFrame], optional): Validation dataset. Defaults to None.
            test_df (Optional[pl.DataFrame], optional): Test dataset. Defaults to None.

        Returns:
            Dict: Boolean masks indicating rows to retain for each dataset split.
        """

        seen = set()
        to_remove = set()
        keep = set()
        
        for row in df.iter_rows(named=True):
            a, dataset_a, b, dataset_b = row["new_id"], row["dataset"], row["near_duplicate_id"], row["near_duplicate_dataset"]
            
            if (b, dataset_b, a, dataset_a) in seen:
                continue
            
            seen.add((a, dataset_a, b, dataset_b))
            seen.add((b, dataset_b, a, dataset_a))
            
            if (a, dataset_a) in keep:
                to_remove.add((b, dataset_b))
            elif (b, dataset_b) in keep:
                to_remove.add((a, dataset_a))
            else:
                keep.add((a, dataset_a))
                to_remove.add((b, dataset_b))
        
        masks: Dict[str, List[bool] | None] = {}
        for split, split_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
            if split_df is not None:
                masks[split] = split_df["new_id"].map_elements(
                    lambda x: (x, split) not in to_remove,
                    return_dtype=pl.Boolean
                ).to_list()
            else:
                masks[split] = None
        
        return masks

    def merge_boolean_masks(self, consider_near_duplicates: bool):
        """Merges boolean masks for outliers and near duplicates to one boolean mask 
        which can be used for cleaning the datasplits.

        Args:
            consider_near_duplicates (bool): Whether to consider near-duplicate removal.

        Returns:
            Dict: Combined boolean masks for train, val, and test datasets.
        """
        combined_masks = {}
        for split in ["train", "val", "test"]:
            outliers_df = getattr(self, f"{split}_outliers")
            if consider_near_duplicates and self.boolean_near_duplicates_removal_masks is not None:
                duplicate_mask = self.boolean_near_duplicates_removal_masks.get(split)
            else:
                duplicate_mask = None
            
            if outliers_df is not None and duplicate_mask is not None:
                merged_mask = (~outliers_df["outlier_mask"]) & pl.Series(duplicate_mask)
                combined_masks[split] = merged_mask.to_list()
            elif outliers_df is None and duplicate_mask is not None:
                combined_masks[split] = pl.Series(duplicate_mask).to_list()
            elif outliers_df is not None and duplicate_mask is None:
                combined_masks[split] = (~outliers_df["outlier_mask"]).to_list()
            else:
                combined_masks[split] = None
        
        return combined_masks

    def check_data_integrity(self):
        """Checks dataset integrity and consistency across splits.
        Ensures that at least one dataset split is provided and all splits have the same column structure.
        """
        dfs = {"train_df": self.train, "val_df": self.val, "test_df": self.test}
        none_dfs = [name for name, df in dfs.items() if df is None]
        assert len(none_dfs) < len(dfs), f"Alle DataFrames sind None! ({', '.join(none_dfs)})"
        
        valid_dfs = {name: df for name, df in dfs.items() if df is not None}
        column_sets = {name: set(df.columns) for name, df in valid_dfs.items()}
        unique_column_sets = {frozenset(cols) for cols in column_sets.values()}
        assert len(unique_column_sets) == 1, f"Die DataFrames haben unterschiedliche Spalten: {column_sets}"

    def _merge_dfs(self) -> pl.DataFrame:
        """Merges available dataset splits into a single DataFrame.

        Returns:
            pl.DataFrame: Merged dataset containing train, val, and test splits.
        """
        dfs = dfs = [df for df in [self.train, self.val, self.test] if df is not None]
        if not dfs:
            raise ValueError("There are no valide DataFrames given for concatenation")
        return pl.concat(dfs, how="vertical")

    def _detect_feature_types(
        self, 
        feature_types: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Detects feature types (`"num"`, `"cat"`, `"txt"`) for each column in the provided dataset.
        
        Args:
            feature_types (Optional[Dict[str, str]], optional): Predefined feature types. Defaults to None.

        Returns:
            Dict: Dictionary mapping feature (column) names to detected feature types.
        """
        
        if feature_types is None: 
           return detect_feature_types(self.dataset, cat_thres=self.cat_thres)
        else:
            temp_feature_types = {}
            for col in self.column_names:
                temp_feature_types[col] = feature_types.get(col) or detect_feature_type(self.dataset[col], cat_thres=self.cat_thres)

            return temp_feature_types

    def plot_correlation_matrix(self):
        """Plots the correlation matrix as a heatmap."""
        assert self.correlation_matrix is not None, "Cannot plot none existing correlation matrix."
        labels = self.correlation_matrix.columns
        labels = [label[:8] + "..." if len(label) > 8 else label for label in labels]
        
        fig = px.imshow(
            self.correlation_matrix,
            x=labels,
            y=labels,
            text_auto=".2f",
        )
        fig.update_traces(textfont_size=8)
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Columns",
            yaxis_title="Columns",
            width=800,
            height=800,
        )
        fig.show()

    def plot_column_statistics(self, columns: List[str] | str) -> None:
        """Displays column statistics and distributions.
        Args:
            columns (List[str] | str): List of column names or a single column name that should be displayed.
        """
        if isinstance(columns, str):
            columns = [columns]

        data_splits = {}
        if self.train is not None:
            data_splits['train'] = self.train
        if self.val is not None:
            data_splits['val'] = self.val
        if self.test is not None:
            data_splits['test'] = self.test

        stats_splits = {}
        if self.train_stats is not None:
            stats_splits['train'] = self.train_stats
        if self.val_stats is not None:
            stats_splits['val'] = self.val_stats
        if self.test_stats is not None:
            stats_splits['test'] = self.test_stats
        
        print("\n------- Columns statistics displayed ---------")

        for col in columns:
            for split_name, stats in stats_splits.items():
                if col in stats:
                    col_stats = stats[col]
                    dformat = col_stats['dformat']
                    print(f"\nColumn: {col} ({col_stats['dformat']}) - {split_name}")
                    for key, value in col_stats.items():
                        if key != 'dformat':
                            if isinstance(value, (float)):
                                print(f"- {key}: {value:.2f}")
                            else:
                                print(f"- {key}: {value}")

                    if dformat == 'num':
                        self.plot_numerical_distribution(data_splits[split_name], col, split_name)
                    elif dformat == 'cat':
                        self.plot_categorical_distribution(col_stats, col, split_name)
                else:
                    print(f"\nColumn: {col} - Not in dataFrame")
    
    def plot_numerical_distribution(
        self,
        df: pl.DataFrame,
        column: str,
        split_name: str
    ) -> None:
        df_pandas = df.to_pandas()
        fig = px.histogram(df_pandas, x=column, title=f'Numerical Feature Distribution: {column} ({split_name})', marginal="box")
        fig.show()

    def plot_categorical_distribution(
        self,
        stats: Dict[str, Dict[str, Any]],
        column: str,
        split_name: str
        ) -> None: 
        cat_data = pl.DataFrame({"Category": list(stats['value_counts'].keys()), "Count": list(stats['value_counts'].values())})
        fig = px.bar(cat_data.to_pandas(), x='Category', y='Count', title=f'Categorical Feature Distribution: {column} ({split_name})')
        fig.show()

    def detailed_correlation_output(
        self,
        columns: List[str] | str = "target",
        num_values: int = 3
    ) -> None:
        """Displays detailed correlation analysis for specified columns.

        Args:
            columns (List[str] | str, optional): A list of columns or the string "target" to display the correlation of the target variable.
            num_values (int, optional): Number of top correlations to display. Defaults to 3.
        """

        if self.correlation_matrix is None:
            warn("The correlation matrix is empty or not available.", UserWarning)
            return

        num_values = min(num_values, len(self.correlation_matrix.columns))
        
        def get_top_correlations(col: str, n: int = num_values):
            """Returns the top `n` highest correlations for a given column, excluding self-correlation."""
            assert self.correlation_matrix is not None, "Cannot plot none existing correlation matrix."
            cor = self.correlation_matrix[col].to_numpy()
            top = list(zip(self.correlation_matrix.columns, cor))
            top = [entry for entry in top if entry[0] != col]
            top = sorted(top, key=lambda x: abs(x[1]), reverse=True)
            return top[:n]

        if self.data_leakage_correlation_based:
            assert self.highly_correlated_features is not None, "Ensure that self.highly_correlated_features was calculated and "
            if len(self.highly_correlated_features) > 3:
                aff_cols = str(self.highly_correlated_features[:3])[:-1] + ", ..."
            else:
                aff_cols = str(self.highly_correlated_features)
            ermerg_text = "\n------ High correlation warning -------"
            ermerg_text += "\n ⚠️  Data leakage detected: Feature correlation above threshold"
            ermerg_text += f"\n -> {len(self.highly_correlated_features)} Instances: {aff_cols}\n"
            print(ermerg_text)
            
        if columns == "target":
            string = f"\n------ Correlations for target column '{self.target}' -------\n"
            top_corrs = get_top_correlations(self.target, num_values)
            for col_name, corr_value in top_corrs:
                string += f"{col_name} (target): {corr_value:.2f}"
                if abs(corr_value) > 0.8:
                    string += " -> Very high correlation, consider removing this feature.\n"
                else:
                    string += "\n"
            print(string)

        elif isinstance(columns, (list, str)):
            if isinstance(columns, str):
                columns = [columns]
            if set(columns).issubset(set(self.correlation_matrix.columns)):
                string = "\n------ Correlation for chosen columns -------"
                for col in columns:
                    print(f"\nTop {num_values} correlations for column '{col}':")
                    top_corrs = get_top_correlations(col, num_values)
                    for idx, (col_name, corr_value) in enumerate(top_corrs):
                        string = f"\n{idx}. {col_name}: {corr_value:.2f}"
                        if abs(corr_value) > 0.8:
                            string += " -> Very high correlation, consider removing this feature."
                print(string)
            else:
                warn("The parameter 'columns' must be a valid column name or list of valid columns.", UserWarning)
        else:
            warn("The parameter 'columns' must be a list of column names or a single column name as a string.", UserWarning)

    def detailed_outlier_output(self) -> Dict[str, Optional[pl.DataFrame]]:
        """Analyzes and outputs detailed information about detected outliers in train, validation, and test datasets.

        Returns:
            Dict: Dict containing only the outliers DataFrames of train, val, and test splits.
        """
        outlier_num = {
            "train": self.train_outliers["outlier_mask"].sum() if self.train_outliers is not None else 0,
            "val": self.val_outliers["outlier_mask"].sum() if self.val_outliers is not None else 0,
            "test": self.test_outliers["outlier_mask"].sum() if self.test_outliers is not None else 0
        }
        
        dfs: Dict[str, Optional[pl.DataFrame]] = {"train": None, "val": None, "test": None}
        
        if self.train_outliers is not None and self.train is not None:
            dfs["train"] = self.train.filter(self.train_outliers["outlier_mask"])
        if self.val_outliers is not None and self.val is not None:
            dfs["val"] = self.val.filter(self.val_outliers["outlier_mask"])
        if self.test_outliers is not None and self.test is not None:
            dfs["test"] = self.test.filter(self.test_outliers["outlier_mask"])
        
        print("\n------ Outlier Detection Results -------")
        print("The dataset contains the following outliers in the different splits:\n")
        for split, num in outlier_num.items():
            print(f"{split}: {num} outliers found." if num != 0 else f"{split}: No outliers found.")
        
        print("\n")
        for split, df in dfs.items():
            if df is not None:
                print(f"Outlier dataframe for {split} split\n--------------------")
                print(df)
            else:
                print(f"There is no outlier dataframe for {split} split\n--------------------")
        
        return dfs

    def detailed_label_issues_output(self) -> Dict[str, pl.DataFrame | None]:
        """
        Identifies and outputs detailed information about detected label issues in training, validation, and test datasets.
        
        This method analyzes potential label issues in different dataset splits and constructs DataFrames containing 
        relevant information, such as predicted labels and original labels.
        
        Returns:
            Dict[str, Optional[pl.DataFrame]]: A dictionary with DataFrames of detected label issues for each split 
            ("train", "val", "test"). If no issues are found in a split, its value will be None.
        """
        outlier_counts = {
            "train": self.train_misslabelings["label_issue_mask"].sum() if self.train_misslabelings is not None else 0,
            "val": self.val_misslabelings["label_issue_mask"].sum() if self.val_misslabelings is not None else 0,
            "test": self.test_misslabelings["label_issue_mask"].sum() if self.test_misslabelings is not None else 0
        }
        
        dfs: Dict[str, Optional[pl.DataFrame]] = {"train": None, "val": None, "test": None}
        
        for split, dataset, misslabelings in zip(
            ["train", "val", "test"], [self.train, self.val, self.test], 
            [self.train_misslabelings, self.val_misslabelings, self.test_misslabelings]
        ):
            if dataset is not None and misslabelings is not None:
                base_df = dataset.drop(self.target)
                new_columns = misslabelings.select(["label_issue_mask", "predicted_label", "original_label"])
                new_columns = new_columns.rename({"predicted_label": "cleanlab_predicted_label", "original_label": self.target})
                dfs[split] = base_df.with_columns(new_columns)
        
        print("\n------ Label Issue Results -------")
        print("The dataset contains the following label issues in the different splits:\n")
        
        for split, count in outlier_counts.items():
            print(f"{split}: {count} label issues found by cleanlab." if count else f"{split}: No label issues found by cleanlab.")
        
        print("\n")
        
        for split, df in dfs.items():
            if df is not None:
                print(f"{split} DataFrame with label issue column and proposed labels by cleanlab.\n--------------------")
                print(df)
            else:
                print(f"No label issues for {split} split. No DataFrame to display.\n--------------------")
        
        return dfs
    
    def detailed_near_duplicates(self) -> Tuple[Optional[pl.DataFrame], Optional[Dict[str, int]], Optional[Dict[str, list]]]:
        """Provides detailed information near duplicates within and across different dataset splits.

        Returns:
            pl.DataFrame | None: DataFrame containing near-duplicate pairs if available.
        """
        duplicates_summary = None
        duplicates_ids = None
        if self.near_duplicates_issues is not None:
            group_sets: DefaultDict[str, Set[int]] = defaultdict(set)
            cross_split_sets: DefaultDict[str, Set[int]] = defaultdict(set)

            for row in self.near_duplicates_issues.iter_rows(named=True):
                if row["dataset"] == row["near_duplicate_dataset"]:
                    group_sets[f"{row['dataset']}-{row['dataset']}"] |= {row["new_id"], row["near_duplicate_id"]}
                else:
                    cross_split_sets[f"{row['dataset']}-{row['near_duplicate_dataset']}"] |= {row["near_duplicate_id"]}

            duplicates_summary = {
                "train-train": len(group_sets.get("train-train", set())),
                "val-val": len(group_sets.get("val-val", set())),
                "test-test": len(group_sets.get("test-test", set())),
                "train-val": len(cross_split_sets.get("train-val", set())),
                "train-test": len(cross_split_sets.get("train-test", set())),
                "val-test": len(cross_split_sets.get("val-test", set())),
            }

            duplicates_ids = {
                "train-train": sorted(group_sets.get("train-train", set())),
                "val-val": sorted(group_sets.get("val-val", set())),
                "test-test": sorted(group_sets.get("test-test", set())),
                "train-val": sorted(cross_split_sets.get("train-val", set())),
                "train-test": sorted(cross_split_sets.get("train-test", set())),
                "val-test": sorted(cross_split_sets.get("val-test", set())),
            }

            same_split_count = duplicates_summary["train-train"] + duplicates_summary["val-val"] + duplicates_summary["test-test"] 
            cross_split_count = duplicates_summary["train-val"] + duplicates_summary["val-test"] + duplicates_summary["train-test"] 
            
            result = "\n------ Near Duplicates Report ------"
            if same_split_count != 0:
                result += f"\n{same_split_count} Near Duplicates within the same split"
                result += f"\n - {duplicates_summary['train-train']} in train split (IDs: {duplicates_ids['train-train']})"
                result += f"\n - {duplicates_summary['val-val']} in validation split (IDs: {duplicates_ids['val-val']})"
                result += f"\n - {duplicates_summary['test-test']} in test split (IDs: {duplicates_ids['test-test']})"
            if cross_split_count != 0:
                result += f"\n⚠️  {cross_split_count} Near Duplicates across different splits"
                result += f"\n - {duplicates_summary['train-val']} between train and validation split (IDs: {duplicates_ids['train-val']})"
                result += f"\n - {duplicates_summary['train-test']} between train and test split (IDs: {duplicates_ids['train-test']})"
                result += f"\n - {duplicates_summary['val-test']} between validation and test split (IDs: {duplicates_ids['val-test']})"
            
            if result == "\n------ Near Duplicates Report ------":
                result += f"\nNo near duplicate issues found."
            else:
                print(result)
                print(self.near_duplicates_issues)
        else:
            result = "\n------ Near Duplicates Report ------"
            result += f"\nNo near duplicate issues found."
            print(result)

        return self.near_duplicates_issues, duplicates_summary, duplicates_ids

    def detailed_semantic_types(self):
        """Displays detected semantic types for dataset features."""
        if self.semantic_types:
            result = "\n------ Semantic Types results ------"
            for colum, sem_type in self.semantic_types.items():
                result += f"\nColumn {colum} could have semantic type: {sem_type}"    
            print(result)
        
        else:
            result = "\n------ Semantic Types results ------"
            result+= "No sementic types were created."

def load_from_data(data: DownloadedData, profiled_data: ProfiledData) -> DataSummary:
    """Loads a DataSummary object from a dataset and precomputed profiling data.
    Args:
        data (DownloadedData): The downloaded dataset.
        profiled_data (ProfiledData): The precomputed profiling results.

    Returns:
        DataSummary: A reconstructed DataSummary object.
    """
    
    data_summary = DataSummary(data)

    attributes = {
        "train_stats": "train_stats",
        "val_stats": "val_stats",
        "test_stats": "test_stats",
        "train_solved_issues": "train_solved_issues",
        "val_solved_issues": "val_solved_issues",
        "test_solved_issues": "test_solved_issues",
        "train_outliers": "train_outlier",
        "val_outliers": "val_outlier",
        "test_outliers": "test_outlier",
        "train_misslabelings": "train_misslabeling",
        "val_misslabelings": "val_misslabeling",
        "test_misslabelings": "test_misslabeling",
        "data_leakage_split_based": "data_leakage_split_based",
        "near_duplicates_issues": "near_duplicated_issues",
        "boolean_near_duplicates_removal_masks": "boolean_near_duplicates_removal_masks",
        "data_leakage_correlation_based": "data_leakage_correlation_based",
        "correlation_matrix": "correlation_matrix",
        "highly_correlated_features": "highly_correlated_features",
        "semantic_types": "semantic_types",
        "data_balance": "data_balancing",
    }
    

    for attr, key in attributes.items():
        setattr(data_summary, attr, profiled_data[key])
    
    return data_summary

def load_data_summary_from_pickle(path_to_summary: str) -> DataSummary:
    """Loads a DataSummary object from a pickle file.

    Args:
        path_to_summary (str): Path to the saved DataSummary pickle file.

    Returns:
        DataSummary: The loaded DataSummary object.
    """
    
    path = Path(path_to_summary).absolute()
    assert path.exists(), f"Der angegebene Pfad {path_to_summary} doesn't exist"
    
    with open(path, "rb") as file:
        summary = pickle.load(file)
        assert isinstance(summary, DataSummary), f"Loaded object is not an DataSummary Object"
    
    return summary

def load_profiled_data_from_pickle(path_to_profiled_data: str) -> ProfiledData:
    """Loads ProfiledData from a pickle file.

    Args:
        path_to_profiled_data (str): Path to the saved ProfiledData pickle file.

    Returns:
        ProfiledData: The loaded ProfiledData object.
    """
    
    path = Path(path_to_profiled_data).absolute()
    assert path.exists(), f"Der angegebene Pfad {path_to_profiled_data} doesn't exist"
    
    with open(path, "rb") as file:
        profiled_data: ProfiledData = pickle.load(file)
        
    if isinstance(profiled_data, dict) and all(key in profiled_data for key in ProfiledData.__annotations__):
            return profiled_data
    else:
        raise TypeError(f"Geladenes Objekt ist kein gültiges `ProfiledData`-Dictionary! (Erhalten: {type(profiled_data)})")