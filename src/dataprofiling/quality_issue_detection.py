import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal
from warnings import warn

import numpy as np
import pandas as pd
import polars as pl
from dython.nominal import associations
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from cleanlab import Datalab

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataprofiling.utils.feature_types import detect_feature_type


def preprocess_numerical_features(df_clean: pl.DataFrame, num_cols: list[str]) -> pl.DataFrame:
    """Preprocesses numerical features by handling missing values and standardizing them.

    This function replaces missing (`NaN` or `null`) values with the median of each numerical column 
    and then applies standard scaling to normalize the features.

    Args:
        df_clean (pl.DataFrame): The dataset containing numerical columns.
        num_cols (list[str]): A list of numerical feature column names.

    Returns:
        pl.DataFrame: A modified DataFrame with missing values imputed and features standardized.
    """
    df_clean = df_clean.with_columns([
        df_clean[cols].fill_nan(df_clean[cols].median()).fill_null(df_clean[cols].median())
        for cols in num_cols
    ])
    
    scaler = StandardScaler()
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols].to_pandas())
    return df_clean

def preprocess_categorical_features(df_clean: pl.DataFrame, cat_cols: list[str], target_col: Optional[str] = None) -> Tuple[pl.DataFrame, Optional[LabelEncoder]]:
    """Encodes categorical features and handles missing values.

    This function replaces missing values in categorical columns with the mode (most frequent value) 
    and applies label encoding. If `target_col` is a categorical column, its encoder is returned separately
    so that potencial corrected labels can be later translated back into there decoded form.

    Args:
        df_clean (pl.DataFrame): The dataset containing categorical columns.
        cat_cols (list[str]): A list of categorical feature column names.
        target_col (Optional[str], optional): The target column that requires special encoding. Defaults to None.

    Returns:
        Tuple: 
            - Processed DataFrame with categorical features encoded as numerical values.
            - A `LabelEncoder` for the target column if `target_col` is provided, otherwise None.
    """
    df_clean = df_clean.with_columns([
        df_clean[col].fill_null(
            df_clean[col].drop_nulls().mode().first()
        ) 
        for col in cat_cols
    ])
    
    le_save = None
    for col in cat_cols:
        if target_col is not None and target_col == col:
            le_save = LabelEncoder()
            encoded_values = le_save.fit_transform(df_clean.select(pl.col(col)).to_series().to_list())
            df_clean = df_clean.with_columns(pl.Series(col, encoded_values))
            continue
        le = LabelEncoder()
        encoded_values = le.fit_transform(df_clean.select(pl.col(col)).to_series().to_list())
        df_clean = df_clean.with_columns(pl.Series(col, encoded_values))

    if target_col is not None:
        return df_clean, le_save
    else:
        return df_clean, None

def preprocess_text_features(
    df_clean: pl.DataFrame, 
    txt_cols: list[str], 
    txt_emb_model: str = "all-MiniLM-L6-v2", 
    max_txt_emb_dim_per_col: int = 50
) -> pl.DataFrame:
    """Encodes text features/columns using sentence embeddings and reduces their dimensionality.

    This function processes specified text columns in a Polars DataFrame by:
    - Replacing missing values with "Unknown text".
    - Computing sentence embeddings using a pre-trained `SentenceTransformer` model.
    - Applying Principal Component Analysis (PCA) to reduce the dimensionality of embeddings,
      ensuring that at most `max_txt_emb_dim_per_col` dimensions per text column are retained.

    Args:
        df_clean (pl.DataFrame): A Polars DataFrame containing the dataset with text columns.
        txt_cols (list[str]): A list of column names corresponding to text features that need embedding.
        txt_emb_model (str, optional): The name of the pre-trained `SentenceTransformer` model used 
            for text embedding. Defaults to `"all-MiniLM-L6-v2"`.
        max_txt_emb_dim_per_col (int, optional): The maximum number of dimensions to retain per 
            text column after PCA. Defaults to 50.

    Returns:
        pl.DataFrame: A transformed DataFrame where text features are replaced with their
            embedding-based representations, with reduced dimensionality.
    """
    df_clean = df_clean.with_columns([
        df_clean[cols]
        .fill_null("Unknown text")
        for cols in txt_cols
    ])
    try:
        transformer = SentenceTransformer(txt_emb_model)
    except Exception as e:
        warn(f"Fehler: The model '{txt_emb_model}' could not be loaded for txt embedding of the columns. Use the 'all-MiniLM-L6-v2' modell instead.")
        transformer = SentenceTransformer("all-MiniLM-L6-v2")
    new_columns = []
    for col in txt_cols:
        text_list = df_clean.select(pl.col(col).cast(str)).to_series().to_list()
        embeddings = transformer.encode(text_list, convert_to_numpy=True)
        pca = PCA(n_components=0.95, svd_solver='full')
        reduced_embeddings = pca.fit_transform(embeddings)
        n_components = min(reduced_embeddings.shape[1], max_txt_emb_dim_per_col)
        reduced_embeddings = reduced_embeddings[:, :n_components]  
        new_cols = [f"{col}_{i}" for i in range(reduced_embeddings.shape[1])]
        new_columns.append(pl.DataFrame(reduced_embeddings, schema=new_cols))
    df_clean = pl.concat([df_clean] + new_columns, how="horizontal")
    df_clean = df_clean.drop(txt_cols)
    return df_clean

def calc_pred_probs(data: np.ndarray, labels: np.ndarray, clf: ClassifierMixin, cv: int = 5) -> np.ndarray:
    """Computes prediction probabilities using cross-validation.

    This function estimates class probabilities for each instance in the dataset 
    by performing cross-validation with `cross_val_predict` using the `predict_proba` method.

    Args:
        data (np.ndarray): A 2D NumPy array of shape (n_samples, n_features) representing the feature matrix.
        labels (np.ndarray): A 1D NumPy array of shape (n_samples,) containing the target class labels.
        clf (ClassifierMixin): A classifier model that implements the `predict_proba` method.
        cv (int, optional): Number of cross-validation folds. Higher values provide a more reliable
            estimate but increase computation time. Defaults to 5.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n_samples, n_classes) containing the predicted class probabilities
            for each instance in the dataset.
    """
    if not isinstance(clf, ClassifierMixin):
        raise TypeError("clf must be a classification model (ClassifierMixin).")

    pred_probs = cross_val_predict(
        clf,
        data,
        labels,
        method="predict_proba",
        cv = cv
    )
    return pred_probs

def calc_knn_graph(
    data: np.ndarray, 
    metric: Literal["euclidean", "cosine"] = "euclidean", 
    n_neighbors: int = 50
) -> csr_matrix:
    """Builds a k-nearest neighbors (KNN) graph for the dataset.

    This function constructs a KNN graph by computing pairwise distances between data points
    using the specified metric and retaining the `n_neighbors` closest neighbors for each point.
    The result is a sparse adjacency matrix where each entry represents the distance between
    connected points.

    Args:
        data (np.ndarray): A 2D NumPy array of shape (n_samples, n_features) representing the dataset.
        metric (Literal["euclidean", "cosine"], optional): The distance metric to use for computing
            nearest neighbors. Supported options are:
            - "euclidean": Uses Euclidean distance.
            - "cosine": Uses cosine similarity as a distance metric.
            Defaults to "euclidean".
        n_neighbors (int, optional): The number of nearest neighbors to consider for each point.
            Higher values result in a denser graph. Defaults to 50.

    Returns:
        csr_matrix: A sparse adjacency matrix (Compressed Sparse Row format) of shape (n_samples, n_samples),
            where each nonzero entry corresponds to the distance between neighboring points.
    """
    KNN = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
    KNN.fit(data)
    knn_graph = KNN.kneighbors_graph(mode="distance")
    return csr_matrix(knn_graph) 

def detect_quality_issues(
    df: pl.DataFrame,
    feature_cols: List[str],
    target_col: str,
    issue_types: Optional[List[str]] = None, 
    feature_types: Dict[str, str] = {},
    task: Optional[str] = None,
    pred_probs: Optional[np.ndarray] = None,
    cat_thres: float = 0.02,
    pred_prob_model: ClassifierMixin = HistGradientBoostingClassifier(),
    cv_folds: int = 5, 
    txt_emb_model: str = "all-MiniLM-L6-v2",
    max_txt_emb_dim_per_col: int = 50, 
    knn_metric: Literal["euclidean", "cosine"] = "euclidean", 
    knn_neighbors: int = 10,
    
) -> pl.DataFrame:
    """Detects data quality issues such as outliers, label issues, and near-duplicates.

    This function uses `cleanlab Datalab` to identify issues in the dataset based on feature values,
    prediction probabilities, and nearest-neighbor relationships.

    Args:
        df (pl.DataFrame): 
            The dataset containing features and labels. The prediction probabilities (`pred_probs`), 
            if provided, should correspond to the exact observations and labels in this DataFrame.
        
        feature_cols (List[str]): List of feature column names.
        target_col (str): Name of the target column.
        issue_types (List[str], optional): List of issue types to detect ("label", "outlier", "near_duplicate"). Defaults to all.
        feature_types (Dict[str, str], optional): Dictionary mapping column names to feature types ("num", "cat", "txt"). Defaults to {}.
        task (Optional[str], optional): Task type ("classification" or "regression"). If None, it is inferred from the target type. Defaults to None.
        
        pred_probs (np.ndarray, optional): 
            Prediction probabilities corresponding exactly to the rows in `df` and the `target_col`. 
            If provided, they must align in order with `df` to ensure correct issue detection. 
            If `pred_probs` is given, no prediction probabilities are computed using `pred_prob_model` and `cv_folds`.
        
        cat_thres (float, optional): 
            The threshold used to classify a numerical feature as categorical if feature_types must be
            calculated. A feature is considered categorical if `num_unique / num_total < cat_thres`.
            Default is `0.02` (i.e., features where unique values are less than 2% of the total count).
        
        pred_prob_model (ClassifierMixin, optional): 
            A scikit-learn compatible classification model used to generate prediction probabilities 
            for label issue detection. If a regression target is detected, this model is ignored.
            Default: `HistGradientBoostingClassifier()`.
        
        cv_folds (int, optional): 
            Number of cross-validation folds used with the pred_prob_model to create own pred_probs for
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

        knn_metric (Literal["euclidean", "cosine"]): 
            Distance metric used for K-Nearest Neighbors graph calculation in issue detection.
            - `"euclidean"`: Better if numerical features/ columns should have higher weight for 
              example for issue detection (e.g. outlier detection) than categorical.
            - `"cosine"`: Better for categorical and text-heavy datasets where those columns should
              have higher weight for example for issue detection (e.g. outlier detection).
            Default is `"euclidean"`.
        
        knn_neighbors (int, optional): 
            Number of nearest neighbors considered when constructing the KNN graph 
            for issue detection (e.g., CleanLab label noise detection).
            Default is `10`.
    Returns:
        pl.DataFrame: The input DataFrame with additional columns indicating detected issues.
    """
    original_df = df.clone()
    feature_types = _get_target_information(df, feature_types, feature_cols + [target_col], cat_thres=cat_thres)
    
    num_cols = [col for col, typ in feature_types.items() if typ == "num"]
    cat_cols = [col for col, typ in feature_types.items() if typ == "cat"]
    txt_cols = [col for col, typ in feature_types.items() if typ == "txt"]
    all_cols = num_cols + cat_cols + txt_cols
    
    if not task:
        if feature_types[target_col] == "num":
            task = "regression"
        elif feature_types[target_col] == "cat":
            task = "classification"
        else:
            raise Exception("Issue detection for text target isn't implemented")
    
    original_df = original_df.with_columns(pl.any_horizontal([original_df[col].is_null() for col in all_cols]).alias("was_imputed"))
    le_save = None
    if num_cols != []:
        df = preprocess_numerical_features(df, num_cols)
    if cat_cols != []:
        if target_col in cat_cols:
            df, le_save = preprocess_categorical_features(df, cat_cols, target_col=target_col)
        else:
            df, le_save = preprocess_categorical_features(df, cat_cols)
    if txt_cols != []:
        df = preprocess_text_features(df, txt_cols, txt_emb_model=txt_emb_model, max_txt_emb_dim_per_col=max_txt_emb_dim_per_col)

    # remove admistrative columns and label column
    X_processed = df
    if "dataset" in X_processed.columns:
        X_processed = X_processed.drop("dataset")
    if "ensemble_id_col" in X_processed.columns:
        X_processed = X_processed.drop("ensemble_id_col")
    labels = X_processed[target_col]
    X_processed = X_processed.drop(target_col)
    
    cleanlab_input = X_processed.to_numpy().astype(np.float64)
    data_dict = {"X": cleanlab_input, "labels": labels}
    lab = Datalab(data_dict, label_name="labels", task=task, verbosity=0)
    
    if issue_types is not None and any(t in ["label", "outlier", "near_duplicate"] for t in issue_types):
        issues: Dict[str, Dict] = {
            issue: {} 
            for issue in issue_types if issue in ["label", "outlier", "near_duplicate"]
        }
    else: 
        issues = {"label": {}, "outlier": {}, "near_duplicate": {}}
    
    if task == "classification":
        if pred_probs is None:
            pred_probs = calc_pred_probs(data = cleanlab_input, labels = labels.to_numpy(), clf = pred_prob_model, cv = cv_folds)
        knn_graph = calc_knn_graph(data = cleanlab_input, metric = knn_metric, n_neighbors = knn_neighbors)
        if knn_graph is not None:
            lab.find_issues(pred_probs=pred_probs, knn_graph=knn_graph, issue_types=issues)
        else:
            lab.find_issues(pred_probs=pred_probs, issue_types=issues, features=cleanlab_input)
            
            
    else:
        lab.find_issues(features=cleanlab_input, issue_types=issues)

    ### OUTLIER ISSUE ###
    outlier_mask = np.full(len(df), False, dtype=bool)
    outlier_scores = np.full(len(df), np.nan)
    try:
        outlier_results = lab.get_issues("outlier").reset_index()
        outlier_mask = outlier_results["is_outlier_issue"].values
        outlier_scores = outlier_results["outlier_score"].values
    except ValueError:
        warn("Either no outliers found or unable to calculate outliers.", UserWarning)

    ### LABEL ISSUES ###
    misslabeling_mask = np.full(len(df), False, dtype=bool)
    misslabeling_scores = np.full(len(df), np.nan)
    predicted_labels = [None] * len(df)
    try:    
        misslabel_results = lab.get_issues("label").reset_index()
        misslabeling_mask = misslabel_results["is_label_issue"].values
        misslabeling_scores = misslabel_results["label_score"].values
        predicted_labels = misslabel_results["predicted_label"]

        # Inverse transform predicted target label to string type value or num type if target label was numeric categorical  
        if target_col in cat_cols and le_save is not None and isinstance(predicted_labels[0], (int, np.integer)):
            predicted_labels = [le_save.inverse_transform([x])[0] for x in predicted_labels]
        
    except ValueError:
        warn("Either no label issues found or unable to calculate label issues.", UserWarning)

    ### NEAR DUPLICATES ###
    near_duplicate_mask = np.full(len(df), False, dtype=bool)
    near_duplicate_scores = np.full(len(df), np.nan)
    near_duplicate_sets = [[]] * len(df)
    try:        
        near_duplicate_results = lab.get_issues("near_duplicate").reset_index()
        near_duplicate_mask = near_duplicate_results["is_near_duplicate_issue"].values
        near_duplicate_scores = near_duplicate_results["near_duplicate_score"].values
        near_duplicate_sets = near_duplicate_results["near_duplicate_sets"].values
    except ValueError:
        warn("Either no near-duplicates found or unable to calculate near-duplicates.", UserWarning)


    df_with_issues = original_df.with_columns([
        pl.Series("is_outlier_issue", outlier_mask),
        pl.Series("outlier_score", outlier_scores),

        pl.Series("is_label_issue", misslabeling_mask),
        pl.Series("label_score", misslabeling_scores),
        pl.Series("predicted_label", predicted_labels),

        pl.Series("is_near_duplicate_issue", near_duplicate_mask),
        pl.Series("near_duplicate_score", near_duplicate_scores),
        pl.Series("near_duplicate_sets", near_duplicate_sets)
    ])

    return df_with_issues

def find_highly_correlated_groups(matrix: np.ndarray, feature_names: list, high_corr_thr: float = 0.95):
    """Identifies groups of highly correlated features.

    This function scans the correlation matrix and groups together features that exceed 
    the given correlation threshold.

    Args:
        matrix (np.ndarray): Correlation matrix.
        
        feature_names (list): List of feature names corresponding to the matrix indices.
        
        high_corr_thr (float): Threshold for detecting high correlation between features.  
            This value defines the correlation level at which potential data leakage is identified,  
            meaning that one feature explains another to an excessive degree. Correlation is considered  
            between numerical-numerical, numerical-categorical, and categorical-categorical feature pairs.
    
    Returns:
        List[tuple]: A list of tuples, where each tuple contains names of highly correlated features.
    """
    n = matrix.shape[0]
    visited = set()
    correlated_groups = []

    for i in range(n):
        if feature_names[i] in visited:
            continue 

        group = {feature_names[i]}
        for j in range(n):
            if i != j and abs(matrix[i, j]) >= high_corr_thr:
                group.add(feature_names[j])
                
        if len(group) > 1:
            sorted_group = tuple(sorted(group))
            if sorted_group not in correlated_groups:
                correlated_groups.append(sorted_group)
        visited.update(group)

    return correlated_groups

def detect_correlation_dataleakage(
    df: pl.DataFrame,
    feature_types: Optional[Dict[str, str]] = None,
    high_corr_thr: float = 0.95
) -> Tuple[pl.DataFrame, List[tuple]]:
    """Detects data leakage based on feature correlation.

    This function computes a correlation matrix for numerical and categorical features, identifies 
    highly correlated feature pairs, and flags them as potential sources of data leakage.

    Args:
        df (pl.DataFrame): The dataset containing features.
        
        feature_types (Optional[Dict[str, str]], optional): Dictionary mapping column names to feature types. 
            If None, feature types are inferred. Defaults to None.
        
        high_corr_thr (float): Threshold for detecting high correlation between features.  
            This value defines the correlation level at which potential data leakage is identified,  
            meaning that one feature explains another to an excessive degree. Correlation is considered  
            between numerical-numerical, numerical-categorical, and categorical-categorical feature pairs.

    Returns:
        Tuple[pl.DataFrame, List[tuple]]:
            - Correlation matrix as a DataFrame.
            - List of highly correlated feature pairs.
    """
    feature_types = _get_target_information(df, feature_types, list(df.columns))
    num_cols = [col for col, typ in feature_types.items() if typ == "num"]
    cat_cols = [col for col, typ in feature_types.items() if typ == "cat"]
    txt_cols = [col for col, typ in feature_types.items() if typ == "txt"]
    
    if txt_cols:
        warn("No correlation measure for text features implemented yet. Dataleakage based on text features will therefore not be taken into account.", UserWarning, stacklevel=0)
    
    cols_to_analyse = num_cols + cat_cols
    df = df.select(cols_to_analyse)
    df_pandas = df.to_pandas()
    
    corr = associations(
        dataset = df_pandas,
        numerical_columns = num_cols,
        nominal_columns = cat_cols,
        plot=False
    )
    
    corr_df: pd.DataFrame = corr["corr"]
    feature_names = corr_df.columns.tolist()
    corr_matrix = corr_df.to_numpy()
    correlated_features = find_highly_correlated_groups(corr_matrix, feature_names, high_corr_thr=high_corr_thr)
    
    return pl.from_pandas(corr_df), correlated_features

def _get_target_information(
    df: pl.DataFrame,
    feature_types: Optional[Dict[str, str]],
    target_cols: List[str],
    cat_thres: float = 0.02
) -> Dict[str, str]:
    """Extracts feature type information for target columns.

    This function determines the data type of each target column and classifies it 
    as numerical, categorical, or textual based on predefined thresholds.

    Args:
        df (pl.DataFrame): The input dataset.
        feature_types (Optional[Dict[str, str]]): A dictionary with predefined feature types for specific columns.
            If None, feature types are detected automatically.
        target_cols (List[str]): The list of columns to analyze.
        cat_thres (float, optional): Threshold for classifying a column as categorical,
            based on the ratio of unique values to the total number of entries. Default: 0.02.

    Returns:
        Dict[str, str]: A dictionary mapping column names to their detected feature types.
    """
    target_feature_types = {}
    for col in target_cols:
        if feature_types is not None and col in feature_types:
            target_feature_types[col] = feature_types[col]
        else:
            target_feature_types[col] = detect_feature_type(df[col], cat_thres=cat_thres)
    return target_feature_types
