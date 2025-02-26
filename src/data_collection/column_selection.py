import json
import os
import random
import re
import sys
import warnings
from pathlib import Path
import openml
import polars as pl

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.custom_data_types import DownloadedData
from src.data_collection.download import huggingface_dataset, kaggle_dataset, openml_dataset
from src.data_collection.prompting import prompt


def remove_all_zero_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Remove all columns that have only zeros.

    Args:
        df (pl.DataFrame): Input dataframe

    Returns:
        pl.DataFrame: DataFrame with columns that have only zeros removed
    """
    if df is None:
        return None
    
    zero_columns = [col for col in df.columns if df[col].n_unique() == 1 and (df[col][0] == 0 or df[col][0] is None)]
    
    if zero_columns:
        warnings.warn(f"⚠️  Removing columns with only zeros or None: {zero_columns}", UserWarning)
    
    return df.drop(zero_columns)

def get_dataset(dataset_link: str, save_path="saves", llm_config = {"type": "API", "model": "gpt-4o-mini"}) -> DownloadedData:
    """Download a dataset from a given link and prompt the user to select the target column, sensitive features, and columns of interest.
    
    Args:
        dataset_link (str): Link to the dataset
        save_path (str, optional): Path to save the downloaded dataset. Defaults to "saves".
        llm_config (dict, optional): Configuration for the language model. Defaults to {"type": "API", "model": "gpt-4o-mini"}.

    Raises:
        ValueError: If the source of the dataset is unknown or no training data is found
        
    Returns:
        DownloadedData: A dictionary containing the train, test, and validation dataframes, the target column, sensitive features, columns of interest, dataset description, dataset title, seed, and path to the local data
    """
    HF_REGEX = r"https://huggingface\.co/datasets/([^/]+/[^/]+)"
    HF_NAME_REGEX = r"https://huggingface\.co/datasets/([^/]+)/([^/]+)"
    KAGGLE_REGEX = r"https://www\.kaggle\.com/datasets/([^/]+/[^/]+)"
    KAGGLE_NAME_REGEX = r"https://www\.kaggle\.com/datasets/([^/]+)/([^/]+)"
    OPENML_REGEX = r"[?&]id=(\d+)"
    OPENML_NAME_REGEX = r"https://www\.openml\.org/search\?.*exact_name=([^&]+)"

    if "huggingface.co" in dataset_link:
        dataset_name_match = re.search(HF_REGEX, dataset_link)
        dataset_name = dataset_name_match.group(1) if dataset_name_match else ""
        save_dir = f"{save_path}/hf.{dataset_name.replace('/', '.')}/datasets"
        downloaded_data = huggingface_dataset(dataset_name, save_dir)
        dataset_title_match = re.search(HF_NAME_REGEX, dataset_link)
        dataset_title = dataset_title_match.group(2) if dataset_title_match else ""
    elif "kaggle.com" in dataset_link:
        dataset_name_match = re.search(KAGGLE_REGEX, dataset_link)
        dataset_name = dataset_name_match.group(1) if dataset_name_match else ""
        save_dir = f"{save_path}/ka.{dataset_name.replace('/', '.')}/datasets"
        downloaded_data = kaggle_dataset(dataset_name, save_dir, llm_config)
        dataset_title_match = re.search(KAGGLE_NAME_REGEX, dataset_link)
        dataset_title = dataset_title_match.group(2) if dataset_title_match else ""
    elif "openml.org" in dataset_link:
        dataset_id_match = re.search(OPENML_REGEX, dataset_link)
        dataset_id = int(dataset_id_match.group(1)) if dataset_id_match else -1
        dataset_title_match = re.search(OPENML_NAME_REGEX, dataset_link)

        dataset_title = None
        if dataset_title_match:
            # try to get title from url
            dataset_title = dataset_title_match.group(1)

        if not dataset_title and os.path.exists(save_path):
            # try to get title from local path
            for f in os.listdir(save_path):
                if f.startswith("om.") and f.endswith(f".{dataset_id}"):
                    dataset_title = f.split('.')[1]

        if not dataset_title:
            # get title from openml
            dataset_title = openml.datasets.get_dataset(dataset_id).name

        save_dir = f"{save_path}/om.{dataset_title.replace('/', '.')}.{dataset_id}/datasets"
        downloaded_data = openml_dataset(dataset_id, dataset_title, save_dir)
    else:
        print(f"🚨 Unknown source: {dataset_link}")
        raise ValueError(f"Unknown source")
    
    if downloaded_data["train"] is None:
        print("🚨 No training data found")
        raise ValueError("training data is None")
    
    splits = {
        "train": remove_all_zero_columns(downloaded_data["train"]),
        "val": remove_all_zero_columns(downloaded_data["val"]),
        "test": remove_all_zero_columns(downloaded_data["test"]),
    }

    # Check if there are exclusive columns
    if not all(
        set(df.columns) == set(splits["train"].columns) 
        for df in [splits["val"], splits["test"]] 
        if df is not None
    ):
        print("⚠️  Exclusive columns detected. Dropping them.")
        non_none_splits = {k: v for k, v in splits.items() if v is not None}
        common_columns = set.intersection(*(set(df.columns) for df in non_none_splits.values()))
        splits = {k: v.select(list(common_columns)) if v is not None else None for k, v in splits.items()}

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    def check_missing_target(target: str):
        if not all(
            (df is None or target in df.columns) for df in [train_df, val_df, test_df]
        ):
            print(f"🚨 Target column {target} missing in one of the splits.")
            raise ValueError("Target column missing")
    
    if os.path.exists(f"{save_dir}/llm-columns.json"):
        # Load llm answer from cache
        with open(os.path.join(save_dir, "llm-columns.json"), "r") as f:
            cache = json.load(f)
            check_missing_target(cache["target"])

            return DownloadedData(
                train=train_df,
                test=test_df,
                val=val_df,
                target=cache["target"],
                columns_of_interest=cache["columns_of_interest"],
                sensitive_features=cache["sensitive_features"],
                description=downloaded_data["description"],
                title=dataset_title,
                seed=random.randint(0, 2**32 - 1),
                path=downloaded_data["path"].parent,
            )
    else:
        answer = json.loads(prompt(f"""
            You are a data scientist tasked with building a predictive model. The dataset has the following columns: {train_df.columns}.
            1. Identify the column to be used as the target variable for the predictive model.
            2. Identify sensitive columns that contain personally identifiable information (PII) or legally protected attributes (e.g., race, gender, age, income, etc.).
            3. This is a large dataset. Select only the most important columns to be used as features for training the model.

            Provide the results in JSON format as follows and do not provide reasoning:
            {{
                "target_column": <name_of_target_column>,
                "sensitive_features": [<list_of_sensitive_columns>],
                "columns_of_interest": [<list_of_potentially_important_columns>]
            }}

            Also, you can use the dataset description to help you make your decision:
            {downloaded_data["description"]}
            """,
            type=llm_config["type"],
            model=llm_config["model"]
        ).strip("```json").strip("```").strip())

        check_missing_target(answer["target_column"])

        # Cache llm answer
        with open(os.path.join(save_dir, "llm-columns.json"), "w") as f:
            json.dump({
                "target": answer["target_column"],
                "columns_of_interest": answer["columns_of_interest"],
                "sensitive_features": answer["sensitive_features"]
            }, f)

        return DownloadedData(
            train=train_df,
            test=test_df,
            val=val_df,
            target=answer["target_column"],
            columns_of_interest=answer["columns_of_interest"],
            sensitive_features=answer["sensitive_features"],
            description=downloaded_data["description"],
            title=dataset_title,
            seed=random.randint(0, 2**32 - 1),
            path=downloaded_data["path"].parent,
        )