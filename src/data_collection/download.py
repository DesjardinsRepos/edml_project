import json
import os
import re
import sys
import tempfile
import contextlib
from pathlib import Path
from typing import Optional
import cchardet
import kaggle
import polars as pl
from datasets import load_dataset
from huggingface_hub import hf_api
from sklearn.datasets import fetch_openml
from contextlib import redirect_stdout

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_collection.prompting import prompt


def load_local_files(file_paths: list[str]) -> Optional[pl.DataFrame]:
    """Load multiple local files into a Polars DataFrame.

    Args:        
        file_paths (list[str]): List of file paths

    Raises:
        ValueError: If the columns of the files do not match

    Returns:
        Optional[pl.DataFrame]: Polars DataFrame
    """
    if not file_paths:
        return None
    
    if isinstance(file_paths, str):
        file_paths = [file_paths]  # Convert single path to list
    
    dfs = []
    column_set = None
    
    for file_path in file_paths:
        if file_path.endswith('.csv'):
            df = pl.read_csv(file_path, use_pyarrow=True)
        elif file_path.endswith('.parquet'):
            df = pl.read_parquet(file_path, use_pyarrow=True)
        elif file_path.endswith('.feather'):
            df = pl.read_ipc(file_path, use_pyarrow=True)
        else:
            continue
        
        if column_set is None:
            column_set = set(df.columns)
        elif column_set != set(df.columns):
            raise ValueError(f"Column mismatch detected in file: {file_path}")
        
        dfs.append(df)
    
    return pl.concat(dfs) if dfs else None

def load_local_file(file_path: str) -> pl.DataFrame:
    """Load a local file into a Polars DataFrame.

    Args:
        file_path (str): Path to the file

    Returns:
        Optional[pl.DataFrame]: Polars DataFrame
    """
    if file_path.endswith('.csv'):
        with open(file_path, "rb") as f:
            autoEnc = cchardet.detect(f.read(100000))["encoding"]
        
        return pl.read_csv(file_path, use_pyarrow=True, encoding=autoEnc)
    if file_path.endswith('.parquet'):
        return pl.read_parquet(file_path)
    if file_path.endswith('.feather'):
        return pl.read_feather(file_path)
    raise ValueError(f"Unsupported file format: {file_path}")

def save_datasets(save_dir: str, train_df: Optional[pl.DataFrame], val_df: Optional[pl.DataFrame], test_df: Optional[pl.DataFrame]):
    """Save datasets to a single Parquet file.

    Args:
        save_dir (str): Directory to save the dataset
        train_df (Optional[pl.DataFrame]): Training dataset to save
        val_df (Optional[pl.DataFrame]): Validation dataset to save
        test_df (Optional[pl.DataFrame]): Test dataset to save
    """
    save_path = os.path.join(save_dir, "datasets.parquet")
    
    # Add a column to distinguish datasets
    df_list = []
    if train_df is not None:
        train_df = train_df.with_columns(pl.lit("train").alias("dataset"))
        df_list.append(train_df)
    if val_df is not None:
        val_df = val_df.with_columns(pl.lit("val").alias("dataset"))
        df_list.append(val_df)
    if test_df is not None:
        test_df = test_df.with_columns(pl.lit("test").alias("dataset"))
        df_list.append(test_df)
    
    if df_list:
        combined_df = pl.concat(df_list)
        combined_df.write_parquet(save_path)

def load_datasets(save_dir: str) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """Load datasets from a single Parquet file.

    Args:
        save_dir (str): Directory where the dataset is saved

    Returns:
        tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]: Tuple of training, validation, and test datasets
    """
    save_path = os.path.join(save_dir, "datasets.parquet")
    if not os.path.exists(save_path):
        return None, None, None
    
    df = pl.read_parquet(save_path)
    
    train_df = df.filter(pl.col("dataset") == "train").drop("dataset") if "train" in df["dataset"].unique().to_list() else None
    val_df = df.filter(pl.col("dataset") == "val").drop("dataset") if "val" in df["dataset"].unique().to_list() else None
    test_df = df.filter(pl.col("dataset") == "test").drop("dataset") if "test" in df["dataset"].unique().to_list() else None
    
    return train_df, val_df, test_df

def save_description(save_dir: str, description: str):
    """Save dataset description to disk.

    Args:
        save_dir (str): Directory to save the description
        description (str): Description of the dataset
    """
    with open(os.path.join(save_dir, "dataset_description.txt"), "w") as f:
        # Remove non-UTF-8 characters to avoid issues on windows
        f.write(re.sub(r"[^\u0000-\uD7FF\uE000-\uFFFF]+", "", description))

def load_description(save_dir: str) -> str:
    """Load dataset description from disk.

    Args:
        save_dir (str): Directory where the description is saved

    Returns:
        str: Description of the dataset
    """
    with open(os.path.join(save_dir, "dataset_description.txt"), "r") as f:
        return f.read()

def huggingface_dataset(dataset_name: str, save_dir: str) -> dict:
    """Download a Hugging Face dataset.

    Args:
        dataset_name (str): Name of the dataset, for example "victor/titanic"

    Raises:
        ValueError: If train and validation splits have different structures

    Returns:
        Optional[dict]: Dictionary containing training, validation, and test datasets, description, and local path to the data
    """
    local_dataset_name = f"hf.{dataset_name.replace('/', '.')}"
    
    if os.path.exists(save_dir):
        # Load from cache
        print(f"📂 Loading dataset '{local_dataset_name}' from cache...")
        train_df, val_df, test_df = load_datasets(save_dir)
        description = load_description(save_dir)

        return {
            "train": train_df, 
            "val": val_df, 
            "test": test_df, 
            "description": description,
            "path": Path(save_dir),
        }
    try:
        # Download Data
        print(f"📂 Downloading dataset '{local_dataset_name}'")
        
        configs = hf_api.HfApi().dataset_info(dataset_name).cardData

        with open("/dev/null", "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            if hasattr(configs, 'configs'):
                config = configs.configs[0]["config_name"] if configs.configs[0]["config_name"] != "encoding" else configs.configs[1]["config_name"]
                dataset = load_dataset(dataset_name, config)
            else:
                try:
                    dataset = load_dataset(dataset_name)
                except Exception as e:
                    if "Config name is missing" in str(e):
                        # If we are here, the dataset is splitted into subsets but the config does not specify them.
                        # However, the exception provides the subset names, so we have to extract them this way.

                        configs = eval(re.search(r"\['(.*?)'\]", str(e)).group(0)[1:-1])
                        config = configs[0] if configs[0] != "encoding" else configs[1]
                        dataset = load_dataset(dataset_name, config)
                    else:
                        raise

        train_df = dataset['train'].to_polars() if 'train' in dataset.keys() else None
        val_df = dataset['validation'].to_polars() if 'validation' in dataset.keys() else None
        test_df = dataset['test'].to_polars() if 'test' in dataset.keys() else None
        
        description = (
            dataset['train'].info.description
            if 'train' in dataset.keys() and hasattr(dataset['train'], 'info') and hasattr(dataset['train'].info, 'description')
            else "No description available."
        )

        for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            if split_df is not None and train_df is not None and set(train_df.columns) != set(split_df.columns):
                raise ValueError(f"🚨 Train and {split_name} splits have different structures.")

        # Cache Data

        os.makedirs(save_dir, exist_ok=True)

        save_datasets(save_dir, train_df, val_df, test_df)
        save_description(save_dir, description)

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "description": description,
            "path": Path(save_dir),
        }

    except Exception as e:
        print(f"🚨 Error downloading Hugging Face dataset: {e}")
        raise

def kaggle_dataset(dataset_name: str, save_dir: str, llm_config: dict) -> dict:
    """Download a Kaggle dataset.
    
    Args:
        dataset_name (str): Name of the dataset, for example "rahulsah06/titanic"
        save_dir (str): Directory to save the dataset
        llm_config (dict): Configuration for the language model

    Raises:
        ValueError: If no supported file format (CSV, Parquet, Feather) found in the downloaded Kaggle dataset
        
    Returns:
        Optional[dict]: Dictionary containing training, validation, and test datasets, description, and local path to the data
    """
    local_dataset_name = f"ka.{dataset_name.replace('/', '.')}"

    try:
        if os.path.exists(save_dir):
            # Load from cache
            print(f"📂 Loading dataset '{local_dataset_name}' from cache...")
            description = load_description(save_dir)
        else:
            # Download Data
            print(f"📂 Downloading dataset '{local_dataset_name}'")
            with tempfile.TemporaryDirectory() as temp_dir:
                metadata_path = os.path.join(temp_dir, "dataset-metadata.json")
                kaggle.api.dataset_metadata(dataset_name, path=temp_dir)
                
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                description = metadata.get("description", "No description available.")

                os.makedirs(save_dir, exist_ok=True)
                save_description(save_dir, description)

                with open(os.devnull, "w") as f, redirect_stdout(f):
                    kaggle.api.dataset_download_files(dataset_name, path=save_dir, unzip=True, quiet=True)

        if os.path.exists(os.path.join(save_dir, "llm-splits.json")):
            # Load from cache
            with open(os.path.join(save_dir, "llm-splits.json"), "r") as f:
                cache = json.load(f)

                return {
                    "train": load_local_files(cache["train"]),
                    "val": load_local_files(cache["val"]),
                    "test": load_local_files(cache["test"]),
                    "description": description,
                    "path": Path(save_dir),
                }

        # Process Data if not available in cache

        file_list = []    
        for root, _, files in os.walk(save_dir):
            for file in files:
                if file.endswith(('.csv', '.parquet', '.feather')):
                    file_path = os.path.join(root, file)
                    file_list.append(f"{file_path} - {os.path.getsize(file_path)/1000}")

        prompt_text = f"""
            This is the file list of a kaggle dataset, with their respective sizes in KB:
            
            {file_list}

            In the next prompt, i will ask you to identify the files which contain the train/val/test data.
            These can be multiple for each split.
            Currently, we only support csv, parqet and feather files.
            For that, give me a list of files with their filepath you want to know the colums of and i will provide them in the next prompt.
            Answer in the following format and do not provide reasoning:

            {{
                "interesting_dataframes": [<list of dataframes you want to know the columns of>]
            }}
        """.replace("\'", '')

        answer = json.loads(prompt(
            prompt_text,
            type=llm_config["type"],
            model=llm_config["model"]
        ).strip("```json").strip("```").strip())
        
        column_list = [f"{x}: {load_local_file(x).columns}" for x in answer["interesting_dataframes"]]

        prompt_text = f"""
            This is the list of files of my kaggle dataset:
            {[x.split(" ")[0] for x in file_list]}

            Your task is now to identify one or multiple files for the train/val/test dataset.
            Test and val may be null.
            Be aware that if multiple files need to be merged, they need to have the same columns.
            Currently, we only support csv, parqet and feather files.

            Here you have a list of column names for some files that might be helpful:
            {column_list}
            
            Finally, answer in the following format and do not provide reasoning.
            Only set val or test if you are sure and always provide the full path to the file:

            {{
                "train": [<path to one or multiple files for the train dataset>],
                "val": <path to one or multiple files for the validation dataset or null>,
                "test": <path to one or multiple files for the test dataset or null>,
            }}
        """.replace("\'", '')

        answer = json.loads(prompt(
            prompt_text,
            type=llm_config["type"],
            model=llm_config["model"]
        ).strip("```json").strip("```").strip())

        with open(os.path.join(save_dir, "llm-splits.json"), "w") as f:
            json.dump({
                "train": answer["train"],
                "test": answer["test"],
                "val": answer["val"],
            }, f)

        if answer["train"] is not None:
            return {
                "train": load_local_files(answer["train"]),
                "test": load_local_files(answer["test"]),
                "val": load_local_files(answer["val"]),
                "description": description,
                "path": Path(save_dir),
            }

        print("🚨 No supported file format (CSV, Parquet, Feather) found in the downloaded Kaggle dataset.")
        raise ValueError("No supported file format (CSV, Parquet, Feather) found in the downloaded Kaggle dataset.")

    except Exception as e:
        print(f"🚨 Error downloading Kaggle dataset: {e}")
        raise

def openml_dataset(dataset_id: int, dataset_name: str, save_dir: str) -> dict:
    """Download an OpenML dataset.

    Args:
        dataset_id (int): ID of the dataset, for example 40945
        dataset_name (str): Name of the dataset, for example "titanic"

    Returns:
        Optional[dict]: Dictionary containing training, validation, and test datasets, description, and local path to the data
    """

    dataset_name = f"om.{dataset_name.replace('/', '.')}.{dataset_id}"
    
    if os.path.exists(save_dir):
        # Load from cache
        print(f"📂 Loading dataset '{dataset_name}' from cache...")
        train_df, val_df, test_df = load_datasets(save_dir)
        description = load_description(save_dir)

        return {
            "train": train_df, 
            "val": val_df, 
            "test": test_df, 
            "description": description,
            "path": Path(save_dir),
        }
    try:
        # Download Data
        print(f"📂 Downloading dataset '{dataset_name}'")

        dataset = fetch_openml(data_id=int(dataset_id), as_frame=True)
        df = pl.from_pandas(dataset.frame)

        description = (
            dataset.description
            if hasattr(dataset, 'description')
            else "No description available."
        )

        train_df = df
        val_df = None
        test_df = None

        if 'train' in dataset.keys():
            train_df = pl.from_pandas(dataset['train'])
        if 'validation' in dataset.keys():
            val_df = pl.from_pandas(dataset['validation'])
        if 'test' in dataset.keys():
            test_df = pl.from_pandas(dataset['test'])

        # Cache Data

        os.makedirs(save_dir, exist_ok=True)
        save_datasets(save_dir, train_df, val_df, test_df)
        save_description(save_dir, description)

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "description": description,
            "path": Path(save_dir),
        }

    except Exception as e:
        print(f"🚨 Error downloading OpenML dataset: {e}")
        raise