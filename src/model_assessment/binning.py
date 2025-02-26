import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl
import psutil
from optbinning import BinningProcess

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.custom_data_types import AutoMLData, ModelAssessmentData
from src.dataprofiling.static_profiling import profile_statistics


def _get_physical_cores():
    try:
        return psutil.cpu_count(logical=False)
    except Exception:
        import multiprocessing

        return multiprocessing.cpu_count()


PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "max": {
        "n_jobs": 1,
    },
    "best": {
        "n_jobs": max(1, _get_physical_cores() // 4),
    },
    "high": {
        "n_jobs": max(1, _get_physical_cores() // 3),
    },
    "good": {
        "n_jobs": max(1, _get_physical_cores() // 2),
    },
    "medium": {
        "n_jobs": -1,
    },
}


class Binning:
    model_assessment_data: ModelAssessmentData

    def __init__(
        self,
        automl_data: AutoMLData,
        preset: str = "max",
        columns_to_bin: Optional[list[str]] = None,
    ):
        self.automl_data = automl_data
        self.train_stats = profile_statistics(automl_data["train"])
        self.test_stats = profile_statistics(automl_data["test"])
        self.preset = preset
        self.columns_to_bin = columns_to_bin

    def _exclude_target(self, data_selection: str = "test"):
        return self.automl_data[data_selection].select(pl.exclude(self.automl_data["target"]))

    def build_binning(
        self,
        n_jobs: Optional[int] = None,
        data_selection: str = "test",
        maxbins: Optional[int] = 20,
        minbins: Optional[int] = 2,
    ) -> pl.DataFrame:
        preset_config = PRESET_CONFIGS.get(self.preset, PRESET_CONFIGS["medium"]).copy()
        if n_jobs is not None:
            preset_config["n_jobs"] = n_jobs
        preset_config["min_n_bins"] = minbins
        preset_config["max_n_bins"] = maxbins
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

        all_columns = self.automl_data[data_selection].columns

        if self.automl_data["problem_type"] == "multiclass":
            # OptBinning doesn't support categorical for multiclass
            base_columns = [
                col
                for col in all_columns
                if (
                    col in self.automl_data[data_selection].schema
                    and self.automl_data[data_selection].schema[col] in numerical_dtypes
                    and col != self.automl_data["target"]
                )
            ]
        else:
            base_columns = [col for col in all_columns if col != self.automl_data["target"]]

        if self.columns_to_bin is not None:
            columns = [col for col in self.columns_to_bin if col in base_columns]
            if len(columns) < len(self.columns_to_bin):
                excluded = set(self.columns_to_bin) - set(columns)
                warnings.warn(
                    f"Warning: {len(excluded)} columns were excluded from binning: {list(excluded)}", UserWarning
                )
        else:
            columns = base_columns

        if not columns:
            warnings.warn("Warning: No columns found for binning. Returning unbinned", UserWarning)
            return self.automl_data[data_selection]

        base_config = {"variable_names": columns}

        if self.automl_data["problem_type"] != "multiclass":
            categorical_dtypes = {pl.Utf8, pl.String, pl.Categorical, pl.Boolean}
            cat_features = []

            cat_features.extend(
                [
                    col
                    for col in columns
                    if col in self.automl_data[data_selection].schema
                    and self.automl_data[data_selection].schema[col] in categorical_dtypes
                ]
            )

            cat_features.extend(
                [
                    col
                    for col in self.train_stats.keys()
                    if (
                        col in columns
                        and col != self.automl_data["target"]
                        and col not in cat_features  # Avoid duplicates
                        and self.train_stats[col].get("dformat", "not_cat") == "cat"
                    )
                ]
            )

            if cat_features:
                base_config["categorical_variables"] = cat_features

        try:
            optb = BinningProcess(**preset_config, **base_config)
            binned = pl.from_pandas(
                optb.fit_transform(
                    X=self.automl_data[data_selection].select(columns).to_pandas(),
                    y=self.automl_data[data_selection].select(self.automl_data["target"]).to_series().to_pandas(),
                    metric="bins",
                )
            )

            remaining_cols = self.automl_data[data_selection].select([col for col in all_columns if col not in columns])
            output = pl.concat([binned, remaining_cols], how="horizontal")

            missing_cols = [col for col in all_columns if col not in output.columns]
            if missing_cols:
                missing_df = self.automl_data[data_selection].select(missing_cols)
                output = pl.concat([output, missing_df], how="horizontal")

            return output

        except Exception as e:
            print(f"Error in binning process: {str(e)}")
            print("Returning original data without binning")
            return self.automl_data[data_selection]

    def auto_bin(self):
        self.model_assessment_data = ModelAssessmentData(
            **self.automl_data,
            binned_train=self.build_binning(data_selection="train"),
            binned_test=self.build_binning(data_selection="test"),
        )
