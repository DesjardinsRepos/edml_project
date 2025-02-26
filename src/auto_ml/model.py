import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import polars as pl
import torch
from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import TabularPredictor

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.custom_data_types import AutoMLData, ProcessedData

NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "max": {
        "presets": "experimental_quality",
        "full_weighted_ensemble_additionally": True,
        "refit_full": False,
        "set_best_to_refit_full": False,
        "save_bag_folds": True,
        "time_limit": 6000,
        "ag_args_fit": {"num_gpus": NUM_GPUS},
    },
    "best": {
        "presets": "best_quality",
        "full_weighted_ensemble_additionally": True,
        "refit_full": False,
        "set_best_to_refit_full": False,
        "save_bag_folds": True,
        "time_limit": 6000,
        "ag_args_fit": {"num_gpus": NUM_GPUS},
    },
    "high": {
        "presets": "high_quality",
        "full_weighted_ensemble_additionally": True,
        "refit_full": False,
        "set_best_to_refit_full": False,
        "save_bag_folds": True,
        "time_limit": 600,
        "ag_args_fit": {"num_gpus": NUM_GPUS},
    },
    "good": {
        "presets": "good_quality",
        "full_weighted_ensemble_additionally": True,
        "refit_full": False,
        "set_best_to_refit_full": False,
        "save_bag_folds": True,
        "time_limit": 600,
        "ag_args_fit": {"num_gpus": NUM_GPUS},
    },
    "medium": {
        "presets": "medium_quality",
        "refit_full": "best",
        "set_best_to_refit_full": True,
        "save_bag_folds": False,
        "time_limit": 300,
        "ag_args_fit": {"num_gpus": NUM_GPUS},
    },
}
MODEL_DIR = Path("model")


class AutoMLModel:
    model: TabularPredictor | MultiModalPredictor
    auto_ml_data: AutoMLData

    def __init__(
        self,
        processed_data: ProcessedData,
        time_limit: Optional[int] = None,
        preset: str = "medium",
        load: bool = False,
        verbosity: int = 0,
        **kwargs,
    ):
        self.processed_data = processed_data
        self.time_limit = time_limit
        self.preset = preset
        self.load = load
        self._build_predictor(verbosity=verbosity, **kwargs)

    def _build_predictor(self, verbosity: int, **kwargs):
        """Build the predictor using ProcessedData."""
        print(">>> building predictor...")
        model_type = self.processed_data.get("model_type", "TabularPredictor")
        problem_type = pt_val if (pt_val := self.processed_data.get("problem_type", "auto")) != "auto" else None
        eval_metric = em_val if (em_val := self.processed_data.get("eval_metric", "auto")) != "auto" else None
        model_path = self.processed_data["path"] / MODEL_DIR

        base_config = {
            "label": self.processed_data["target"],
            "problem_type": problem_type,
            "eval_metric": eval_metric,
            "verbosity": verbosity,
            "path": model_path,
        }
        self._base_config = base_config
        if self.load:
            if self.retrieve_model():
                return
        match model_type:
            case "TabularPredictor":
                self.model = TabularPredictor(**base_config, **kwargs)
            case "MultiModalPredictor":
                # not supported yet
                self.model = MultiModalPredictor(**base_config, **kwargs)
            case "TimeSeriesPredictor":
                # not supported yet
                raise NotImplementedError("TimeSeriesPredictor is not implemented yet")
            case _:
                self.model = TabularPredictor(**base_config)
        print(">>> predictor built")

    def _update_problem_type(self) -> None:
        """Update the problem type in the processed data."""
        if self.processed_data.get("problem_type") == "auto":
            print(f"Problem type is auto, setting to {self.model.problem_type}")
            self.processed_data["problem_type"] = self.model.problem_type

    def train_predictor(self, **kwargs):
        """Train the predictor on the processed data."""
        config = PRESET_CONFIGS.get(self.preset, PRESET_CONFIGS["medium"]).copy()
        if self.time_limit is not None:
            config["time_limit"] = self.time_limit
        for key, value in kwargs.items():
            if key in config:
                warnings.warn(f"Overriding preset config '{key}={config[key]}' with '{key}={value}'", UserWarning)
            config[key] = value
        self.model = self.model.fit(self.processed_data["train"].to_pandas(), **config)
        self._update_problem_type()

    def predict(self, data_selection: Literal["test", "train"] = "test", **kwargs) -> pl.DataFrame | pl.Series:
        """Get predictions from the model.

        Args:
            data_selection (str, optional): specify which dataset to use for predictions. Defaults to "test".
            **kwargs: additional arguments to pass to the model's predict method.

        Returns:
            pl.DataFrame | pl.Series: Predictions from the model.
        """
        predictions: pl.Series = pl.from_pandas(
            self.model.predict(self.processed_data[data_selection].to_pandas(), **kwargs)
        )
        # cast predictions to same data type as target column
        predictions = predictions.cast(self.processed_data[data_selection][self.processed_data["target"]].dtype)
        if not hasattr(self, "auto_ml_data"):
            self.auto_ml_data = AutoMLData(**self._filtered_processed_data())
        if data_selection == "test":
            self.auto_ml_data["predictions"] = predictions
        return predictions

    def predict_proba(self, data_selection: Literal["test", "train"] = "test", **kwargs) -> Optional[pl.DataFrame]:
        """Get prediction probabilities from the model.

        Args:
            data_selection (str, optional): specify which dataset to use for predictions. Defaults to "test".
            **kwargs: additional arguments to pass to the model's predict_proba method.

        Returns:
            pl.DataFrame: Prediction probabilities from the model.
        """
        if self.model.problem_type in ["binary", "multiclass"]:
            prediction_probs = pl.from_pandas(
                self.model.predict_proba(self.processed_data[data_selection].to_pandas(), **kwargs)
            )
            if not hasattr(self, "auto_ml_data"):
                self.auto_ml_data = AutoMLData(**self._filtered_processed_data())
            if data_selection == "test":
                self.auto_ml_data["prediction_probs"] = prediction_probs
            return prediction_probs
        else:
            warnings.warn(f"Problem type {self.model.problem_type} does not support predict_proba", UserWarning)
            return None

    def _filtered_processed_data(self) -> Dict[str, Any]:
        """Filter out 'model_type' from processed_data."""
        return {k: v for k, v in self.processed_data.items() if k != "model_type"}

    def retrieve_model(self) -> bool:
        """Load the model from the processed data path."""
        model_type = self.processed_data.get("model_type", "TabularPredictor")
        model_path = self.processed_data.get("path", Path("model")) / MODEL_DIR
        try:
            match model_type:
                case "TabularPredictor":
                    self.model = TabularPredictor(**self._base_config).load(model_path)
                case "MultiModalPredictor":
                    self.model = MultiModalPredictor(**self._base_config).load(model_path)
                case _:
                    self.model = TabularPredictor(**self._base_config).load(model_path)
            self._update_problem_type()
            return True
        except Exception as e:
            self.load = False
            warnings.warn(f"Error loading model from path {str(model_path)}: {str(e)}", UserWarning)
            return False

    def transform_features(self):
        """Transform features into new feature space and retain target column."""
        self.auto_ml_data["tranformed_train"] = pl.from_pandas(
            self.model.transform_features(self.processed_data["train"].to_pandas())
        ).with_columns(self.processed_data["train"].select(self.processed_data["target"]))
        self.auto_ml_data["transformed_test"] = pl.from_pandas(
            self.model.transform_features(self.processed_data["test"].to_pandas())
        ).with_columns(self.processed_data["test"].select(self.processed_data["target"]))

    def run_auto_ml(self):
        """Run the AutoML pipeline."""
        try:
            if not self.load:
                print(">>> training the model...")
                self.train_predictor()
            print(">>> making predictions...")
            self.predict()
            if self.model.problem_type not in ["regression", "quantile"]:
                print(">>> predicting probabilities...")
                self.predict_proba()
            print(">>> transforming features...")
            self.transform_features()
        except Exception as e:
            raise Exception(f"Error running AutoML pipeline: {str(e)}") from e

    def get_feature_importance(self, model=None, subsample_size=5000, num_top_features=20, **kwargs) -> pl.DataFrame:
        """Get and print feature importance from the trained model.

        Args:
            model (str, optional): Name of model to get feature importance from.
                If None, gets feature importance from the best model. Defaults to None.
            subsample_size (int, optional): Size of data subsample to use for feature importance calculation.
                Larger values give more accurate results but take longer. Defaults to 5000.
            num_top_features (int, optional): Number of top features to print. Defaults to 20.
            **kwargs: Additional arguments to pass to model.get_feature_importance()

        Returns:
            pl.DataFrame: Feature importance DataFrame with columns ['feature', 'importance']
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        print(f">>> calculating feature importance (subsample_size={subsample_size})...")

        try:
            importance_df = pl.from_pandas(
                self.model.get_feature_importance(model=model, subsample_size=subsample_size, **kwargs)
            )

            importance_df = importance_df.sort("importance", descending=True)

            print(f"\nTop {min(num_top_features, len(importance_df))} important features:")
            print("-" * 50)
            return importance_df

        except Exception as e:
            warnings.warn(f"Error calculating feature importance: {str(e)}", UserWarning)

            return pl.DataFrame({"feature": [], "importance": []})
