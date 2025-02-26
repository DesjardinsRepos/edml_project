import sys
from pathlib import Path
from typing import Optional

import polars as pl
from sliceline.slicefinder import Slicefinder

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.custom_data_types import ModelAssessmentData


class Slicer:
    sf: Slicefinder

    def __init__(self, data: ModelAssessmentData):
        self.data = data

    def _compute_log_loss(self, probs: pl.Series, true_labels: pl.Series, eps: float = 1e-15) -> pl.Series:
        clipped_probs = probs.clip(eps, 1 - eps)
        return -(true_labels * clipped_probs.log() + (1 - true_labels) * (1 - clipped_probs).log())

    def _compute_rmse(self, preds: pl.Series, true_labels: pl.Series) -> pl.Series:
        return ((preds - true_labels).pow(2)).sqrt()

    def find_slices(
        self,
        eval_metric: Optional[str] = None,  # for now only for regression use logloss for classification
        data_selection: str = "binned_test",
        max_l: Optional[int] = None,
        alpha: float = 0.95,
        k: int = 1,
        min_support: int = 1,
        verbose: bool = False,
    ):
        if max_l is None:
            max_l = self.data[data_selection].select(pl.exclude(self.data["target"])).shape[1]
        self.sf = Slicefinder(alpha=alpha, k=k, max_l=max_l, min_sup=min_support, verbose=verbose)
        if self.data["problem_type"] == "binary":
            self.sf.fit(
                X=self.data[data_selection].select(pl.exclude(self.data["target"])),
                errors=self._compute_log_loss(
                    probs=self.data["prediction_probs"]["1"], true_labels=self.data["test"][self.data["target"]]
                ),
            )

    def get_top_slices(self):
        return pl.DataFrame(
            {col: values for col, values in zip(self.sf.feature_names_in_, self.sf.top_slices_.T)}
        ).with_row_index()
