from src.data_collection.column_selection import get_dataset
from src.dataprofiling.data_summary import DataSummary
from src.auto_ml.model import AutoMLModel
from src.model_assessment.fairness import FairnessAssessor
from src.data_preprocessing.preprocessing import autoML_prep, feature_generation

__all__ = ["get_dataset", "DataSummary", "AutoMLModel", "FairnessAssessor", "autoML_prep", "feature_generation"]

__version__ = "1.0.0"