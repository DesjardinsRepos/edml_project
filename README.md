# Automated Data Profiling & Issue Correction Library

Ensuring high-quality data is essential for building reliable and fair machine learning models. 
In our project, we designed and implemented an easy-to-use library for data profiling and quality assessment. 
This tool enables users to quickly gain insights into datasets from OpenML, Kaggle, and Hugging Face.

> [!NOTE]
> This project is part of the Engineering Data for Machine Learning course at **Technical University Berlin**.
> The project is expected to be submitted on **26.02.2025**. After submission, it will **no longer be actively maintained or updated**.

## Key Features
✅ Profile datasets and detect issues fullly automatically <br>
✅ Automatically resolve data quality issues <br>
✅ Integrated fairness analysis & performance evaluation using AutoGluon

## Usage
```python
from edml_project import get_dataset, autoML_prep, DataSummary, AutoMLModel, FairnessAssessor

# Load dataset
data = get_dataset("https://www.kaggle.com/datasets/yasserh/titanic-dataset")

# Profile data & print a summary
summary = DataSummary(data)
summary.create_summary()
profiled_data = summary.export()
print(summary.quick_summary()["string_output"])

# AutoML on original dataset (baseline)
processed_data = autoML_prep(data)
baseline_model = AutoMLModel(processed_data, time_limit=300, preset="good", load=False)
baseline_model.run_auto_ml()
baseline_automl_data = baseline_model.auto_ml_data

# Apply data quality fixes
cleaned_data = summary.solve_issues(return_results=True)
for split in ["train", "val", "test"]:
    if cleaned_data.get(split) is not None:
        data[split] = cleaned_data[split]

# AutoML with improved data
cleaned_processed_data = autoML_prep(data) 
enhanced_model = AutoMLModel(cleaned_processed_data, time_limit=300, preset="good", load=False)
enhanced_model.run_auto_ml()
enhanced_automl_data = enhanced_model.auto_ml_data

# Fairness analysis of original dataset
fairness_baseline = FairnessAssessor(baseline_automl_data, profiled_data)
fairness_baseline.analyze_all_categorical()
# fairness_baseline.analyze_all_sens()
baseline_fairness_metrics = fairness_baseline.get_all_metrics()

# Fairness analysis with improved data
fairness_enhanced = FairnessAssessor(enhanced_automl_data, profiled_data)
fairness_enhanced.analyze_all_categorical()
# fairness_enhanced.analyze_all_sens()
enhanced_fairness_metrics = fairness_enhanced.get_all_metrics()
```

## License
This project is licensed under the MIT License.
