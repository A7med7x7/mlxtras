# mlxtras

[![CI](https://img.shields.io/github/actions/workflow/status/A7med7x7/mlxtras/ci.yml?branch=main&label=CI)](https://github.com/A7med7x7/mlxtras/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlxtras.svg)](https://pypi.org/project/mlxtras/)

Mlxtras is a Python utility library with helpers for data processing, model validation, and MLflow experiment tracking. It collects recurring boilerplate across pandas, scikit-learn, and LightGBM workflows into small functions.

### How it started

This project started in 2023 when my mentor suggested packaging custom functions to learn Python library design. I shaped it around utilities I kept rewriting across projects: shrinking DataFrame memory usage, validating baseline models across folds, and capturing environment details in MLflow runs.

## Installation

```bash
pip install mlxtras
```

> [!TIP]
> To log NVIDIA GPU metrics with MLflow, install the optional GPU extra:
> ```bash
> pip install "mlxtras[gpu]"
> ```

## Usage

### Downcast DataFrame memory

Downcast numeric columns to smaller representations that fit the value ranges:

```python
import pandas as pd
from mlxtras import reduce_memory_usage

df = pd.read_csv("dataset.csv")
df = reduce_memory_usage(df, convert_object_to_category=True)
```

### Log run environment to MLflow

Track hardware, installed packages, and git commit details in an active MLflow run:

```python
import mlflow
from mlxtras.mlflow import log_gpu, log_python, log_git

with mlflow.start_run():
    log_gpu()     # Detects NVIDIA or AMD hardware and logs driver and device info
    log_python()  # Logs Python version, system info, and pip freeze
    log_git()     # Logs commit hash, current branch, and dirty status
```

### Validate baseline models

Evaluate models across folds with cross-validation:

```python
from lightgbm import LGBMClassifier
from mlxtras.feature_selection import validate_model

model = LGBMClassifier()
results = validate_model(model, X, y, metric="accuracy", folds=5)
```

## Acknowledgments

* Inspired by machine learning work at [AirQo](https://airqo.net).
* Built on top of pandas, scikit-learn, and NumPy.
