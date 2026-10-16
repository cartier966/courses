# Lesson 5 - Machine Learning with scikit-learn

A small Python project scaffold for exploring data analysis and machine
learning using `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`,
and `tensorflow-recommenders`.

## Project structure

```
lesson5/
├── src/
│   └── main.py                         # demo script importing all required packages
├── lesson_5_interactive_notebook.ipynb # teaching notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

A virtual environment has already been created at `.venv`. To use it:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the demo script

```powershell
.\.venv\Scripts\python.exe src\main.py
```

The script loads the scikit-learn breast cancer dataset, explores it with
pandas/numpy, trains a `RandomForestClassifier`, evaluates it, and saves a
chart with matplotlib to `output/feature_importances.png`.

## Using TensorFlow Recommenders

`tensorflow-recommenders` currently only supports the legacy Keras 2 backend,
so `TF_USE_LEGACY_KERAS=1` must be set **before** `tensorflow` is imported.
The notebook's setup cell already does this with:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_recommenders as tfrs
```

If running a script from the terminal instead, set the environment variable
first:

```powershell
$env:TF_USE_LEGACY_KERAS = "1"
.\.venv\Scripts\python.exe your_script.py
```
