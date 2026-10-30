# Lesson 11: Embeddings and Vector Databases

This project contains the interactive notebook
`embeddings_vector_databases_interactive_notebook.ipynb`.

## Requirements

- Python 3.11 or newer
- The packages listed in `requirements.txt`

## Setup on Windows

From this folder, create and activate a virtual environment, then install the
dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start JupyterLab and open the notebook:

```powershell
python -m jupyter lab embeddings_vector_databases_interactive_notebook.ipynb
```

Choose the Python 3 kernel associated with this environment if prompted. ChromaDB
stores its local database in `chroma_db/`; its default embedding model may be
downloaded the first time a text query runs.

## Run as a Python script

`app.py` contains all executable code cells from the notebook in their original
order. Run it from this folder:

```powershell
python app.py
```

The script may open Matplotlib windows. ChromaDB's default embedding model may
also need to download the first time the script runs a text query.
