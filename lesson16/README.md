# Lesson 16: Embeddings, ChromaDB, and Chat Memory

This project contains a Streamlit RAG lab and its companion Jupyter notebook.

## Setup

Create and activate a virtual environment from this directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` with your OpenAI API key:

```toml
OPENAI_API_KEY = "your-key-here"
```

## Run the app

```powershell
streamlit run app.py
```

The notebook uses the same project dependencies. Select this project's `.venv`
as its kernel in VS Code before running notebook cells.
