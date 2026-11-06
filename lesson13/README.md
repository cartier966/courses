# Lessons 11–13: Live Embeddings and Semantic Search

A Streamlit teaching app for generating text embeddings, comparing semantic
similarity, storing records in ChromaDB, and searching them by meaning.

## Requirements

- Python 3.10 or newer
- An OpenAI API key for live embedding features

## Setup on Windows

From this folder, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the Streamlit secrets file and add your API key:

```powershell
New-Item -ItemType Directory -Force .streamlit
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit\secrets.toml`:

```toml
OPENAI_API_KEY = "your_key_here"
```

Start the app:

```powershell
streamlit run app.py
```

The app stores its local ChromaDB data in `chroma_db\`. The interactive
notebook is also included as `lesson11_13_live_embeddings_interactive_notebook.ipynb`.
The HTML guides and ZIP archive are lesson materials and are not needed to run
the app.

Never commit a real API key or `.streamlit\secrets.toml`.
