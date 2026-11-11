# Interactive RAG Learning Lab

Streamlit app walking through load → clean → chunk → embed → retrieve → augment → generate.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit\secrets.toml` and set your `OPENAI_API_KEY`.

## Run

```powershell
streamlit run app.py
```

A simpler version (no OpenAI needed; requires only `streamlit` and `pandas`) is in
`rag_cleaning_chunking_visualizer/`.
