# Lesson 12: Multimodal AI and CLIP

This project contains the interactive notebook
[`multimodal_ai_clip_interactive_notebook.ipynb`](./multimodal_ai_clip_interactive_notebook.ipynb)
and the packages needed to run it and its demonstrated examples.

## Set up on Windows

The notebook metadata specifies Python 3.11. From this folder, create and activate
a virtual environment, then install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open the notebook in VS Code or launch Jupyter:

```powershell
python -m notebook multimodal_ai_clip_interactive_notebook.ipynb
```

If prompted, select the Python interpreter at `.venv\Scripts\python.exe`.

## Run the Streamlit app

The app turns the notebook's code examples into interactive lesson sections:

```powershell
python -m streamlit run app.py
```

The simulated vector search, token counting, and cost examples run without an
API key.

## Optional OpenAI example

The app's sample embedding request is optional. To enable it, create
`.streamlit\secrets.toml` with your own key:

```toml
OPENAI_API_KEY = "your-api-key"
```

That secrets file is ignored by Git. Never commit a real API key.
