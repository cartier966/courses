# Lesson 10: NLP Fundamentals & Tokenization

A Streamlit app based on the NLP tokenization lesson notebook. It demonstrates
building a vocabulary, converting words to token IDs, handling unknown words,
padding batches, comparing tokenization levels, and estimating token counts.

## Setup

Create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m streamlit run app.py
```

The exact `tiktoken` example is optional and is not part of the pinned
dependencies. The app remains usable without it.
