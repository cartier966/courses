# Lesson 14: LLM Prompt Engineering and AI Ethics

This project contains an interactive Jupyter notebook covering LLM behavior,
prompt engineering, AI ethics, prompt injection, and grounded RAG.

## Set up on Windows

From this folder, create a virtual environment and install the project
requirements:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name lesson14 --display-name "Python (lesson14)"
```

Start Jupyter Lab with the project environment:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

Open `llm_prompt_engineering_ai_ethics_interactive_notebook.ipynb` and run the
cells from top to bottom. The examples are self-contained and do not require an
API key.
