# Data science starter

A small Streamlit app using NumPy, Pandas, and scikit-learn to generate data and fit a linear regression model.

## Set up and run

In PowerShell, create and activate a virtual environment, install the pinned dependencies, and start Streamlit:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

If PowerShell blocks environment activation, run Streamlit through the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Change the sample count and noise level in the sidebar to explore the model's test score and predictions.
