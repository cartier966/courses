# lesson3

A small Python project using NumPy and pandas.

## Practice datasets

The CSV files are synthetic and intentionally messy so they can be cleaned before training a model:

- `customer_churn.csv`: predict `churned`. Practice missing values, invalid numbers, inconsistent text casing, and duplicate rows.
- `house_prices.csv`: predict `price`. Practice currency strings, missing values, inconsistent neighborhood names, and invalid years.
- `student_performance.csv`: predict `passed` or `final_score`. Practice percentages, words in numeric columns, missing values, out-of-range values, and duplicate rows.

Useful cleaning steps include `read_csv()`, `head()`, `info()`, `isna()`, `drop_duplicates()`, `fillna()`, `astype()`, `to_numeric()`, and `str.strip()`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```
