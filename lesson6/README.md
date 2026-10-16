# Lesson 6 Python Project

This project accompanies `lesson_6_interactive_rl_notebook.ipynb` and includes
the core data science dependencies used by the lesson:

- pandas
- numpy
- matplotlib
- scikit-learn
- streamlit
- tensorflow
- streamlit

## Setup

Create and activate a virtual environment, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

The dependencies can also be installed directly with:

```powershell
python -m pip install -r requirements.txt
```

## Run the neural-network example

The notebook's purchase-intent example is also available as a command-line
program. Install the project dependencies, then run:

```powershell
python main.py
```

Training runs for 60 epochs by default and displays a training/validation loss
plot. To change the number of epochs or skip the plot:

```powershell
python main.py --epochs 30 --no-plot
```

## Verify imports

```powershell
python -m lesson6.dependencies
```
