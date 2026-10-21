# Lesson 9

A Streamlit app that turns the interactive notebook's NumPy MNIST classifier into a
runnable training and evaluation workflow. The app includes dense-layer
backpropagation, accuracy and confusion-matrix reporting, sample inference, a
convolution-gradient demonstration, and saving/loading model weights as `.npy` files.

## Setup

Create and activate a virtual environment in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Streamlit app

```powershell
python -m streamlit run app.py
```

The first training or inference action downloads the MNIST dataset through
TensorFlow/Keras. Training uses the subset and settings selected in the app's sidebar;
the trained weights are saved in `lesson9_numpy_weights`.
