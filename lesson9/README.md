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

## Handwritten digit recognizer

Run the drawing app from this directory:

```powershell
python -m streamlit run digiter.py
```

Keep `lesson9_saved_mnist_cnn.keras` next to `digiter.py`. The app uses that
trained model directly; it does not download MNIST or train a new model.
Draw a digit with a mouse, pen, or touch, adjust pen thickness if needed, and
use **Clear drawing** to start again. Predictions update after each stroke.

The app converts the white-on-black drawing to grayscale, crops and centers the
strokes in a 28 × 28 image, and normalizes pixels to 0–1. It displays the
processed image, the highest-probability digit, its confidence, and all ten
probabilities. The model was trained on MNIST, so unusual handwriting may be
misclassified even when its confidence is high.

### Drawing tests and reflection

These are individual browser tests, not an accuracy benchmark. The first four
drawings were intended as the digit 1; the last was an ambiguous scribble.

| Drawing style | Result |
| --- | --- |
| Centered, normal pen | Predicted 1 with 99.6% confidence. |
| Near the corner | Predicted 1 with 69.2% confidence; cropping brought it to the center. |
| Thin, pen width 5 | Predicted 1 with 90.8% confidence. |
| Thick, pen width 40 | Predicted 1 with 87.6% confidence. |
| Messy, unclear strokes | Predicted 7 with 71.7% confidence despite the ambiguous input. |

Centering helps the model handle placement, but stroke thickness and shape
still affect its predictions. The unclear example illustrates that the highest
probability is not proof that the drawing is a recognizable digit.

See [the running app screenshot](digit-recognizer-screenshot.png), stitched
from consecutive viewport captures. It shows an example of a misclassification,
not the centered test above. Run the preprocessing and model checks with:

```powershell
python -m unittest test_digiter
```
