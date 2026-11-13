# Handwritten digit recognizer

Draw a digit on the canvas and a trained convolutional neural network predicts
which digit it is. Predictions update after each stroke. Use **Pen thickness**
to change the width of new strokes and **Clear drawing** to start over.

## Setup and run

Python 3.13 was used for testing. From this folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

Keep `mnist_cnn.keras` beside `app.py`. It is a copy of the trained MNIST CNN
from the repository's Lesson 9; inference does not download the dataset or
train a model. Installing the dependencies requires internet access.

## How it works

The transparent canvas is composited onto black and converted to grayscale.
The nonempty drawing is cropped, scaled to fit within 20 x 20 pixels, centered
in a 28 x 28 image, and normalized to values between 0 and 1. The model receives
an array shaped `(1, 28, 28, 1)` and returns probabilities for digits 0-9.
The app shows the top prediction, its confidence, the complete probability
chart, and the processed image. A blank canvas is not classified.

## Drawing tests and reflection

These are individual browser tests, not an accuracy benchmark. The first four
drawings were vertical strokes intended as 1; the last was an ambiguous scribble.
Results may differ with your handwriting.

| Drawing style | Observed result |
| --- | --- |
| Centered, normal pen (20) | Predicted 1, 99.9% confidence. |
| Near the upper-left corner, normal pen | Predicted 1, 100.0% displayed confidence; cropping centered the stroke. |
| Thin, pen width 5 | Predicted 1, 99.9% confidence. |
| Thick, pen width 40 | Predicted 1, 99.2% confidence. |
| Messy, unclear strokes | Predicted 7, 58.1%; 3 received about 40.9%. |

Centering reduces sensitivity to where a digit is drawn, but stroke thickness
and shape still affect the input. The model must choose one of ten digits even
for an ambiguous scribble; confidence is not a guarantee of correctness.
The probability chart makes competing guesses visible, while the processed
image helps explain what the model received.

See the [app screenshot](screenshot.png), stitched from consecutive browser
viewport captures of the same drawing.

## Automated tests

After setup, run:

```powershell
python -m unittest discover -s tests -v
```

These tests check blank input, centering and normalization, invalid image input,
and the saved model's ten output probabilities.
