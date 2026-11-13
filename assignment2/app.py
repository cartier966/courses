"""Draw a digit and classify it with a trained MNIST convolutional network."""

import base64
import binascii
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

MODEL_PATH = Path(__file__).with_name("mnist_cnn.keras")

DRAWING_CANVAS = st.components.v2.component(
    "digit_drawing_canvas",
    html="""
        <div class="drawing">
            <canvas width="280" height="280" role="img"
                aria-label="Drawing area. Draw a digit with a mouse, pen, or touch."></canvas>
            <button type="button">Clear drawing</button>
        </div>
    """,
    css="""
        .drawing { display: flex; flex-direction: column; align-items: flex-start; gap: 12px; }
        canvas { width: min(280px, 100%); height: auto; aspect-ratio: 1;
            background: #000; border: 2px solid #777; border-radius: 8px;
            touch-action: none; cursor: crosshair; }
        button { padding: 8px 14px; cursor: pointer; border: 1px solid #777;
            border-radius: 6px; background: #fff; color: #111; }
        button:focus-visible { outline: 3px solid #ff4b4b; outline-offset: 2px; }
    """,
    js="""
        export default function ({parentElement, data, setStateValue}) {
            const canvas = parentElement.querySelector("canvas");
            const button = parentElement.querySelector("button");
            const ctx = canvas.getContext("2d");
            const image = data.image || "";

            if (canvas.dataset.image !== image) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (image) {
                    const saved = new Image();
                    saved.onload = () => ctx.drawImage(saved, 0, 0);
                    saved.src = image;
                }
                canvas.dataset.image = image;
            }

            function position(event) {
                const rect = canvas.getBoundingClientRect();
                return {
                    x: (event.clientX - rect.left) * canvas.width / rect.width,
                    y: (event.clientY - rect.top) * canvas.height / rect.height
                };
            }
            canvas.onpointerdown = (event) => {
                event.preventDefault();
                canvas.setPointerCapture(event.pointerId);
                const point = position(event);
                ctx.strokeStyle = "white";
                ctx.fillStyle = "white";
                ctx.lineWidth = data.strokeWidth;
                ctx.lineCap = "round";
                ctx.lineJoin = "round";
                ctx.beginPath();
                ctx.arc(point.x, point.y, ctx.lineWidth / 2, 0, 2 * Math.PI);
                ctx.fill();
                ctx.beginPath();
                ctx.moveTo(point.x, point.y);
            };
            canvas.onpointermove = (event) => {
                if (!canvas.hasPointerCapture(event.pointerId)) return;
                const point = position(event);
                ctx.lineTo(point.x, point.y);
                ctx.stroke();
            };
            canvas.onpointerup = (event) => {
                if (!canvas.hasPointerCapture(event.pointerId)) return;
                canvas.releasePointerCapture(event.pointerId);
                const value = canvas.toDataURL("image/png");
                canvas.dataset.image = value;
                setStateValue("image", value);
            };
            canvas.onpointercancel = canvas.onpointerup;
            button.onclick = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                canvas.dataset.image = "";
                setStateValue("image", "");
            };
        }
    """,
)


def preprocess_image(data_url: str) -> np.ndarray:
    """Turn a transparent canvas PNG into a centered, normalized MNIST image."""
    header, encoded = data_url.split(",", 1)
    if header != "data:image/png;base64":
        raise ValueError("The drawing must be a PNG image.")
    image_bytes = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGBA")
        background = Image.new("RGBA", image.size, "black")
        image = Image.alpha_composite(background, image).convert("L")

    bounding_box = image.getbbox()
    if bounding_box is None:
        return np.zeros((28, 28), dtype=np.float32)

    digit = image.crop(bounding_box)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    image = Image.new("L", (28, 28))
    image.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return np.asarray(image, dtype=np.float32) / 255.0


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)


def main() -> None:
    st.set_page_config(page_title="Digit recognizer")
    st.title("Handwritten digit recognizer")
    st.write(
        "Draw a digit from 0 to 9. A trained MNIST model predicts after each "
        "stroke and shows its confidence for every digit."
    )

    stroke_width = st.slider("Pen thickness", 5, 40, 20, help="Change the width of new strokes.")
    state = st.session_state.get("digit_canvas", {})
    drawing = DRAWING_CANVAS(
        key="digit_canvas",
        data={"image": state.get("image", ""), "strokeWidth": stroke_width},
        on_image_change=lambda: None,
    )

    if not drawing.image:
        st.info("Draw a digit above to see a prediction.")
        return

    try:
        pixels = preprocess_image(drawing.image)
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as exc:
        st.error(f"Could not read the drawing: {exc}")
        return

    if not np.any(pixels):
        st.info("Draw a digit above to see a prediction.")
        return

    try:
        model = load_model()
    except (OSError, ValueError) as exc:
        st.error(f"Could not load the trained model at {MODEL_PATH}: {exc}")
        return

    probabilities = np.asarray(model(pixels.reshape(1, 28, 28, 1), training=False))[0]
    predicted_digit = int(np.argmax(probabilities))

    st.subheader("Prediction")
    left, right = st.columns(2)
    left.metric("Predicted digit", str(predicted_digit))
    right.metric("Confidence", f"{probabilities[predicted_digit]:.1%}")
    st.caption("Confidence is the model's probability, not a guarantee of correctness.")

    st.subheader("Probability for each digit")
    st.bar_chart(
        pd.DataFrame({"Probability": probabilities}, index=pd.Index(range(10), name="Digit")),
        alt="Bar chart showing the model's probability for each digit from 0 to 9",
    )

    st.subheader("Processed image")
    st.image(
        (pixels * 255).astype(np.uint8),
        width=196,
        caption="28 × 28 pixels seen by the model",
        alt="Preprocessed handwritten digit, white on black",
    )
    st.caption("The drawing is cropped, centered, resized, and normalized before prediction.")


if __name__ == "__main__":
    main()
