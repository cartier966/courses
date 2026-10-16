import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
except Exception:
    st.error("TensorFlow is required. Install it with: pip install tensorflow")
    st.stop()

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st.error("streamlit-drawable-canvas is required. Install it with: pip install streamlit-drawable-canvas")
    st.stop()


MODEL_PATH = "lesson8_mnist_cnn.keras"

st.set_page_config(page_title="Lesson 8 MNIST CNN Classifier", layout="wide")

st.title("Lesson 8: CNN MNIST Digit Classifier")
st.write("Draw a digit from 0 to 9. This app uses a CNN trained on MNIST.")


@st.cache_resource
def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        return model, "Loaded saved model."

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    model.save(MODEL_PATH)

    return model, f"Trained new model. Test accuracy: {test_accuracy:.2%}"


def preprocess_canvas_image(canvas_image):
    image = Image.fromarray(canvas_image.astype("uint8"), mode="RGBA")
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    image_array = np.array(image) / 255.0
    model_input = image_array.reshape(1, 28, 28, 1)

    return image, image_array, model_input

with st.spinner("Loading or training the CNN model..."):
    model, status_message = train_or_load_model()

st.success(status_message)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("Draw a Digit")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="lesson8_canvas",
        return_image_data=True
    )

    st.caption("Tip: Draw one large digit in the center of the box.")

with right_col:
    st.header("Prediction")

    if canvas_result.image_data is not None:
        processed_pil, processed_array, model_input = preprocess_canvas_image(canvas_result.image_data)

        prediction = model.predict(model_input, verbose=0)[0]
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        st.subheader(f"Predicted Digit: {predicted_digit}")
        st.metric("Confidence", f"{confidence:.2%}")

        st.write("Model confidence by digit:")
        st.bar_chart(prediction)

        st.write("Processed 28×28 image sent to the CNN:")
        st.image(processed_pil.resize((140, 140)), caption="28×28 processed image", width=140)

        with st.expander("View raw 28×28 normalized pixel matrix"):
            st.write(np.round(processed_array, 2))
    else:
        st.info("Draw a digit to get a prediction.")

st.divider()
st.header("How This App Connects to Lesson 8")

st.write(
    """
    Lesson 8 workflow:

    1. User draws a digit.
    2. Drawing becomes a grayscale image.
    3. Image is resized to 28×28.
    4. Pixel values are normalized between 0 and 1.
    5. CNN predicts probabilities for digits 0–9.
    6. The highest probability becomes the predicted digit.
    """
)

st.code(
    """
Drawing
↓
Grayscale
↓
Resize to 28×28
↓
Normalize
↓
CNN Prediction
↓
Digit + Confidence
    """,
    language="text"
)

st.subheader("Student Testing Ideas")
st.write(
    """
    Try:
    - Drawing in the center
    - Drawing in a corner
    - Thin strokes
    - Thick strokes
    - A messy digit that could look like two numbers
    """
)