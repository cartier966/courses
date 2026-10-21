"""
Lesson 9 Streamlit App
MNIST Digit Classifier with Model Persistence and Inference Focus

How to run:
1. pip install streamlit tensorflow numpy pillow streamlit-drawable-canvas scikit-learn
2. streamlit run lesson9_mnist_persistence_streamlit_app.py

Focus:
- Training vs inference
- Test accuracy
- Confusion matrix
- Misclassification analysis
- Saving/loading trained model and .npy weights
"""

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

try:
    from sklearn.metrics import confusion_matrix
except Exception:
    confusion_matrix = None


MODEL_PATH = "lesson9_saved_mnist_cnn.keras"
WEIGHTS_DIR = "lesson9_saved_weights"

st.set_page_config(page_title="Lesson 9 MNIST Inference and Persistence", layout="wide")

st.title("Lesson 9: CNN Inference, Evaluation, and Weight Persistence")
st.write(
    "This app focuses on using a trained model for inference, evaluating performance, and saving/loading learned parameters."
)


def export_layer_weights_to_numpy(model, folder=WEIGHTS_DIR):
    os.makedirs(folder, exist_ok=True)
    saved_files = []

    for layer_index, layer in enumerate(model.layers):
        weights = layer.get_weights()

        if len(weights) == 0:
            continue

        for weight_index, weight_array in enumerate(weights):
            filename = f"layer_{layer_index}_param_{weight_index}.npy"
            path = os.path.join(folder, filename)
            np.save(path, weight_array)
            saved_files.append(path)

    return saved_files


@st.cache_resource
def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        return model, "Loaded saved trained model for inference."

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1), name="conv_1"),
        keras.layers.MaxPooling2D((2, 2), name="pool_1"),
        keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv_2"),
        keras.layers.MaxPooling2D((2, 2), name="pool_2"),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(10, activation="softmax", name="output")
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

    model.save(MODEL_PATH)
    export_layer_weights_to_numpy(model)

    return model, "Trained new model and saved model + .npy weights."


@st.cache_data
def load_test_data(limit=1000):
    (_, _), (X_test, y_test) = mnist.load_data()

    X_test_norm = X_test / 255.0
    X_test_cnn = X_test_norm.reshape(-1, 28, 28, 1)

    return X_test[:limit], X_test_cnn[:limit], y_test[:limit]


def preprocess_canvas_image(canvas_image):
    image = Image.fromarray(canvas_image.astype("uint8"), mode="RGBA")
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    image_array = np.array(image) / 255.0
    model_input = image_array.reshape(1, 28, 28, 1)

    return image, image_array, model_input


def evaluate_model(model):
    X_test_raw, X_test_cnn, y_test = load_test_data(limit=1000)

    probabilities = model.predict(X_test_cnn, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = np.mean(y_pred == y_test)

    return X_test_raw, X_test_cnn, y_test, y_pred, probabilities, accuracy


with st.spinner("Loading trained model or training a new one..."):
    model, status_message = train_or_load_model()

st.success(status_message)

tab1, tab2, tab3, tab4 = st.tabs([
    "Live Inference",
    "Model Evaluation",
    "Saved Weights",
    "Lesson 9 Explanation"
])

with tab1:
    st.header("Live Digit Classification")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=18,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="lesson9_canvas"
        )

        st.caption("Draw one digit from 0 to 9.")

    with right_col:
        if canvas_result.image_data is not None:
            processed_pil, processed_array, model_input = preprocess_canvas_image(canvas_result.image_data)

            prediction = model.predict(model_input, verbose=0)[0]
            predicted_digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.subheader(f"Predicted Digit: {predicted_digit}")
            st.metric("Confidence", f"{confidence:.2%}")
            st.bar_chart(prediction)

            st.image(processed_pil.resize((140, 140)), caption="Processed 28×28 image", width=140)

            with st.expander("View normalized pixel matrix"):
                st.write(np.round(processed_array, 2))
        else:
            st.info("Draw a digit to run inference.")

with tab2:
    st.header("Model Evaluation on Unseen Test Data")

    if st.button("Run Evaluation"):
        X_test_raw, X_test_cnn, y_test, y_pred, probabilities, accuracy = evaluate_model(model)

        st.metric("Test Accuracy on 1000 MNIST Test Images", f"{accuracy:.2%}")

        if confusion_matrix is not None:
            cm = confusion_matrix(y_test, y_pred)

            st.subheader("Confusion Matrix")
            st.write("Rows = actual digit. Columns = predicted digit.")
            st.dataframe(cm)

            pairs = []

            for actual in range(10):
                for predicted in range(10):
                    if actual != predicted:
                        count = cm[actual, predicted]
                        pairs.append((count, actual, predicted))

            pairs = sorted(pairs, reverse=True)

            st.subheader("Top 3 Misclassification Pairs")
            for count, actual, predicted in pairs[:3]:
                st.write(f"Actual {actual} predicted as {predicted}: {count} times")
        else:
            st.warning("Install scikit-learn to view confusion matrix: pip install scikit-learn")

        st.subheader("Example Misclassified Image")

        misclassified = np.where(y_pred != y_test)[0]

        if len(misclassified) > 0:
            index = int(misclassified[0])
            st.image(X_test_raw[index], width=140)
            st.write(f"Actual: {y_test[index]}")
            st.write(f"Predicted: {y_pred[index]}")
            st.write(f"Confidence: {np.max(probabilities[index]):.2%}")
        else:
            st.success("No misclassifications found in this subset.")

with tab3:
    st.header("Weight Persistence")

    st.write("Lesson 9 focuses on saving trained parameters so the app can use the model without retraining.")

    st.subheader("Saved Keras Model")
    st.code(MODEL_PATH, language="text")

    if st.button("Export Layer Weights as .npy Files"):
        saved_files = export_layer_weights_to_numpy(model)

        st.success(f"Saved {len(saved_files)} parameter files.")

        for file in saved_files:
            st.code(file, language="text")

    st.subheader("Example NumPy Save/Load Pattern")
    st.code(
        """
# Save weights
np.save("dense_weights.npy", dense_layer.weights)
np.save("dense_biases.npy", dense_layer.biases)

# Load weights
dense_layer.weights = np.load("dense_weights.npy")
dense_layer.biases = np.load("dense_biases.npy")
        """,
        language="python"
    )

with tab4:
    st.header("How This Connects to Lesson 9")

    st.write(
        """
        Lesson 8 focused on building and using a CNN.

        Lesson 9 focuses on what happens after training:

        1. Evaluate the model on unseen data.
        2. Calculate accuracy.
        3. Use a confusion matrix to inspect errors.
        4. Save the trained model or weights.
        5. Load the trained model later.
        6. Run inference without retraining.
        """
    )

    st.code(
        """
Training:
Forward Pass
↓
Loss
↓
Backpropagation
↓
Gradient Descent
↓
Updated Weights

Inference:
Load Trained Weights
↓
Forward Pass Only
↓
Prediction
        """,
        language="text"
    )

    st.subheader("Classroom Questions")
    st.write(
        """
        - Is this app training every time you draw a digit?
        - What is the difference between training and inference?
        - Why do we save weights?
        - What does the confusion matrix show that accuracy does not?
        - Which digit pairs does the model confuse most often?
        """
    )
