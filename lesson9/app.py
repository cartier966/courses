from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import streamlit as st
from tensorflow.keras.datasets import mnist


MODEL_DIRECTORY = Path(__file__).resolve().parent / "lesson9_numpy_weights"
WEIGHT_FILES = {
    "dense1_weights": "dense1_weights.npy",
    "dense1_biases": "dense1_biases.npy",
    "dense2_weights": "dense2_weights.npy",
    "dense2_biases": "dense2_biases.npy",
}
IMAGE_SIZE = 28
NUM_CLASSES = 10


class DenseLayer:
    """A fully connected layer with cached values for backpropagation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        rng: np.random.Generator,
    ) -> None:
        self.weights = rng.normal(0.0, 0.01, size=(input_size, output_size)).astype(
            np.float32
        )
        self.biases = np.zeros((1, output_size), dtype=np.float32)
        self.inputs: np.ndarray | None = None
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("Call forward before backward on a dense layer.")
        self.d_weights = self.inputs.T @ d_output
        self.d_biases = np.sum(d_output, axis=0, keepdims=True)
        return d_output @ self.weights.T

    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases


class ReLU:
    """Rectified linear activation with a backward pass."""

    def __init__(self) -> None:
        self.inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("Call forward before backward on ReLU.")
        d_inputs = d_output.copy()
        d_inputs[self.inputs <= 0] = 0
        return d_inputs


class SimpleNumPyClassifier:
    """Two-layer NumPy network used by the notebook's MNIST training loop."""

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.dense1 = DenseLayer(IMAGE_SIZE * IMAGE_SIZE, 64, rng)
        self.relu1 = ReLU()
        self.dense2 = DenseLayer(64, NUM_CLASSES, rng)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        hidden = self.relu1.forward(self.dense1.forward(inputs))
        return softmax(self.dense2.forward(hidden))

    def backward(self, d_logits: np.ndarray) -> None:
        d_hidden = self.dense2.backward(d_logits)
        self.dense1.backward(self.relu1.backward(d_hidden))

    def update(self, learning_rate: float) -> None:
        self.dense1.update(learning_rate)
        self.dense2.update(learning_rate)

    def predict_probabilities(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_probabilities(inputs), axis=1)


@dataclass(frozen=True)
class EvaluationResults:
    accuracy: float
    predictions: np.ndarray
    confusion: np.ndarray
    misclassified_indices: np.ndarray


@dataclass(frozen=True)
class TrainingSettings:
    train_size: int
    test_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted_logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    correct_probabilities = probabilities[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.maximum(correct_probabilities, 1e-9))))


def softmax_cross_entropy_backward(
    probabilities: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    d_logits = probabilities.copy()
    d_logits[np.arange(labels.size), labels] -= 1
    return d_logits / labels.size


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def create_batches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = rng.permutation(images.shape[0])
    for start in range(0, images.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images[batch_indices], labels[batch_indices]


def train_model(
    model: SimpleNumPyClassifier,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 42,
) -> dict[str, list[float]]:
    history = {"loss": [], "train_accuracy": [], "test_accuracy": []}
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        batch_losses = []
        for image_batch, label_batch in create_batches(
            train_images, train_labels, batch_size, rng
        ):
            probabilities = model.forward(image_batch)
            batch_losses.append(cross_entropy_loss(probabilities, label_batch))
            model.backward(
                softmax_cross_entropy_backward(probabilities, label_batch)
            )
            model.update(learning_rate)

        history["loss"].append(float(np.mean(batch_losses)))
        history["train_accuracy"].append(
            calculate_accuracy(train_labels, model.predict(train_images))
        )
        history["test_accuracy"].append(
            calculate_accuracy(test_labels, model.predict(test_images))
        )

    return history


@st.cache_data(show_spinner=False)
def load_mnist_data(
    train_size: int, test_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_size = min(train_size, len(train_images))
    test_size = min(test_size, len(test_images))

    train_raw = train_images[:train_size].copy()
    test_raw = test_images[:test_size].copy()
    x_train = train_raw.reshape(train_size, -1).astype(np.float32) / 255.0
    x_test = test_raw.reshape(test_size, -1).astype(np.float32) / 255.0
    return (
        train_raw,
        x_train,
        train_labels[:train_size].copy(),
        test_raw,
        x_test,
        test_labels[:test_size].copy(),
    )


def build_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def evaluate_model(
    model: SimpleNumPyClassifier, y_true: np.ndarray, test_images: np.ndarray
) -> EvaluationResults:
    predictions = model.predict(test_images)
    confusion = build_confusion_matrix(y_true, predictions)
    return EvaluationResults(
        accuracy=calculate_accuracy(y_true, predictions),
        predictions=predictions,
        confusion=confusion,
        misclassified_indices=np.flatnonzero(predictions != y_true),
    )


def get_misclassification_pairs(confusion: np.ndarray) -> list[tuple[int, int, int]]:
    pairs = [
        (int(confusion[actual, predicted]), actual, predicted)
        for actual in range(NUM_CLASSES)
        for predicted in range(NUM_CLASSES)
        if actual != predicted
    ]
    return sorted(pairs, key=lambda pair: (-pair[0], pair[1], pair[2]))


def save_model_weights(
    model: SimpleNumPyClassifier, directory: Path = MODEL_DIRECTORY
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    parameters = {
        "dense1_weights": model.dense1.weights,
        "dense1_biases": model.dense1.biases,
        "dense2_weights": model.dense2.weights,
        "dense2_biases": model.dense2.biases,
    }
    saved_paths = []
    for name, values in parameters.items():
        path = directory / WEIGHT_FILES[name]
        np.save(path, values)
        saved_paths.append(path)
    return saved_paths


def load_model_weights(
    model: SimpleNumPyClassifier, directory: Path = MODEL_DIRECTORY
) -> None:
    def load_array(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
        values = np.load(path, allow_pickle=False)
        if values.shape != expected_shape:
            raise ValueError(
                f"Invalid shape in {path.name}: expected {expected_shape}, "
                f"received {values.shape}."
            )
        return values.astype(np.float32, copy=False)

    model.dense1.weights = load_array(
        directory / WEIGHT_FILES["dense1_weights"], model.dense1.weights.shape
    )
    model.dense1.biases = load_array(
        directory / WEIGHT_FILES["dense1_biases"], model.dense1.biases.shape
    )
    model.dense2.weights = load_array(
        directory / WEIGHT_FILES["dense2_weights"], model.dense2.weights.shape
    )
    model.dense2.biases = load_array(
        directory / WEIGHT_FILES["dense2_biases"], model.dense2.biases.shape
    )


def initialize_session_state() -> None:
    if "model" in st.session_state:
        return

    model = SimpleNumPyClassifier()
    weight_paths = [MODEL_DIRECTORY / filename for filename in WEIGHT_FILES.values()]
    existing_paths = [path.exists() for path in weight_paths]
    if any(existing_paths) and not all(existing_paths):
        missing = [
            path.name for path, exists in zip(weight_paths, existing_paths) if not exists
        ]
        raise FileNotFoundError(
            "Saved weights are incomplete. Missing: " + ", ".join(missing)
        )

    if all(existing_paths):
        load_model_weights(model)
        st.session_state["model_status"] = "Loaded saved NumPy model weights."
    else:
        st.session_state["model_status"] = "No saved model found. Train a model to begin."
    st.session_state["model"] = model
    st.session_state["training_history"] = None
    st.session_state["evaluation_results"] = None


def train_from_sidebar(settings: TrainingSettings) -> None:
    with st.spinner("Loading MNIST and training the NumPy model..."):
        (
            _,
            x_train,
            y_train,
            _,
            x_test,
            y_test,
        ) = load_mnist_data(settings.train_size, settings.test_size)
        model = SimpleNumPyClassifier(seed=settings.seed)
        history = train_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            seed=settings.seed,
        )
        save_model_weights(model)

    st.session_state["model"] = model
    st.session_state["model_status"] = "Trained a new model and saved its weights."
    st.session_state["training_history"] = history
    st.session_state["evaluation_results"] = None


def render_sidebar() -> TrainingSettings:
    st.sidebar.header("Training settings")
    train_size = st.sidebar.slider("Training images", 500, 6000, 2000, 500)
    test_size = st.sidebar.slider("Test images", 100, 2000, 500, 100)
    epochs = st.sidebar.slider("Epochs", 1, 20, 8)
    batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)
    learning_rate = st.sidebar.select_slider(
        "Learning rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1
    )
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
    return TrainingSettings(
        train_size=train_size,
        test_size=test_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=int(seed),
    )


def render_training_tab(settings: TrainingSettings) -> None:
    st.header("Train the NumPy classifier")
    st.write(
        "The network learns from a small MNIST subset using mini-batch gradient "
        "descent. Training also saves the learned parameters for later use."
    )
    st.info(st.session_state["model_status"])
    if st.button("Train model and save weights", type="primary"):
        train_from_sidebar(settings)
        st.success(st.session_state["model_status"])

    history = st.session_state["training_history"]
    if history:
        st.subheader("Training history")
        st.line_chart({"Loss": history["loss"]})
        st.line_chart(
            {
                "Training accuracy": history["train_accuracy"],
                "Test accuracy": history["test_accuracy"],
            }
        )
        st.metric("Final test accuracy", f"{history['test_accuracy'][-1]:.2%}")


def render_evaluation_tab(settings: TrainingSettings) -> None:
    st.header("Evaluate on unseen test images")
    st.write("Review accuracy, a confusion matrix, and common misclassification pairs.")

    if st.button("Run evaluation"):
        with st.spinner("Evaluating the model..."):
            _, _, _, _, x_test, y_test = load_mnist_data(
                settings.train_size, settings.test_size
            )
            results = evaluate_model(st.session_state["model"], y_test, x_test)
            st.session_state["evaluation_results"] = results
            st.session_state["evaluation_settings"] = (
                settings.train_size,
                settings.test_size,
            )

    results: EvaluationResults | None = st.session_state["evaluation_results"]
    if results is None or st.session_state.get("evaluation_settings") != (
        settings.train_size,
        settings.test_size,
    ):
        st.info("Run evaluation to measure this model with the current data settings.")
        return

    st.metric("Test accuracy", f"{results.accuracy:.2%}")
    st.caption("Rows are actual digits; columns are predicted digits.")
    st.dataframe(results.confusion, use_container_width=True)

    st.subheader("Most common misclassification pairs")
    pairs = get_misclassification_pairs(results.confusion)[:3]
    for count, actual, predicted in pairs:
        st.write(f"Actual **{actual}**, predicted **{predicted}**: {count}")

    if results.misclassified_indices.size:
        st.subheader("Example misclassified image")
        index = int(results.misclassified_indices[0])
        _, _, _, test_raw, _, y_test = load_mnist_data(
            settings.train_size, settings.test_size
        )
        st.image(
            test_raw[index],
            caption=(
                f"Actual: {int(y_test[index])} | "
                f"Predicted: {int(results.predictions[index])}"
            ),
            width=140,
        )
    else:
        st.success("No misclassifications in this test subset.")


def render_inference_tab(settings: TrainingSettings) -> None:
    st.header("Run inference on an MNIST test image")
    st.write("Inference uses only the saved or currently loaded model parameters.")
    if st.button("Load test images for inference"):
        with st.spinner("Loading MNIST test images..."):
            _, _, _, test_raw, x_test, y_test = load_mnist_data(
                settings.train_size, settings.test_size
            )
        st.session_state["inference_data"] = (test_raw, x_test, y_test)
        st.session_state["inference_settings"] = (
            settings.train_size,
            settings.test_size,
        )

    inference_data = st.session_state.get("inference_data")
    if inference_data is None or st.session_state.get("inference_settings") != (
        settings.train_size,
        settings.test_size,
    ):
        st.info("Load the test images to choose a sample for inference.")
        return

    test_raw, x_test, y_test = inference_data
    index = st.slider("Test image index", 0, len(test_raw) - 1, 0)
    probabilities = st.session_state["model"].predict_probabilities(
        x_test[index : index + 1]
    )[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(
            test_raw[index],
            caption=f"Actual digit: {int(y_test[index])}",
            width=196,
        )
    with right_column:
        st.metric("Predicted digit", predicted_digit)
        st.metric("Confidence", f"{confidence:.2%}")
        st.bar_chart(probabilities)


def conv2d_forward_single(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    output_height = image.shape[0] - kernel.shape[0] + 1
    output_width = image.shape[1] - kernel.shape[1] + 1
    output = np.zeros((output_height, output_width), dtype=np.float64)
    for row in range(output_height):
        for column in range(output_width):
            patch = image[
                row : row + kernel.shape[0], column : column + kernel.shape[1]
            ]
            output[row, column] = np.sum(patch * kernel)
    return output


def conv2d_filter_gradient_single(
    image: np.ndarray, kernel: np.ndarray, d_output: np.ndarray
) -> np.ndarray:
    gradient = np.zeros_like(kernel, dtype=np.float64)
    for row in range(d_output.shape[0]):
        for column in range(d_output.shape[1]):
            patch = image[
                row : row + kernel.shape[0], column : column + kernel.shape[1]
            ]
            gradient += patch * d_output[row, column]
    return gradient


def render_backpropagation_tab() -> None:
    st.header("Backpropagation building blocks")
    st.write(
        "For a dense layer, the backward pass calculates gradients for its weights, "
        "biases, and inputs."
    )
    inputs = np.array([[1.0, 2.0, 3.0]])
    weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    biases = np.array([[0.1, -0.1]])
    d_output = np.array([[1.0, -1.0]])
    st.code(
        "output = inputs @ weights + biases\n"
        "d_weights = inputs.T @ d_output\n"
        "d_biases = sum(d_output)\n"
        "d_inputs = d_output @ weights.T",
        language="python",
    )
    st.write("Dense output:", inputs @ weights + biases)
    st.write("Weight gradient:", inputs.T @ d_output)
    st.write("Bias gradient:", np.sum(d_output, axis=0, keepdims=True))
    st.write("Gradient passed to previous layer:", d_output @ weights.T)

    st.subheader("Convolution filter gradient")
    image = np.array(
        [[0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]],
        dtype=np.float64,
    )
    kernel = np.array([[1, 0], [0, -1]], dtype=np.float64)
    convolution = conv2d_forward_single(image, kernel)
    filter_gradient = conv2d_filter_gradient_single(
        image, kernel, np.ones_like(convolution)
    )
    st.write("Input image:", image)
    st.write("Filter:", kernel)
    st.write("Convolution output:", convolution)
    st.write("Filter gradient for an all-ones output gradient:", filter_gradient)


def render_persistence_tab() -> None:
    st.header("Saved model weights")
    st.write(
        "The model is stored as four NumPy arrays, matching the notebook's "
        "`np.save()` / `np.load()` persistence exercise."
    )
    st.code(str(MODEL_DIRECTORY), language="text")
    if st.button("Save current weights"):
        saved_paths = save_model_weights(st.session_state["model"])
        st.success(f"Saved {len(saved_paths)} weight and bias arrays.")

    weight_paths = [MODEL_DIRECTORY / filename for filename in WEIGHT_FILES.values()]
    if st.button(
        "Reload saved weights",
        disabled=not all(path.exists() for path in weight_paths),
    ):
        load_model_weights(st.session_state["model"])
        st.session_state["evaluation_results"] = None
        st.success("Reloaded weights from disk.")

    st.subheader("Files")
    for filename in WEIGHT_FILES.values():
        path = MODEL_DIRECTORY / filename
        st.write(f"{filename}: {'saved' if path.exists() else 'not saved'}")


def main() -> None:
    st.set_page_config(
        page_title="Lesson 9 - NumPy MNIST",
        page_icon="📊",
        layout="wide",
    )
    initialize_session_state()

    st.title("Lesson 9: CNN Training Concepts and NumPy MNIST Classifier")
    st.write(
        "Train a small NumPy neural network, inspect its performance, run inference, "
        "and persist its learned weights."
    )
    settings = render_sidebar()
    training_tab, evaluation_tab, inference_tab, backprop_tab, persistence_tab = st.tabs(
        [
            "Training",
            "Evaluation",
            "Inference",
            "Backpropagation",
            "Weight persistence",
        ]
    )
    with training_tab:
        render_training_tab(settings)
    with evaluation_tab:
        render_evaluation_tab(settings)
    with inference_tab:
        render_inference_tab(settings)
    with backprop_tab:
        render_backpropagation_tab()
    with persistence_tab:
        render_persistence_tab()


if __name__ == "__main__":
    main()
