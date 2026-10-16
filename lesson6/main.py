"""Train and run the notebook's purchase-intent recommendation model."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "customer_purchase_2000.csv"
FEATURE_COLUMNS = [
    "product_interest",
    "website_visits",
    "time_on_site",
    "previous_purchases",
    "added_to_cart",
    "discount_viewed",
    "email_open_rate",
    "days_since_last_visit",
]
TARGET_COLUMN = "will_buy"
DEFAULT_PROSPECT = [5, 12, 10, 3, 1, 1, 80, 5]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the customer purchase CSV and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    data = pd.read_csv(path)
    missing = [c for c in [*FEATURE_COLUMNS, TARGET_COLUMN] if c not in data.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
    return data.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])


def inspect_dataset(data: pd.DataFrame) -> None:
    """Print the head, schema, and descriptive statistics."""
    print("Sample rows:")
    print(data.head())
    print("\nDataset information:")
    data.info()
    print("\nDescriptive statistics:")
    print(data.describe())


def split_and_scale_data(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split features and labels, fitting the scaler only on training features."""
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler


def build_model(number_of_features: int, layers_config: list[int]) -> tf.keras.Model:
    """Create and compile the binary classification network."""
    layers = []
    for layer in layers_config:
        layers.append(tf.keras.layers.Dense(layer, activation="relu"))

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(number_of_features,)),
            *layers,
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
) -> tf.keras.callbacks.History:
    """Fit the model and retain training and validation metrics."""

    
    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.20,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )],
        shuffle=True,
        verbose=1
    )


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Evaluate the trained model and print its test metrics."""
    metrics = cast(
        dict[str, float],
        model.evaluate(x_test, y_test, verbose=0, return_dict=True),
    )
    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    return float(loss), float(accuracy)


def recommend(
    model: tf.keras.Model,
    scaler: StandardScaler,
    prospect_features: Sequence[float],
) -> tuple[float, str]:
    """Predict purchase likelihood and convert it to a recommendation."""
    if len(prospect_features) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} prospect features in this order: "
            f"{', '.join(FEATURE_COLUMNS)}"
        )

    features = pd.DataFrame([prospect_features], columns=FEATURE_COLUMNS)
    scaled_features = scaler.transform(features)
    probability = float(model.predict(scaled_features, verbose=0)[0][0])
    recommendation = "Recommend" if probability >= 0.5 else "Do Not Recommend"
    return probability, recommendation


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Plot training and validation loss across epochs."""
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train the example purchase-intent recommendation model."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 60).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to the customer purchase CSV (default: next to main.py).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the training-loss plot.",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    
    layer_options = [[8], [8, 4], [16, 8], [32, 16, 8]]
    epoch_options = [30, 100, 200]
    batch_size_options = [16, 32, 64]
    simple_network_configs = [
        {
            "layers": layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "accuracy_rate": 0.0,
            "loss_rate": 0.0,
        }
        for layers in layer_options
        for epochs in epoch_options
        for batch_size in batch_size_options
    ]
    """Run data inspection, training, evaluation, and recommendation."""
    for config in simple_network_configs:
        args = parse_args(argv)
        data = load_dataset(args.data)
        inspect_dataset(data)

        x_train, x_test, y_train, y_test, scaler = split_and_scale_data(data)
        model = build_model(len(FEATURE_COLUMNS), config["layers"])
        history = train_model(model, x_train, y_train, config["epochs"], config["batch_size"])
        loss, accuracy = evaluate_model(model, x_test, y_test)
        config["loss_rate"] = loss
        config["accuracy_rate"] = accuracy

    sorted_network_configs = sorted(
        simple_network_configs,
        key=lambda config: (-config["loss_rate"], config["accuracy_rate"]),
    )
    for config in sorted_network_configs:
        print(config)
        print("-" * 40)
    # probability, recommendation = recommend(model, scaler, DEFAULT_PROSPECT)
    # print(f"\nPurchase probability: {probability:.4f}")
    # print(recommendation)

    # if not args.no_plot:
    #     plot_training_history(history)


if __name__ == "__main__":
    main()
