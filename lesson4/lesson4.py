# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///

"""Machine learning lesson using the real diabetic dataset.

This script follows the notebook workflow:
1. load the diabetes dataset,
2. inspect the data,
3. separate features from the target,
4. encode labels,
5. split into training/testing sets,
6. scale the data,
7. train several classifiers,
8. compare their accuracy,
9. choose the best model,
10. use it to predict a new patient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET_COLUMN = "Diabetic"
EXPECTED_COLUMNS = [
    "NumberOfPregnancies",
    "GlucoseConcentration",
    "BloodPressure",
    "SkinThickness",
    "InsulinLevel",
    "BodyMassIndex",
    "DiabetesPedigree",
    "Age",
]


def load_dataset() -> pd.DataFrame:
    """Load the real dataset used in this lesson."""
    csv_path = Path(__file__).resolve().parent / "DiabeticData.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected dataset not found: {csv_path}. "
            "Make sure DiabeticData.csv is in the same folder as this script."
        )

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"The dataset does not contain the target column '{TARGET_COLUMN}'. "
            f"Available columns: {list(df.columns)}"
        )

    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"The dataset is missing expected features: {missing}. "
            "Expected columns: "
            f"{EXPECTED_COLUMNS}"
        )

    return df


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Print the most important facts about the dataset before modeling."""
    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Create a simple bar chart showing how many patients are diabetic vs not diabetic."""
    counts = df[TARGET_COLUMN].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#4c72b0", "#dd8452"])
    plt.title("Distribution of Diabetic vs Non-Diabetic Patients")
    plt.xlabel("Diabetic")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig("target_distribution.png")
    plt.close()
    print("Saved target distribution chart to target_distribution.png")


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into the input features (X) and the output label (y)."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def encode_target_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Convert text labels like "Yes"/"No" into integers expected by ML models."""
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)
    print("\nEncoded classes:", encoder.classes_)
    print("First 10 encoded labels:", encoded_y[:10])
    return encoded_y, encoder


def split_and_scale_data(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Train/test split followed by feature scaling.

    A scaler is fit only on the training data and then applied to both training and test data.
    This helps prevent data leakage and is especially important for models like SVM and KNN.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nScaled sample values (first 3 rows):")
    print(X_train_scaled[:3])
    return X_train, X_test, y_train, y_test, scaler


def create_model_dictionary() -> Dict[str, object]:
    """Return the set of models we want to compare in the lesson."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
    }


def evaluate_models(
    model_dict: Dict[str, object],
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train each model, predict on the test set, and compute test accuracy."""
    results = []
    trained_models = {}

    for model_name, model in model_dict.items():
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({"Model": model_name, "Accuracy": accuracy})

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print("\n=== MODEL COMPARISON ===")
    print(results_df)

    model_bar_chart(results_df)
    return results_df, trained_models


def model_bar_chart(results_df: pd.DataFrame) -> None:
    """Plot model accuracy as a bar chart so it is easy to compare them."""
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["Model"], results_df["Accuracy"], color="#2ca02c")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    plt.close()
    print("Saved model accuracy comparison to model_accuracy_comparison.png")


def choose_best_model(results_df: pd.DataFrame, model_dict: Dict[str, object]) -> tuple[str, object, float]:
    """Pick the best-performing model based on the highest accuracy score."""
    best_model_name = results_df.iloc[0]["Model"]
    best_model = model_dict[best_model_name]
    best_accuracy = results_df.iloc[0]["Accuracy"]

    print("\nBest model:", best_model_name)
    print("Best accuracy:", best_accuracy)
    return best_model_name, best_model, best_accuracy


def print_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """Show the evaluation metrics: precision, recall, F1-score, and support."""
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    display.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Saved confusion matrix to confusion_matrix.png")


def calculate_feature_importance(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Train a random forest and inspect which features mattered most."""
    forest_model = RandomForestClassifier(random_state=42)
    forest_model.fit(X, y)

    feature_importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": forest_model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(feature_importance)

    plt.figure(figsize=(10, 5))
    plt.bar(feature_importance["Feature"], feature_importance["Importance"])
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    print("Saved feature importance chart to feature_importance.png")

    return feature_importance


def create_new_patient() -> pd.DataFrame:
    """Build a sample patient record in the same format as the real dataset."""
    return pd.DataFrame(
        {
            "NumberOfPregnancies": [6],
            "GlucoseConcentration": [148],
            "BloodPressure": [72],
            "SkinThickness": [35],
            "InsulinLevel": [0],
            "BodyMassIndex": [33.6],
            "DiabetesPedigree": [0.627],
            "Age": [50],
        }
    )


def predict_new_patient(
    model: object,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    patient_df: pd.DataFrame,
) -> str:
    """Use the trained best model to predict the class for a new patient."""
    patient_scaled = scaler.transform(patient_df)
    prediction = model.predict(patient_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]


def main() -> None:
    """Run the full teaching workflow from data loading to prediction."""
    print("Starting the machine learning lesson...")
    print(f"numpy {np.__version__}")
    print(f"pandas {pd.__version__}")
    print(f"scikit-learn {sklearn_version}")

    df = load_dataset()
    print_dataset_overview(df)
    plot_target_distribution(df)

    X, y_raw = prepare_features_and_target(df)
    y_encoded, label_encoder = encode_target_labels(y_raw)

    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y_encoded)

    model_dict = create_model_dictionary()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results_df, trained_models = evaluate_models(
        model_dict,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        label_encoder,
    )

    best_model_name, best_model, best_accuracy = choose_best_model(results_df, model_dict)
    y_pred_best = best_model.predict(scaler.transform(X_test))
    print_classification_summary(y_test, y_pred_best, label_encoder)

    # The random forest is trained on the original unscaled data because feature importance
    # is often computed from the naturally encoded inputs, and this branch is meant to show
    # how to interpret the model rather than to compare against other classifiers.
    calculate_feature_importance(X_train, y_train)

    patient_df = create_new_patient()
    prediction = predict_new_patient(best_model, scaler, label_encoder, patient_df)
    print("\nPrediction for a new patient:", prediction)

    print("\nLesson complete. The model has been trained, evaluated, and used for inference.")


if __name__ == "__main__":
    main()
