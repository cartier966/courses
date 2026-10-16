# /// script
# requires-python = ">=3.9,<3.14"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "pycaret",
# ]
# ///

"""Diabetes prediction lesson rewritten with PyCaret.

This version keeps the same learning flow as lesson4.py but uses PyCaret to automate
training, comparison, evaluation, and prediction. The goal is to make the workflow
simpler while still teaching the same machine learning concepts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

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
FOLD_COUNT = 5
CV_SPLITTER = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=42)


def load_dataset() -> pd.DataFrame:
    """Load the real diabetic dataset used in this lesson."""
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
            f"Expected columns: {EXPECTED_COLUMNS}"
        )

    return df


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Show the basic dataset information before modeling."""
    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Create a target distribution chart."""
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


def initialize_pycaret(df: pd.DataFrame) -> ClassificationExperiment:
    """Prepare the dataset for PyCaret classification workflow."""
    experiment = ClassificationExperiment(
        target=TARGET_COLUMN,
        session_id=42,
        train_size=0.8,
        normalize=True,
        fold=FOLD_COUNT,
        verbose=False,
    )
    experiment.fit(df)
    return experiment


def compare_models_with_pycaret(
    experiment: ClassificationExperiment,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, object]]:
    """Train and compare several models using PyCaret.

    We still keep the same model set as the original lesson so the comparison is familiar:
    logistic regression, decision tree, random forest, KNN, and SVM.
    """
    model_codes = {
        "Logistic Regression": "lr",
        "Decision Tree": "dt",
        "Random Forest": "rf",
        "KNN": "knn",
        "Support Vector Machine": "svm",
    }

    results = []
    trained_models = {}
    for display_name, model_key in model_codes.items():
        model_result = experiment.create_model(
            model_key, fold=CV_SPLITTER, verbose=False
        )
        trained_models[model_key] = model_result.pipeline
        accuracy = float(model_result.metrics.loc["Mean", "Accuracy"])
        results.append({"Model": display_name, "Accuracy": accuracy})
        print(f"{display_name}: accuracy = {accuracy:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print("\n=== MODEL COMPARISON ===")
    print(results_df)
    return results_df, model_codes, trained_models


def restore_prediction_labels(
    predicted_labels: pd.Series, target_labels: pd.Series
) -> pd.Series:
    """Convert PyCaret's encoded labels back to the dataset's original labels."""
    known_labels = set(pd.unique(target_labels))
    if set(pd.unique(predicted_labels)).issubset(known_labels):
        return predicted_labels

    return predicted_labels.map(dict(enumerate(sorted(known_labels))))


def evaluate_best_model(
    experiment: ClassificationExperiment, best_model, model_name: str
) -> None:
    """Generate evaluation metrics for the best model using the holdout test set."""
    test_data = experiment.get_config("X_test").copy()
    test_data[TARGET_COLUMN] = experiment.get_config("y_test")

    predictions = experiment.predict_model(best_model, data=test_data).predictions
    y_true = predictions[TARGET_COLUMN]
    y_pred = restore_prediction_labels(predictions["prediction_label"], y_true)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    tick_labels = ["No", "Yes"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig("pycaret_confusion_matrix.png")
    plt.close(fig)
    print("Saved confusion matrix to pycaret_confusion_matrix.png")


def show_feature_importance(
    experiment: ClassificationExperiment, best_model
) -> None:
    """Plot feature importance only for models that support it."""
    model = best_model.steps[-1][1]
    model_name = model.__class__.__name__

    if hasattr(model, "feature_importances_"):
        experiment.plot_model(
            best_model, plot="feature", save="feature_importance_pycaret.png"
        )
        print("Saved feature importance chart to feature_importance_pycaret.png")
    else:
        print(
            f"Skipping feature-importance chart because {model_name} does not expose "
            "feature_importances_."
        )


def create_new_patient() -> pd.DataFrame:
    """Create a sample patient row in the same format as the dataset."""
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


def predict_patient(experiment: ClassificationExperiment, best_model) -> str:
    """Use the trained model to predict diabetic status for a new patient."""
    patient_df = create_new_patient()
    prediction = experiment.predict_model(best_model, data=patient_df).predictions
    target_labels = experiment.get_config("y_train")
    predicted_label = str(
        restore_prediction_labels(prediction["prediction_label"], target_labels).iloc[0]
    )
    print("\nPrediction for a new patient:", predicted_label)
    return predicted_label


def main() -> None:
    """Run the full lesson using PyCaret instead of manual scikit-learn code."""
    print("Starting the machine learning lesson with PyCaret...")

    df = load_dataset()
    print_dataset_overview(df)
    plot_target_distribution(df)

    experiment = initialize_pycaret(df)
    print("\nPyCaret setup complete.")

    results_df, model_codes, trained_models = compare_models_with_pycaret(experiment)
    best_model_name = results_df.iloc[0]["Model"]
    best_model_code = model_codes[best_model_name]
    best_model = trained_models[best_model_code]

    print(f"\nBest model: {best_model_name}")
    print(f"Best accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

    evaluate_best_model(experiment, best_model, best_model_name)
    final_model = experiment.finalize_model(best_model).pipeline
    show_feature_importance(experiment, final_model)
    predict_patient(experiment, final_model)

    print("\nLesson complete. The workflow used PyCaret for model comparison and prediction.")


if __name__ == "__main__":
    main()
