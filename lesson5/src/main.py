"""Demo script showing pandas, numpy, matplotlib, and scikit-learn working together.

Loads the breast cancer dataset, explores it with pandas/numpy, trains a
RandomForestClassifier, evaluates it, and saves a feature-importance chart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_dataset() -> pd.DataFrame:
    """Load the breast cancer dataset into a pandas DataFrame."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df.rename(columns={"target": "diagnosis"}, inplace=True)
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Print a quick numpy/pandas-based summary of the dataset."""
    print(f"Dataset shape: {df.shape}")
    print(f"Class balance:\n{df['diagnosis'].value_counts(normalize=True)}")
    print(f"Mean values (numpy):\n{np.round(df.drop(columns='diagnosis').mean(), 2).head()}")


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, pd.Series]:
    """Train a RandomForestClassifier and return it along with feature importances."""
    X = df.drop(columns="diagnosis")
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, target_names=["malignant", "benign"]))

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    return model, importances


def plot_feature_importances(importances: pd.Series, output_path: Path) -> None:
    """Plot the top 10 feature importances with matplotlib and save to disk."""
    top_features = importances.head(10)

    plt.figure(figsize=(8, 6))
    plt.barh(top_features.index[::-1], top_features.values[::-1])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved chart to {output_path}")


def main() -> None:
    df = load_dataset()
    explore_dataset(df)
    _, importances = train_model(df)
    plot_feature_importances(importances, Path("output/feature_importances.png"))


if __name__ == "__main__":
    main()
