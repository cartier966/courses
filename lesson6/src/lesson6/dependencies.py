"""Verify that the project's data science dependencies are importable."""

import matplotlib
import numpy
import pandas
import sklearn
import streamlit
import tensorflow


def check_dependencies() -> dict[str, str]:
    """Return the installed versions of the required packages."""
    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit-learn": sklearn.__version__,
        "streamlit": streamlit.__version__,
        "tensorflow": tensorflow.__version__,
    }


if __name__ == "__main__":
    for package, version in check_dependencies().items():
        print(f"{package} {version}")
