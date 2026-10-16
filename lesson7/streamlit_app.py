import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Data science starter", page_icon=":material/monitoring:")
st.title("Data science starter")
st.write("Explore a simple linear regression model built with NumPy, Pandas, and scikit-learn.")

with st.sidebar:
    st.header("Dataset settings")
    sample_count = st.slider("Number of samples", min_value=50, max_value=1000, value=250, step=50)
    noise_level = st.slider("Noise level", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

rng = np.random.default_rng(seed=42)
feature = rng.uniform(0, 100, size=sample_count)
target = 3.2 * feature + 5 + rng.normal(0, noise_level, size=sample_count)
dataset = pd.DataFrame({"Feature": feature, "Target": target})

x_train, x_test, y_train, y_test = train_test_split(
    dataset[["Feature"]],
    dataset["Target"],
    test_size=0.2,
    random_state=42,
)
model = LinearRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

st.metric("Test R² score", f"{score:.3f}")
st.caption(f"Learned relationship: target = {model.coef_[0]:.2f} × feature + {model.intercept_:.2f}")

st.subheader("Generated dataset")
st.scatter_chart(dataset, x="Feature", y="Target", alt="Scatter plot of generated feature and target values")

preview = pd.DataFrame(
    {
        "Feature": x_test["Feature"],
        "Actual target": y_test,
        "Predicted target": model.predict(x_test),
    }
).sort_values("Feature")
st.subheader("Test predictions")
st.dataframe(preview.head(20), hide_index=True)
