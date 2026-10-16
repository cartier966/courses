import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from recommender import build_ratings, train_svd

st.title("Lesson 8 - Recommender Demo")

ratings = build_ratings()
st.subheader("Sample ratings")
st.dataframe(ratings.head(20))

model = train_svd(ratings)

user_id = st.selectbox("User", sorted(ratings["user"].unique()))
items = sorted(ratings["item"].unique())
preds = pd.DataFrame(
    {"item": items, "predicted_rating": [model.predict(user_id, i).est for i in items]}
).sort_values("predicted_rating", ascending=False)

st.subheader("Top recommendations")
st.dataframe(preds.head(10))

pivot = ratings.pivot_table(index="user", columns="item", values="rating").fillna(0)
sim = cosine_similarity(pivot.values)
st.subheader("User similarity (cosine)")
st.dataframe(pd.DataFrame(sim, index=pivot.index, columns=pivot.index).round(2))
