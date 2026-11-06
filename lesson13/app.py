
"""
Lessons 11–13 Streamlit Project
Live Embedding Model Selection + Semantic Search Dashboard

Run:
    pip install -r requirements.txt
    streamlit run app.py

Setup:
    1. Create .streamlit/secrets.toml
    2. Add:
       OPENAI_API_KEY = "your_key_here"

What this app teaches:
- Lesson 11: embeddings, vector length, cosine similarity
- Lesson 12: shared vector-space thinking and semantic meaning
- Lesson 13: OpenAI API + ChromaDB + Streamlit production interface
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
from openai import OpenAI


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Live Embeddings + Semantic Search",
    layout="wide"
)

st.title("Lessons 11–13: Live Embeddings + Semantic Search Dashboard")
st.write(
    "Select an embedding model, generate vectors live, compare semantic similarity, "
    "store records in ChromaDB, and search by meaning."
)


# -----------------------------
# Constants
# -----------------------------

EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]

DEFAULT_RECORDS = [
    {
        "id": "img_01",
        "description": "A cat sitting near a sunny window",
        "uri": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800",
        "category": "animals",
    },
    {
        "id": "img_02",
        "description": "A golden retriever playing in a park",
        "uri": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=800",
        "category": "animals",
    },
    {
        "id": "img_03",
        "description": "A snowy forest trail during winter morning",
        "uri": "https://images.unsplash.com/photo-1477601263568-180e2c6d046e?w=800",
        "category": "nature",
    },
    {
        "id": "img_04",
        "description": "A hot cup of coffee on a wooden table",
        "uri": "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=800",
        "category": "food",
    },
    {
        "id": "img_05",
        "description": "A tall steel crane at a construction site",
        "uri": "https://images.unsplash.com/photo-1504917595217-d4dc5ebe6122?w=800",
        "category": "urban",
    },
    {
        "id": "img_06",
        "description": "A crane bird standing near a lake",
        "uri": "https://images.unsplash.com/photo-1555169062-013468b47731?w=800",
        "category": "animals",
    },
]


# -----------------------------
# Helpers
# -----------------------------

@st.cache_resource
def get_openai_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to .streamlit/secrets.toml or Streamlit Cloud secrets."
        )
    return OpenAI(api_key=api_key)


@st.cache_resource
def get_chroma_collection(collection_name: str):
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name=collection_name)


def mask_key(key: str) -> str:
    if len(key) < 8:
        return "****"
    return key[:4] + "..." + key[-4:]


def get_embedding(text: str, model: str) -> List[float]:
    client = get_openai_client()
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_cost(total_tokens: int, price_per_million: float = 0.02) -> float:
    return (total_tokens / 1_000_000) * price_per_million


def upsert_records(records: List[Dict[str, Any]], model: str, collection_name: str) -> int:
    collection = get_chroma_collection(collection_name)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for record in records:
        embedding = get_embedding(record["description"], model)
        ids.append(record["id"])
        documents.append(record["description"])
        embeddings.append(embedding)
        metadatas.append({
            "uri": record["uri"],
            "category": record["category"],
            "model": model,
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)


def query_collection(query: str, model: str, collection_name: str, n_results: int):
    collection = get_chroma_collection(collection_name)
    query_embedding = get_embedding(query, model)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Controls")

selected_model = st.sidebar.selectbox(
    "Embedding Model",
    EMBEDDING_MODELS,
    index=0
)

collection_name = st.sidebar.text_input(
    "ChromaDB Collection Name",
    value="lesson11_13_demo"
)

n_results = st.sidebar.slider(
    "Number of Search Results",
    min_value=1,
    max_value=6,
    value=3
)

price_per_million = st.sidebar.number_input(
    "Embedding Price per 1M Tokens",
    min_value=0.0,
    value=0.02,
    step=0.01,
    format="%.4f"
)

with st.sidebar.expander("API Key Check"):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if api_key:
            st.success(f"Loaded: {mask_key(api_key)}")
        else:
            st.warning("OPENAI_API_KEY missing.")
    except Exception as e:
        st.error("Could not read secrets.")
        st.write(e)


# -----------------------------
# Explanation
# -----------------------------

with st.expander("How this project connects Lessons 11, 12, and 13", expanded=True):
    st.markdown(
        """
        **Lesson 11:** You learned that embeddings are vectors that represent meaning.

        **Lesson 12:** You learned that multimodal AI uses shared vector spaces to connect text and images.

        **Lesson 13:** You use OpenAI embeddings, ChromaDB, and Streamlit to build a semantic search app.

        This app lets you see live vectors, compare similarity, populate a vector database, and retrieve records using natural language search.
        """
    )

st.divider()


# -----------------------------
# Tabs
# -----------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Live Vector",
    "2. Similarity Lab",
    "3. Database Builder",
    "4. Semantic Search",
    "5. Token + Cost Audit",
])


# -----------------------------
# Tab 1: Live Vector
# -----------------------------

with tab1:
    st.header("Live Embedding Vector Generator")

    text = st.text_area(
        "Enter text to embed",
        value="A cat sitting near a sunny window.",
        height=100
    )

    if st.button("Generate Embedding", key="generate_embedding"):
        try:
            embedding = get_embedding(text, selected_model)
            vector = np.array(embedding)

            st.success("Embedding generated.")
            st.metric("Selected Model", selected_model)
            st.metric("Vector Length", len(embedding))

            col1, col2, col3 = st.columns(3)
            col1.metric("Min", f"{vector.min():.4f}")
            col2.metric("Max", f"{vector.max():.4f}")
            col3.metric("Mean", f"{vector.mean():.4f}")

            st.subheader("First 25 Vector Values")
            st.dataframe(pd.DataFrame({
                "dimension": list(range(25)),
                "value": embedding[:25]
            }))

            st.subheader("First 100 Dimensions")
            chart_df = pd.DataFrame({
                "dimension": list(range(min(100, len(embedding)))),
                "value": embedding[:100]
            })
            st.line_chart(chart_df, x="dimension", y="value")

        except Exception as e:
            st.error("Could not generate embedding.")
            st.write(e)


# -----------------------------
# Tab 2: Similarity Lab
# -----------------------------

with tab2:
    st.header("Semantic Similarity Lab")

    col_a, col_b = st.columns(2)

    with col_a:
        text_a = st.text_area("Text A", value="doctor", height=80)

    with col_b:
        text_b = st.text_area("Text B", value="physician", height=80)

    if st.button("Compare Similarity"):
        try:
            emb_a = get_embedding(text_a, selected_model)
            emb_b = get_embedding(text_b, selected_model)

            similarity = cosine_similarity(emb_a, emb_b)

            st.metric("Cosine Similarity", f"{similarity:.4f}")

            if similarity > 0.8:
                st.success("These texts are very semantically similar.")
            elif similarity > 0.5:
                st.info("These texts are somewhat related.")
            else:
                st.warning("These texts are weakly related.")

        except Exception as e:
            st.error("Could not compare texts.")
            st.write(e)


# -----------------------------
# Tab 3: Database Builder
# -----------------------------

with tab3:
    st.header("Build the ChromaDB Semantic Search Database")

    st.write(
        "This section upserts default image/text records into ChromaDB. "
        "Each record has an ID, description, URI, category, and embedding."
    )

    df = pd.DataFrame(DEFAULT_RECORDS)
    st.dataframe(df)

    total_description_tokens = sum(count_tokens(r["description"]) for r in DEFAULT_RECORDS)
    estimated = estimate_cost(total_description_tokens, price_per_million)

    col1, col2 = st.columns(2)
    col1.metric("Description Tokens", total_description_tokens)
    col2.metric("Estimated Embedding Cost", f"${estimated:.8f}")

    if st.button("Upsert Default Records into ChromaDB"):
        try:
            count = upsert_records(DEFAULT_RECORDS, selected_model, collection_name)
            st.success(f"Upserted {count} records into collection '{collection_name}'.")
        except Exception as e:
            st.error("Could not upsert records.")
            st.write(e)

    with st.expander("Add Your Own Record"):
        custom_id = st.text_input("Unique ID", value="custom_01")
        custom_desc = st.text_area("Description", value="A peaceful mountain lake at sunrise")
        custom_uri = st.text_input("Image URI", value="https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?w=800")
        custom_category = st.text_input("Category", value="nature")

        if st.button("Upsert Custom Record"):
            try:
                custom_record = [{
                    "id": custom_id,
                    "description": custom_desc,
                    "uri": custom_uri,
                    "category": custom_category,
                }]
                count = upsert_records(custom_record, selected_model, collection_name)
                st.success(f"Upserted {count} custom record.")
            except Exception as e:
                st.error("Could not upsert custom record.")
                st.write(e)


# -----------------------------
# Tab 4: Semantic Search
# -----------------------------

with tab4:
    st.header("Semantic Image/Text Search")

    query = st.chat_input("Search by meaning, e.g. 'feline', 'winter morning', 'construction machine'")

    if query:
        try:
            results = query_collection(query, selected_model, collection_name, n_results)

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not documents:
                st.warning("No results returned. Try upserting records first.")
            else:
                st.subheader(f"Results for: {query}")

                cols = st.columns(3)

                for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                    with cols[idx % 3]:
                        st.image(meta["uri"], use_container_width=True)
                        st.markdown(f"**{doc}**")
                        st.caption(f"Category: {meta.get('category', 'unknown')}")
                        st.caption(f"Distance: {dist:.4f}")
                        st.caption(f"Model used to store: {meta.get('model', 'unknown')}")

        except Exception as e:
            st.error("Search failed.")
            st.write(e)


# -----------------------------
# Tab 5: Token + Cost Audit
# -----------------------------

with tab5:
    st.header("Token and Cost Audit")

    audit_text = st.text_area(
        "Paste descriptions or prompts to estimate token cost",
        value="A cat sitting near a sunny window.\\nA snowy forest trail during winter morning.",
        height=150
    )

    if st.button("Run Token Audit"):
        lines = [line.strip() for line in audit_text.splitlines() if line.strip()]
        rows = []

        total_tokens = 0

        for line in lines:
            token_count = count_tokens(line)
            total_tokens += token_count
            rows.append({
                "text": line,
                "tokens": token_count
            })

        st.dataframe(pd.DataFrame(rows))

        cost = estimate_cost(total_tokens, price_per_million)

        col1, col2 = st.columns(2)
        col1.metric("Total Tokens", total_tokens)
        col2.metric("Estimated Cost", f"${cost:.8f}")

st.divider()
st.caption("Never commit API keys to GitHub. Use Streamlit secrets locally and in Streamlit Cloud.")
