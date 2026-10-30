"""Runnable Streamlit companion app for the Lesson 12 notebook."""

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="Multimodal AI and CLIP", page_icon="🔎", layout="wide")

modalities = {
    "text": ["prompt", "caption", "product description"],
    "image": ["photo", "drawing", "diagram"],
    "audio": ["speech", "music", "sound effect"],
    "video": ["frames", "motion", "scene"],
}

# These are teaching examples, not real CLIP embeddings.
text_embeddings = {
    "winter morning": np.array([0.10, 0.90]),
    "hot drink": np.array([0.45, 0.55]),
    "desert heat": np.array([0.90, 0.10]),
    "construction crane": np.array([0.80, 0.30]),
    "bird crane": np.array([0.20, 0.70]),
}

image_embeddings = {
    "snowy forest image": np.array([0.12, 0.88]),
    "hot coffee image": np.array([0.48, 0.57]),
    "heavy coat image": np.array([0.18, 0.82]),
    "desert image": np.array([0.88, 0.12]),
    "steel crane image": np.array([0.78, 0.32]),
    "bird crane image": np.array([0.22, 0.72]),
}

candidate_images = {
    "A - snowy forest": image_embeddings["snowy forest image"],
    "B - hot coffee": image_embeddings["hot coffee image"],
    "C - heavy coat": image_embeddings["heavy coat image"],
    "D - desert landscape": image_embeddings["desert image"],
}

ambiguous_pairs = [
    {
        "word": "batter",
        "meaning_1": "cake batter in a mixing bowl",
        "meaning_2": "baseball batter holding a bat",
        "visual_context_1": "bowl, flour, spoon, kitchen",
        "visual_context_2": "baseball field, helmet, bat",
    },
    {
        "word": "jam",
        "meaning_1": "fruit jam in a jar",
        "meaning_2": "traffic jam",
        "visual_context_1": "jar, toast, fruit",
        "visual_context_2": "cars, road, congestion",
    },
    {
        "word": "crane",
        "meaning_1": "bird near water",
        "meaning_2": "construction machine",
        "visual_context_1": "feathers, water, long legs",
        "visual_context_2": "steel, hook, construction site",
    },
]

image_items = [
    {
        "id": "img1",
        "description": "snowy forest trail during winter morning",
        "uri": "./images/snowy_forest.jpg",
        "category": "winter",
    },
    {
        "id": "img2",
        "description": "hot cup of coffee on a wooden table",
        "uri": "./images/coffee.jpg",
        "category": "morning",
    },
    {
        "id": "img3",
        "description": "person wearing a heavy coat in cold weather",
        "uri": "./images/heavy_coat.jpg",
        "category": "winter",
    },
    {
        "id": "img4",
        "description": "desert landscape under bright sun",
        "uri": "./images/desert.jpg",
        "category": "desert",
    },
]

descriptions = [
    "A snowy forest trail during a quiet winter morning.",
    "A hot cup of coffee on a wooden table near a window.",
    "A person wearing a heavy coat while walking in cold weather.",
    "A desert landscape under bright sunlight with sand dunes.",
]

my_descriptions = [
    "A waterproof hiking jacket on a mountain trail.",
    "A breathable windbreaker for cool spring weather.",
    "A heavy winter parka designed for snowstorms.",
    "Trail running shoes with strong grip and support.",
    "A compact camping lantern glowing inside a tent.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity, treating a zero-length vector as unrelated."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def search_images_by_text(
    query_vector: np.ndarray,
    image_vectors: dict[str, np.ndarray],
    n_results: int = 3,
) -> list[tuple[float, str]]:
    results = [
        (cosine_similarity(query_vector, image_vector), image_name)
        for image_name, image_vector in image_vectors.items()
    ]
    return sorted(results, reverse=True)[:n_results]


def calculate_embedding_cost(
    total_tokens: int, price_per_million: float = 0.02
) -> float:
    return (total_tokens / 1_000_000) * price_per_million


st.title("Multimodal AI, CLIP, and Shared Vector Spaces")
st.caption(
    "An interactive, runnable companion to the Lesson 12 notebook. "
    "The plotted vectors below are simulated teaching examples, not real CLIP embeddings."
)

section = st.sidebar.radio(
    "Explore the lesson",
    [
        "Overview",
        "Shared vector space",
        "Text-to-image search",
        "Ambiguity and context",
        "Search pipeline",
        "Data mapping",
        "API key safety",
        "Tokens and cost",
        "Project worksheet",
        "Final reflection",
    ],
)

if section == "Overview":
    st.header("What is multimodal AI?")
    st.write(
        "A modality is a type of data. Multimodal AI works with more than one type; "
        "this lesson focuses on connecting text and images."
    )
    columns = st.columns(4)
    for column, (modality, examples) in zip(columns, modalities.items()):
        column.subheader(modality.title())
        column.write(", ".join(examples))
    st.subheader("What CLIP demonstrates")
    st.write(
        "CLIP maps text and images into a shared vector space. Cosine similarity "
        "can rank images by how closely their vectors match a text query."
    )
    st.info(
        "This app uses small, hand-written 2D vectors to demonstrate the idea. "
        "It does not load a CLIP model or create production embeddings."
    )

elif section == "Shared vector space":
    st.header("Shared vector space")
    st.write("Text points use x markers; image points use circles.")
    fig, axis = plt.subplots(figsize=(9, 7))
    for label, vector in text_embeddings.items():
        axis.scatter(vector[0], vector[1], marker="x", s=100, color="#a23b72")
        axis.text(vector[0] + 0.01, vector[1] + 0.01, "TEXT: " + label, fontsize=9)
    for label, vector in image_embeddings.items():
        axis.scatter(vector[0], vector[1], marker="o", s=80, color="#2878b5")
        axis.text(vector[0] + 0.01, vector[1] - 0.03, "IMG: " + label, fontsize=9)
    axis.set_xlabel("Dimension 1")
    axis.set_ylabel("Dimension 2")
    axis.set_title("Simulated Shared Vector Space")
    axis.grid(True)
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Practice 1: read the shared space"):
        st.markdown(
            "- Which image is closest to **winter morning**?\n"
            "- Which image is closest to **desert heat**?\n"
            "- Which two points represent the two meanings of “crane”?\n"
            "- Why are the vectors labeled simulated?"
        )

elif section == "Text-to-image search":
    st.header("Cosine similarity for text-to-image matching")
    query_name = st.selectbox("Choose a text query", list(text_embeddings))
    query_vector = text_embeddings[query_name]
    top_n = st.slider("Number of image results", min_value=1, max_value=6, value=3)
    ranked_results = search_images_by_text(query_vector, image_embeddings, top_n)
    st.write(f"Query: **{query_name}**")
    for rank, (score, image_name) in enumerate(ranked_results, start=1):
        st.write(f"{rank}. **{image_name}** — similarity: `{score:.4f}`")
        st.progress(max(0.0, min(1.0, score)))
    st.caption(
        "Cosine similarity compares vector direction; these illustrative values "
        "should not be interpreted as model confidence."
    )
    with st.expander("Practice 2: compare desert heat"):
        desert_results = search_images_by_text(
            text_embeddings["desert heat"], candidate_images
        )
        for rank, (score, image_name) in enumerate(desert_results, start=1):
            st.write(f"{rank}. {image_name} | similarity: {score:.4f}")

elif section == "Ambiguity and context":
    st.header("Ambiguity and context")
    query_key = st.radio(
        "Compare the meanings of “crane”",
        ["bird crane", "construction crane"],
        format_func=lambda value: (
            "Bird" if value == "bird crane" else "Construction machine"
        ),
        horizontal=True,
    )
    crane_images = {
        "steel crane image": image_embeddings["steel crane image"],
        "bird crane image": image_embeddings["bird crane image"],
    }
    for score, image_name in search_images_by_text(
        text_embeddings[query_key], crane_images, n_results=2
    ):
        st.write(f"{score:.4f} — {image_name}")
    st.subheader("Other ambiguous words")
    for word, meanings in {
        "bark": ["tree bark", "dog barking"],
        "bat": ["baseball bat", "flying animal"],
        "seal": ["marine animal", "stamp/seal"],
    }.items():
        st.write(f"**{word}**: {', '.join(meanings)}")
    with st.expander("Practice 3: semantic database mapping"):
        for pair in ambiguous_pairs:
            st.markdown(
                f"**{pair['word']}**  \n"
                f"1. {pair['meaning_1']} — context: {pair['visual_context_1']}  \n"
                f"2. {pair['meaning_2']} — context: {pair['visual_context_2']}"
            )

elif section == "Search pipeline":
    st.header("Simulated multimodal search function")
    st.write(
        "A practical pipeline accepts a text prompt, embeds it, searches a vector "
        "database, and returns matching image references."
    )
    with st.form("pipeline_form"):
        niche = st.text_input("Niche", value="Outdoor Gear")
        prompt = st.text_input(
            "Text prompt", value="waterproof jacket for mountain hiking"
        )
        embedding_model = st.selectbox(
            "Embedding model",
            ["OpenAI embedding model", "CLIP-style model"],
        )
        vector_database = st.text_input("Vector database", value="ChromaDB")
        retrieval_target = st.text_input(
            "Retrieval target", value="image URIs of matching gear"
        )
        submitted = st.form_submit_button("Show pipeline")
    if submitted:
        pipeline = {
            "niche": niche,
            "text_prompt": prompt,
            "embedding_model": embedding_model,
            "vector_database": vector_database,
            "retrieval_target": retrieval_target,
            "expected_results": [
                "rain jacket image",
                "windbreaker image",
                "waterproof hiking shell image",
            ],
        }
        st.json(pipeline)
    st.subheader("Example simulated search")
    for score, image_name in search_images_by_text(
        text_embeddings["winter morning"], image_embeddings, n_results=3
    ):
        st.write(f"{score:.4f} — {image_name}")

elif section == "Data mapping":
    st.header("Semantic database mapping")
    st.write("Example image records from the notebook:")
    st.json(image_items)
    st.subheader("Practice 4: design a pipeline")
    st.json(
        {
            "niche": "Outdoor Gear",
            "text_prompt": "waterproof jacket for mountain hiking",
            "embedding_model": "OpenAI embedding model or CLIP-style model",
            "vector_database": "ChromaDB",
            "retrieval_target": "image URIs of matching gear",
            "expected_results": [
                "rain jacket image",
                "windbreaker image",
                "waterproof hiking shell image",
            ],
        }
    )

elif section == "API key safety":
    st.header("Secure API key use")
    fake_api_key = "sk-demo1234567890abcdef"
    masked_demo_key = fake_api_key[:4] + "..." + fake_api_key[-4:]
    st.code(f"Masked demo key: {masked_demo_key}\nNever print the full key.")
    st.warning(
        "The value above is fake. Store real credentials in "
        "`.streamlit/secrets.toml` or the `OPENAI_API_KEY` environment variable; "
        "never hardcode or commit them."
    )
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        api_key = None
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if api_key:
        st.success(f"API key is configured: {api_key[:4]}...{api_key[-4:]}")
        if st.button("Request a sample text embedding"):
            from openai import APIError, OpenAI

            try:
                client = OpenAI(api_key=api_key)
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input="A snowy forest trail during winter.",
                )
            except APIError as error:
                st.error(f"OpenAI API request failed: {error}")
            else:
                embedding = response.data[0].embedding
                st.write(f"Embedding dimensions: {len(embedding)}")
                st.caption("The embedding values are not displayed.")
    else:
        st.info(
            "No API key is configured. The rest of the app runs without one. "
            "Add `OPENAI_API_KEY = \"your-api-key\"` to "
            "`.streamlit/secrets.toml` to enable the optional API request."
        )
    with st.expander("Notebook Streamlit secret-check example"):
        st.code(
            'import streamlit as st\n\n'
            'st.title("API Key Check")\n'
            'api_key = st.secrets["OPENAI_API_KEY"]\n'
            'masked_key = api_key[:4] + "..." + api_key[-4:]\n'
            'st.success(f"API key loaded: {masked_key}")',
            language="python",
        )

elif section == "Tokens and cost":
    st.header("Token counting and cost estimation")
    st.write("Token counts use `tiktoken` when available.")
    try:
        import tiktoken
    except ImportError:
        tiktoken = None

    token_rows: list[tuple[str, int, list[int] | None]] = []
    if tiktoken is not None:
        encoding = tiktoken.get_encoding("cl100k_base")
        for description in descriptions:
            tokens = encoding.encode(description)
            token_rows.append((description, len(tokens), tokens))
        total_tokens = sum(row[1] for row in token_rows)
        st.caption("Encoding: cl100k_base")
    else:
        for description in descriptions:
            estimated_count = int(len(description.split()) * (1000 / 750))
            token_rows.append((description, estimated_count, None))
        total_tokens = sum(row[1] for row in token_rows)
        st.warning(
            "tiktoken is not installed; token counts are word-count estimates. "
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        )
    for description, token_count, token_ids in token_rows:
        st.write(f"**{description}** — {token_count} tokens")
        if token_ids is not None and st.checkbox(
            f"Show token IDs: {description}", key=f"tokens-{description}"
        ):
            st.code(str(token_ids))
    st.metric("Total tokens", total_tokens)

    st.subheader("Embedding cost")
    price_per_million = st.number_input(
        "Price per 1 million tokens (USD)",
        min_value=0.0,
        value=0.02,
        step=0.01,
        format="%.4f",
    )
    estimated_cost = calculate_embedding_cost(total_tokens, price_per_million)
    st.metric("Estimated cost for these descriptions", f"${estimated_cost:.8f}")

    st.subheader("Practice 5: cost audit")
    total_words = sum(len(description.split()) for description in my_descriptions)
    estimated_practice_tokens = int(total_words * (1000 / 750))
    estimated_practice_cost = calculate_embedding_cost(
        estimated_practice_tokens, price_per_million
    )
    st.write(f"Total words: {total_words}")
    st.write(f"Estimated tokens: {estimated_practice_tokens}")
    st.write(f"Estimated embedding cost: ${estimated_practice_cost:.8f}")

    example_total_tokens = 5000
    example_price_per_million = 0.02
    st.caption(
        f"Notebook example: {example_total_tokens:,} tokens at "
        f"${example_price_per_million:.2f}/1M = "
        f"${calculate_embedding_cost(example_total_tokens, example_price_per_million):.4f}."
    )
    with st.expander("Notebook OpenAI embedding example"):
        st.code(
            'from openai import OpenAI\n'
            'import streamlit as st\n\n'
            'client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])\n'
            'response = client.embeddings.create(\n'
            '    model="text-embedding-3-small",\n'
            '    input="A snowy forest trail during winter."\n'
            ')\n'
            'embedding = response.data[0].embedding\n'
            'print(len(embedding))',
            language="python",
        )

elif section == "Project worksheet":
    st.header("Final project planning worksheet")
    with st.form("project_plan_form"):
        project_niche = st.text_input("Niche", value="Outdoor Gear")
        user_query_example = st.text_input(
            "Example user query",
            value="waterproof jacket for hiking in cold weather",
        )
        retrieval_goal = st.text_area(
            "Retrieval goal",
            value=(
                "Return the most semantically relevant product images "
                "for the user's text query."
            ),
        )
        saved = st.form_submit_button("Show plan")
    if saved:
        project_plan = {
            "niche": project_niche,
            "user_query_example": user_query_example,
            "data_items": [
                {
                    "id": "item1",
                    "description": "Lightweight raincoat for wet mountain trails",
                    "uri": "./images/raincoat.jpg",
                },
                {
                    "id": "item2",
                    "description": "Insulated winter parka for snow and wind",
                    "uri": "./images/parka.jpg",
                },
                {
                    "id": "item3",
                    "description": "Breathable windbreaker for spring hikes",
                    "uri": "./images/windbreaker.jpg",
                },
            ],
            "retrieval_goal": retrieval_goal,
        }
        st.json(project_plan)

elif section == "Final reflection":
    st.header("Final reflection")
    st.write("Use these prompts to summarize what you learned:")
    reflection_prompts = [
        "What does multimodal mean?",
        "How does a shared vector space connect text and images?",
        "What does cosine similarity measure?",
        "How does text-to-image retrieval rank results?",
        "Why can context change the meaning of an ambiguous word?",
        "What information belongs in an image record?",
        "Why should API keys never be hardcoded?",
        "Where should application secrets be stored?",
        "Why should a key be masked when verifying setup?",
        "What does a token count help estimate?",
        "Why is the 2D embedding demonstration only a simulation?",
    ]
    for index, prompt in enumerate(reflection_prompts, start=1):
        st.text_area(prompt, key=f"reflection-{index}", height=70)
