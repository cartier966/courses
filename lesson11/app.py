"""Runnable script containing the executable cells from the Lesson 11 notebook."""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# %% Notebook cell 1
token_ids = {
    "cat": 45,
    "kitten": 9002,
    "banana": 17
}

print(token_ids)

print("\nImportant:")
print("These IDs are labels. They do not contain meaning by themselves.")

# %% Notebook cell 2
import numpy as np
import matplotlib.pyplot as plt

embeddings_2d = {
    "doctor": np.array([0.91, 0.10]),
    "physician": np.array([0.89, 0.12]),
    "nurse": np.array([0.82, 0.18]),
    "banana": np.array([0.05, 0.93]),
    "apple": np.array([0.08, 0.88]),
    "train": np.array([0.55, 0.45])
}

for word, vector in embeddings_2d.items():
    print(word, "→", vector)

# %% Notebook cell 3
plt.figure(figsize=(7, 6))

for word, vector in embeddings_2d.items():
    plt.scatter(vector[0], vector[1])
    plt.text(vector[0] + 0.01, vector[1] + 0.01, word)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Simple 2D Embedding Space")
plt.grid(True)
plt.show()

# %% Notebook cell 4
# Practice 1

embeddings_2d["surgeon"] = np.array([0.93, 0.08])
embeddings_2d["orange"] = np.array([0.10, 0.90])
embeddings_2d["airplane"] = np.array([0.60, 0.40])

plt.figure(figsize=(7, 6))

for word, vector in embeddings_2d.items():
    plt.scatter(vector[0], vector[1])
    plt.text(vector[0] + 0.01, vector[1] + 0.01, word)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Updated 2D Embedding Space")
plt.grid(True)
plt.show()

# %% Notebook cell 5
semantic_space = {
    "house cat": np.array([8, 9]),
    "kitten": np.array([7, 9]),
    "tiger": np.array([9, 1]),
    "lion": np.array([9, 2]),
    "robot vacuum": np.array([6, 8]),
    "cloud": np.array([2, 1])
}

plt.figure(figsize=(7, 6))

for word, vector in semantic_space.items():
    plt.scatter(vector[0], vector[1])
    plt.text(vector[0] + 0.1, vector[1] + 0.1, word)

plt.xlabel("Physicality")
plt.ylabel("Domesticity")
plt.title("Example Semantic Space")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid(True)
plt.show()

# %% Notebook cell 6
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

cat = semantic_space["house cat"]
kitten = semantic_space["kitten"]
tiger = semantic_space["tiger"]

print("Distance between house cat and kitten:", euclidean_distance(cat, kitten))
print("Distance between house cat and tiger:", euclidean_distance(cat, tiger))

# %% Notebook cell 7
# Practice 2

pairs = [
    ("tiger", "lion"),
    ("house cat", "robot vacuum"),
    ("cloud", "tiger")
]

for a, b in pairs:
    distance = euclidean_distance(semantic_space[a], semantic_space[b])
    print(f"Distance between {a} and {b}: {distance:.2f}")

# %% Notebook cell 8
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

doctor = embeddings_2d["doctor"]
physician = embeddings_2d["physician"]
banana = embeddings_2d["banana"]

print("Cosine similarity: doctor / physician:", cosine_similarity(doctor, physician))
print("Cosine similarity: doctor / banana:", cosine_similarity(doctor, banana))

# %% Notebook cell 9
# Practice 3

pairs = [
    ("banana", "apple"),
    ("doctor", "nurse"),
    ("train", "airplane")
]

for a, b in pairs:
    similarity = cosine_similarity(embeddings_2d[a], embeddings_2d[b])
    print(f"Cosine similarity between {a} and {b}: {similarity:.4f}")

# %% Notebook cell 10
A = np.array([1, 1])
B = np.array([2, 2])
C = np.array([1, 0])

print("Euclidean A-B:", euclidean_distance(A, B))
print("Cosine A-B:", cosine_similarity(A, B))

print("\nEuclidean A-C:", euclidean_distance(A, C))
print("Cosine A-C:", cosine_similarity(A, C))

# %% Notebook cell 11
documents = [
    "Lightweight raincoat for wet weather",
    "Insulated winter parka for snow",
    "Breathable windbreaker for spring hikes",
    "Trail running shoes with strong grip",
    "Stainless steel water bottle for camping"
]

query = "waterproof outerwear"

# Simple keyword search
keyword_results = []

for doc in documents:
    if query.lower() in doc.lower():
        keyword_results.append(doc)

print("Keyword search results:")
print(keyword_results)

# %% Notebook cell 12
# Simple hand-made vectors for demonstration
# Dimensions: [waterproof, warmth, hiking, footwear, drinkware]

document_vectors = {
    "Lightweight raincoat for wet weather": np.array([0.95, 0.10, 0.50, 0.00, 0.00]),
    "Insulated winter parka for snow": np.array([0.60, 0.95, 0.30, 0.00, 0.00]),
    "Breathable windbreaker for spring hikes": np.array([0.50, 0.20, 0.80, 0.00, 0.00]),
    "Trail running shoes with strong grip": np.array([0.10, 0.00, 0.75, 0.95, 0.00]),
    "Stainless steel water bottle for camping": np.array([0.05, 0.00, 0.50, 0.00, 0.95])
}

query_vector = np.array([0.90, 0.10, 0.40, 0.00, 0.00])

scores = []

for doc, vector in document_vectors.items():
    score = cosine_similarity(query_vector, vector)
    scores.append((score, doc))

scores = sorted(scores, reverse=True)

print("Semantic search results:")
for score, doc in scores:
    print(f"{score:.4f} → {doc}")

# %% Notebook cell 13
# Practice 4
# Dimensions: [waterproof, warmth, hiking, footwear, drinkware]

warm_winter_query = np.array([0.40, 0.95, 0.20, 0.00, 0.00])

scores = []

for doc, vector in document_vectors.items():
    score = cosine_similarity(warm_winter_query, vector)
    scores.append((score, doc))

scores = sorted(scores, reverse=True)

for score, doc in scores:
    print(f"{score:.4f} → {doc}")

# %% Notebook cell 14
def semantic_search(query_vector, document_vectors, n_results=3):
    scores = []

    for doc, vector in document_vectors.items():
        score = cosine_similarity(query_vector, vector)
        scores.append((score, doc))

    scores = sorted(scores, reverse=True)
    return scores[:n_results]

results = semantic_search(query_vector, document_vectors, n_results=3)

for score, doc in results:
    print(f"{score:.4f} → {doc}")

# %% Notebook cell 15
try:
    import chromadb
    print("ChromaDB is installed.")
except ImportError:
    print("ChromaDB is not installed.")
    print("Install it with: pip install chromadb")

# %% Notebook cell 16
try:
    import chromadb

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="student_memories")

    print("ChromaDB client and collection are ready.")

except ImportError:
    print("ChromaDB not available. Skip ChromaDB sections or install chromadb.")

# %% Notebook cell 17
try:
    collection.add(
        documents=[
            "My favorite movie genre is sci-fi",
            "I enjoy working out in the morning",
            "Pizza is my go-to comfort food"
        ],
        ids=["memory1", "memory2", "memory3"]
    )

    print("Documents added.")

except NameError:
    print("Collection is not available.")
except Exception as e:
    print("Something happened while adding documents:")
    print(e)
    print("This may happen if IDs already exist. Try changing the IDs.")

# %% Notebook cell 18
try:
    results = collection.query(
        query_texts=["What food do I like the most?"],
        n_results=2
    )

    print(results)

except NameError:
    print("Collection is not available.")
except Exception as e:
    print("Query failed:")
    print(e)

# %% Notebook cell 19
# Practice 5

try:
    collection.add(
        documents=[
            "I like studying at quiet coffee shops",
            "I prefer rainy weather for reading books",
            "My favorite weekend activity is hiking near mountains"
        ],
        ids=["fact1_unique", "fact2_unique", "fact3_unique"]
    )

    results = collection.query(
        query_texts=["Where do I enjoy learning?"],
        n_results=2
    )

    print(results)

except NameError:
    print("Collection is not available.")
except Exception as e:
    print("This may happen if the IDs already exist. Try changing the IDs.")
    print(e)

# %% Notebook cell 20
try:
    query = "What activity do I enjoy outdoors?"

    results_1 = collection.query(
        query_texts=[query],
        n_results=1
    )

    results_3 = collection.query(
        query_texts=[query],
        n_results=3
    )

    print("n_results=1")
    print(results_1)

    print("\nn_results=3")
    print(results_3)

except NameError:
    print("Collection is not available.")
except Exception as e:
    print(e)

# %% Notebook cell 21
try:
    product_collection = client.get_or_create_collection(name="product_search_demo")

    product_collection.add(
        documents=[
            "Vintage leather jacket with silver zippers",
            "Lightweight raincoat for wet weather",
            "Trail running shoes with strong grip",
            "Insulated winter parka for snow",
            "Breathable windbreaker for spring hikes"
        ],
        ids=["product1", "product2", "product3", "product4", "product5"],
        metadatas=[
            {"category": "clothing", "uri": "./images/leather_jacket.jpg"},
            {"category": "clothing", "uri": "./images/raincoat.jpg"},
            {"category": "footwear", "uri": "./images/trail_shoes.jpg"},
            {"category": "clothing", "uri": "./images/winter_parka.jpg"},
            {"category": "clothing", "uri": "./images/windbreaker.jpg"}
        ]
    )

    print("Products added.")

except NameError:
    print("ChromaDB client is not available.")
except Exception as e:
    print("This may happen if IDs already exist. Try changing IDs or collection name.")
    print(e)

# %% Notebook cell 22
try:
    results = product_collection.query(
        query_texts=["waterproof outerwear"],
        n_results=3
    )

    print(results)

except NameError:
    print("Product collection is not available.")
except Exception as e:
    print(e)

# %% Notebook cell 23
metadata_map = [
    {"id": "product1", "uri": "./images/leather_jacket.jpg"},
    {"id": "product2", "uri": "./images/raincoat.jpg"},
    {"id": "product3", "uri": "./images/trail_shoes.jpg"},
    {"id": "product4", "uri": "./images/winter_parka.jpg"},
    {"id": "product5", "uri": "./images/windbreaker.jpg"}
]

for item in metadata_map:
    print(item["id"], "→", item["uri"])

# %% Notebook cell 24
# Word Neighbors Activity

theme = "Outdoor Adventure Gear"

anchor = "A lightweight jacket for breezy spring weather"

neighbors = [
    "Breathable windbreaker for mild hiking days",
    "Thin rain shell for wet spring walks",
    "Insulated winter parka for freezing snow",
    "Cotton t-shirt for summer workouts",
    "Camping lantern with rechargeable battery"
]

print("Theme:", theme)
print("Anchor:", anchor)
print("\nRank these by semantic closeness:")

for i, item in enumerate(neighbors, start=1):
    print(i, item)

# %% Notebook cell 25
# Practice 6: Write your ranking as comments

# 1.
# 2.
# 3.
# 4.
# 5.

# Explanation:

# %% Notebook cell 26
anchor = "A lightweight jacket for breezy spring weather"
keyword_trap = "A light for a jacket pocket"

print("Anchor:", anchor)
print("Keyword trap:", keyword_trap)
print("\nQuestion: Why might keyword search be fooled?")
print("Answer: It sees similar words but may miss the different meaning.")

# %% Notebook cell 27
# Final Reflection Answers

# 1.
# 2.
# 3.
# 4.
# 5.
# 6.
# 7.
# 8.
# 9.
# 10.
# 11.
# 12.
# 13.
