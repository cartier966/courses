
from __future__ import annotations
import math, re
from collections import Counter
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAG Preparation Visualizer", layout="wide")
st.title("RAG Preparation Visualizer")
st.caption("Visualize cleaning, chunking, retrieval, and prompt augmentation.")

DEFAULT_TEXT = """EMPLOYEE HANDBOOK
1

Vacation Policy

Full-time employees receive 15 paid
vacation days every calendar year.     Employees must submit
requests to their manager.

EMPLOYEE HANDBOOK
2

Remote Work Policy

Employees may work remotely up to two days
per week.	Manager approval is required.

EMPLOYEE HANDBOOK
3

Security Policy

Passwords must not be shared. Confidential files
must be stored in approved systems.
"""

def clean_text(text, header, remove_pages, repair):
    stages = []
    text = text.replace("\\x00", "").replace("\\u00a0", " ")
    stages.append(("Control characters", text))
    if header:
        text = text.replace(header, "")
        stages.append(("Repeated header", text))
    if remove_pages:
        text = "\n".join(line for line in text.splitlines() if not line.strip().isdigit())
        stages.append(("Page numbers", text))
    text = re.sub(r"[ \t]+", " ", text)
    stages.append(("Spaces and tabs", text))
    if repair:
        text = re.sub(r"(?<!\n)\n(?=[a-z])", " ", text)
        stages.append(("Mid-sentence line breaks", text))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    stages.append(("Blank lines", text))
    return text, stages

def fixed_chunks(text, size):
    return [text[i:i+size] for i in range(0, len(text), size)]

def overlapping_chunks(text, size, overlap):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks

def sentence_chunks(text, count):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [" ".join(sentences[i:i+count]) for i in range(0, len(sentences), count)]

def recursive_split(text, max_size, separators=None):
    separators = separators or ["\n\n", ". ", " ", ""]
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]
    sep = separators[0]
    if sep == "":
        return fixed_chunks(text, max_size)
    chunks, current = [], ""
    for part in text.split(sep):
        candidate = part if not current else current + sep + part
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks += recursive_split(current, max_size, separators[1:])
            current = part
    if current:
        chunks += recursive_split(current, max_size, separators[1:])
    return [c.strip() for c in chunks if c.strip()]

STOPWORDS = {"the","a","an","and","or","to","of","in","is","are","be","for","on","with","this","that","how","many","do","does","what"}

def tokenize(text):
    return [w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if w not in STOPWORDS]

def score(query, chunk):
    q, c = Counter(tokenize(query)), Counter(tokenize(chunk))
    shared = set(q) & set(c)
    if not shared:
        return 0.0
    return sum(min(q[w], c[w]) for w in shared) / math.sqrt(max(1, len(tokenize(chunk))))

def retrieve(query, chunks, top_k):
    rows = [{"chunk":i, "text":c, "score":score(query,c)} for i,c in enumerate(chunks)]
    return sorted(rows, key=lambda x:x["score"], reverse=True)[:top_k]

def prompt(query, rows):
    context = "\n\n".join(f"[Chunk {r['chunk']}]\n{r['text']}" for r in rows)
    return (
        "You are a document-grounded assistant.\n\n"
        "Use only the context below.\n"
        'If unsupported, say: "I do not know based on the uploaded documents."\n\n'
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
    )

with st.sidebar:
    uploaded = st.file_uploader("Upload TXT", type=["txt"])
    header = st.text_input("Header to remove", "EMPLOYEE HANDBOOK")
    remove_pages = st.checkbox("Remove page numbers", True)
    repair = st.checkbox("Repair line breaks", True)

source = uploaded.getvalue().decode("utf-8") if uploaded else DEFAULT_TEXT
source = st.text_area("Raw document", source, height=240)
cleaned, stages = clean_text(source, header, remove_pages, repair)

a,b,c = st.columns(3)
a.metric("Raw characters", len(source))
b.metric("Cleaned characters", len(cleaned))
c.metric("Removed", len(source)-len(cleaned))

t1,t2,t3,t4 = st.tabs(["Cleaning","Chunking","Retrieval","Pipeline"])

with t1:
    left,right = st.columns(2)
    with left:
        st.markdown("**Raw**"); st.code(source)
    with right:
        st.markdown("**Cleaned**"); st.code(cleaned)
    for i,(name,value) in enumerate(stages,1):
        with st.expander(f"{i}. {name}"):
            st.code(value)

with t2:
    x,y,z = st.columns(3)
    with x:
        strategy = st.selectbox("Strategy", ["Fixed","Sliding window","Sentence","Recursive"])
    with y:
        size = st.slider("Chunk size", 50, 1000, 220, 10)
    with z:
        overlap = st.slider("Overlap", 0, max(0,size-10), min(50,max(0,size-10)), 10)
    sentences = st.slider("Sentences per chunk", 1, 8, 2)

    if strategy == "Fixed":
        chunks = fixed_chunks(cleaned,size)
    elif strategy == "Sliding window":
        chunks = overlapping_chunks(cleaned,size,overlap)
    elif strategy == "Sentence":
        chunks = sentence_chunks(cleaned,sentences)
    else:
        chunks = recursive_split(cleaned,size)

    lengths = [len(chunk) for chunk in chunks]
    q1,q2,q3 = st.columns(3)
    q1.metric("Chunks",len(chunks))
    q2.metric("Average",round(sum(lengths)/len(lengths),1) if lengths else 0)
    q3.metric("Largest",max(lengths) if lengths else 0)

    st.bar_chart(pd.DataFrame({"chunk":range(1,len(chunks)+1),"characters":lengths}),
                 x="chunk",y="characters")
    for i,chunk in enumerate(chunks,1):
        with st.expander(f"Chunk {i} · {len(chunk)} characters", expanded=i<=2):
            st.write(chunk)

with t3:
    query = st.text_input("Question","How many vacation days do employees receive?")
    top_k = st.slider("Retrieved chunks",1,min(10,max(1,len(chunks))),min(3,max(1,len(chunks))))
    rows = retrieve(query,chunks,top_k)
    for rank,row in enumerate(rows,1):
        st.markdown(f"**Rank {rank} · Chunk {row['chunk']} · Score {row['score']:.3f}**")
        st.write(row["text"])
    st.subheader("Augmented prompt")
    st.code(prompt(query,rows),language="text")
    if any(row["score"]>=0.15 for row in rows):
        st.success("The retrieved context may support a grounded answer.")
    else:
        st.warning('The assistant should say: "I do not know based on the uploaded documents."')

with t4:
    pipeline = [
        ("1","Load","Read the document"),
        ("2","Clean","Remove extraction noise"),
        ("3","Chunk","Create focused units"),
        ("4","Embed","Convert chunks into vectors"),
        ("5","Store","Save vectors and metadata"),
        ("6","Retrieve","Find relevant chunks"),
        ("7","Augment","Insert context into the prompt"),
        ("8","Generate","Answer from the evidence"),
    ]
    for n,title,desc in pipeline:
        left,right = st.columns([1,8])
        with left: st.markdown(f"### {n}")
        with right:
            st.markdown(f"**{title}**")
            st.caption(desc)
    st.info("Retrieval is simulated with word overlap. Replace it later with embeddings and ChromaDB.")
