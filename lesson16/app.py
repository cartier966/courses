import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.decomposition import PCA

st.set_page_config(page_title='Lesson 16 RAG Lab', page_icon='🧠', layout='wide')
st.title('🧠 Lesson 16 — Embeddings, ChromaDB, Top-K, and Chat Memory')
st.caption('Build a realistic RAG pipeline with live OpenAI embeddings, persistent ChromaDB, Top-K retrieval, merged context, and Streamlit session memory.')

CHUNKS = [
    {'id':'vacation_001','text':'Full-time employees receive 15 paid vacation days every calendar year.','section':'Vacation Policy','source':'employee_handbook.pdf'},
    {'id':'vacation_002','text':'Vacation requests should be submitted to the manager at least two weeks in advance.','section':'Vacation Policy','source':'employee_handbook.pdf'},
    {'id':'vacation_003','text':'Unused vacation days do not carry forward into the next calendar year.','section':'Vacation Policy','source':'employee_handbook.pdf'},
    {'id':'remote_001','text':'Employees may work remotely up to two days per week with manager approval.','section':'Remote Work','source':'employee_handbook.pdf'},
    {'id':'remote_002','text':'Remote employees must remain available during core hours from 10 AM to 3 PM.','section':'Remote Work','source':'employee_handbook.pdf'},
    {'id':'security_001','text':'Passwords must not be shared and confidential files must be stored in approved systems.','section':'Security','source':'employee_handbook.pdf'},
    {'id':'security_002','text':'Security incidents must be reported to the IT team immediately.','section':'Security','source':'employee_handbook.pdf'},
]

def read_key():
    try:
        return st.secrets.get('OPENAI_API_KEY')
    except Exception:
        return None

@st.cache_resource
def openai_client(api_key):
    return OpenAI(api_key=api_key)

@st.cache_resource
def collection(name):
    client = chromadb.PersistentClient(path='./lesson16_chroma_db')
    return client.get_or_create_collection(name=name)

def get_embedding(client, text, model):
    return client.embeddings.create(model=model, input=text).data[0].embedding

def grounded_answer(client, model, question, rows):
    context = '\n\n'.join(
        f"[Chunk {r['rank']} | {r['section']}]\n{r['text']}" for r in rows
    )
    instructions = (
        'You are a document-grounded assistant. '
        'Use only the supplied context. '
        'If unsupported, say exactly: "I do not know based on the provided documents." '
        'Cite relevant evidence using [Chunk N]. Do not invent policies or citations.'
    )
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=f'CONTEXT:\n{context}\n\nQUESTION:\n{question}'
    )
    return response.output_text

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'all_vectors' not in st.session_state:
    st.session_state.all_vectors = None

api_key = read_key()
client = openai_client(api_key) if api_key else None

with st.sidebar:
    st.header('⚙️ Controls')
    if api_key:
        st.success(f"OpenAI key loaded: {api_key[:5]}...{api_key[-4:]}")
    else:
        st.error('OpenAI key not found.')
        st.code('.streamlit/secrets.toml\n\nOPENAI_API_KEY = "your_key_here"', language='toml')
    embedding_model = st.selectbox('Embedding model', ['text-embedding-3-small','text-embedding-3-large'])
    generation_model = st.text_input('Generation model', value='gpt-5-mini')
    collection_name = st.text_input('Chroma collection', value='lesson16_knowledge')
    top_k = st.slider('Top-K chunks', 1, 7, 3)
    if st.button('Clear Chat'):
        st.session_state.messages = []
        st.rerun()

col = collection(collection_name)

with st.expander('What are we building today?', expanded=True):
    st.markdown('''
In Lesson 15, documents were cleaned and chunked.

Today the flow is:

**Chunk → Embedding → ChromaDB → Top-K Retrieval → Merged Context → LLM Answer**

We also add **Streamlit Session State** so the conversation remains visible across reruns.
''')

t1,t2,t3,t4,t5,t6 = st.tabs(['1. Embeddings','2. ChromaDB','3. Top-K Lab','4. RAG Chat','5. Session State','6. Full Pipeline'])

with t1:
    st.header('Step 1 — Convert Text Chunks into Embeddings')
    idx = st.selectbox('Choose a chunk', range(len(CHUNKS)), format_func=lambda i: f"{CHUNKS[i]['section']} — {CHUNKS[i]['text'][:60]}")
    record = CHUNKS[idx]
    st.code(record['text'])
    if st.button('Generate Live Embedding', type='primary'):
        if not client:
            st.error('Add your API key first.')
        else:
            vec = get_embedding(client, record['text'], embedding_model)
            arr = np.asarray(vec)
            a,b,c = st.columns(3)
            a.metric('Dimensions', len(vec)); b.metric('Min', f'{arr.min():.4f}'); c.metric('Max', f'{arr.max():.4f}')
            st.dataframe(pd.DataFrame({'Dimension':range(1,26),'Value':vec[:25]}), hide_index=True)
            st.line_chart(pd.DataFrame({'Dimension':range(1,min(101,len(vec)+1)),'Value':vec[:100]}).set_index('Dimension'))

    st.subheader('Visualize All Chunk Embeddings')
    if st.button('Generate All Embeddings'):
        if not client:
            st.error('Add your API key first.')
        else:
            st.session_state.all_vectors = [get_embedding(client, r['text'], embedding_model) for r in CHUNKS]
            st.success('All vectors generated.')
    if st.session_state.all_vectors:
        points = PCA(n_components=2).fit_transform(np.asarray(st.session_state.all_vectors))
        df = pd.DataFrame({'Chunk':[r['id'] for r in CHUNKS],'Section':[r['section'] for r in CHUNKS],'PCA 1':points[:,0],'PCA 2':points[:,1]})
        st.scatter_chart(df, x='PCA 1', y='PCA 2')
        st.caption('PCA compresses the high-dimensional vectors into 2D for teaching.')

with t2:
    st.header('Step 2 — Store Embeddings in Persistent ChromaDB')
    st.dataframe(pd.DataFrame(CHUNKS), hide_index=True, use_container_width=True)
    if st.button('Embed and Upsert All Chunks', type='primary'):
        if not client:
            st.error('Add your API key first.')
        else:
            with st.status('Building vector database...', expanded=True) as status:
                vectors = []
                for i,r in enumerate(CHUNKS,1):
                    st.write(f'Embedding chunk {i}/{len(CHUNKS)}')
                    vectors.append(get_embedding(client, r['text'], embedding_model))
                col.upsert(
                    ids=[r['id'] for r in CHUNKS],
                    documents=[r['text'] for r in CHUNKS],
                    embeddings=vectors,
                    metadatas=[{'source':r['source'],'section':r['section'],'chunk_index':i,'embedding_model':embedding_model} for i,r in enumerate(CHUNKS)]
                )
                status.update(label='Knowledge base ready', state='complete')
            st.success(f'Stored {col.count()} chunks.')
    st.metric('Collection count', col.count())
    if st.button('Peek at Stored Records'):
        st.json(col.peek())


def run_query(question, k):
    qvec = get_embedding(client, question, embedding_model)
    results = col.query(query_embeddings=[qvec], n_results=min(k, max(1,col.count())))
    rows = []
    docs = results.get('documents',[[]])[0]
    metas = results.get('metadatas',[[]])[0]
    distances = results.get('distances',[[]])[0]
    for rank,(doc,meta,distance) in enumerate(zip(docs,metas,distances),1):
        rows.append({'rank':rank,'text':doc,'section':meta.get('section','Unknown'),'source':meta.get('source','Unknown'),'distance':distance})
    return rows

with t3:
    st.header('Step 3 — Top-K Retrieval and Context Merging')
    q = st.text_input('Question', value='Explain the vacation policy.', key='topk_q')
    if st.button('Compare K = 1, 3, and 5'):
        if col.count() == 0:
            st.error('Populate ChromaDB first.')
        elif not client:
            st.error('Add your API key first.')
        else:
            summary = []
            for k in [1,3,5]:
                rows = run_query(q,k)
                merged = '\n\n'.join(r['text'] for r in rows)
                summary.append({'Top-K':k,'Retrieved Chunks':len(rows),'Context Characters':len(merged)})
                with st.expander(f'Top-{k} Context', expanded=(k==3)):
                    for r in rows:
                        st.markdown(f"**Rank {r['rank']} · {r['section']}**")
                        st.write(r['text'])
                    st.code(merged, language='text')
            st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
    st.warning("Too much context can bury useful evidence. This is related to the 'lost in the middle' problem.")

with t4:
    st.header('Step 4 — Persistent RAG Chat')
    for m in st.session_state.messages:
        with st.chat_message(m['role']):
            st.write(m['content'])
            if m.get('sources'):
                with st.expander('Sources'):
                    for s in m['sources']:
                        st.write(f'- {s}')
    question = st.chat_input('Ask the knowledge base a question')
    if question:
        if col.count() == 0:
            st.error('Populate ChromaDB first.')
        elif not client:
            st.error('Add your API key first.')
        else:
            st.session_state.messages.append({'role':'user','content':question})
            with st.chat_message('user'):
                st.write(question)
            with st.chat_message('assistant'):
                with st.spinner('Retrieving and generating...'):
                    rows = run_query(question, top_k)
                    answer = grounded_answer(client, generation_model, question, rows)
                st.write(answer)
                with st.expander('Retrieved Evidence'):
                    for r in rows:
                        st.markdown(f"**Chunk {r['rank']} · {r['section']}**")
                        st.write(r['text'])
                        st.caption(f"Distance: {r['distance']:.4f}")
            sources = [f"{r['source']} — {r['section']}" for r in rows]
            st.session_state.messages.append({'role':'assistant','content':answer,'sources':sources})
    st.code('''How many vacation days do employees receive?
How early should vacation be requested?
Can employees work remotely?
What are the required remote work hours?
What should happen after a security incident?
Does the company provide dental insurance?''', language='text')

with t5:
    st.header('Step 5 — Understand Streamlit Session State')
    st.markdown('Streamlit reruns the whole script after most interactions. Session state keeps the chat history alive.')
    st.code('''if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state.messages.append({"role": "user", "content": question})
st.session_state.messages.append({"role": "assistant", "content": answer})''', language='python')
    st.write('Messages stored:', len(st.session_state.messages))
    if st.session_state.messages:
        st.dataframe(pd.DataFrame([{'Order':i+1,'Role':m['role'],'Content':m['content']} for i,m in enumerate(st.session_state.messages)]), hide_index=True, use_container_width=True)
    st.info('Session state preserves UI memory. Prior messages must still be explicitly sent to the model if you want the LLM itself to use them as conversational context.')

with t6:
    st.header('The Complete Lesson 16 Architecture')
    stages = [
        ('1','Clean and chunk','Prepare focused text units.'),
        ('2','Embed chunks','Convert text into vectors.'),
        ('3','Persist in ChromaDB','Store vectors, text, IDs, and metadata.'),
        ('4','Embed the question','Place the query in the same vector space.'),
        ('5','Top-K retrieval','Find the nearest chunks.'),
        ('6','Merge context','Combine multiple relevant chunks.'),
        ('7','Ground generation','Answer using only retrieved evidence.'),
        ('8','Store chat history','Preserve conversation with session state.'),
    ]
    for number,title,desc in stages:
        left,right = st.columns([1,8])
        with left:
            st.markdown(f'## {number}')
        with right:
            st.markdown(f'### {title}')
            st.write(desc)
