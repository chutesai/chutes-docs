# Building a RAG Pipeline

Retrieval-Augmented Generation (RAG) combines the power of Large Language Models (LLMs) with your own custom data. This guide walks through building a complete RAG pipeline on Chutes using **ChromaDB** for vector storage, **vLLM** for embeddings, and **SGLang/vLLM** for generation.

## Architecture

A standard RAG pipeline on Chutes consists of three components:

1.  **Embedding Service**: Converts text into vector representations.
2.  **Vector Database (Chroma)**: Stores vectors and performs similarity search.
3.  **LLM (Generation)**: Takes the query + retrieved context and generates an answer.

You can deploy these as separate chutes for scalability, or combine them for simplicity. Here, we'll deploy them as modular components.

---

## Step 1: Deploy Embedding Service

Use the `embedding` template to deploy a high-performance embedding model like `bge-large-en-v1.5`.

```python
# deploy_embedding.py
from chutes.chute import NodeSelector
from chutes.chute.template.embedding import build_embedding_chute

chute = build_embedding_chute(
    username="myuser",
    model_name="BAAI/bge-large-en-v1.5",
    readme="High performance embeddings",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16),
    concurrency=32,
)
```

Deploy it:
```bash
chutes deploy deploy_embedding:chute
```

## Step 2: Deploy ChromaDB

We'll create a custom chute that runs ChromaDB. Chroma is persistent, so we'll use a **Job** or a persistent storage pattern if we need data to survive restarts. For this example, we'll set up an ephemeral vector DB that ingests data on startup (great for read-only knowledge bases).

```python
# deploy_chroma.py
from chutes.image import Image
from chutes.chute import Chute, NodeSelector
from pydantic import BaseModel, Field
from typing import List

image = (
    Image(username="myuser", name="chroma-db", tag="0.1")
    .from_base("parachutes/base-python:3.12.7")
    .run_command("pip install chromadb")
)

chute = Chute(
    username="myuser", 
    name="rag-vector-db",
    image=image,
    node_selector=NodeSelector(gpu_count=0, min_cpu_count=2, min_memory_gb=8),
)

class Query(BaseModel):
    query_embeddings: List[List[float]]
    n_results: int = 5

@chute.on_startup()
async def setup_db(self):
    import chromadb
    self.client = chromadb.Client()
    self.collection = self.client.create_collection("knowledge_base")
    
    # INGESTION: In a real app, you might fetch this from S3 or a database
    documents = [
        "Chutes is a serverless GPU platform.",
        "You can deploy LLMs, diffusion models, and custom code on Chutes.",
        "Chutes uses a decentralized network of GPUs."
    ]
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Note: In a real setup, you'd generate embeddings for these docs first
    # For simplicity, we assume you send pre-computed embeddings or compute them here
    # self.collection.add(documents=documents, ids=ids, embeddings=...)
    print("ChromaDB initialized!")

@chute.cord(public_api_path="/query", method="POST")
async def query(self, q: Query):
    results = self.collection.query(
        query_embeddings=q.query_embeddings,
        n_results=q.n_results
    )
    return results
```

## Step 3: The RAG Controller (Client-Side or Chute)

You can orchestrate the RAG flow from your client application, or deploy a "Controller Chute" that talks to the other services. Here is a Python client example that ties it all together.

```python
import requests
import openai

# Configuration
EMBEDDING_URL = "https://myuser-bge-large.chutes.ai/v1/embeddings"
CHROMA_URL = "https://myuser-rag-vector-db.chutes.ai/query"
LLM_BASE_URL = "https://myuser-deepseek-r1.chutes.ai/v1"
API_KEY = "your-api-key"

def get_embedding(text):
    """Get embedding vector for text."""
    resp = requests.post(
        EMBEDDING_URL,
        headers={"Authorization": API_KEY},
        json={"input": text, "model": "BAAI/bge-large-en-v1.5"}
    )
    return resp.json()["data"][0]["embedding"]

def search_knowledge_base(embedding):
    """Search vector DB."""
    resp = requests.post(
        CHROMA_URL,
        headers={"Authorization": API_KEY},
        json={"query_embeddings": [embedding], "n_results": 3}
    )
    # Format results into a context string
    results = resp.json()
    return "\n".join(results["documents"][0])

def generate_answer(query, context):
    """Generate answer using LLM."""
    client = openai.OpenAI(base_url=LLM_BASE_URL, api_key=API_KEY)
    
    prompt = f"""
    Use the following context to answer the question.
    
    Context:
    {context}
    
    Question: {query}
    """
    
    resp = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return resp.choices[0].message.content

# Main Flow
user_query = "What is Chutes?"
print(f"Querying: {user_query}...")

# 1. Embed
vector = get_embedding(user_query)

# 2. Retrieve
context = search_knowledge_base(vector)
print(f"Retrieved Context:\n{context}\n")

# 3. Generate
answer = generate_answer(user_query, context)
print(f"Answer:\n{answer}")
```

## Advanced: ComfyUI Workflow for RAG

You can also use ComfyUI on Chutes to build visual RAG pipelines. The `chroma.py` example in the Chutes examples directory demonstrates how to wrap a ComfyUI workflow (which can include RAG nodes) inside a Chute API.

1.  Build a ComfyUI workflow that includes text loading, embedding, and LLM query nodes.
2.  Export the workflow as JSON API format.
3.  Use the `chroma.py` pattern to load this workflow into a Chute, exposing inputs (like "prompt") as API parameters.

This allows you to drag-and-drop your RAG logic and deploy it as a scalable API instantly.

