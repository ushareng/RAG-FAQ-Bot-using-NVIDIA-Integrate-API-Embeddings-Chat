import os
import requests
import numpy as np

# ----------------------------
# Config
# ----------------------------
API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    raise SystemExit("❌ NVIDIA_API_KEY not set. Export it first.")

BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Embeddings model (works for you; you saw dim=1024)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

# Chat model (callable for your account; your curl returned HTTP/2 200)
CHAT_MODEL = os.getenv("CHAT_MODEL", "meta/llama-3.1-8b-instruct")

TOP_K = int(os.getenv("TOP_K", "4"))

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ----------------------------
# Synthetic FAQ dataset
# ----------------------------
FAQ = [
    ("What is NeMo Retriever?",
     "NeMo Retriever is a retrieval component used for RAG. It fetches relevant context and can use reranking to improve grounding."),
    ("What is RAG?",
     "RAG (Retrieval-Augmented Generation) combines retrieval from a knowledge store with generation by an LLM."),
    ("How do I reduce hallucinations?",
     "Use retrieval with good chunking and instruct the model to only answer from context; add guardrails if needed."),
    ("What is an embedding?",
     "An embedding is a numeric vector representation of text used for semantic similarity search."),
    ("What is reranking?",
     "Reranking re-orders retrieved passages using a stronger relevance model for better ranking."),
    ("What should I do if the answer is not found?",
     "Return 'I don't know based on the provided FAQ' instead of guessing.")
]
DOCS = [f"Q: {q}\nA: {a}" for q, a in FAQ]

SYSTEM = (
    "You are a FAQ assistant.\n"
    "Answer ONLY using the provided context.\n"
    "If the answer is not in the context, say: \"I don't know based on the provided FAQ.\"\n"
    "Keep it short and practical."
)

# ----------------------------
# Embeddings (REST)
# ----------------------------
def embed_texts(texts, input_type: str):
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "input_type": input_type,  # "query" or "passage"
    }
    r = requests.post(f"{BASE_URL}/embeddings", headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    return [item["embedding"] for item in r.json()["data"]]

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

print("Indexing synthetic FAQ with embeddings…")
doc_vecs = np.array([
    normalize(np.array(v, dtype=np.float32))
    for v in embed_texts(DOCS, input_type="passage")
])
print(f"✅ Indexed {len(DOCS)} chunks")

# ----------------------------
# Retrieve
# ----------------------------
def retrieve(question: str, k: int = TOP_K):
    qv = normalize(np.array(embed_texts([question], input_type="query")[0], dtype=np.float32))
    scores = doc_vecs @ qv
    idx = np.argsort(-scores)[:k]
    return [DOCS[i] for i in idx]

# ----------------------------
# Chat (REST)
# ----------------------------
def chat_complete(system: str, user: str) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }
    r = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def answer(question: str) -> str:
    ctx = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(retrieve(question, TOP_K))])
    user_prompt = f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    return chat_complete(SYSTEM, user_prompt)

# ----------------------------
# CLI loop
# ----------------------------
print("\n🤖 Synthetic FAQ Bot ready. Type 'exit' to quit.\n")
while True:
    q = input("You: ").strip()
    if not q or q.lower() in {"exit", "quit"}:
        break
    try:
        print("\nBot:", answer(q), "\n")
    except Exception as e:
        print("\n❌ Error:", e, "\n")
