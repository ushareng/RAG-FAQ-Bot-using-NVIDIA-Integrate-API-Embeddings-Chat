# RAG FAQ Bot using NVIDIA Integrate API (Embeddings + Chat)

A **minimal Retrieval-Augmented Generation (RAG) FAQ assistant** built using the **NVIDIA Integrate API**.  
This project demonstrates how to combine **semantic embeddings**, **in-memory vector search**, and **LLM chat completion** to answer questions strictly from retrieved context.

🔗 **GitHub Repository:**  
https://github.com/ushareng/RAG-FAQ-Bot-using-NVIDIA-Integrate-API-Embeddings-Chat

---

## 🚀 Overview

This project implements a lightweight RAG pipeline:

1. FAQ documents are embedded using **NVIDIA embeddings**
2. Embeddings are stored in memory (NumPy)
3. User queries are embedded and matched via cosine similarity
4. Top-K relevant chunks are injected into an LLM prompt
5. The LLM answers **only from retrieved context**

The goal is to provide a **clear, dependency-light RAG reference** without LangChain, vector databases, or orchestration frameworks.

---

## 🧠 Architecture

```
FAQ Documents
     ↓
NVIDIA Embeddings API
     ↓
In-Memory Vector Index (NumPy)
     ↓
Cosine Similarity Retrieval
     ↓
Context Injection
     ↓
NVIDIA Chat Completions API (Llama 3.1)
     ↓
Answer
```

---

## 🧰 Technologies Used

- **Python 3.9+**
- **NVIDIA Integrate API**
  - `nvidia/nv-embedqa-e5-v5` (Embeddings)
  - `meta/llama-3.1-8b-instruct` (Chat)
- `requests`
- `numpy`

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/ushareng/RAG-FAQ-Bot-using-NVIDIA-Integrate-API-Embeddings-Chat.git
cd RAG-FAQ-Bot-using-NVIDIA-Integrate-API-Embeddings-Chat
```

### 2. Install dependencies
```bash
pip install numpy requests
```

---

## 🔑 Configuration

Set your **NVIDIA API key** as an environment variable:

```bash
export NVIDIA_API_KEY="nvapi-xxxxxxxxxxxxxxxx"
```

Optional overrides:

```bash
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
export EMBED_MODEL="nvidia/nv-embedqa-e5-v5"
export CHAT_MODEL="meta/llama-3.1-8b-instruct"
export TOP_K=4
```

---

## ▶️ Run the Application

```bash
python rag_faq_bot.py
```

---

## 🛡️ Hallucination Control

The system prompt enforces:

- Answers only from retrieved context
- Explicit fallback when the answer is not found

---

## 📁 Project Structure

```
.
├── rag_faq_bot.py
└── README.md
```

---

## 🔮 Possible Extensions

- Add a persistent vector store
- Add NeMo Retriever reranking
- Load documents from PDFs
- Add citations
- Deploy as an API

---

## 📜 License

Provided for educational and demonstration purposes.
