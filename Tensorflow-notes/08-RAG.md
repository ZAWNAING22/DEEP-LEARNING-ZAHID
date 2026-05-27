# What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines:

1. **Retrieval system** → searches relevant information
2. **LLM (Large Language Model)** → generates the final answer

Instead of relying only on what the model memorized during training, RAG lets the model **look up external knowledge** before answering.

---

# Simple Idea

Without RAG:

```text
Question → LLM → Answer
```

With RAG:

```text
Question
   ↓
Retriever searches documents
   ↓
Relevant chunks are found
   ↓
LLM reads those chunks
   ↓
Final answer generated
```

---

# Real Example

Suppose you build a chatbot for your university PDFs.

User asks:

> “What is the attendance policy?”

RAG workflow:

1. System searches PDFs
2. Finds relevant paragraph
3. Sends paragraph + question to LLM
4. LLM answers using that paragraph

So the model becomes:

* more accurate
* up-to-date
* domain-specific

---

# Why RAG is Important

LLMs alone have problems:

| Problem                  | RAG Solution               |
| ------------------------ | -------------------------- |
| Hallucination            | Uses real retrieved data   |
| Old knowledge            | Can access latest docs     |
| Cannot know private data | Can read your company PDFs |
| Expensive retraining     | No need to retrain model   |

---

# Core Components of RAG

---

## 1. Documents / Knowledge Base

Your data source:

* PDFs
* websites
* database
* research papers
* company docs
* CSVs
* Notion
* SharePoint

Example:

```text
AI.pdf
Rules.docx
database records
```

---

## 2. Chunking

LLMs cannot read huge documents directly.

So documents are split into small pieces called **chunks**.

Example:

```text
Document → 500-word chunks
```

Why?

Because retrieval works better on small sections.

---

## 3. Embeddings

Each chunk is converted into a numerical vector.

This captures semantic meaning.

Example:

```text
"The cat sits"
→ [0.12, -0.88, 0.45, ...]
```

Embedding models place similar meanings close together in vector space.

---

# Embedding Concept

Words/sentences with similar meaning become nearby vectors.

Example:

```text
"car"
"vehicle"
"automobile"
```

will have similar embeddings.

---

## 4. Vector Database

Embeddings are stored in a vector database.

Popular vector DBs:

* Pinecone
* Weaviate
* Chroma
* Milvus
* Qdrant
* FAISS

The DB performs:

```text
Similarity Search
```

---

## 5. Retriever

When user asks a question:

```text
"What is CNN?"
```

Question is also embedded.

Then vector DB finds:

```text
Most similar chunks
```

using:

* cosine similarity
* nearest neighbor search

---

## 6. LLM Generation

Finally:

```text
Question + Retrieved Chunks
```

are given to the LLM.

Example prompt:

```text
Context:
[CNN explanation chunk]

Question:
What is CNN?

Answer:
```

Then LLM generates grounded answer.

---

# Full RAG Pipeline

```text
                OFFLINE STAGE
Documents
   ↓
Chunking
   ↓
Embedding Model
   ↓
Vector Database


                ONLINE STAGE
User Query
   ↓
Embedding
   ↓
Similarity Search
   ↓
Top Relevant Chunks
   ↓
LLM
   ↓
Final Answer
```

---

# Types of RAG

---

## 1. Naive RAG

Basic retrieval + LLM.

Simplest architecture.

---

## 2. Advanced RAG

Adds:

* reranking
* filtering
* query rewriting
* hybrid search

Better accuracy.

---

## 3. Agentic RAG

LLM can:

* decide which tool to use
* search multiple sources
* reason step-by-step

Very modern approach.

---

# Important RAG Concepts

---

## Top-K Retrieval

Retriever returns top matching chunks.

Example:

```text
Top 5 chunks
```

---

## Similarity Search

Measures semantic closeness.

Common metric:

```text
Cosine Similarity
```

---

## Reranking

After retrieval:

Another model sorts results by relevance.

Improves quality.

---

## Hybrid Search

Combines:

* semantic search (embeddings)
* keyword search (BM25)

Very powerful.

---

# Where RAG is Used

---

## AI Chatbots

Examples:

* company support bots
* PDF QA bots
* university assistants

---

## Search Engines

Modern AI search systems use RAG.

Examples:

* OpenAI ChatGPT browsing
* Perplexity AI
* Google AI Search

---

## Healthcare

Medical document retrieval.

---

## Legal Systems

Search laws/contracts.

---

## Coding Assistants

Retrieve documentation before coding.

Examples:

* GitHub Copilot-style systems

---

# How RAG Differs from Fine-Tuning

| RAG                          | Fine-Tuning             |
| ---------------------------- | ----------------------- |
| External knowledge retrieval | Changes model weights   |
| Easy updates                 | Expensive retraining    |
| Good for dynamic data        | Good for behavior/style |
| Faster to maintain           | More costly             |

---

# Modern RAG Tools (2026)

# 1. Frameworks

These help build RAG pipelines.

---

## [LangChain](https://www.langchain.com?utm_source=chatgpt.com)

Most popular RAG framework.

Features:

* document loaders
* chunking
* vector DB integration
* agents
* memory

Good for beginners.

---

## [LlamaIndex](https://www.llamaindex.ai?utm_source=chatgpt.com)

Focused heavily on RAG/data retrieval.

Excellent for:

* PDFs
* structured docs
* enterprise search

Very popular for production RAG.

---

## [Haystack](https://haystack.deepset.ai?utm_source=chatgpt.com)

Enterprise-grade RAG framework.

Strong for:

* search pipelines
* production deployment

---

# 2. Vector Databases

---

## [Pinecone](https://www.pinecone.io?utm_source=chatgpt.com)

Managed cloud vector DB.

Very easy to use.

---

## [Weaviate](https://weaviate.io?utm_source=chatgpt.com)

Open-source + cloud.

Supports hybrid search.

---

## [Chroma](https://www.trychroma.com?utm_source=chatgpt.com)

Very beginner-friendly.

Good local experiments.

---

## [FAISS](https://github.com/facebookresearch/faiss?utm_source=chatgpt.com)

Fast local vector similarity library by Meta.

Excellent for research.

---

# 3. Embedding Models

---

## [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings?utm_source=chatgpt.com)

Very strong quality.

---

## [Sentence Transformers](https://www.sbert.net/?utm_source=chatgpt.com)

Open-source embeddings.

Widely used.

---

## [Cohere Embeddings](https://cohere.com/embed?utm_source=chatgpt.com)

Enterprise-grade embeddings.

---

# 4. LLMs Used in RAG

* OpenAI GPT models
* Anthropic Claude
* Google Gemini
* Meta Llama models
* Mistral AI Mistral

---

# Typical Modern Stack

Very common today:

```text
Frontend:
React / Next.js

Backend:
FastAPI

RAG Framework:
LangChain or LlamaIndex

Embedding:
OpenAI embeddings

Vector DB:
Pinecone / Chroma

LLM:
GPT-4 / Claude / Llama
```

---

# Simple Python Example (Modern RAG)

Using [LangChain](https://www.langchain.com?utm_source=chatgpt.com) + [Chroma](https://www.trychroma.com?utm_source=chatgpt.com)

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load PDF
loader = PyPDFLoader("book.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Create embeddings
embedding = OpenAIEmbeddings()

# Store in vector DB
vectordb = Chroma.from_documents(
    chunks,
    embedding
)

# Retriever
retriever = vectordb.as_retriever()

# LLM
llm = ChatOpenAI()

# RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Ask question
response = qa.run("Explain CNN")

print(response)
```

---

# How You Should Learn RAG

Since you already study deep learning and TensorFlow, learn RAG in this order:

---

## Stage 1 — Basics

Learn:

* embeddings
* cosine similarity
* vector DB
* chunking

---

## Stage 2 — Simple RAG

Build:

* PDF chatbot
* website QA bot

Use:

* LangChain
* Chroma

---

## Stage 3 — Advanced RAG

Learn:

* hybrid search
* reranking
* metadata filtering
* query expansion

---

## Stage 4 — Production RAG

Learn:

* FastAPI
* streaming
* caching
* evaluation
* latency optimization

---

## Stage 5 — Agentic RAG

Learn:

* AI agents
* tool calling
* memory
* multi-step retrieval

---

# Best Beginner Stack

If you are starting today:

| Component  | Recommendation                 |
| ---------- | ------------------------------ |
| Framework  | LangChain                      |
| Vector DB  | Chroma                         |
| Embeddings | OpenAI or SentenceTransformers |
| LLM        | GPT or Llama                   |
| Backend    | FastAPI                        |

---

# Important Reality

RAG is currently one of the most important skills in AI engineering.

Many modern AI apps are basically:

```text
RAG + LLM + Agents
```

instead of training huge models from scratch.
