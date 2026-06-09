# 🌍 Semantic Climate Policy RAG Engine & Ingestion Pipeline

An end-to-end, local, open-weights Retrieval-Augmented Generation (RAG) pipeline engineered to semantically query and summarize dense, unstructured climate policy frameworks and sustainability guidelines. 

## 🚀 Project Overview
* **Architected an open-weights RAG pipeline** utilizing token-based recursive chunking and a FAISS vector store to achieve sub-millisecond semantic retrieval latency over dense climate policy frameworks.
* **Deployed a 4-bit quantized Qwen2.5-7B model** on an NVIDIA T4 GPU, integrated with an interactive Gradio web interface to deliver secure, context-bounded analysis for policy researchers.

---

## 🛠️ Tech Stack & Architecture

* **Orchestration Framework:** LangChain & LangChain-Community
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
* **Large Language Model:** `Qwen/Qwen2.5-7B-Instruct` (Quantized via BitsAndBytes)
* **Frontend UI:** Gradio Web Interface
* **Compute Environment:** Google Colab / NVIDIA T4 GPU

---

## 🔑 Key Features

### 1. Token-Aware Native Ingestion Engine
Unlike standard character-based splitters, this engine utilizes a native Hugging Face tokenizer wrapper to recursively chunk PDF text corpora by precise token counts (**1000-token chunks with a 200-token overlap**). This maximizes the context retention of heavy environmental policy documents.

### 2. High-Performance Semantic Vector Indexing
Leverages a lightweight embedding model to construct a mathematical vector space of text segments. FAISS indexing provides sub-millisecond $k$-nearest neighbors ($k$-NN) retrieval latency to isolate relevant policy context blocks dynamically.

### 3. Hardware-Optimized LLM Execution
Implements **4-bit NormalFloat (NF4) quantization** with double quantization support via `bitsandbytes`. This compression maps a high-performance 7-billion parameter model safely inside 14GB of VRAM on a commodity T4 GPU without degrading legal reasoning capacities.

### 4. Guardrailed Context-Bounded Logic
Designed with a zero-hallucination prompt topology. The model evaluates user inputs strictly against the retrieved semantic vectors. If a localized scientific or geographical query falls outside the uploaded framework document (e.g., specific regional weather tracking), the system executes a graceful, context-bound rejection rather than fabricating text.

---

## 📦 Getting Started & Usage

### Prerequisites
To run this pipeline locally or in a notebook environment, ensure you have an NVIDIA GPU (minimum 16GB VRAM for non-quantized or 8GB VRAM for 4-bit quantized execution) and Python 3.10+. using google colab  is advise as it is easy to learn 

