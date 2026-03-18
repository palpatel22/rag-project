# 🤖 RAG (Intelligent Document Q&A System)

A Python-based Retrieval Augmented Generation system# 🔧 Technology Employed
LLM: LLaMA 3.1 (through Groq)
BM25 + FAISS Hybrid Search for Retrieval
SentenceTransformers: **Embeddings
PDF, TXT, and CSV files are supported.🚀 How to Run
1. Use pip install -r requirement.txt to install dependencies.

2. Include your Groq API key in the `.env`: GROQ_API_KEY=your_key_here

3. Add files to the folder "data."

4. Execute: python rag_pipeline.py ## Features
BM25 + FAISS hybrid retrieval
Self-assessment (completeness, relevance, faithfulness)
PDF, TXT, and CSV files are supported.
