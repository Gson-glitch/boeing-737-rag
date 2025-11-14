# Boeing 737 Operations Manual RAG System

A Retrieval-Augmented Generation (RAG) system for querying the Boeing 737 Operations Manual.

## 🎯 Features

- **Hybrid Retrieval**: Combines semantic search (BGE-M3) with keyword search (BM25)
- **Contextual Chunking**: Uses Gemini to add context to document chunks
- **Cross-Encoder Reranking**: Improves retrieval accuracy
- **Citation Tracking**: Returns relevant page numbers with answers
- **REST API**: FastAPI endpoint for easy integration

## 📊 Performance

- **Hit Rate@1**: 80%
- **Hit Rate@3**: 100%
- **MRR@10**: 0.90
- **nDCG@10**: 0.90
- **Page Precision**: 100%

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- UV package manager
- Google Gemini API key
- poppler-utils
- tesseract-ocr

### Installation
```bash
# Clone repository
git clone https://github.com/Gson-glitch/boeing-737-rag.git
cd boeing-737-rag

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Setup
```bash
# 1. Process the manual (chunking + contextualization)
python scripts/process_manual.py

# 2. Build search indices
python scripts/build_index.py
```

### Run API Server
```bash
python main.py
```

Server runs at `http://localhost:8000`

### Query Example
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the amber STAIRS OPER light indicate?"}'
```

Response:
```json
{
  "answer": "The amber STAIRS Operating (OPER) light indicates the airstair is in transit [Document 1].",
  "pages": [126]
}
```

## 🧪 Testing
### Run Evaluation
```bash
python scripts/evaluate_system.py
```

## 📁 Project Structure
```
boeing-737-rag/
├── data/
│   ├── raw/              # Original PDF
│   └── processed/        # Chunks and indices
├── src/
│   ├── ingestion/        # PDF processing and chunking
│   ├── retrieval/        # Hybrid search and reranking
│   ├── generation/       # LLM answer generation
│   └── api/              # FastAPI routes
├── scripts/              # Setup and testing scripts
└── main.py               # API entry point
```

## 🛠️ Tech Stack

- **Embeddings**: BAAI/bge-m3
- **Reranker**: BAAI/bge-reranker-v2-m3
- **LLM**: Google Gemini 2.5 Flash
- **Vector DB**: ChromaDB
- **Keyword Search**: BM25
- **API**: FastAPI
