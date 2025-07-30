# Real Estate RAG System

A Retrieval-Augmented Generation (RAG) system for real estate platform that combines structured property data and unstructured policy guidelines to provide intelligent responses to user queries.

## Features

- Processes both structured (CSV) and unstructured (PDF) data
- Uses OpenAI embeddings for semantic search
- FAISS-based vector storage for efficient retrieval
- GPT-3.5-turbo integration for natural language responses
- Handles property listings and policy guidelines queries

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. Place your data files:
- `properties.csv` - Structured property listings
- `guidelines.pdf` - Real estate policy guidelines

## Usage

1. Build the knowledge base:
```python
from rag_system import RAGSystem

system = RAGSystem(csv_path='properties.csv', pdf_path='guidelines.pdf')
system.build_knowledge_base()
```

2. Run the CLI:
```python
python rag_system.py
```

## Example Queries

- "Which 2BHK properties under ₹80L in Velachery with gym and metro nearby?"
- "What are the rules for constructing floors near rivers in Tamil Nadu?"
- "List semi-furnished apartments with pool and school access in Adyar under 1.2 Cr."

## Data Processing

- Properties CSV:
  - Normalizes prices (e.g., "₹1.2 Cr" → 12000000)
  - Converts amenities and nearby locations to lists
  - Filters out sold properties
  - Creates vector embeddings using OpenAI

- Guidelines PDF:
  - Extracts text and creates chunks
  - Preserves section information
  - Creates embeddings using OpenAI
