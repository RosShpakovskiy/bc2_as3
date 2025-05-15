# bc2_as3

A Streamlit-based AI assistant that enables users to interact with the **Constitution of the Republic of Kazakhstan** using natural language. Users can query specific articles or legal phrases, and the assistant will retrieve the relevant sections using both keyword matching and semantic search.

---

## Project Structure
```bash
.
AI-Constitution-Assistant
├── app.py
├── data/
│ └── akorda.kz-Constitution of the Republic of Kazakhstan.pdf
```

## Features

- **Article Retrieval**: Get exact constitutional articles by number (e.g. "Article 15")
- **Semantic Search**: Ask legal questions in natural language
- **Exact Phrase Matching**: Find articles containing specific quoted phrases
- **Historical Context**: Includes footnotes and amendments
- **Multi-modal Access**: Combine direct quotes with AI explanations

## Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/AI-Constitution-Assistant.git
cd AI-Constitution-Assistant
```

2. Install dependencies:

```bash
pip install streamlit chromadb langchain pypdf sentence-transformers ollama langchain-community
```

3. Program use:

```bash
ollama serve
ollama pull llama2
streamlit run app.py --server.fileWatcherType none  
```

Ask questions such as:

"What does Article 15 say about right to life?"

"Find provisions about presidential elections"

"Quote 'human dignity shall be inviolable'"

## How it works
Document Processing:
1. Loads Constitution PDF
2. Splits text into articles/sections
3. Extracts metadata (article numbers, sections)

Embedding:
1. Uses all-MiniLM-L6-v2 model for embeddings
2. Stores vectors in ChromaDB

Query Handling:
1. Direct article number matching
2. Exact phrase search for quoted text
3. Semantic search with hybrid LLM response (Llama2)
