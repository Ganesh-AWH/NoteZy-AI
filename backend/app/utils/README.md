# Utils Module

Contains utility functions used across the backend.

# Utils Work Flow

```
Document Text
↓
Chunking
↓
Embeddings
↓
Vector Store (Index)
↓
Similarity Search
↓
Relevant Context → LLM
```

## Files

### File_loader.py

    Loads PDFs and text files and extracting texts from it for this we use hybrid technique 
    1. Text-Based Extraction 
        Used when the document contains machine-readable text.
    2. OCR-Based Extraction (Fallback)
        Used when the document is scanned or text extraction fails

## Text_cleaner.py → Cleans and normalizes text

    It is responsible for cleaning, normalizing, and preparing raw text extracted from documents

    Core Responsibilities
        1. Normalize whitespace and line breaks
        2. Remove unwanted special characters
        3. Fix encoding issues
        4. Standardize punctuation
        5. Prepare text for chunking and embeddings

## Chunker.py

    It is responsible for splitting large blocks of cleaned text into smaller, manageable chunks that can be efficiently processed by embedding models and Large Language Models (LLMs).

## Embeddings.py → Generates vector embeddings

    It is responsible for converting text chunks into numerical vector representations (embeddings) that capture the semantic meaning of the text.

    Responsibilites:
        1. Receive text chunks from chunker.py
        2. Generate vector embeddings using a transformer model
        3. Normalize vectors for cosine similarity
        4. Return embeddings for storage or querying

## Vector_store.py

    The Vector Store is responsible for storing, indexing, and retrieving vector embeddings generated from document chunks. It enables fast semantic search by finding text that is meaningfully similar to a user’s query.
    
    For vector storing we used FAISS(Facebook AI Semantic Search) vector database since it is very fast and accurate 

    High-Level Workflow
        1. Receive embeddings from embeddings.py
        2. Store vectors in an indexed structure
        3. Convert user query into an embedding
        4. Search for nearest vectors
        5. Return most relevant text chunks

## Usage

These functions are stateless and reused by multiple services.
