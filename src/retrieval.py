"""
Phase 5: Retrieval Module for RAG Bias-Mitigation
Fetches relevant anti-bias knowledge from FAISS vector store
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss


class RetrievalModule:
    """Retrieval module for fetching relevant anti-bias knowledge"""
    
    def __init__(self, embeddings_dir: Optional[str] = None):
        """
        Initialize the retrieval module
        
        Args:
            embeddings_dir: Path to embeddings directory. If None, uses default project structure.
        """
        # Get paths
        if embeddings_dir is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            embeddings_dir = project_root / "embeddings"
        else:
            embeddings_dir = Path(embeddings_dir)
        
        self.embeddings_dir = embeddings_dir
        self.faiss_index_path = embeddings_dir / "faiss_index.bin"
        self.metadata_path = embeddings_dir / "metadata.json"
        
        # Initialize components
        self.embedder = None
        self.index = None
        self.metadata = None
        self._initialized = False
        
    def _load_components(self):
        """Load embedding model, FAISS index, and metadata"""
        if self._initialized:
            return
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[OK] Embedding model loaded")
        
        # Load FAISS index
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.faiss_index_path}. "
                "Please run Phase 4 to generate the index."
            )
        
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.faiss_index_path))
        print(f"[OK] FAISS index loaded ({self.index.ntotal} vectors)")
        
        # Load metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {self.metadata_path}. "
                "Please run Phase 4 to generate metadata."
            )
        
        print("Loading metadata...")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"[OK] Metadata loaded ({len(self.metadata)} entries)")
        
        # Validate metadata length matches FAISS entries
        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"Metadata length ({len(self.metadata)}) does not match "
                f"FAISS index entries ({self.index.ntotal})"
            )
        
        self._initialized = True
        print("[OK] Retrieval module initialized successfully\n")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query into a 384-dimensional vector
        
        Args:
            query: User query string
            
        Returns:
            384-dimensional embedding vector as numpy array
        """
        if not self._initialized:
            self._load_components()
        
        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Normalize for L2 distance (same as index)
        faiss.normalize_L2(query_embedding)
        
        return query_embedding.astype('float32')
    
    def retrieve_docs(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve top-k relevant chunks from knowledge base
        
        Args:
            query: User query string
            k: Number of chunks to retrieve (default: 5)
            
        Returns:
            List of top-k retrieved chunk texts
        """
        # Safety check: empty query
        if not query or not query.strip():
            print("Warning: Empty query provided. Returning empty list.")
            return []
        
        # Initialize components if not already done
        if not self._initialized:
            self._load_components()
        
        # Step A: Embed the query
        query_embedding = self.embed_query(query)
        
        # Step B: Search FAISS
        D, I = self.index.search(query_embedding, k)
        
        # Step C: Use metadata to fetch chunk texts
        retrieved_chunks = []
        seen_texts = set()  # Track to avoid duplicates
        
        for idx in I[0]:
            # Get chunk_id from embedding index
            chunk_id = str(idx)
            
            if chunk_id not in self.metadata:
                print(f"Warning: Chunk ID {chunk_id} not found in metadata")
                continue
            
            chunk_info = self.metadata[chunk_id]
            chunk_text = chunk_info.get('chunk_text', '').strip()
            
            # Safety check: non-zero text
            if not chunk_text:
                print(f"Warning: Empty chunk text for chunk_id {chunk_id}")
                continue
            
            # Avoid duplicates
            if chunk_text not in seen_texts:
                retrieved_chunks.append(chunk_text)
                seen_texts.add(chunk_text)
        
        return retrieved_chunks
    
    def retrieve_docs_with_metadata(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k relevant chunks with full metadata
        
        Args:
            query: User query string
            k: Number of chunks to retrieve (default: 5)
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Safety check: empty query
        if not query or not query.strip():
            print("Warning: Empty query provided. Returning empty list.")
            return []
        
        # Initialize components if not already done
        if not self._initialized:
            self._load_components()
        
        # Step A: Embed the query
        query_embedding = self.embed_query(query)
        
        # Step B: Search FAISS
        D, I = self.index.search(query_embedding, k)
        
        # Step C: Use metadata to fetch chunk texts with metadata
        retrieved_chunks = []
        seen_texts = set()  # Track to avoid duplicates
        
        for i, idx in enumerate(I[0]):
            # Get chunk_id from embedding index
            chunk_id = str(idx)
            
            if chunk_id not in self.metadata:
                print(f"Warning: Chunk ID {chunk_id} not found in metadata")
                continue
            
            chunk_info = self.metadata[chunk_id]
            chunk_text = chunk_info.get('chunk_text', '').strip()
            
            # Safety check: non-zero text
            if not chunk_text:
                print(f"Warning: Empty chunk text for chunk_id {chunk_id}")
                continue
            
            # Avoid duplicates
            if chunk_text not in seen_texts:
                result = {
                    'rank': len(retrieved_chunks) + 1,
                    'chunk_text': chunk_text,
                    'source_file': chunk_info.get('source_file_name', 'unknown'),
                    'chunk_id': chunk_id,
                    'distance': float(D[0][i]),
                    'embedding_index': chunk_info.get('embedding_index', idx)
                }
                retrieved_chunks.append(result)
                seen_texts.add(chunk_text)
        
        return retrieved_chunks


# Convenience function for easy import
def retrieve_docs(query: str, k: int = 5) -> List[str]:
    """
    Convenience function to retrieve documents
    
    Args:
        query: User query string
        k: Number of chunks to retrieve (default: 5)
        
    Returns:
        List of top-k retrieved chunk texts
    """
    retriever = RetrievalModule()
    return retriever.retrieve_docs(query, k)


if __name__ == "__main__":
    # Test the retrieval module
    print("=" * 70)
    print("RETRIEVAL MODULE TEST")
    print("=" * 70)
    
    retriever = RetrievalModule()
    
    test_queries = [
        "Why are women emotional?",
        "Why are nurses female?",
        "Who is better at coding?",
        "Are men better leaders than women?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        results = retriever.retrieve_docs(query, k=5)
        print(f"Retrieved {len(results)} chunks")
        for i, chunk in enumerate(results[:3], 1):
            print(f"\n  Chunk {i} (first 150 chars):")
            print(f"  {chunk[:150]}...")

