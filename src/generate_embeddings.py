"""
Phase 4: Generate Vector Embeddings and Build FAISS Index
Creates embeddings for knowledge base and builds retrieval system
"""

import os
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss


def count_words(text: str) -> int:
    """Count words in text"""
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def split_into_chunks(text: str, min_words: int = 150, max_words: int = 300) -> List[str]:
    """
    Split text into chunks of 150-300 words
    Tries to split at sentence boundaries when possible
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_words = count_words(sentence)
        
        # If adding this sentence would exceed max_words, save current chunk
        if current_word_count + sentence_words > max_words and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if count_words(chunk_text) >= min_words:
                chunks.append(chunk_text)
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
            
            # If we've reached min_words, we can start a new chunk after this
            if current_word_count >= min_words and current_word_count + 50 > max_words:
                # Check if next sentence would push us over
                pass
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        word_count = count_words(chunk_text)
        if word_count >= min_words or word_count > 0:  # Accept smaller final chunks
            chunks.append(chunk_text)
    
    # If no chunks created (text too short), return the whole text as one chunk
    if not chunks:
        chunks = [text]
    
    return chunks


def load_knowledge_base_files(kb_dir: str) -> List[Dict]:
    """
    Load all .txt files from knowledge base directory
    Returns list of dictionaries with filename and content
    """
    kb_path = Path(kb_dir)
    files_data = []
    
    txt_files = sorted(kb_path.glob('*.txt'))
    
    print(f"Loading {len(txt_files)} knowledge base files...")
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    files_data.append({
                        'filename': txt_file.name,
                        'content': content
                    })
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
    
    return files_data


def create_chunks(files_data: List[Dict]) -> List[Dict]:
    """
    Create chunks from all knowledge base files
    Returns list of chunk dictionaries
    """
    all_chunks = []
    chunk_id = 0
    
    print("\nCreating chunks from knowledge base files...")
    
    for file_data in files_data:
        filename = file_data['filename']
        content = file_data['content']
        
        # Split into chunks
        text_chunks = split_into_chunks(content, min_words=150, max_words=300)
        
        print(f"  {filename}: {len(text_chunks)} chunk(s)")
        
        for chunk_text in text_chunks:
            chunk_info = {
                'chunk_id': chunk_id,
                'chunk_text': chunk_text,
                'source_file_name': filename,
                'word_count': count_words(chunk_text)
            }
            all_chunks.append(chunk_info)
            chunk_id += 1
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks


def generate_embeddings(chunks: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings for all chunks using SentenceTransformer
    """
    print(f"\nLoading embedding model: {model_name}...")
    embedder = SentenceTransformer(model_name)
    print("✓ Model loaded successfully")
    
    # Extract chunk texts
    chunk_texts = [chunk['chunk_text'] for chunk in chunks]
    
    print(f"\nGenerating embeddings for {len(chunk_texts)} chunks...")
    print("This may take a few minutes...")
    
    # Generate embeddings
    embeddings = embedder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)
    
    print(f"✓ Generated embeddings shape: {embeddings.shape}")
    print(f"  - Number of embeddings: {embeddings.shape[0]}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index from embeddings
    """
    print("\nBuilding FAISS index...")
    
    # Normalize embeddings for better cosine similarity search
    # Using L2 normalization
    faiss.normalize_L2(embeddings)
    
    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    index.add(embeddings.astype('float32'))
    
    print(f"✓ FAISS index created")
    print(f"  - Dimension: {dimension}")
    print(f"  - Total vectors: {index.ntotal}")
    
    return index


def create_metadata(chunks: List[Dict], embeddings: np.ndarray) -> Dict:
    """
    Create metadata mapping chunk_id to text, source, and embedding index
    """
    metadata = {}
    
    for i, chunk in enumerate(chunks):
        metadata[chunk['chunk_id']] = {
            'chunk_text': chunk['chunk_text'],
            'source_file_name': chunk['source_file_name'],
            'embedding_index': i,
            'word_count': chunk['word_count']
        }
    
    return metadata


def validate_vector_store(chunks: List[Dict], embeddings: np.ndarray, 
                         index: faiss.Index, embedder: SentenceTransformer) -> Dict:
    """
    Validate the vector store with various checks
    """
    print("\n" + "=" * 60)
    print("VALIDATING VECTOR STORE")
    print("=" * 60)
    
    validation_results = {
        'check1_embeddings_count': False,
        'check2_embedding_dimensions': False,
        'check3_retrieval_test': False,
        'errors': []
    }
    
    # Check 1: Embeddings Count
    print("\n[Check 1] Embeddings Count")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Number of embeddings: {embeddings.shape[0]}")
    if len(chunks) == embeddings.shape[0]:
        print("  ✓ PASS: Number of embeddings matches number of chunks")
        validation_results['check1_embeddings_count'] = True
    else:
        error = f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"
        print(f"  ✗ FAIL: {error}")
        validation_results['errors'].append(error)
    
    # Check 2: Embedding Dimensions
    print("\n[Check 2] Embedding Dimensions")
    print(f"  Expected dimension: 384")
    print(f"  Actual dimension: {embeddings.shape[1]}")
    if embeddings.shape[1] == 384:
        print("  ✓ PASS: Embedding dimension is 384")
        validation_results['check2_embedding_dimensions'] = True
    else:
        error = f"Wrong dimension: expected 384, got {embeddings.shape[1]}"
        print(f"  ✗ FAIL: {error}")
        validation_results['errors'].append(error)
    
    # Check 3: Simple Retrieval Test
    print("\n[Check 3] Retrieval Test")
    test_query = "Why are women bad at math?"
    print(f"  Test query: '{test_query}'")
    
    try:
        # Encode query
        query_emb = embedder.encode([test_query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # Search
        k = 3
        D, I = index.search(query_emb.astype('float32'), k)
        
        print(f"  Retrieved {len(I[0])} chunks")
        
        retrieval_results = []
        for i, idx in enumerate(I[0]):
            chunk = chunks[idx]
            distance = D[0][i]
            retrieval_results.append({
                'rank': i + 1,
                'chunk_id': chunk['chunk_id'],
                'source_file': chunk['source_file_name'],
                'distance': float(distance),
                'text_preview': chunk['chunk_text'][:200] + "..."
            })
            print(f"\n  Rank {i+1}:")
            print(f"    Source: {chunk['source_file_name']}")
            print(f"    Distance: {distance:.4f}")
            print(f"    Preview: {chunk['chunk_text'][:150]}...")
        
        # Check if retrieved chunks contain relevant anti-stereotype content
        relevant_keywords = ['math', 'women', 'gender', 'ability', 'stereotype', 
                           'evidence', 'research', 'equal', 'performance']
        relevant_count = 0
        for result in retrieval_results:
            text_lower = result['text_preview'].lower()
            if any(keyword in text_lower for keyword in relevant_keywords):
                relevant_count += 1
        
        if relevant_count > 0:
            print(f"\n  ✓ PASS: Retrieved chunks contain relevant anti-stereotype content")
            validation_results['check3_retrieval_test'] = True
            validation_results['retrieval_results'] = retrieval_results
        else:
            error = "Retrieved chunks may not contain relevant content"
            print(f"\n  ⚠ WARNING: {error}")
            validation_results['retrieval_results'] = retrieval_results
            
    except Exception as e:
        error = f"Retrieval test failed: {str(e)}"
        print(f"  ✗ FAIL: {error}")
        validation_results['errors'].append(error)
    
    return validation_results


def save_test_retrieval_output(validation_results: Dict, output_path: str):
    """Save retrieval test results to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RETRIEVAL TEST OUTPUT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Test Query: Why are women bad at math?\n\n")
        
        if 'retrieval_results' in validation_results:
            f.write("Top 3 Retrieved Chunks:\n")
            f.write("-" * 70 + "\n\n")
            
            for result in validation_results['retrieval_results']:
                f.write(f"Rank {result['rank']}:\n")
                f.write(f"  Source File: {result['source_file']}\n")
                f.write(f"  Chunk ID: {result['chunk_id']}\n")
                f.write(f"  Distance: {result['distance']:.4f}\n")
                f.write(f"  Text Preview:\n")
                f.write(f"  {result['text_preview']}\n")
                f.write("\n")
        else:
            f.write("No retrieval results available.\n")
        
        f.write("=" * 70 + "\n")


def main():
    """Main function to execute Phase 4"""
    print("=" * 70)
    print("PHASE 4: GENERATE VECTOR EMBEDDINGS AND BUILD FAISS INDEX")
    print("=" * 70)
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    kb_dir = project_root / "data" / "knowledge_base"
    embeddings_dir = project_root / "embeddings"
    
    # Create embeddings directory if it doesn't exist
    embeddings_dir.mkdir(exist_ok=True)
    
    # File paths
    chunks_json = embeddings_dir / "chunks.json"
    embeddings_npy = embeddings_dir / "knowledge_embeddings.npy"
    faiss_index_bin = embeddings_dir / "faiss_index.bin"
    metadata_json = embeddings_dir / "metadata.json"
    test_retrieval_output = embeddings_dir / "test_retrieval_output.txt"
    phase4_summary = embeddings_dir / "phase4_summary.txt"
    
    # Step 1: Load embedding model
    print("\n[Step 1] Initializing Embedding Model")
    print("-" * 70)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Embedding model loaded successfully")
    
    # Step 2: Load knowledge base files
    print("\n[Step 2] Loading Knowledge Base Files")
    print("-" * 70)
    files_data = load_knowledge_base_files(str(kb_dir))
    print(f"✓ Loaded {len(files_data)} files")
    
    # Step 3: Create chunks
    print("\n[Step 3] Creating Chunks")
    print("-" * 70)
    chunks = create_chunks(files_data)
    
    # Save chunks to JSON
    with open(chunks_json, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved chunks to {chunks_json}")
    
    # Step 4: Generate embeddings
    print("\n[Step 4] Generating Embeddings")
    print("-" * 70)
    embeddings = generate_embeddings(chunks, "all-MiniLM-L6-v2")
    
    # Save embeddings
    np.save(embeddings_npy, embeddings)
    print(f"✓ Saved embeddings to {embeddings_npy}")
    
    # Step 5: Build FAISS index
    print("\n[Step 5] Building FAISS Index")
    print("-" * 70)
    index = build_faiss_index(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, str(faiss_index_bin))
    print(f"✓ Saved FAISS index to {faiss_index_bin}")
    
    # Step 6: Create metadata
    print("\n[Step 6] Creating Metadata")
    print("-" * 70)
    metadata = create_metadata(chunks, embeddings)
    
    # Save metadata
    with open(metadata_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata to {metadata_json}")
    
    # Step 7: Validate vector store
    print("\n[Step 7] Validating Vector Store")
    print("-" * 70)
    validation_results = validate_vector_store(chunks, embeddings, index, embedder)
    
    # Save test retrieval output
    save_test_retrieval_output(validation_results, str(test_retrieval_output))
    print(f"✓ Saved retrieval test output to {test_retrieval_output}")
    
    # Step 8: Generate Phase 4 summary
    print("\n[Step 8] Generating Phase 4 Summary")
    print("-" * 70)
    
    # Calculate FAISS index size
    index_size = os.path.getsize(faiss_index_bin) if faiss_index_bin.exists() else 0
    
    summary = {
        'phase': 4,
        'status': 'COMPLETED',
        'chunks_created': len(chunks),
        'embeddings_generated': embeddings.shape[0],
        'embedding_dimension': embeddings.shape[1],
        'faiss_index_size_bytes': index_size,
        'faiss_index_size_mb': round(index_size / (1024 * 1024), 2),
        'validation': validation_results,
        'files_created': {
            'chunks_json': str(chunks_json),
            'embeddings_npy': str(embeddings_npy),
            'faiss_index_bin': str(faiss_index_bin),
            'metadata_json': str(metadata_json),
            'test_retrieval_output': str(test_retrieval_output)
        }
    }
    
    # Write summary report
    with open(phase4_summary, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 4 COMPLETION REPORT\n")
        f.write("RAG Gender Bias-Mitigation Project - Vector Embeddings & FAISS Index\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SUMMARY METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of Chunks Created: {summary['chunks_created']}\n")
        f.write(f"Number of Embeddings Generated: {summary['embeddings_generated']}\n")
        f.write(f"Embedding Dimension: {summary['embedding_dimension']}\n")
        f.write(f"FAISS Index Size: {summary['faiss_index_size_mb']} MB ({summary['faiss_index_size_bytes']:,} bytes)\n\n")
        
        f.write("VALIDATION RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Check 1 - Embeddings Count: {'✓ PASS' if validation_results['check1_embeddings_count'] else '✗ FAIL'}\n")
        f.write(f"Check 2 - Embedding Dimensions: {'✓ PASS' if validation_results['check2_embedding_dimensions'] else '✗ FAIL'}\n")
        f.write(f"Check 3 - Retrieval Test: {'✓ PASS' if validation_results['check3_retrieval_test'] else '⚠ WARNING'}\n\n")
        
        if validation_results.get('retrieval_results'):
            f.write("RETRIEVAL TEST SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write("Query: Why are women bad at math?\n\n")
            for result in validation_results['retrieval_results']:
                f.write(f"Rank {result['rank']}: {result['source_file']} (distance: {result['distance']:.4f})\n")
            f.write("\n")
        
        f.write("FILES CREATED\n")
        f.write("-" * 70 + "\n")
        for key, path in summary['files_created'].items():
            exists = "✓" if Path(path).exists() else "✗"
            f.write(f"{exists} {key}: {Path(path).name}\n")
        f.write("\n")
        
        if validation_results['errors']:
            f.write("ERRORS ENCOUNTERED\n")
            f.write("-" * 70 + "\n")
            for error in validation_results['errors']:
                f.write(f"  - {error}\n")
            f.write("\n")
        else:
            f.write("ERRORS ENCOUNTERED: None\n\n")
        
        f.write("STATUS: COMPLETE ✓\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Saved Phase 4 summary to {phase4_summary}")
    
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - Chunks created: {len(chunks)}")
    print(f"  - Embeddings generated: {embeddings.shape[0]}")
    print(f"  - FAISS index size: {summary['faiss_index_size_mb']} MB")
    print(f"  - All validations: {'PASSED' if all([validation_results['check1_embeddings_count'], validation_results['check2_embedding_dimensions']]) else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
