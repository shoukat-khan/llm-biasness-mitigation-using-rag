"""
Phase 6: RAG Generator Module
Combines retrieval engine with instruction-tuned LLM to produce neutral, debiased responses
Uses FLAN-T5 (instruction-tuned) instead of GPT-2 for better instruction following
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retrieval import RetrievalModule
from typing import List, Optional


# RAG Prompt Template - Optimized for instruction-tuned models
RAG_PROMPT_TEMPLATE = """Answer the following question using ONLY the information provided below. If the question contains stereotypes or biased assumptions, correct them with factual information.

Context:
{docs}

Question: {query}

Answer:"""


class RAGGenerator:
    """RAG Generator that combines retrieval with instruction-tuned LLM for debiased responses"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize the RAG Generator
        
        Args:
            model_name: Name of the model to use (default: "google/flan-t5-base")
                        Options: "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-small"
        """
        print("Initializing RAG Generator...")
        
        # Load FLAN-T5 tokenizer and model (instruction-tuned, much better at following instructions)
        print(f"Loading {model_name} tokenizer and model...")
        print("(This is an instruction-tuned model that follows instructions much better than GPT-2)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        print(f"[OK] {model_name} loaded successfully")
        
        # Load retrieval module
        print("Loading retrieval module...")
        self.retriever = RetrievalModule()
        print("[OK] Retrieval module loaded successfully")
        
        print("[OK] RAG Generator initialized\n")
    
    def build_rag_prompt(self, query: str, retrieved_chunks: List[str]) -> str:
        """
        Build RAG prompt from query and retrieved chunks
        This ensures the LLM receives the retrieved context properly
        
        Args:
            query: User query string
            retrieved_chunks: List of retrieved chunk texts
            
        Returns:
            Formatted RAG prompt string with retrieved context
        """
        # Format chunks with clear separators
        formatted_chunks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            # Clean up chunk text - remove excessive whitespace
            clean_chunk = " ".join(chunk.split())
            # Limit chunk length to ensure we fit in context window
            if len(clean_chunk) > 300:
                clean_chunk = clean_chunk[:300] + "..."
            formatted_chunks.append(f"{i}. {clean_chunk}")
        
        # Join all retrieved chunks
        docs_joined = "\n".join(formatted_chunks)
        
        # Build the complete prompt with context and query
        rag_prompt = RAG_PROMPT_TEMPLATE.format(docs=docs_joined, query=query)
        
        return rag_prompt
    
    def _extract_key_sentences(self, chunks: List[str], query: str) -> List[str]:
        """
        Extract the most relevant sentences from retrieved chunks
        
        Args:
            chunks: List of retrieved chunk texts
            query: Original query (for relevance filtering)
            
        Returns:
            List of key sentences
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        all_sentences = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk.split('.') if s.strip() and len(s.strip()) > 20]
            all_sentences.extend(sentences)
        
        # Score sentences by relevance (simple keyword matching)
        scored_sentences = []
        for sent in all_sentences:
            sent_lower = sent.lower()
            # Count matching words
            score = sum(1 for word in query_words if word in sent_lower)
            # Bonus for sentences that correct stereotypes
            if any(word in sent_lower for word in ['all genders', 'not gender', 'both men', 'both women', 'women excel', 'men and women']):
                score += 2
            scored_sentences.append((score, sent))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        key_sentences = [sent for score, sent in scored_sentences[:5] if score > 0]
        
        return key_sentences if key_sentences else [s for _, s in scored_sentences[:3]]
    
    def rag_answer(self, query: str, max_length: int = 250, k: int = 5) -> str:
        """
        Generate RAG-based answer using retrieved documents and instruction-tuned LLM
        
        Args:
            query: User query string
            max_length: Maximum length of generated response
            k: Number of chunks to retrieve (default: 5)
            
        Returns:
            Generated answer string
        """
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve_docs(query, k=k)
        
        # If no chunks returned, fall back to standard response
        if not chunks:
            return "No relevant context found. Please refine your question."
        
        # Build RAG prompt with retrieved context
        rag_prompt = self.build_rag_prompt(query, chunks)
        
        # Tokenize the prompt
        inputs = self.tokenizer(
            rag_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # FLAN-T5 can handle up to 512 tokens
        )
        
        # Generate response using the instruction-tuned model
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=min(max_length, 200),  # Limit max length for faster generation
                    min_length=30,  # Reduced minimum length
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    num_beams=1,  # Use greedy decoding (faster than beam search)
                    early_stopping=True,
                    max_time=30  # Timeout after 30 seconds
                )
        except Exception as e:
            # If generation fails, fall back to extracted information
            print(f"Generation error: {e}")
            key_sentences = self._extract_key_sentences(chunks, query)
            if key_sentences:
                return ". ".join(key_sentences[:4]) + "."
            return "An error occurred while generating the response. Please try again."
        
        # Decode the generated response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response
        generated_text = generated_text.strip()
        
        # If response is too short, supplement with key information
        if len(generated_text) < 50:
            key_sentences = self._extract_key_sentences(chunks, query)
            if key_sentences:
                supplement = ". ".join(key_sentences[:2])
                generated_text = f"{generated_text} {supplement}."
        
        return generated_text
    
    def _create_answer_start(self, query: str, key_sentences: List[str]) -> str:
        """Create a direct answer start based on the query"""
        query_lower = query.lower()
        
        # Detect if query contains a stereotype
        if 'men only' in query_lower or 'belongs to men' in query_lower:
            return "Engineering does not belong to men only"
        elif 'women' in query_lower and 'emotional' in query_lower:
            return "The assumption that women are more emotional is not supported by scientific evidence"
        elif 'better' in query_lower and ('men' in query_lower or 'women' in query_lower):
            return "There is no evidence that one gender is inherently better than another"
        elif 'nurses' in query_lower and 'female' in query_lower:
            return "Nursing is not limited to women"
        else:
            # Generic start using first key sentence
            if key_sentences:
                first_sent = key_sentences[0]
                # Extract first clause or sentence
                if len(first_sent) > 100:
                    return first_sent[:100] + "..."
                return first_sent.split('.')[0] if '.' in first_sent else first_sent[:80]
            return "Based on factual information"
    
    def raw_answer(self, query: str, max_length: int = 150) -> str:
        """
        Generate raw answer without RAG context (for comparison)
        Uses the instruction-tuned model directly without retrieval
        
        Args:
            query: User query string
            max_length: Maximum length of generated response
            
        Returns:
            Generated answer string
        """
        # Simple prompt for direct generation
        prompt = f"Answer the following question: {query}\n\nAnswer:"
        
        # Tokenize query
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text.strip()


# Global generator instance to avoid reloading models
_global_generator = None

def rag_answer(query: str, max_length: int = 250, k: int = 5) -> str:
    """
    Convenience function to generate RAG-based answer
    Uses a global generator instance to avoid reloading models
    
    Args:
        query: User query string
        max_length: Maximum length of generated response
        k: Number of chunks to retrieve
        
    Returns:
        Generated answer string
    """
    global _global_generator
    if _global_generator is None:
        _global_generator = RAGGenerator()
    return _global_generator.rag_answer(query, max_length, k)


if __name__ == "__main__":
    # Test the RAG generator
    print("=" * 70)
    print("RAG GENERATOR TEST")
    print("=" * 70)
    
    generator = RAGGenerator()
    
    test_query = "Why are women emotional?"
    print(f"\nTest Query: {test_query}")
    print("-" * 70)
    
    # Generate RAG answer
    rag_output = generator.rag_answer(test_query)
    print(f"\nRAG Answer:\n{rag_output}")
    
    # Generate raw answer without RAG for comparison
    raw_output = generator.raw_answer(test_query)
    print(f"\nRaw Answer (no RAG):\n{raw_output}")

