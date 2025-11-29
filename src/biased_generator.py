"""
Phase 7: Biased Generator Module
Produces raw LLM outputs with no retrieval or debiasing.
Uses FLAN-T5 (same as RAG mode) but without retrieval context.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load FLAN-T5 tokenizer and model (standalone, no retrieval)
MODEL_NAME = "google/flan-t5-base"

print("[BiasedGenerator] Loading FLAN-T5 tokenizer and model...")
print("(Using FLAN-T5 without retrieval - this shows raw model behavior)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
print("[BiasedGenerator] FLAN-T5 loaded successfully.")


def generate_biased_output(query: str, max_length: int = 200) -> str:
    """
    Generate a raw LLM output with NO retrieval and NO debiasing.

    - Accepts a plain user query
    - Generates output with no bias filtering or context
    - Returns exactly what the model produces without retrieval
    - Serves as the baseline (may still be better than GPT-2 due to instruction tuning)
    """
    # Create a simple prompt for direct generation (no context)
    prompt = f"Answer the following question: {query}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            min_length=30,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            num_beams=2,
            early_stopping=True
        )

    # Decode the generated response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text.strip()


def biased_mode_api(query: str) -> str:
    """
    Wrapper for web/API integration in later phases.

    Usage in web layer (Phase 8):
        from biased_generator import biased_mode_api
        answer = biased_mode_api(user_query)
    """
    return generate_biased_output(query)


if __name__ == "__main__":
    # Simple manual sanity check (optional)
    test_query = "Why are women emotional?"
    print("\n[BiasedGenerator] Self-test with query:")
    print(f"  {test_query}\n")
    output = generate_biased_output(test_query)
    print("Raw LLM output (no retrieval):")
    print(output)