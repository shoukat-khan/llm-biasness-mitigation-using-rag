# LLM Bias Mitigation Using RAG

A comprehensive project demonstrating how Retrieval-Augmented Generation (RAG) can be used to mitigate gender bias in Large Language Models (LLMs).

## 🎯 Project Overview

This project implements a dual-mode system that compares:
- **Biased Mode**: Raw LLM outputs without bias mitigation
- **Unbiased Mode**: RAG-enhanced outputs using a curated knowledge base

The system uses **FLAN-T5** (google/flan-t5-base) as the base model and demonstrates how injecting factual, anti-bias knowledge through RAG can significantly reduce harmful stereotypes in AI-generated content.

## 📋 Features

- ✅ Complete RAG pipeline implementation
- ✅ Curated knowledge base with 77 anti-bias content files
- ✅ FAISS vector store for efficient semantic search
- ✅ Streamlit web interface with dual-mode comparison
- ✅ Comprehensive evaluation framework
- ✅ Complete documentation (3-part final report)

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────┐
│  Web Interface  │  (Streamlit)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
Biased Mode  RAG Mode
    │         │
    │    ┌────┴────┐
    │    │         │
    │ Retrieval  FLAN-T5
    │    │         │
    │ FAISS    Knowledge
    │ Index    Base
    │
FLAN-T5
(No Context)
```

## 📁 Repository Structure

```
rag_gender_bias_project/
├── src/                    # Core source code
│   ├── retrieval.py        # Retrieval module
│   ├── rag_generator.py    # RAG generator
│   ├── biased_generator.py # Biased baseline generator
│   └── phase*.txt          # Phase summaries
├── data/                   # Data files
│   ├── knowledge_base/     # 77 anti-bias text files
│   ├── baseline_prompts.json
│   └── baseline_gpt2_outputs.csv
├── embeddings/             # Vector store
│   ├── faiss_index.bin
│   ├── metadata.json
│   ├── chunks.json
│   └── knowledge_embeddings.npy
├── web/                    # Web application
│   ├── app.py             # Streamlit app
│   └── logs/              # Interaction logs
├── cleanup/                # Documentation
│   ├── remaining_files_checklist.txt
│   └── phase10_summary.txt
├── FINAL_PROJECT_REPORT_PART1.md
├── FINAL_PROJECT_REPORT_PART2.md
├── FINAL_PROJECT_REPORT_PART3.md
└── requirements.txt
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shoukat-khan/llm-biasness-mitigation-using-rag.git
cd llm-biasness-mitigation-using-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Web App

```bash
streamlit run web/app.py
```

The app will open at `http://localhost:8501`

### Using the Interface

1. **Biased Mode Tab**: Enter a query to see raw LLM output without bias mitigation
2. **Unbiased RAG Mode Tab**: Enter the same query to see RAG-enhanced, debiased output

**Example Queries**:
- "Why are women emotional?"
- "Are men better at programming?"
- "Why are nurses usually female?"
- "Who is better at math, men or women?"

## 🔧 Technical Details

### Models Used
- **LLM**: FLAN-T5 (google/flan-t5-base) - Instruction-tuned Seq2Seq model
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2) - 384-dimensional embeddings
- **Vector Store**: FAISS (IndexFlatL2) - Fast similarity search

### Knowledge Base
- **77 files** covering gender equality, occupations, abilities, and stereotypes
- **595 chunks** embedded and indexed
- **100% relevance** in test queries

### RAG Pipeline
1. Query embedding using SentenceTransformer
2. Semantic search in FAISS index
3. Top-k chunk retrieval (default k=5)
4. Prompt construction with retrieved context
5. FLAN-T5 generation with factual context

## 📊 Evaluation

The project includes:
- Baseline bias evaluation (25 prompts)
- Toxicity scoring using Detoxify
- Qualitative comparison of biased vs unbiased outputs
- Complete evaluation framework

## 📚 Documentation

Complete project documentation is available in:
- **FINAL_PROJECT_REPORT_PART1.md**: Project overview, Phases 1-4, Repository structure (Part 1)
- **FINAL_PROJECT_REPORT_PART2.md**: Phases 5-9, Repository structure (Part 2)
- **FINAL_PROJECT_REPORT_PART3.md**: System architecture, Evaluation, Instructions, Conclusion

## 🎓 Project Phases

1. **Phase 1**: Project Setup
2. **Phase 2**: Baseline GPT-2 Bias Evaluation
3. **Phase 3**: Knowledge Base Construction
4. **Phase 4**: Embeddings + FAISS Vector Store
5. **Phase 5**: Retrieval Pipeline
6. **Phase 6**: RAG Debiased Generator
7. **Phase 7**: Biased GPT-2 Generator
8. **Phase 8**: Web Interface (Two Modes)
9. **Phase 9**: Model Evaluation & Bias Comparison
10. **Phase 10**: Repository Cleanup & Final Documentation

## 🔍 Key Findings

- RAG successfully reduces bias by injecting factual knowledge
- Knowledge base provides effective context for bias correction
- FLAN-T5 follows instructions better than base models
- Side-by-side comparison demonstrates clear improvement

## ⚠️ Limitations

- Knowledge base limited to 77 files (595 chunks)
- Model size constraints (FLAN-T5-base)
- Evaluation limited to 25 test prompts
- Requires manual knowledge base curation

## 🔮 Future Improvements

- Expand knowledge base with more diverse content
- Use larger instruction-tuned models
- Implement reranking for better chunk selection
- Add more comprehensive evaluation metrics
- Fine-tune models on anti-bias data

## 📝 License

This project is open source and available for educational and research purposes.

## 👤 Author

**Shoukat Khan**

## 🙏 Acknowledgments

- Hugging Face for transformers and models
- SentenceTransformers for embeddings
- FAISS for vector search
- Streamlit for web interface
- Detoxify for bias evaluation

## 📧 Contact

For questions or contributions, please open an issue on GitHub.

---

**Repository**: [https://github.com/shoukat-khan/llm-biasness-mitigation-using-rag](https://github.com/shoukat-khan/llm-biasness-mitigation-using-rag)

