# FINAL PROJECT REPORT - PART 1
## RAG Gender Bias-Mitigation Project
## Sections A, B (Phases 1-4), and C (Repository Structure - Part 1)

---

## SECTION A — PROJECT OVERVIEW

### Why GPT-2 Produces Bias

GPT-2, released in 2019, is a base language model trained on a large corpus of internet text. It produces bias for several reasons:

1. **Training Data Bias**: GPT-2 was trained on web-scraped text that contains societal biases, stereotypes, and discriminatory content present in online discourse.

2. **No Instruction Tuning**: GPT-2 is a base model without instruction tuning, meaning it doesn't understand or follow explicit instructions to be unbiased or factual.

3. **Statistical Patterns**: The model learns statistical patterns from training data, including harmful stereotypes about gender, race, and other protected characteristics.

4. **No Factual Grounding**: Without external knowledge, GPT-2 relies solely on patterns in its training data, which may include outdated or biased information.

5. **Completion vs. Instruction**: GPT-2 is designed for text completion, not for following instructions or correcting biases in queries.

**Note**: The current implementation uses **FLAN-T5** instead of GPT-2 for better instruction following, but the biased mode still demonstrates baseline behavior without retrieval context.

### Why RAG is Used for Debiasing

Retrieval-Augmented Generation (RAG) is used for debiasing because:

1. **Factual Knowledge Injection**: RAG retrieves factual, anti-bias information from a curated knowledge base and injects it into the generation context.

2. **Grounding in Evidence**: Instead of relying on potentially biased training data, the model uses evidence-based information from the knowledge base.

3. **Explicit Bias Correction**: The RAG prompt template explicitly instructs the model to correct stereotypes and use only factual information.

4. **Dynamic Context**: For each query, relevant anti-bias content is retrieved and provided to the model, ensuring contextually appropriate responses.

5. **Knowledge Base Control**: The knowledge base is carefully curated with factual, anti-stereotype content, giving control over what information the model uses.

### Goal of the System

The goal of this system is to:

1. **Demonstrate Bias**: Show how base language models (like GPT-2/FLAN-T5) can produce biased outputs when used without mitigation.

2. **Mitigate Bias**: Use RAG to provide factual, anti-bias context that guides the model to produce neutral, evidence-based responses.

3. **Enable Comparison**: Allow side-by-side comparison between biased (no retrieval) and unbiased (RAG-enhanced) outputs.

4. **Provide Evidence-Based Answers**: Ensure responses are grounded in factual knowledge rather than potentially biased training data.

5. **Educational Tool**: Serve as a demonstration of how RAG can be used to reduce bias in AI systems.

### Summary of Biased vs Unbiased Modes

**Biased Mode (Baseline)**:
- Uses FLAN-T5 without any retrieval or context
- Generates responses based solely on model's training data
- May contain stereotypes, biased assumptions, or harmful content
- Serves as the control/baseline for comparison
- Demonstrates raw model behavior

**Unbiased Mode (RAG-Enhanced)**:
- Uses FLAN-T5 with RAG (Retrieval-Augmented Generation)
- Retrieves relevant anti-bias knowledge from curated knowledge base
- Generates responses using factual, evidence-based context
- Corrects stereotypes and provides neutral, factual answers
- Demonstrates bias mitigation through knowledge injection

---

## SECTION B — COMPLETE PHASE DOCUMENTATION

### PHASE 1 — PROJECT SETUP

**Objective**: Establish the project environment, directory structure, and dependencies.

**Components Created**:
- Project directory structure (`rag_gender_bias_project/`)
- Source code directory (`src/`)
- Data directory (`data/`)
- Embeddings directory (`embeddings/`)
- Tests directory (`tests/`)
- Web directory (`web/`)
- Requirements file (`requirements.txt`)

**Files Produced**:
- `requirements.txt` - Python dependencies list
- Project directory structure

**Logic Implemented**:
- Directory structure for organized code organization
- Dependency management setup

**Validation Steps**:
- Verified all directories created
- Confirmed requirements.txt contains necessary packages
- Tested Python environment setup

**Output of the Phase**:
- Complete project structure ready for development
- All dependencies documented

**Why the Phase Matters**:
- Establishes foundation for all subsequent phases
- Ensures consistent project organization
- Enables reproducible environment setup

---

### PHASE 2 — BASELINE GPT-2 BIAS EVALUATION

**Objective**: Evaluate and document baseline bias in GPT-2 outputs to establish a comparison baseline.

**Components Created**:
- Baseline evaluation script
- Bias detection using Detoxify library
- Evaluation prompts dataset

**Files Produced**:
- `data/baseline_prompts.json` - 25 biased test prompts
- `data/baseline_gpt2_outputs.csv` - GPT-2 outputs with toxicity scores

**Logic Implemented**:
- Generated responses from GPT-2 for 25 biased prompts
- Evaluated outputs using Detoxify (toxicity, identity attack, overall bias scores)
- Documented baseline bias metrics

**Validation Steps**:
- Generated outputs for all 25 prompts
- Calculated average toxicity scores
- Verified bias detection metrics

**Output of the Phase**:
- Baseline bias metrics established
- Evidence of GPT-2 producing biased outputs
- Quantitative bias scores for comparison

**Evaluation Results** (from `baseline_gpt2_outputs.csv`):
- 25 prompts tested
- Average toxicity scores calculated
- Examples of biased outputs documented

**Why the Phase Matters**:
- Provides quantitative baseline for bias measurement
- Demonstrates the problem that needs to be solved
- Enables comparison with RAG-debiased outputs

---

### PHASE 3 — KNOWLEDGE BASE CONSTRUCTION

**Objective**: Create a curated knowledge base of factual, anti-bias content to use for RAG.

**Components Created**:
- Knowledge base directory structure
- 77 anti-bias content files
- Content covering gender equality, occupations, abilities, roles

**Files Produced**:
- `data/knowledge_base/*.txt` - 77 text files with anti-bias content
- Files covering topics such as:
  - Gender equality and bias definitions
  - Occupational performance (engineering, coding, nursing, teaching, etc.)
  - Gender roles and stereotypes
  - Educational access and STEM fields
  - Leadership and abilities
  - Emotional expression and biology

**Logic Implemented**:
- Curated factual content from reliable sources
- Organized by topic (occupations, gender concepts, abilities)
- Structured for easy chunking and embedding

**Validation Steps**:
- Verified all 77 files created
- Checked content quality and relevance
- Confirmed anti-bias messaging in all files

**Output of the Phase**:
- 77 knowledge base files
- Comprehensive coverage of gender bias topics
- Factual, evidence-based content ready for embedding

**Why the Phase Matters**:
- Provides the factual foundation for RAG
- Ensures high-quality, unbiased information
- Enables retrieval of relevant context for any query

---

### PHASE 4 — EMBEDDINGS + FAISS VECTOR STORE

**Objective**: Convert knowledge base text into embeddings and create a searchable vector store.

**Components Created**:
- Embedding generation script
- FAISS vector index
- Metadata mapping system

**Files Produced**:
- `embeddings/knowledge_embeddings.npy` - 384-dimensional embeddings (595 vectors)
- `embeddings/faiss_index.bin` - FAISS index for similarity search
- `embeddings/metadata.json` - Maps chunk IDs to text and source files
- `embeddings/chunks.json` - Text chunks with metadata

**Logic Implemented**:
- Used SentenceTransformer model (`all-MiniLM-L6-v2`) to generate embeddings
- Chunked knowledge base files into smaller segments
- Created 595 text chunks from 77 source files
- Built FAISS index (IndexFlatL2) for fast similarity search
- Created metadata mapping for retrieval

**Validation Steps**:
- Verified 595 embeddings generated
- Confirmed FAISS index contains all vectors
- Tested similarity search functionality
- Validated metadata accuracy

**Output of the Phase**:
- 595 embedded chunks ready for retrieval
- FAISS index for fast similarity search
- Complete metadata system for chunk lookup

**Why the Phase Matters**:
- Enables semantic search over knowledge base
- Provides fast retrieval of relevant context
- Foundation for RAG retrieval pipeline

---

## SECTION C — REPOSITORY STRUCTURE DOCUMENTATION (Part 1)

### Core Source Files

#### `src/retrieval.py`
- **File Path**: `rag_gender_bias_project/src/retrieval.py`
- **What it does**: Retrieves relevant anti-bias knowledge documents from FAISS vector store based on user queries
- **Created in**: Phase 5
- **Used in**: Phase 6 (RAG Generator), Phase 8 (Web Interface)
- **Importance**: Core component of RAG pipeline - enables semantic search over knowledge base
- **Key Functions**:
  - `RetrievalModule.__init__()` - Loads embedding model, FAISS index, and metadata
  - `embed_query(query)` - Converts query to 384-dimensional embedding
  - `retrieve_docs(query, k=5)` - Retrieves top-k relevant chunks
  - `retrieve_docs_with_metadata(query, k=5)` - Returns chunks with full metadata

#### `src/rag_generator.py`
- **File Path**: `rag_gender_bias_project/src/rag_generator.py`
- **What it does**: Combines retrieval with FLAN-T5 LLM to generate debiased responses using retrieved context
- **Created in**: Phase 6
- **Used in**: Phase 8 (Web Interface - Unbiased Mode)
- **Importance**: Core RAG implementation - generates unbiased responses using knowledge base
- **Key Components**:
  - `RAGGenerator` class - Main RAG generator
  - `RAG_PROMPT_TEMPLATE` - Template for instruction-tuned models
  - `rag_answer(query, max_length, k)` - Main function for RAG generation
  - Uses FLAN-T5 (`google/flan-t5-base`) for generation
  - Retrieves context, builds prompt, generates response

#### `src/biased_generator.py`
- **File Path**: `rag_gender_bias_project/src/biased_generator.py`
- **What it does**: Generates raw LLM outputs without retrieval or debiasing (baseline mode)
- **Created in**: Phase 7
- **Used in**: Phase 8 (Web Interface - Biased Mode)
- **Importance**: Provides baseline for comparison - shows raw model behavior
- **Key Functions**:
  - `generate_biased_output(query, max_length)` - Generates response without context
  - Uses FLAN-T5 (`google/flan-t5-base`) without retrieval
  - No bias mitigation or factual context

#### `web/app.py`
- **File Path**: `rag_gender_bias_project/web/app.py`
- **What it does**: Streamlit web interface with dual-mode tabs (Biased vs Unbiased)
- **Created in**: Phase 8
- **Used in**: User interaction, system demonstration
- **Importance**: User-facing interface for comparing biased and unbiased outputs
- **Key Features**:
  - Two tabs: "Biased Output" and "Unbiased RAG Output"
  - Sidebar with example queries
  - Logging to `web/logs/`
  - Error handling and user feedback

### Data Files

#### `data/baseline_prompts.json`
- **File Path**: `rag_gender_bias_project/data/baseline_prompts.json`
- **What it does**: Contains 25 biased test prompts for evaluation
- **Created in**: Phase 2
- **Used in**: Phase 2 (Baseline evaluation), testing
- **Importance**: Standardized test set for bias evaluation
- **Content**: 25 prompts covering gender stereotypes, occupational bias, emotional bias, leadership bias, STEM bias

#### `data/baseline_gpt2_outputs.csv`
- **File Path**: `rag_gender_bias_project/data/baseline_gpt2_outputs.csv`
- **What it does**: Contains GPT-2 outputs and bias scores for 25 prompts
- **Created in**: Phase 2
- **Used in**: Phase 9 (Evaluation comparison)
- **Importance**: Baseline metrics for bias comparison
- **Columns**: prompt, gpt2_output_raw, toxicity_score, identity_attack_score, overall_bias_score

#### `data/knowledge_base/*.txt` (77 files)
- **File Path**: `rag_gender_bias_project/data/knowledge_base/*.txt`
- **What it does**: Contains factual, anti-bias content for RAG retrieval
- **Created in**: Phase 3
- **Used in**: Phase 4 (Embedding), Phase 5 (Retrieval), Phase 6 (RAG)
- **Importance**: Source of factual information for bias mitigation
- **Content**: 77 files covering gender equality, occupations, abilities, roles, stereotypes

### Embedding Files

#### `embeddings/faiss_index.bin`
- **File Path**: `rag_gender_bias_project/embeddings/faiss_index.bin`
- **What it does**: FAISS vector index for fast similarity search
- **Created in**: Phase 4
- **Used in**: Phase 5 (Retrieval), Phase 6 (RAG), Phase 8 (Web)
- **Importance**: Enables fast semantic search over knowledge base
- **Size**: 913,965 bytes
- **Index Type**: IndexFlatL2 (L2 distance)
- **Vectors**: 595

#### `embeddings/knowledge_embeddings.npy`
- **File Path**: `rag_gender_bias_project/embeddings/knowledge_embeddings.npy`
- **What it does**: NumPy array of 384-dimensional embeddings
- **Created in**: Phase 4
- **Used in**: Phase 4 (Index creation)
- **Importance**: Embedding vectors for all knowledge base chunks
- **Size**: 914,048 bytes
- **Dimensions**: 595 x 384

#### `embeddings/metadata.json`
- **File Path**: `rag_gender_bias_project/embeddings/metadata.json`
- **What it does**: Maps chunk IDs to text, source files, and embedding indices
- **Created in**: Phase 4
- **Used in**: Phase 5 (Retrieval), Phase 6 (RAG)
- **Importance**: Enables retrieval of actual text from embedding indices
- **Size**: 1,049,299 bytes
- **Entries**: 595

#### `embeddings/chunks.json`
- **File Path**: `rag_gender_bias_project/embeddings/chunks.json`
- **What it does**: Contains all text chunks with metadata
- **Created in**: Phase 4
- **Used in**: Reference and debugging
- **Importance**: Complete record of all embedded chunks
- **Size**: 1,041,079 bytes

### Phase Summary Files

#### `src/phase5_summary.txt`
- **File Path**: `rag_gender_bias_project/src/phase5_summary.txt`
- **What it does**: Documents Phase 5 completion (Retrieval Module)
- **Created in**: Phase 5
- **Used in**: Documentation, reference
- **Importance**: Records retrieval module implementation and testing

#### `src/phase6_summary.txt`
- **File Path**: `rag_gender_bias_project/src/phase6_summary.txt`
- **What it does**: Documents Phase 6 completion (RAG Generator)
- **Created in**: Phase 6
- **Used in**: Documentation, reference
- **Importance**: Records RAG generator implementation and validation

#### `src/phase7_summary.txt`
- **File Path**: `rag_gender_bias_project/src/phase7_summary.txt`
- **What it does**: Documents Phase 7 completion (Biased Generator)
- **Created in**: Phase 7
- **Used in**: Documentation, reference
- **Importance**: Records biased generator implementation and baseline outputs

---

**END OF PART 1**

*Continue to FINAL_PROJECT_REPORT_PART2.md for Phases 5-9 and Repository Structure (Part 2)*

