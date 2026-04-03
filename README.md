# Multi-Agent Network Troubleshooting System
### Ericsson AI Research Intern – Portfolio Project | ELTE CS

---

## Architecture Overview



┌─────────────────────────────────────────────────────────┐
│                    LangGraph State Machine               │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   Agent A    │───▶│   Agent B    │───▶│  Agent C  │  │
│  │  Researcher  │    │   Analyst    │    │Coordinator│  │
│  └──────────────┘    └──────────────┘    └───────────┘  │
│   ChromaDB RAG        Scikit-learn RF     Gemini 1.5     │
│   (NO LLM)           (deterministic)    (grounded only)  │
└─────────────────────────────────────────────────────────┘


## Agents

| Agent | Tool | Role | Hallucination Risk |
|-------|------|------|--------------------|
| A – Researcher | ChromaDB | Semantic search over site docs | **Zero** – returns verbatim chunks |
| B – Analyst | Random Forest | Predict network congestion | **Zero** – deterministic ML output |
| C – Coordinator | Gemini 1.5 Flash | Write acceptance report | **Minimised** – system prompt enforces citation & grounding |

## Hallucination Prevention (4-Layer Defence)

### Layer 1 – Retrieval Grounding (Agent A)
Agent A **never invokes an LLM**. It queries ChromaDB using TF-IDF embeddings and returns raw document chunks.

### Layer 2 – Deterministic ML (Agent B)
Agent B runs a Scikit-learn `RandomForestClassifier`. The output is a mathematical prediction, ensuring the "brain" of the agent remains objective.

### Layer 3 – Constrained Prompt Chaining (Agent C)
The coordinator uses LangChain `SystemMessage` to enforce strict grounding rules:
- **Citations Required**: Every claim must have a [RAG] or [ML] tag.
- **No Inference**: Missing data results in a `⚠️ DATA NOT AVAILABLE` label.

### Layer 4 – Evidence Audit Trail
The shared `GraphState` accumulates an `evidence_log`, providing a full "chain of custody" for every decision made by the system.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key in .env
# GOOGLE_API_KEY=AIza...

# 3. Run the pipeline
python agents.py
## 👤 Author

Built as a portfolio project for the **Multi-Agents Network Troubleshooting System**

CS Student · 
[github.com/Chiyuka](https://github.com/Chiyuka) · [www.linkedin.com/in/phannarong-tuon-734267296](https://www.linkedin.com/in/phannarong-tuon-734267296)