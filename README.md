# Multi-Agent Network Troubleshooting System

A LangGraph-based multi-agent pipeline that automates network site acceptance reporting by combining RAG document retrieval, traditional ML inference, and a grounded LLM coordinator.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph State Machine               │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   Agent A    │───▶│   Agent B    │───▶│  Agent C  │  │
│  │  Researcher  │    │   Analyst    │    │Coordinator│  │
│  └──────────────┘    └──────────────┘    └───────────┘  │
│   ChromaDB RAG        Scikit-learn RF     Groq / Llama   │
│   (NO LLM)           (deterministic)    (grounded only)  │
└─────────────────────────────────────────────────────────┘
```

| Agent | Tool | Role | Hallucination Risk |
|-------|------|------|--------------------|
| A – Researcher | ChromaDB + TF-IDF | Retrieve site documentation | **Zero** – verbatim chunks, no generation |
| B – Analyst | RandomForestClassifier | Predict network congestion | **Zero** – deterministic ML inference |
| C – Coordinator | Groq (Llama-3.3-70b) | Write acceptance report | **Minimised** – hard citation rules enforced |

---

## Hallucination Prevention

The system uses a 4-layer defence to ensure every claim in the output is traceable to a real source.

**Layer 1 – Retrieval grounding (Agent A)**
No LLM is invoked. ChromaDB returns raw document chunks via TF-IDF cosine similarity. The text Agent C receives is literal source material.

**Layer 2 – Deterministic ML (Agent B)**
A trained `RandomForestClassifier` produces a binary prediction and a calibrated probability. A mathematical model cannot fabricate an answer.

**Layer 3 – Constrained prompt chaining (Agent C)**
The system prompt enforces four hard rules:
- `RULE 1` Every factual claim must be traceable to `[RAG]` or `[ML]`
- `RULE 2` Missing data → `⚠️ DATA NOT AVAILABLE`, never inferred
- `RULE 3` Inline citation tag after every factual sentence
- `RULE 4` PASS/FAIL verdict derived only from ML label + RAG acceptance criteria

**Layer 4 – Evidence audit trail**
Every agent appends to `evidence_log` in the shared `GraphState`, creating a full chain of custody from raw document to final report claim.

---

## Project Structure

```
.
├── agents.py          # Full pipeline — all agents, graph, and entry point
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/Chiyuka/multi-agent-network-troubleshooting
cd multi-agent-network-troubleshooting
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)

# 4. Run the pipeline
python agents.py
```

**Expected output:**
```
✅ System initialised with Groq (llama-3.3-70b-versatile)
================================================================
FINAL SITE ACCEPTANCE REPORT
================================================================
# Site Acceptance Report – ERB-BUD-042
...
--- EVIDENCE AUDIT TRAIL ---
 • [RAG] Retrieved chunk: 'Acceptance Criteria ...'
 • [ML]  Random Forest prediction: CONGESTED (confidence 98.0%)
 • [COORDINATOR] Report generated via Groq (llama-3.3-70b-versatile); grounded in RAG+ML context only.
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph |
| Vector database | ChromaDB (in-memory, TF-IDF embeddings) |
| ML model | Scikit-learn RandomForestClassifier |
| LLM | Groq API — Llama-3.3-70b-versatile (free tier) |
| LLM interface | LangChain / langchain-openai |

---

## Model swap

To use a different LLM, change the two constants at the top of `agents.py`:

```python
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "llama-3.3-70b-versatile"   # or "llama-3.1-8b-instant", "mixtral-8x7b-32768"
```

Any OpenAI-compatible provider works — just update `GROQ_BASE_URL` and the corresponding API key in `.env`.

---

## 👤 Author

[github.com/Chiyuka](https://github.com/Chiyuka) · [linkedin.com/in/phannarong-tuon-734267296](https://www.linkedin.com/in/phannarong-tuon-734267296)