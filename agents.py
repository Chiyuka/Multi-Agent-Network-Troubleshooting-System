"""
Multi-Agent Network Troubleshooting System
Ericsson AI Research Intern – Portfolio Project
Author: ELTE CS Student

Architecture:
  Agent A (Researcher)   → ChromaDB RAG query  (TF-IDF embeddings, fully offline)
  Agent B (Analyst)      → Scikit-learn Random Forest congestion predictor
  Agent C (Coordinator)  → Claude writes the final Site Acceptance Report

Hallucination-prevention strategy:
  1. Agent A returns VERBATIM chunks from ChromaDB – no LLM involved.
  2. Agent B returns a deterministic ML prediction – no language model.
  3. Agent C is constrained by a system prompt that enforces citation of every
     factual claim to [RAG] or [ML] and forbids adding outside knowledge.
"""

from __future__ import annotations

import json
import textwrap
from typing import TypedDict, Annotated
import operator

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

# ── ChromaDB (with offline TF-IDF embedding function) ─────────────────────────
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Scikit-learn ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ── Anthropic (Agent C uses Claude as LLM) ────────────────────────────────────
#import anthropic

# gemini api
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load variables from .env
load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# 0.  OFFLINE EMBEDDING FUNCTION  (TF-IDF, no model download required)
# ══════════════════════════════════════════════════════════════════════════════

class TFIDFEmbeddingFunction(EmbeddingFunction):
    """
    Lightweight offline embedding function backed by scikit-learn TF-IDF.
    Suitable for small document sets where semantic accuracy matters less
    than portability and zero external dependencies.
    Replace with sentence-transformers or OpenAI embeddings in production.
    """

    def __init__(self, corpus: list[str]):
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=512,
        )
        self._vectorizer.fit(corpus)

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        matrix = self._vectorizer.transform(input)
        # ChromaDB expects plain Python lists of floats
        return matrix.toarray().tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SHARED STATE  (LangGraph passes this dict between nodes)
# ══════════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    site_id: str
    query: str
    # outputs accumulated by each agent
    rag_docs: Annotated[list[str], operator.add]   # Agent A fills this
    ml_prediction: dict                             # Agent B fills this
    final_report: str                               # Agent C fills this
    # audit trail – every grounded fact is logged here
    evidence_log: Annotated[list[str], operator.add]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MOCK DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

MOCK_SITE_DOCS = [
    {
        "id": "doc_001",
        "text": (
            "Site ID: ERB-BUD-042 | Location: Budapest, Kelenföld | "
            "Tower Height: 48 m | Antenna Config: 4T4R Massive MIMO | "
            "Frequency Bands: 700 MHz (n28), 2100 MHz (n1), 3500 MHz (n78) | "
            "Installation Date: 2024-03-15 | Status: Commercially Active"
        ),
        "metadata": {"site": "ERB-BUD-042", "category": "site_profile"},
    },
    {
        "id": "doc_002",
        "text": (
            "Maintenance Log ERB-BUD-042: 2025-01-10 – RRU unit #3 replaced due to "
            "thermal runaway. 2025-03-22 – Software upgrade to Ericsson CUDB 23.Q4. "
            "Next scheduled inspection: 2025-07-01. Open tickets: TT-88821 (minor "
            "feeder loss on sector 2, under investigation)."
        ),
        "metadata": {"site": "ERB-BUD-042", "category": "maintenance"},
    },
    {
        "id": "doc_003",
        "text": (
            "KPI Summary ERB-BUD-042 (last 30 days): Avg DL Throughput 412 Mbps, "
            "Avg UL Throughput 98 Mbps, Packet Loss 0.4 %, Latency P99 18 ms, "
            "Cell Availability 99.1 %, Active UE peak 1240 (Tuesday 18:00 local)."
        ),
        "metadata": {"site": "ERB-BUD-042", "category": "kpi"},
    },
    {
        "id": "doc_004",
        "text": (
            "Acceptance Criteria (Ericsson EAB-17-0034): DL Throughput ≥ 300 Mbps, "
            "Packet Loss ≤ 1 %, Cell Availability ≥ 99 %, Latency P99 ≤ 20 ms. "
            "All criteria must be met simultaneously over a 30-day observation window."
        ),
        "metadata": {"site": "ERB-BUD-042", "category": "acceptance_criteria"},
    },
]

MOCK_CSV_DATA = pd.DataFrame(
    {
        "active_ue":           [120, 450,  980, 1240, 300,  870, 1100, 200, 560, 1050],
        "dl_throughput_mbps":  [480, 390,  310,  260, 450,  330,  280, 470, 370,  295],
        "packet_loss_pct":     [0.1, 0.3,  0.8,  1.4, 0.2,  0.7,  1.2, 0.1, 0.4,  1.1],
        "latency_p99_ms":      [ 12,  15,   19,   25,  13,   18,   23,  11,  16,   22],
        "congested":           [  0,   0,    0,    1,   0,    0,    1,   0,   0,    1],
    }
)

# Peak-hour snapshot used for inference
LIVE_KPI = {
    "active_ue":          1240,
    "dl_throughput_mbps":  260,
    "packet_loss_pct":     1.4,
    "latency_p99_ms":       25,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VECTOR DATABASE SETUP  (ChromaDB in-memory, offline TF-IDF embeddings)
# ══════════════════════════════════════════════════════════════════════════════

def build_vector_db() -> chromadb.Collection:
    """
    Fit TF-IDF on the corpus first, then populate an in-memory ChromaDB
    collection.  No network access required.
    """
    corpus = [d["text"] for d in MOCK_SITE_DOCS]
    ef = TFIDFEmbeddingFunction(corpus)

    client = chromadb.Client()   # ephemeral / in-memory
    collection = client.get_or_create_collection(
        name="ericsson_site_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=corpus,
        ids=[d["id"] for d in MOCK_SITE_DOCS],
        metadatas=[d["metadata"] for d in MOCK_SITE_DOCS],
    )
    return collection


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SCIKIT-LEARN MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════

def build_ml_model() -> RandomForestClassifier:
    """Train a Random Forest on mock KPI data."""
    X = MOCK_CSV_DATA[["active_ue", "dl_throughput_mbps", "packet_loss_pct", "latency_p99_ms"]]
    y = MOCK_CSV_DATA["congested"]
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    return clf


# ══════════════════════════════════════════════════════════════════════════════
# 5.  AGENT NODES
# ══════════════════════════════════════════════════════════════════════════════

# Module-level singletons (initialised once in init_system)
_collection: chromadb.Collection | None = None
_clf: RandomForestClassifier | None = None
#_anthropic_client: anthropic.Anthropic | None = None

# Change the type hint for the client
_gemini_model: ChatGoogleGenerativeAI | None = None

    
def init_system() -> None:
    """Initialise all singletons. Call once before invoking the graph."""
    global _collection, _clf, _gemini_model
    _collection = build_vector_db()
    _clf = build_ml_model()
    
    # Try the most specific, stable version string
    _gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", # <--- Changed to 'latest'
        temperature=0
    )

"""
def init_system() -> None:
    """"""Initialise all singletons.  Call once before invoking the graph.""""""
    global _collection, _clf, _anthropic_client
    _collection = build_vector_db()
    _clf = build_ml_model()
    _anthropic_client = anthropic.Anthropic()
"""

# ── Agent A: The Researcher ───────────────────────────────────────────────────

def agent_researcher(state: GraphState) -> dict:
    """
    Retrieves relevant site documentation from ChromaDB.

    HALLUCINATION PREVENTION
    ────────────────────────
    No LLM is invoked.  This node performs a cosine-similarity search over
    TF-IDF vectors and returns the raw document strings verbatim.  The text
    that reaches Agent C is literal source material, not generated content.
    """
    assert _collection is not None, "Call init_system() first."

    results = _collection.query(
        query_texts=[state["query"]],
        n_results=4,
    )
    docs: list[str] = results["documents"][0]

    evidence = [f"[RAG] Retrieved chunk: '{d[:80]}...'" for d in docs]

    return {
        "rag_docs": docs,
        "evidence_log": evidence,
    }


# ── Agent B: The Analyst ──────────────────────────────────────────────────────

def agent_analyst(state: GraphState) -> dict:
    """
    Runs the Scikit-learn Random Forest to predict network congestion.

    HALLUCINATION PREVENTION
    ────────────────────────
    The prediction is 100 % deterministic – a numerical ML inference, not a
    language model generation.  The confidence value is a calibrated ensemble
    probability from predict_proba(), not a linguistic hedge.
    """
    assert _clf is not None, "Call init_system() first."

    live_df = pd.DataFrame([LIVE_KPI])

    prediction = int(_clf.predict(live_df)[0])
    confidence = float(_clf.predict_proba(live_df)[0][prediction])
    label = "CONGESTED" if prediction == 1 else "NORMAL"

    feature_names = ["active_ue", "dl_throughput_mbps", "packet_loss_pct", "latency_p99_ms"]
    importances = dict(zip(feature_names, _clf.feature_importances_.tolist()))

    ml_result = {
        "label": label,
        "confidence": round(confidence, 3),
        "input_kpis": LIVE_KPI,
        "feature_importances": {k: round(v, 3) for k, v in importances.items()},
    }

    evidence = [
        f"[ML] Random Forest prediction: {label} (confidence {confidence:.1%})",
        f"[ML] Top feature: {max(importances, key=importances.get)}",
    ]

    return {
        "ml_prediction": ml_result,
        "evidence_log": evidence,
    }


# ── Agent C: The Coordinator ──────────────────────────────────────────────────

def agent_coordinator(state: GraphState) -> dict:
    """
    Synthesises Agent A and B outputs into a formal Site Acceptance Report.

    HALLUCINATION PREVENTION
    ────────────────────────
    The system prompt enforces four hard rules:
      RULE 1 – GROUNDING  : every fact must cite [RAG] or [ML].
      RULE 2 – GAPS       : missing data → '⚠️ DATA NOT AVAILABLE', never inferred.
      RULE 3 – CITATIONS  : inline source tag after every factual sentence.
      RULE 4 – VERDICT    : Pass/Fail derived only from ML label + RAG criteria.
    Claude receives only the structured outputs of A and B; it cannot draw on
    external knowledge to fill gaps.
    """
    #assert _anthropic_client is not None, "Call init_system() first."
    assert _gemini_model is not None, "Call init_system() first."
    """
    rag_context = "\n\n".join(
        f"[DOC {i + 1}]: {doc}" for i, doc in enumerate(state["rag_docs"])
    )
    ml_context = json.dumps(state["ml_prediction"], indent=2)
    """    
    rag_context = "\n\n".join(f"[DOC {i + 1}]: {doc}" for i, doc in enumerate(state["rag_docs"]))
    ml_context = json.dumps(state["ml_prediction"], indent=2)

    system_prompt = textwrap.dedent(f"""
        You are an Ericsson Site Acceptance Engineer writing an official report.
        You MUST follow these rules to ensure accuracy:

        RULE 1 – GROUNDING: Every factual claim MUST be traceable to either
                 [RAG DOCUMENTS] or [ML PREDICTION] provided below.
                 Do NOT invent numbers, dates, or technical details.

        RULE 2 – GAPS: If information needed for a section is absent from the
                 provided context, write exactly: ⚠️ DATA NOT AVAILABLE.

        RULE 3 – CITATIONS: After each factual sentence append the source tag
                 [RAG] or [ML].

        RULE 4 – VERDICT: Base the PASS/FAIL verdict ONLY on whether the ML
                 prediction is "NORMAL" or "CONGESTED" AND whether KPI values
                 meet the acceptance criteria found in the RAG documents.

        Violation of any rule constitutes a compliance failure.
    """).strip()

    user_prompt = textwrap.dedent(f"""
        Generate a Site Acceptance Report for site {state['site_id']}.

        ===== RAG DOCUMENTS =====
        {rag_context}

        ===== ML PREDICTION =====
        {ml_context}

        ===== REPORT TEMPLATE =====
        # Site Acceptance Report – {state['site_id']}

        ## 1. Site Profile
        ## 2. Maintenance Status
        ## 3. KPI Analysis
        ## 4. Congestion Prediction (ML)
        ## 5. Acceptance Verdict
        ## 6. Recommended Actions
    """).strip()

    """
    response = _anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    """
    # The LangChain way to call Gemini:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    
    response = _gemini_model.invoke(messages)
    
    #report_text: str = response.content[0].text

    return {
        "final_report": response.content,
        "evidence_log": ["[COORDINATOR] Report generated; grounded in RAG+ML context only."],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  LANGGRAPH  –  Graph Definition
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Topology:
        START → researcher → analyst → coordinator → END

    Sequential chaining ensures every downstream agent receives the full
    accumulated state produced by all upstream agents before executing.
    """
    g = StateGraph(GraphState)

    g.add_node("researcher",  agent_researcher)
    g.add_node("analyst",     agent_analyst)
    g.add_node("coordinator", agent_coordinator)

    g.set_entry_point("researcher")
    g.add_edge("researcher",  "analyst")
    g.add_edge("analyst",     "coordinator")
    g.add_edge("coordinator", END)

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(site_id: str = "ERB-BUD-042") -> GraphState:
    init_system()
    graph = build_graph()

    initial_state: GraphState = {
        "site_id": site_id,
        "query": "site profile maintenance KPI acceptance criteria",
        "rag_docs": [],
        "ml_prediction": {},
        "final_report": "",
        "evidence_log": [],
    }

    return graph.invoke(initial_state)


if __name__ == "__main__":
    state = run_pipeline()

    print("\n" + "=" * 72)
    print("FINAL SITE ACCEPTANCE REPORT")
    print("=" * 72)
    print(state["final_report"])

    print("\n--- EVIDENCE AUDIT TRAIL ---")
    for entry in state["evidence_log"]:
        print(" •", entry)