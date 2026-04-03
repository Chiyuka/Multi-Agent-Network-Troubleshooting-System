"""
Multi-Agent Network Troubleshooting System
Ericsson AI Research Intern – Portfolio Project
Author: ELTE CS Student

Architecture:
  Agent A (Researcher)   → ChromaDB RAG query  (TF-IDF embeddings, fully offline)
  Agent B (Analyst)      → Scikit-learn Random Forest congestion predictor
  Agent C (Coordinator)  → Gemini writes the final Site Acceptance Report

Hallucination-prevention strategy:
  1. Agent A returns VERBATIM chunks from ChromaDB – no LLM involved.
  2. Agent B returns a deterministic ML prediction – no language model.
  3. Agent C is constrained by a system prompt that enforces citation of every
     factual claim to [RAG] or [ML] and forbids adding outside knowledge.
"""

from __future__ import annotations

import json
import os
import time
import textwrap
from typing import Annotated, TypedDict
import operator

from dotenv import load_dotenv

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph

# ── ChromaDB (offline TF-IDF embedding function – no model download) ──────────
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Scikit-learn ──────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ── Gemini via LangChain ──────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── Model fallback chain ───────────────────────────────────────────────────────
# Two groups, each built with the correct API version for that model generation.
#
#   v1beta endpoint → gemini-2.x models  (langchain-google-genai default)
#   v1     endpoint → gemini-1.5 models  (requires api_version override)
#
# Models are tried in order; the first one that responds becomes active.
_MODELS_V1BETA = [
    "gemini-2.0-flash-lite",  # lightest, highest free-tier RPM
    "gemini-2.0-flash",       # standard free tier
]
_MODELS_V1 = [
    "gemini-1.5-flash",       # stable, separate daily quota bucket
    "gemini-1.5-flash-8b",    # smallest / most quota-generous of the 1.5 family
]
GEMINI_FALLBACK_MODELS = _MODELS_V1BETA + _MODELS_V1  # for reference / logging

# Errors that mean "skip this model, try the next one"
_SKIPPABLE = ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  OFFLINE EMBEDDING FUNCTION  (TF-IDF, no model download required)
# ══════════════════════════════════════════════════════════════════════════════

class TFIDFEmbeddingFunction(EmbeddingFunction):
    """
    Lightweight offline embedding function backed by scikit-learn TF-IDF.
    Suitable for small document sets where portability matters most.
    Swap for sentence-transformers or Google text-embedding-004 in production.
    """

    def __init__(self, corpus: list[str]) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=512,
        )
        self._vectorizer.fit(corpus)

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        matrix = self._vectorizer.transform(input)
        return matrix.toarray().tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SHARED STATE  (LangGraph passes this dict between nodes)
# ══════════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    site_id: str
    query: str
    rag_docs: Annotated[list[str], operator.add]     # filled by Agent A
    ml_prediction: dict                               # filled by Agent B
    final_report: str                                 # filled by Agent C
    evidence_log: Annotated[list[str], operator.add]  # accumulated by all agents


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
        "active_ue":          [120, 450,  980, 1240, 300,  870, 1100, 200, 560, 1050],
        "dl_throughput_mbps": [480, 390,  310,  260, 450,  330,  280, 470, 370,  295],
        "packet_loss_pct":    [0.1, 0.3,  0.8,  1.4, 0.2,  0.7,  1.2, 0.1, 0.4,  1.1],
        "latency_p99_ms":     [ 12,  15,   19,   25,  13,   18,   23,  11,  16,   22],
        "congested":          [  0,   0,    0,    1,   0,    0,    1,   0,   0,    1],
    }
)

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
    """Fit TF-IDF on the corpus, then load it into an in-memory ChromaDB collection."""
    corpus = [d["text"] for d in MOCK_SITE_DOCS]
    ef = TFIDFEmbeddingFunction(corpus)

    client = chromadb.Client()
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
    """Train a Random Forest classifier on mock KPI data."""
    X = MOCK_CSV_DATA[["active_ue", "dl_throughput_mbps", "packet_loss_pct", "latency_p99_ms"]]
    y = MOCK_CSV_DATA["congested"]
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    return clf


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SINGLETONS & MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_collection: chromadb.Collection | None = None
_clf: RandomForestClassifier | None = None
_gemini_model: ChatGoogleGenerativeAI | None = None
_active_model_name: str = ""


def _is_skippable(err: str) -> bool:
    """True if the error means 'this model is unavailable – try the next one'."""
    return any(code in err for code in _SKIPPABLE)


def _build_client(model: str) -> ChatGoogleGenerativeAI:
    """
    Build a ChatGoogleGenerativeAI client with the correct API version.

    google-genai SDK 1.x defaults to v1beta, which only serves gemini-2.x.
    The gemini-1.5 family is only available on the stable v1 endpoint.
    We pass the api_version through client_args → http_options.
    """
    api_version = "v1" if model.startswith("gemini-1.5") else "v1beta"
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        client_args={"http_options": {"api_version": api_version}},
    )


def init_system() -> None:
    """
    Initialise all singletons.

    Probes each model in GEMINI_FALLBACK_MODELS with a lightweight call.
    Skips models that are quota-exhausted (429) or not found (404).
    Fails fast on auth errors (403) or unexpected exceptions.
    """
    global _collection, _clf, _gemini_model, _active_model_name

    _collection = build_vector_db()
    _clf = build_ml_model()

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to your .env file.")

    for model_name in GEMINI_FALLBACK_MODELS:
        print(f"   Trying {model_name} ...", end=" ", flush=True)
        client = _build_client(model_name)
        try:
            client.invoke([HumanMessage(content="ping")])
            _gemini_model = client
            _active_model_name = model_name
            print("✅")
            print(f"✅ System initialised with {model_name}")
            return
        except Exception as e:
            err = str(e)
            if _is_skippable(err):
                reason = "quota exhausted" if "429" in err else "model not found"
                print(f"⚠️  {reason}, trying next...")
            else:
                print(f"❌  fatal error: {err[:120]}")
                raise

    raise RuntimeError(
        "\n\nAll Gemini models are unavailable right now.\n"
        "Most likely cause: free-tier daily quota exhausted for all models.\n\n"
        "Options:\n"
        "  1. Wait for midnight Pacific time — quota resets daily.\n"
        "  2. Go to https://aistudio.google.com/app/apikey, create a NEW project,\n"
        "     generate a fresh API key, and update GOOGLE_API_KEY in your .env.\n"
        "  3. Enable billing at console.cloud.google.com (free tier remains).\n"
    )


def _invoke_with_fallback(messages: list) -> str:
    """
    Call the active model. Rotates to the next fallback if a skippable error
    occurs mid-pipeline. Returns the response text as a plain string.
    """
    global _gemini_model, _active_model_name

    candidates = [_active_model_name] + [
        m for m in GEMINI_FALLBACK_MODELS if m != _active_model_name
    ]

    for model_name in candidates:
        if model_name != _active_model_name:
            print(f"   ↩ Rotating to fallback: {model_name}")
            _gemini_model = _build_client(model_name)
            _active_model_name = model_name

        try:
            return _gemini_model.invoke(messages).content
        except Exception as e:
            err = str(e)
            if _is_skippable(err):
                print(f"   ⚠️ {model_name} unavailable mid-call, rotating...")
                time.sleep(2)
            else:
                raise

    raise RuntimeError("All fallback models exhausted during pipeline execution.")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  AGENT NODES
# ══════════════════════════════════════════════════════════════════════════════

def agent_researcher(state: GraphState) -> dict:
    """
    Retrieves relevant site documentation from ChromaDB.

    HALLUCINATION PREVENTION
    ────────────────────────
    No LLM is invoked. Cosine-similarity search over TF-IDF vectors returns
    raw document strings verbatim – literal source material, not generated text.
    """
    assert _collection is not None, "Call init_system() first."

    results = _collection.query(query_texts=[state["query"]], n_results=4)
    docs: list[str] = results["documents"][0]

    return {
        "rag_docs": docs,
        "evidence_log": [f"[RAG] Retrieved chunk: '{d[:80]}...'" for d in docs],
    }


def agent_analyst(state: GraphState) -> dict:
    """
    Predicts network congestion with a Scikit-learn Random Forest.

    HALLUCINATION PREVENTION
    ────────────────────────
    100 % deterministic ML inference – no language model, no sampling,
    no token generation. Confidence is a calibrated ensemble probability.
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

    return {
        "ml_prediction": ml_result,
        "evidence_log": [
            f"[ML] Random Forest prediction: {label} (confidence {confidence:.1%})",
            f"[ML] Top feature: {max(importances, key=importances.get)}",
        ],
    }


def agent_coordinator(state: GraphState) -> dict:
    """
    Synthesises Agent A and B outputs into a formal Site Acceptance Report.

    HALLUCINATION PREVENTION
    ────────────────────────
    System prompt enforces four hard rules:
      RULE 1 – GROUNDING  : every fact must cite [RAG] or [ML].
      RULE 2 – GAPS       : missing data → '⚠️ DATA NOT AVAILABLE', never inferred.
      RULE 3 – CITATIONS  : inline source tag after every factual sentence.
      RULE 4 – VERDICT    : PASS/FAIL derived only from ML label + RAG criteria.
    Gemini receives only the structured outputs of A and B as context.
    """
    assert _gemini_model is not None, "Call init_system() first."

    rag_context = "\n\n".join(
        f"[DOC {i + 1}]: {doc}" for i, doc in enumerate(state["rag_docs"])
    )
    ml_context = json.dumps(state["ml_prediction"], indent=2)

    system_prompt = textwrap.dedent("""
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

    report_text = _invoke_with_fallback([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {
        "final_report": report_text,
        "evidence_log": [
            f"[COORDINATOR] Report generated via {_active_model_name}; "
            "grounded in RAG+ML context only."
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  LANGGRAPH GRAPH DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Topology:  START → researcher → analyst → coordinator → END

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
# 8.  ENTRY POINT
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
    try:
        state = run_pipeline()

        print("\n" + "=" * 72)
        print("FINAL SITE ACCEPTANCE REPORT")
        print("=" * 72)
        print(state["final_report"])

        print("\n--- EVIDENCE AUDIT TRAIL ---")
        for entry in state["evidence_log"]:
            print(" •", entry)

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise