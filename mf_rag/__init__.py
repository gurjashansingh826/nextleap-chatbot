"""MF Facts RAG Chatbot — a facts-only retrieval-augmented assistant for HDFC Mutual Fund
scheme pages on Groww.

Pipeline stages map one-to-one onto modules so any stage can be traced to a file and a log
line (architecture §4):

    STAGE 1  loaders.py             fetch + clean public pages
    STAGE 2  chunkers.py            split documents into self-describing chunks
    STAGE 3  embedder.py            text -> 384-dim vector
    STAGE 4  store.py               persist vectors in ChromaDB
    STAGE 5  guards.py              PII + advice/returns refusal (runs FIRST)
             retriever.py           query -> vector -> top-k -> MMR -> threshold
    STAGE 6  prompts.py             system prompt + citation wording
             answerer.py            LLM call + post-checks + extractive fallback
    STAGE 7  app.py                 Streamlit chat UI

Academic class demo. Facts-only. Not investment advice.
"""

__version__ = "1.0.0"
