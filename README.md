# HK Policy RAG

A Retrieval-Augmented Generation (RAG) assistant for Hong Kong government policies, built with Streamlit and OpenAI.

The app provides a chat interface where users can ask questions about HK policies. The RAG backend is designed to retrieve relevant document chunks and inject them into the LLM prompt — retrieval sources are pluggable stubs ready to be wired up.

---

## Features

- Conversational chat UI with message history
- Streaming LLM responses via OpenAI (`gpt-4o-mini`)
- Parallel RAG retrieval scaffold (pluggable backends)
- Automatic summarisation of older conversation history
- Suggestion pills for common questions
- Debug mode (`?debug=true`) to inspect full prompts

---

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- An OpenAI API key

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/hk-policy-rag.git
cd hk-policy-rag
```

**2. Create a `.env` file**

```bash
cp .env.example .env   # then fill in your keys
```

```env
OPENAI_API_KEY=sk-...
```

**3. Install dependencies**

With uv (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

## Running

```bash
# uv
uv run streamlit run streamlit_app.py

# pip / activated venv
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`.

---

## Project Structure

```
hk-policy-rag/
├── streamlit_app.py   # Streamlit UI + LLM integration
├── main.py            # Entry point (placeholder)
├── pyproject.toml     # Project metadata & dependencies (uv)
├── requirements.txt   # Pinned direct dependencies (pip)
├── .env               # API keys (not committed)
└── .env.example       # Template for .env
```

---

## Wiring up RAG

The two stub functions in `streamlit_app.py` are the integration points:

```python
def search_relevant_docs(query: str) -> str:
    """Return relevant policy document chunks for the query."""
    ...

def search_extra_context(query: str) -> str:
    """Return supplementary context for the query."""
    ...
```

Both are called in parallel via `ThreadPoolExecutor`. Return a non-empty string to have the content injected into the LLM prompt automatically.
