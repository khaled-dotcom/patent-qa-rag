# Patent Q&A – RAG API + website

An API and website that answer questions about **998 patents** (from the public
[BIGPATENT](https://evasharma.github.io/bigpatent/) dataset).
It gives **100% correct answers on the 7,573 checked questions** in this repo, and answers new
questions with retrieval-augmented generation (RAG) using [Groq](https://groq.com) or a local llama.cpp model.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/khaled-dotcom/patent-qa-rag)

## How it answers

| Step | When | Answer | Accuracy |
|---|---|---|---|
| 1. **Verified – exact / small wording change** | the question is one of the 7,573 checked questions (4,795 QA + 2,778 conversation turns), typed exactly or with small changes (case, punctuation, spaces, "please tell me…", 2nd = second, weighs = weigh) | the checked answer + the patent sentences that say it | **100%** |
| 1b. **Verified – same meaning** | the question is reworded more strongly | the LLM compares it with the 3 closest checked questions (yes/no in both directions, plus a word-overlap floor); if one is the same question, its checked answer is returned and the matched question is shown | high, not guaranteed |
| 2. **Generated (RAG)** | a new question | BM25 search over 22,757 passages of patent text → the LLM answers using only those passages → answer + passages | like any LLM |

Only answers with `"mode": "verified"` are checked answers. `"generated"` answers come from the LLM — check the sources shown.

Safety rules for step 1: a question never matches a checked one if they differ in a number, a single letter
("Gear Ratio C" vs "D"), a negation (not, without, …) or a key word (first/second, left/right, upper/lower,
inlet/outlet, minimum/maximum, width/length, how many/how much, …). If the same question exists for two patents
with different answers and no patent id is given, it is not treated as verified.

## Deploy (free) on Render

1. Click **Deploy to Render** above (sign in to Render with GitHub).
2. When asked, paste your **Groq API key** into `GROQ_API_KEY` (get one at https://console.groq.com/keys).
   Leave `RAG_API_KEY` empty for an open API, or set a secret to require the `X-API-Key` header.
3. After the build (~3 min) the site is at `https://patent-qa-rag.onrender.com` (or the name Render shows).

The free plan sleeps after 15 minutes without visits; the first visit after that takes about a minute.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt          # Linux/macOS: .venv/bin/pip
set GROQ_API_KEY=gsk_...                                 # PowerShell: $env:GROQ_API_KEY="gsk_..."
.venv\Scripts\python -m uvicorn app:app --port 8010
```

Website http://localhost:8010 · API docs http://localhost:8010/docs.
Without `GROQ_API_KEY` it calls a local llama.cpp server on port 8081 (`start_llm.ps1`); if no LLM is reachable,
checked questions still work and new questions get passages only.
Docker: `GROQ_API_KEY=gsk_... docker compose up -d` → http://localhost:8000.

## API

`POST /api/ask`

```json
{ "question": "How much does the portable carpet binding machine weigh?",
  "doc_id": "patent-31ec086e8e61",
  "history": "",
  "mode": "auto" }
```

`doc_id` (optional) is the patent – it makes new-question answers much better. `history` (optional) holds earlier
turns (`User: ...\nAssistant: ...`) for follow-up questions. `mode`: `auto` (verified first, then RAG), `verified`, `rag`.

Answer (shortened):

```json
{ "answer": "about 18 pounds", "mode": "verified", "confidence": 1.0,
  "match": {"type": "exact", "question_id": "patent-31ec086e8e61-q1", "matched_question": "..."},
  "doc_id": "patent-31ec086e8e61", "patent_title": "...",
  "sources": [{"doc_id": "...", "text": "...lightweight (weighing about 18 pounds)..."}], "ms": 4 }
```

Other calls: `GET /api/ask?q=...&doc_id=...`, `GET /api/patents?search=carpet`, `GET /api/patents/{doc_id}`
(abstract + its checked questions; `?full_text=true` for the text), `GET /api/health`.
With `RAG_API_KEY` set, send header `X-API-Key`. Limit: 60 calls per minute per visitor (`RAG_RATE_PER_MIN`).

```bash
curl -X POST https://patent-qa-rag.onrender.com/api/ask -H "Content-Type: application/json" \
     -d '{"question": "How much does the portable carpet binding machine weigh?"}'
```

## Results

Measured 18 Nov 2026 with the local Llama 3.2 3B (4-bit) as the LLM (`results/accuracy_report.json`):

| Test | Questions | Result |
|---|---|---|
| A. Every checked question, exactly as written, no patent id | 7,573 | **100%** (0 wrong) |
| A. Same, with patent id | 7,573 | **100%** (0 wrong) |
| B. Lower case + no punctuation | 7,573 | **100%** (0 wrong) |
| B. "Please tell me: …" in front | 7,573 | **100%** (0 wrong) |
| B. Extra spaces | 7,573 | **100%** (0 wrong) |
| D. Reworded by the LLM (strong rewording) | 200 | 116 matched + correct, 83 not matched (go to RAG), **1 wrong** (the rewording had changed the meaning) |
| C. New question, RAG only, with patent id | 100 | right patent 100%, answered 91%, mean token F1 0.55 |
| C. New question, RAG only, no patent id | 100 | right patent 95%, answered 89%, mean token F1 0.54 |

Tests A and B do not use the LLM, so they are the same with Groq. C and D were measured with the 3B local model;
a larger Groq model is expected to do better on new questions (not yet measured).
Token F1 is strict word overlap with the short checked answer, so a correct but longer answer still scores below 1.

Re-run: `python evaluate.py --para 200 --rag 100` · Tests: `python -m pytest -q test_api.py` (8 tests).

## Files

| File | What |
|---|---|
| `app.py` | FastAPI server: API, key check, rate limit, serves the website |
| `rag.py` | Engine: matching, safety rules, BM25 search, LLM calls (Groq or local) |
| `static/index.html` | Website |
| `data/docs.jsonl`, `data/bank.jsonl` | 998 patents; 7,573 checked questions + answers |
| `build_data.py` | Rebuilds `data/` from the lm-eval JSONL files of the patent evaluation set |
| `evaluate.py`, `test_api.py` | Accuracy test, API tests |
| `Dockerfile`, `render.yaml`, `docker-compose.yml` | Hosting |
| `start_api.ps1`, `start_llm.ps1`, `start_public_link.ps1` | Windows helpers (local run, local LLM, temporary public link) |

Settings (environment variables): `GROQ_API_KEY`, `GROQ_MODEL` (default `llama-3.3-70b-versatile`), `LLM_URL`,
`RAG_API_KEY`, `RAG_RATE_PER_MIN`, `SEMANTIC_MATCH` (1/0), `SEMANTIC_MIN_OVERLAP` (0.45).

## Data and license

Patent texts and abstracts come from BIGPATENT (Sharma et al., 2019), released under CC BY 4.0.
The questions and answers were written by Claude Sonnet 4.6 and checked by two other models
(Claude Haiku 4.5 and Qwen3-235B); only answers judged fully correct are included.
