"""Patent Q&A API + website.

Run:   python -m uvicorn app:app --host 0.0.0.0 --port 8010
Site:  http://localhost:8010        API docs: http://localhost:8010/docs
Set RAG_API_KEY to require the header  X-API-Key: <key>  on /api/* calls.
"""
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import LLM_NAME, Engine

HERE = Path(__file__).resolve().parent
API_KEY = os.environ.get("RAG_API_KEY", "")
RATE_PER_MIN = int(os.environ.get("RAG_RATE_PER_MIN", "60"))

engine = Engine()
app = FastAPI(title="Patent Q&A API", version="1.0",
              description="Answers questions about 997 BigPatent patents. Checked questions get the verified "
                          "answer; new questions are answered by retrieval + an LLM (Groq or local), with sources.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_hits = defaultdict(deque)


def guard(request: Request, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "missing or wrong X-API-Key")
    # behind a host's proxy (Render etc.) the visitor's address is the first one in X-Forwarded-For
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
    q, now = _hits[ip], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        raise HTTPException(429, "too many requests, try again in a minute")
    q.append(now)


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["What term describes the seal with a closed perimeter used in the second manufacturing method?"])
    doc_id: Optional[str] = Field(None, description="Patent id, e.g. patent-431431adde0b. Optional but makes answers much better.")
    history: str = Field("", max_length=8000, description="Earlier turns for follow-up questions ('User: ...\\nAssistant: ...')")
    mode: str = Field("auto", pattern="^(auto|verified|rag)$", description="auto = verified first, then RAG")


@app.post("/api/ask", dependencies=[Depends(guard)])
def ask(body: AskIn):
    try:
        return engine.ask(body.question, body.doc_id or None, body.history, body.mode)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/ask", dependencies=[Depends(guard)])
def ask_get(q: str = Query(..., min_length=3), doc_id: Optional[str] = None, mode: str = "auto"):
    return ask(AskIn(question=q, doc_id=doc_id, mode=mode))


@app.get("/api/patents", dependencies=[Depends(guard)])
def patents(search: str = "", limit: int = Query(20, le=200)):
    if search.strip():
        hits = engine.guess_patent(search)
        ids = [d for d, _ in hits]
        # also allow searching by id
        ids += [d for d in engine.docs if search.strip().lower() in d][:5]
        ids = list(dict.fromkeys(ids))[:limit]
    else:
        ids = list(engine.docs)[:limit]
    return [{"doc_id": d, "title": engine.docs[d]["title"], "cpc": engine.docs[d]["cpc"],
             "questions": len(engine.by_doc.get(d, []))} for d in ids]


@app.get("/api/patents/{doc_id}", dependencies=[Depends(guard)])
def patent(doc_id: str, full_text: bool = False):
    d = engine.docs.get(doc_id)
    if not d:
        raise HTTPException(404, "unknown patent id")
    out = {"doc_id": doc_id, "title": d["title"], "cpc": d["cpc"], "abstract": d["abstract"],
           "questions": [{"id": engine.bank[i]["id"], "kind": engine.bank[i]["kind"],
                          "question": engine.bank[i]["question"]} for i in engine.by_doc.get(doc_id, [])]}
    if full_text:
        out["text"] = d["text"]
    return out


@app.get("/api/health")
def health():
    return {"ok": True, "patents": len(engine.docs), "checked_questions": len(engine.bank),
            "chunks": len(engine.chunks), "llm": LLM_NAME, "llm_online": engine.llm_ok(), "auth_required": bool(API_KEY),
            "load_seconds": engine.load_seconds}


@app.get("/", include_in_schema=False)
def site():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
