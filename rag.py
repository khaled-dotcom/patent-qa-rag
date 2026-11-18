"""Answer engine: verified-answer lookup first, then retrieval + local LLM (RAG).

Layer 1  "verified"  : the question matches one of the 7,573 checked questions
                       (exact, or the same question with small wording changes)
                       -> the checked answer + the patent sentences that support it.
Layer 2  "generated" : BM25 search over all patent text -> local LLM (llama.cpp
                       server, OpenAI-style API) answers from the passages only.
Layer 3  "retrieval_only": the LLM server is not running -> best passages only.
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

DATA = Path(__file__).resolve().parent / "data"

# LLM: Groq (cloud) if GROQ_API_KEY is set, else a local llama.cpp server (OpenAI-style API)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_URL = os.environ.get("LLM_URL", "https://api.groq.com/openai/v1/chat/completions" if GROQ_API_KEY
                         else "http://127.0.0.1:8081/v1/chat/completions")
LLM_NAME = f"Groq {GROQ_MODEL}" if GROQ_API_KEY else "local llama.cpp"


def chat(messages, max_tokens=200):
    """One chat call to the LLM (Groq or local). Returns the reply text."""
    payload = {"messages": messages, "temperature": 0, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json", "User-Agent": "patent-qa-rag/1.0"}
    if GROQ_API_KEY:
        payload["model"] = GROQ_MODEL
        headers["Authorization"] = f"Bearer {GROQ_API_KEY}"
    req = urllib.request.Request(LLM_URL, data=json.dumps(payload).encode(), headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:  # rate limit: wait and retry
                time.sleep(float(e.headers.get("retry-after", 2)))
                continue
            raise
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))
# step 1b: let the LLM decide if a reworded question is the same as a checked one
SEMANTIC_MATCH = os.environ.get("SEMANTIC_MATCH", "1") == "1"
SEMANTIC_MIN_OVERLAP = float(os.environ.get("SEMANTIC_MIN_OVERLAP", "0.45"))

# a reworded question must be at least this similar to a checked one
FUZZY_MIN_ANY_PATENT = 0.85   # when no patent is given
FUZZY_MIN_IN_PATENT = 0.75    # when the patent is given (only ~8 questions to choose from)
NEGATIONS = {"not", "no", "never", "without", "cannot", "except", "neither", "nor"}
STOP = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were", "be",
    "by", "with", "as", "at", "that", "this", "it", "its", "from", "what", "which", "how", "why",
    "does", "do", "did", "patent", "invention", "described", "according", "please", "tell", "me",
    "can", "could", "you", "explain", "i", "want", "know", "would", "like", "here", "there",
}
# words that change the meaning: two questions that differ in one of these are different questions
KEY_WORDS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "last",
    "left", "right", "upper", "lower", "top", "bottom", "inner", "outer", "front", "rear", "back", "side",
    "minimum", "maximum", "min", "max", "least", "most", "before", "after", "increase", "decrease",
    "positive", "negative", "input", "output", "inlet", "outlet", "male", "female", "primary", "secondary",
    "prior", "conventional", "new", "old", "preferred", "alternative", "other", "same", "different",
    "advantage", "disadvantage", "problem", "purpose", "why", "how", "many", "much", "when", "where", "who",
    "width", "length", "height", "thickness", "diameter", "weight", "temperature", "pressure", "range",
}
ORDINALS = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth", "5th": "fifth",
            "6th": "sixth", "7th": "seventh", "8th": "eighth", "9th": "ninth", "10th": "tenth"}


def norm(s: str) -> str:
    s = html.unescape(s).lower().replace("’", "'")
    words = re.sub(r"[^\w]+", " ", s).split()
    return " ".join(ORDINALS.get(w, w) for w in words)


def squash(s: str) -> str:
    """Second key: punctuation removed instead of spaced ("U-shaped" -> "ushaped")."""
    s = html.unescape(s).lower().replace("’", "'")
    return norm(re.sub(r"[^\w\s]", "", s))


def is_key(w: str) -> bool:
    return w in KEY_WORDS or w in NEGATIONS or len(w) == 1 or any(ch.isdigit() for ch in w)


def numbers(s: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_words(s: str) -> set:
    return {stem(w) for w in norm(s).split() if w not in STOP}


class BM25:
    def __init__(self, texts, k1=1.4, b=0.75):
        self.cv = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b", stop_words="english", lowercase=True)
        X = self.cv.fit_transform(texts).tocsc().astype(np.float32)
        n = X.shape[0]
        dl = np.asarray(X.sum(axis=1)).ravel()
        df = np.diff(X.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm_len = k1 * (1 - b + b * dl / max(dl.mean(), 1))
        # pre-compute the BM25 weight of every (chunk, term) pair
        X = X.tocoo()
        w = X.data * (k1 + 1) / (X.data + norm_len[X.row]) * self.idf[X.col]
        from scipy.sparse import csc_matrix
        self.W = csc_matrix((w, (X.row, X.col)), shape=X.shape)
        self.n = n

    def scores(self, query: str) -> np.ndarray:
        ids = [self.cv.vocabulary_[t] for t in set(self.cv.build_analyzer()(query)) if t in self.cv.vocabulary_]
        if not ids:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.W[:, ids].sum(axis=1)).ravel()


def split_chunks(text: str, size=1100):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size * 1.5:  # very long paragraph -> cut at sentence ends
            cut = p.rfind(". ", 0, size)
            cut = cut + 1 if cut > 200 else size
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if len(cur) + len(p) + 1 > size and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if len(s.strip()) > 20]


class Engine:
    def __init__(self, data_dir: Path = DATA):
        t0 = time.time()
        self.docs = {}
        with open(data_dir / "docs.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.docs[d["doc_id"]] = d
        with open(data_dir / "bank.jsonl", encoding="utf-8") as f:
            self.bank = [json.loads(line) for line in f]

        # --- layer 1: checked questions
        self.qnorm = [norm(b["question"]) for b in self.bank]
        self.exact = defaultdict(list)
        self.by_doc = defaultdict(list)
        for i, (q, b) in enumerate(zip(self.qnorm, self.bank)):
            self.exact[q].append(i)
            sq = squash(b["question"])
            if sq != q:
                self.exact[sq].append(i)
            self.by_doc[b["doc_id"]].append(i)
        self.qvec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        self.QM = self.qvec.fit_transform(self.qnorm)
        self.cvec = TfidfVectorizer(analyzer=lambda x: sorted(content_words(x)), sublinear_tf=True)
        self.CM = self.cvec.fit_transform([b["question"] for b in self.bank])

        # --- layer 2: patent chunks for search
        self.chunks = []  # (doc_id, text)
        self.doc_chunks = defaultdict(list)
        for d in self.docs.values():
            for c in split_chunks(d["text"]):
                self.doc_chunks[d["doc_id"]].append(len(self.chunks))
                self.chunks.append((d["doc_id"], c))
        self.bm25 = BM25([c for _, c in self.chunks])
        self.load_seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------ layer 1
    def match_verified(self, question: str, doc_id: str | None = None):
        q = norm(question)
        cands = list(dict.fromkeys(self.exact.get(q, []) + self.exact.get(squash(question), [])))
        if doc_id:
            cands = [i for i in cands if self.bank[i]["doc_id"] == doc_id]
        if cands:
            answers = {norm(self.bank[i]["answer"]) for i in cands}
            if len(answers) == 1 or doc_id:
                return cands[0], 1.0, "exact"
            return None, 1.0, "ambiguous"  # same question, different patents, different answers

        pool = self.by_doc.get(doc_id, []) if doc_id else None
        v = self.qvec.transform([q])
        sims = (self.QM[pool] if pool is not None else self.QM).dot(v.T).toarray().ravel()
        if not len(sims):
            return None, 0.0, "none"
        order = np.argsort(-sims)[:3]
        best = order[0]
        idx = pool[best] if pool is not None else int(best)
        score = float(sims[best])
        need = FUZZY_MIN_IN_PATENT if doc_id else FUZZY_MIN_ANY_PATENT
        cand = self.bank[idx]["question"]
        cw_c, cw_q = content_words(cand), content_words(question)
        # also accept: almost the same content words (at most 1 differs, never a key word)
        near = len(cw_c) >= 4 and len(cw_c ^ cw_q) <= 1 and score >= 0.5
        if score < need and not near:
            return None, score, "none"
        # safety checks: numbers and negations must agree, few content words may differ
        if numbers(cand) != numbers(question):
            return None, score, "numbers_differ"
        if (NEGATIONS & set(norm(cand).split())) != (NEGATIONS & set(q.split())):
            return None, score, "negation_differs"
        diff = {stem(w) for w in norm(cand).split()} ^ {stem(w) for w in q.split()}
        if any(is_key(w) for w in diff):
            return None, score, "key_word_differs"
        if len(cw_c ^ cw_q) > 3:
            return None, score, "words_differ"
        if pool is None and len(order) > 1:
            second = int(order[1])
            if sims[second] > score - 0.02 and self.bank[second]["doc_id"] != self.bank[idx]["doc_id"] \
                    and norm(self.bank[second]["answer"]) != norm(self.bank[idx]["answer"]):
                return None, score, "ambiguous"
        return idx, score, "reworded"

    def semantic_match(self, question: str, doc_id: str | None = None, exclude=(), k=5):
        """Ask the LLM which of the k closest checked questions asks for the same thing (or none)."""
        pool = self.by_doc.get(doc_id, []) if doc_id else list(range(len(self.bank)))
        pool = [i for i in pool if i not in exclude]
        if not pool:
            return None, "none"
        sims = self.CM[pool].dot(self.cvec.transform([question]).T).toarray().ravel()
        # word-overlap floor 0.45: measured on 120 LLM rewordings + 120 trap questions it removed
        # 11 of 12 wrong LLM picks while keeping 70 of 91 right ones (see README)
        order = [pool[j] for j in np.argsort(-sims)[:k] if sims[j] >= SEMANTIC_MIN_OVERLAP]
        # numbers in the user question must appear in the checked question
        order = [i for i in order if numbers(question) <= numbers(self.bank[i]["question"])]
        if not order:
            return None, "none"
        def yes(a, b):
            p = (f"Question A: {a}\nQuestion B: {b}\n\n"
                 "Would the correct answer to Question B also be the correct answer to Question A? Answer Yes or No.")
            return chat([{"role": "user", "content": p}], max_tokens=3).lower().startswith("yes")

        # closest first; accept only a clear Yes in both directions
        for i in order[:3]:
            cand = self.bank[i]["question"]
            if yes(question, cand) and yes(cand, question):
                return i, "llm_same_meaning"
        return None, "llm_none"

    def evidence(self, doc_id: str, question: str, answer: str, k=2):
        sents = split_sentences(self.docs[doc_id]["text"])
        aw, qw = content_words(answer), content_words(question)
        best = sorted(
            ((2 * len(aw & content_words(s)) + len(qw & content_words(s)), j) for j, s in enumerate(sents)),
            reverse=True)[:k]
        return [sents[j] for sc, j in sorted(best, key=lambda x: x[1]) if sc > 0]

    # ------------------------------------------------------------ layer 2
    def retrieve(self, query: str, doc_id: str | None = None, k=5):
        s = self.bm25.scores(query)
        if doc_id:
            ids = self.doc_chunks.get(doc_id, [])
            ids = sorted(ids, key=lambda i: -s[i])[:k]
        else:
            ids = list(np.argsort(-s)[:k])
        return [{"doc_id": self.chunks[i][0], "score": round(float(s[i]), 2), "text": self.chunks[i][1]}
                for i in ids]

    def guess_patent(self, query: str):
        """Which patent is a question about? Sum of top chunk scores per patent."""
        s = self.bm25.scores(query)
        top = np.argsort(-s)[:30]
        agg = defaultdict(float)
        for i in top:
            agg[self.chunks[i][0]] += float(s[i])
        return sorted(agg.items(), key=lambda x: -x[1])[:5]

    def llm(self, passages, question, history=""):
        ctx = "\n\n".join(f"[{i+1}] (patent {p['doc_id']})\n{p['text']}" for i, p in enumerate(passages))
        system = ("You answer questions about patents. Use ONLY the numbered passages. "
                  "Answer briefly and exactly (a short phrase or one or two sentences). "
                  "If the passages do not contain the answer, reply exactly: Not found in the patent text.")
        user = f"Passages:\n{ctx}\n\n"
        if history:
            user += f"Earlier conversation:\n{history}\n\n"
        user += f"Question: {question}\nAnswer:"
        return chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=200)

    def llm_ok(self):
        if GROQ_API_KEY:
            return True  # checked on first call; avoids spending requests on health checks
        try:
            url = LLM_URL.split("/v1/")[0] + "/health"
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------ main entry
    def ask(self, question: str, doc_id: str | None = None, history: str = "", mode: str = "auto"):
        t0 = time.time()
        question = (question or "").strip()
        if doc_id and doc_id not in self.docs:
            raise KeyError(f"unknown patent id: {doc_id}")
        res = {"question": question, "doc_id": doc_id}

        if mode in ("auto", "verified"):
            idx, score, how = self.match_verified(question, doc_id)
            if idx is not None:
                b = self.bank[idx]
                res.update({
                    "answer": b["answer"], "mode": "verified", "confidence": 1.0,
                    "match": {"type": how, "score": round(score, 3), "question_id": b["id"],
                              "matched_question": b["question"]},
                    "doc_id": b["doc_id"], "patent_title": self.docs[b["doc_id"]]["title"],
                    "sources": [{"doc_id": b["doc_id"], "text": s} for s in self.evidence(b["doc_id"], b["question"], b["answer"])],
                })
                res["ms"] = int((time.time() - t0) * 1000)
                return res
            res["match"] = {"type": how, "score": round(score, 3)}
            if SEMANTIC_MATCH and how != "ambiguous":
                try:
                    idx, how2 = self.semantic_match(question, doc_id)
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                    idx, how2 = None, "llm_offline"
                if idx is not None:
                    b = self.bank[idx]
                    res.update({
                        "answer": b["answer"], "mode": "verified", "confidence": 0.9,
                        "match": {"type": how2, "score": round(score, 3), "question_id": b["id"],
                                  "matched_question": b["question"]},
                        "doc_id": b["doc_id"], "patent_title": self.docs[b["doc_id"]]["title"],
                        "sources": [{"doc_id": b["doc_id"], "text": s} for s in self.evidence(b["doc_id"], b["question"], b["answer"])],
                    })
                    res["ms"] = int((time.time() - t0) * 1000)
                    return res
                res["match"]["semantic"] = how2
            if mode == "verified":
                res.update({"answer": None, "mode": "not_found", "confidence": 0.0, "sources": []})
                res["ms"] = int((time.time() - t0) * 1000)
                return res

        query = question if not history else question + " " + history.splitlines()[-1]
        passages = self.retrieve(query, doc_id, k=5)
        res["sources"] = passages
        if passages and not doc_id:
            res["patent_guess"] = [{"doc_id": d, "title": self.docs[d]["title"], "score": round(s, 1)}
                                   for d, s in self.guess_patent(query)]
        try:
            ans = self.llm(passages, question, history)
            found = not ans.lower().startswith("not found")
            support = len(content_words(ans) & set().union(*[content_words(p["text"]) for p in passages])) \
                / max(len(content_words(ans)), 1) if found else 0.0
            res.update({"answer": ans, "mode": "generated", "confidence": round(0.6 * support, 2) if found else 0.0})
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            res.update({"answer": None, "mode": "retrieval_only", "confidence": 0.0,
                        "note": f"LLM server not reachable ({type(e).__name__}); showing best passages only"})
        if res.get("doc_id") is None and passages:
            res["doc_id"] = passages[0]["doc_id"]
        if res.get("doc_id"):
            res["patent_title"] = self.docs[res["doc_id"]]["title"]
        res["ms"] = int((time.time() - t0) * 1000)
        return res
