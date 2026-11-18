"""Quick tests:  D:\TDAC_local\rag_venv\Scripts\python -m pytest -q test_api.py"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

import app as appmod

c = TestClient(appmod.app)
BANK = [json.loads(l) for l in open(Path(__file__).parent / "data" / "bank.jsonl", encoding="utf-8")]


def test_health():
    h = c.get("/api/health").json()
    assert h["ok"] and h["checked_questions"] == 7573 and h["patents"] == 998


def test_site_and_docs():
    assert "Patent Q&amp;A" in c.get("/").text
    assert c.get("/docs").status_code == 200


def test_exact_question_gets_checked_answer():
    for b in BANK[::500]:
        r = c.post("/api/ask", json={"question": b["question"], "mode": "verified"}).json()
        assert r["mode"] == "verified" and r["answer"] == b["answer"] and r["doc_id"] == b["doc_id"]


def test_small_wording_changes():
    r = c.post("/api/ask", json={"question": "please tell me how much the portable carpet binding machine weighs",
                                 "mode": "verified"}).json()
    assert r["mode"] == "verified" and "18 pounds" in r["answer"]


def test_key_word_change_is_not_matched_to_wrong_answer():
    # "Gear Ratio C" exists; a made-up "Gear Ratio Z" must not get C's or D's answer
    r = c.post("/api/ask", json={"question": "What is the example value given for Gear Ratio Z in the example calculation?",
                                 "mode": "verified"}).json()
    assert r["mode"] != "verified"


def test_unknown_patent_404():
    assert c.post("/api/ask", json={"question": "what is it?", "doc_id": "patent-nope"}).status_code == 404


def test_patent_endpoints():
    lst = c.get("/api/patents", params={"search": "carpet binding"}).json()
    assert lst and "doc_id" in lst[0]
    p = c.get(f"/api/patents/{lst[0]['doc_id']}").json()
    assert p["questions"]


def test_api_key(monkeypatch):
    monkeypatch.setattr(appmod, "API_KEY", "secret")
    assert c.get("/api/patents").status_code == 401
    assert c.get("/api/patents", headers={"X-API-Key": "secret"}).status_code == 200
