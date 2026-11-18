"""Measure accuracy of the Q&A API.

A) every checked question (7,573), asked exactly as written, with and without the patent id
B) the same questions with small changes (lower case, no punctuation, "Please tell me ...")
C) RAG layer alone on a sample (mode=rag): what a NEW question gets. Scored by token F1 and
   "answer contains the key words"; outputs saved to results/rag_sample.jsonl for reading.

Run:  python evaluate.py                     (in-process, fast)
      python evaluate.py --url http://localhost:8000 [--key KEY]   (through the real HTTP API)
      python evaluate.py --rag 100           (also run part C on 100 questions; needs the LLM server)
      python evaluate.py --para 200          (part D: LLM-reworded questions; needs the LLM server)

D) questions reworded by the local LLM (cached in results/paraphrases.jsonl): how many still get the
   verified answer, and - most important - how many get a WRONG verified answer (must be 0).
"""
import argparse
import json
import random
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def toks(s):
    return re.findall(r"\w+", (s or "").lower())


def f1(pred, gold):
    p, g = toks(pred), toks(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def same(a, b):
    return " ".join(toks(a)) == " ".join(toks(b))


def make_asker(url, key):
    if not url:
        from rag import Engine
        eng = Engine()
        return lambda q, d=None, h="", m="auto": eng.ask(q, d, h, m), eng.bank

    def ask(q, d=None, h="", m="auto"):
        body = json.dumps({"question": q, "doc_id": d, "history": h, "mode": m}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/api/ask", data=body,
                                     headers={"Content-Type": "application/json", **({"X-API-Key": key} if key else {})})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    bank = [json.loads(l) for l in open(HERE / "data" / "bank.jsonl", encoding="utf-8")]
    return ask, bank


VARIANTS = {
    "lower_no_punct": lambda q: re.sub(r"[^\w\s]", "", q.lower()),
    "please_tell_me": lambda q: "Please tell me: " + q[0].lower() + q[1:],
    "extra_spaces": lambda q: "  " + q.replace(" ", "  ") + "  ",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--key", default="")
    ap.add_argument("--rag", type=int, default=0, help="sample size for part C (0 = skip)")
    ap.add_argument("--para", type=int, default=0, help="sample size for part D (0 = skip)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-ab", action="store_true")
    a = ap.parse_args()
    ask, bank = make_asker(a.url, a.key)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    rep_path = out / "accuracy_report.json"
    report = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
    report.update({"date": time.strftime("%Y-%m-%d %H:%M"), "via": a.url or "in-process"})

    # A + B
    for label, with_doc, variant in [("A_exact_no_patent_id", False, None), ("A_exact_with_patent_id", True, None)] + \
            [(f"B_{n}", False, n) for n in VARIANTS] if not a.skip_ab else []:
        t0, ok, wrong, missed, by_kind = time.time(), 0, [], 0, Counter()
        for b in bank:
            q = VARIANTS[variant](b["question"]) if variant else b["question"]
            r = ask(q, b["doc_id"] if with_doc else None, "", "verified")
            if r["mode"] != "verified":
                missed += 1
            elif same(r["answer"], b["answer"]):
                ok += 1
                by_kind[b["kind"]] += 1
            else:
                wrong.append({"id": b["id"], "q": q, "got": r["answer"], "gold": b["answer"]})
        n = len(bank)
        report[label] = {"questions": n, "correct": ok, "accuracy_pct": round(100 * ok / n, 2),
                         "wrong_answer": len(wrong), "not_matched": missed, "by_kind": dict(by_kind),
                         "seconds": round(time.time() - t0, 1)}
        if wrong:
            report[label]["wrong_examples"] = wrong[:10]
        print(label, report[label]["accuracy_pct"], "%", f"(wrong {len(wrong)}, not matched {missed})")

    # C
    if a.rag:
        random.seed(a.seed)
        qa = [b for b in bank if b["kind"] == "qa"]
        sample = random.sample(qa, a.rag)
        rows = []
        for i, b in enumerate(sample):
            for with_doc in (True, False):
                r = ask(b["question"], b["doc_id"] if with_doc else None, "", "rag")
                rows.append({"id": b["id"], "with_patent_id": with_doc, "question": b["question"], "gold": b["answer"],
                             "answer": r.get("answer"), "mode": r["mode"], "f1": round(f1(r.get("answer"), b["answer"]), 3),
                             "right_patent": r.get("doc_id") == b["doc_id"] or any(s["doc_id"] == b["doc_id"] for s in r.get("sources", [])[:1])})
            print(f"  RAG {i+1}/{a.rag}", end="\r")
        with open(out / "rag_sample.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for with_doc in (True, False):
            rs = [r for r in rows if r["with_patent_id"] == with_doc]
            key = "C_rag_new_question_" + ("with_patent_id" if with_doc else "no_patent_id")
            report[key] = {"questions": len(rs), "mean_token_f1": round(sum(r["f1"] for r in rs) / len(rs), 3),
                           "f1_at_least_0.5_pct": round(100 * sum(r["f1"] >= 0.5 for r in rs) / len(rs), 1),
                           "found_right_patent_pct": round(100 * sum(r["right_patent"] for r in rs) / len(rs), 1),
                           "answered_pct": round(100 * sum(r["mode"] == "generated" and not (r["answer"] or "").lower().startswith("not found") for r in rs) / len(rs), 1)}
            print(key, report[key])

    # D
    if a.para:
        from rag import LLM_URL
        cache_p = out / "paraphrases.jsonl"
        cache = {}
        if cache_p.exists():
            cache = {r["id"]: r["reworded"] for r in map(json.loads, open(cache_p, encoding="utf-8"))}
        random.seed(a.seed + 1)
        sample = random.sample(bank, a.para)
        res = Counter()
        rows = []
        for i, b in enumerate(sample):
            if b["id"] not in cache:
                body = json.dumps({"messages": [{"role": "user", "content":
                    "Rewrite this question in different words with exactly the same meaning. "
                    "Reply with the new question only.\n\nQuestion: " + b["question"]}],
                    "temperature": 0.7, "max_tokens": 80, "seed": i}).encode()
                req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    cache[b["id"]] = json.loads(r.read())["choices"][0]["message"]["content"].strip().strip('"').splitlines()[0]
                with open(cache_p, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": b["id"], "question": b["question"], "reworded": cache[b["id"]]}, ensure_ascii=False) + "\n")
            q2 = cache[b["id"]]
            for with_doc in (False, True):
                r = ask(q2, b["doc_id"] if with_doc else None, "", "verified")
                tag = ("with_id_" if with_doc else "no_id_") + (
                    "not_matched" if r["mode"] != "verified" else "correct" if same(r["answer"], b["answer"]) else "WRONG")
                res[tag] += 1
                if tag.endswith("WRONG"):
                    rows.append({"id": b["id"], "original": b["question"], "reworded": q2, "matched": r["match"]["matched_question"],
                                 "got": r["answer"], "gold": b["answer"]})
            print(f"  reworded {i+1}/{a.para}", end="\r")
        report["D_llm_reworded"] = {"questions": a.para, **dict(sorted(res.items())), "wrong_examples": rows[:20],
                                    "note": "WRONG may also be a different question that has the same meaning; read the examples"}
        print("D_llm_reworded", dict(sorted(res.items())))

    (out / "accuracy_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", out / "accuracy_report.json")


if __name__ == "__main__":
    main()
