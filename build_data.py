"""Build the compact data files the API uses, from the patent delivery package.

Output (in ./data):
  docs.jsonl  - one line per patent: doc_id, cpc, title, abstract, text
  bank.jsonl  - one line per checked question: id, doc_id, kind (qa/conversation),
                question, answer, history (conversation only), meta
Run:  python build_data.py [--src <03_Dataset folder>]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "03_packages_unzipped" / "TDAC_Patents_Delivery_2026-11-13" / "03_Dataset" / "lmeval_data"


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def make_title(abstract: str) -> str:
    first = abstract.split(". ")[0].strip()
    return (first[:117] + "...") if len(first) > 120 else first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    a = ap.parse_args()
    src = Path(a.src)
    out = HERE / "data"
    out.mkdir(exist_ok=True)

    docs, bank = {}, []
    abstracts = {r["doc_id"]: r.get("abstract", "") for r in read_jsonl(src / "patent_summary.jsonl")}
    summ_text = {r["doc_id"]: r["document"] for r in read_jsonl(src / "patent_summary.jsonl")}

    for kind, fname in (("qa", "patent_qa.jsonl"), ("conversation", "patent_conversation.jsonl")):
        for r in read_jsonl(src / fname):
            if r.get("judge") != "correct":
                continue  # only answers checked as fully correct go in the bank
            d = r["doc_id"]
            docs.setdefault(d, {"doc_id": d, "cpc": r.get("cpc", ""), "text": r["context"]})
            item = {
                "id": r["id"], "doc_id": d, "kind": kind,
                "question": r["question"], "answer": r["answer"],
            }
            if kind == "conversation":
                item["history"] = r.get("history", "")
                item["turn"] = r.get("turn")
            else:
                item["meta"] = {k: r.get(k) for k in ("qtype", "task_type", "difficulty")}
            bank.append(item)

    # patents that only appear in the summary file are still searchable
    for d, text in summ_text.items():
        docs.setdefault(d, {"doc_id": d, "cpc": "", "text": text})
    for d, doc in docs.items():
        doc["abstract"] = abstracts.get(d, "")
        doc["title"] = make_title(doc["abstract"]) if doc["abstract"] else doc["text"][:100]

    with open(out / "docs.jsonl", "w", encoding="utf-8") as f:
        for doc in docs.values():
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with open(out / "bank.jsonl", "w", encoding="utf-8") as f:
        for item in bank:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"patents: {len(docs)}  checked questions: {len(bank)} "
          f"(qa {sum(b['kind']=='qa' for b in bank)}, conversation {sum(b['kind']=='conversation' for b in bank)})")


if __name__ == "__main__":
    main()
