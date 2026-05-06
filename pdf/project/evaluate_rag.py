import argparse
import csv
import json
import math
import time
from pathlib import Path

from evals import grounding_metrics, provider_name
from pipeline import DocumentLegalPipeline
from rag import VectorStore, chunks_from_records
from utils import write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "evaluation" / "rag_eval_dataset.json"
DEFAULT_OUTPUT = ROOT.parent / "results" / "rag_offline_eval.json"
DEFAULT_CSV = ROOT.parent / "results" / "rag_offline_eval.csv"

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def norm(text):
    return " ".join(str(text or "").lower().split())

def source_match(chunk, target):
    expected = norm(target.get("source_file"))
    actual = norm(chunk.get("source_file"))
    return bool(expected and actual.endswith(expected))

def term_hits(text, terms):
    low = norm(text)
    return [term for term in terms if norm(term) in low]

def target_hit(chunk, target):
    if not source_match(chunk, target):
        return False
    terms = target.get("terms", [])
    return len(term_hits(chunk.get("text", ""), terms)) == len(terms)

def relevant(chunk, targets):
    for target in targets:
        if source_match(chunk, target) and term_hits(chunk.get("text", ""), target.get("terms", [])):
            return True
    return False

def target_hits_at(retrieved, targets, k):
    selected = retrieved[:k]
    hits = []
    for target in targets:
        rank = None
        best_terms = []
        for i, chunk in enumerate(selected, 1):
            if source_match(chunk, target):
                hits_for_chunk = term_hits(chunk.get("text", ""), target.get("terms", []))
                best_terms = sorted(set(best_terms) | set(hits_for_chunk))
                if len(best_terms) == len(target.get("terms", [])):
                    rank = i
                    break
        hits.append({
            "source_file": target.get("source_file"),
            "terms": target.get("terms", []),
            "required": target.get("required", True),
            "hit": rank is not None,
            "rank": rank,
            "matched_terms": best_terms,
            "term_coverage": round(len(best_terms) / max(1, len(target.get("terms", []))), 4)
        })
    return hits

def dcg(values):
    total = 0.0
    for i, value in enumerate(values, 1):
        total += float(value) / math.log2(i + 1)
    return total

def metrics_at_k(retrieved, targets, k):
    selected = retrieved[:k]
    rels = [1 if relevant(chunk, targets) else 0 for chunk in selected]
    target_hits = target_hits_at(retrieved, targets, k)
    required_hits = [item for item in target_hits if item["required"]]
    target_recall = sum(1 for item in required_hits if item["hit"]) / max(1, len(required_hits))
    first_rank = None
    for i, value in enumerate(rels, 1):
        if value:
            first_rank = i
            break
    ideal = [1] * min(sum(rels), k)
    ndcg = dcg(rels) / dcg(ideal) if ideal else 0.0
    return {
        "k": k,
        "precision": round(sum(rels) / max(1, k), 4),
        "recall": round(target_recall, 4),
        "hit": bool(target_recall > 0),
        "mrr": round(1 / first_rank, 4) if first_rank else 0.0,
        "ndcg": round(ndcg, 4),
        "target_hits": target_hits
    }

def answer_eval(answer, retrieved):
    data = grounding_metrics(answer, retrieved)
    data["answer_chars"] = len(answer or "")
    return data

def build_store(pipeline, case_id):
    records = pipeline.law_records()
    chunks = chunks_from_records(records, case_id)
    store = VectorStore(case_id)
    store.build(chunks)
    return store, chunks

def retrieve_for_item(pipeline, store, chunks, query, guard, max_k):
    scope = guard.get("scope", "tunisia")
    raw = store.retrieve(pipeline.legal_focus_query(query), top_k=max_k, pool_k=max(max_k * 4, max_k))
    ranked = pipeline.prioritize_reference_scope(raw, scope)
    return pipeline.ensure_scope_core(ranked, chunks, scope, limit=max_k)

def evaluate_item(pipeline, store, chunks, item, k_values, use_llm):
    query = item["query"]
    expected_in_scope = bool(item.get("expected_in_scope", True))
    targets = item.get("expected_targets", [])
    guard = pipeline.guard.evaluate(query, has_document=False)
    guard_correct = guard.get("in_scope") == expected_in_scope
    row = {
        "id": item["id"],
        "query": query,
        "expected_in_scope": expected_in_scope,
        "guardrail": guard,
        "guard_correct": guard_correct,
        "retrieved": [],
        "retrieval": {},
        "answer": {},
        "passed": False
    }
    if not guard.get("in_scope") or not expected_in_scope:
        row["passed"] = guard_correct
        return row
    max_k = max(k_values)
    retrieved = retrieve_for_item(pipeline, store, chunks, query, guard, max_k)
    row["retrieved"] = [
        {
            "rank": chunk.get("rank"),
            "chunk_id": chunk.get("chunk_id"),
            "source_file": chunk.get("source_file"),
            "score": chunk.get("score"),
            "text_preview": " ".join(chunk.get("text", "").split())[:260]
        }
        for chunk in retrieved
    ]
    scores = {str(k): metrics_at_k(retrieved, targets, k) for k in k_values}
    row["retrieval"] = {
        "engine": getattr(store, "last_retrieval", {}).get("engines", []),
        "latency_ms": getattr(store, "last_retrieval", {}).get("latency_ms", 0.0),
        "metrics_at_k": scores,
        "top_source_file": retrieved[0].get("source_file") if retrieved else None
    }
    if use_llm:
        answer = pipeline.feedback.generate(query, retrieved)
        llm = dict(pipeline.feedback.last_metrics)
        row["answer"] = {"llm": llm, "grounding": answer_eval(answer, retrieved)}
    else:
        row["answer"] = {"llm": {"provider": provider_name(), "attempted": False, "response_mode": "generation_not_requested"}, "grounding": {}}
    top_k = str(max_k)
    row["passed"] = guard_correct and scores[top_k]["recall"] >= 1.0
    return row

def aggregate(results, k_values):
    total = len(results)
    guard_correct = sum(1 for row in results if row["guard_correct"])
    in_scope = [row for row in results if row["expected_in_scope"]]
    blocked = [row for row in results if not row["expected_in_scope"]]
    summary = {
        "items_total": total,
        "in_scope_items": len(in_scope),
        "blocked_items": len(blocked),
        "guardrail_accuracy": round(guard_correct / max(1, total), 4),
        "blocked_prompt_accuracy": round(sum(1 for row in blocked if row["guard_correct"]) / max(1, len(blocked)), 4),
        "retrieval": {},
        "passed_items": sum(1 for row in results if row["passed"]),
        "failed_items": [row["id"] for row in results if not row["passed"]]
    }
    for k in k_values:
        key = str(k)
        rows = [row for row in in_scope if row.get("retrieval", {}).get("metrics_at_k", {}).get(key)]
        summary["retrieval"][f"precision@{k}"] = round(sum(row["retrieval"]["metrics_at_k"][key]["precision"] for row in rows) / max(1, len(rows)), 4)
        summary["retrieval"][f"recall@{k}"] = round(sum(row["retrieval"]["metrics_at_k"][key]["recall"] for row in rows) / max(1, len(rows)), 4)
        summary["retrieval"][f"hit_rate@{k}"] = round(sum(1 for row in rows if row["retrieval"]["metrics_at_k"][key]["hit"]) / max(1, len(rows)), 4)
        summary["retrieval"][f"mrr@{k}"] = round(sum(row["retrieval"]["metrics_at_k"][key]["mrr"] for row in rows) / max(1, len(rows)), 4)
        summary["retrieval"][f"ndcg@{k}"] = round(sum(row["retrieval"]["metrics_at_k"][key]["ndcg"] for row in rows) / max(1, len(rows)), 4)
    return summary

def write_csv(path, results, k_values):
    fields = ["id", "expected_in_scope", "guard_in_scope", "guard_correct", "scope", "passed"]
    for k in k_values:
        fields.extend([f"precision@{k}", f"recall@{k}", f"hit@{k}", f"mrr@{k}", f"ndcg@{k}"])
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            flat = {
                "id": row["id"],
                "expected_in_scope": row["expected_in_scope"],
                "guard_in_scope": row["guardrail"].get("in_scope"),
                "guard_correct": row["guard_correct"],
                "scope": row["guardrail"].get("scope"),
                "passed": row["passed"]
            }
            for k in k_values:
                metrics = row.get("retrieval", {}).get("metrics_at_k", {}).get(str(k), {})
                flat[f"precision@{k}"] = metrics.get("precision", "")
                flat[f"recall@{k}"] = metrics.get("recall", "")
                flat[f"hit@{k}"] = metrics.get("hit", "")
                flat[f"mrr@{k}"] = metrics.get("mrr", "")
                flat[f"ndcg@{k}"] = metrics.get("ndcg", "")
            writer.writerow(flat)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()
    dataset = load_json(args.dataset)
    k_values = dataset.get("k_values", [1, 3, 5, 8])
    started = time.perf_counter()
    pipeline = DocumentLegalPipeline()
    store, chunks = build_store(pipeline, "offline_rag_eval")
    results = [evaluate_item(pipeline, store, chunks, item, k_values, args.use_llm) for item in dataset["items"]]
    summary = aggregate(results, k_values)
    output = {
        "dataset": {"name": dataset.get("name"), "items": len(dataset["items"]), "k_values": k_values},
        "mode": {"llm_generation": bool(args.use_llm), "provider": provider_name()},
        "summary": summary,
        "results": results,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 2)
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    write_csv(args.csv, results, k_values)
    print(json.dumps({"output": str(args.output), "csv": str(args.csv), "summary": summary}, indent=2))

if __name__ == "__main__":
    main()
