import re
from collections import Counter
from config import RAG_EMBEDDINGS, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

CITATION_RE = re.compile(r"\[([A-Za-z0-9_]+_C\d{5})\]")
PLACEHOLDER_RE = re.compile(r"\[insert[^\]]*\]|insert article number|article number\]|\bTBD\b|\bTODO\b|citation needed|\[source\]", re.IGNORECASE)

DOMAIN_TERMS = {
    "tunisia", "tunisian", "organic law", "law no. 58", "loi", "article", "legal", "law",
    "lawyer", "court", "judge", "police", "complaint", "protection", "evidence", "deadline",
    "procedure", "jurisdiction", "rights", "victim", "survivor", "violence", "woman", "women",
    "domestic", "rape", "sexual assault", "harassment", "threat", "blackmail", "stalking",
    "injury", "medical", "witness", "safety", "risk", "case", "document", "pdf", "ocr",
    "image", "screenshot", "handwritten", "handwriting", "signature", "tamper", "authentic",
    "extract", "summarize", "summary", "classify", "facts", "next steps", "gaps", "vawa",
    "united states", "u.s.", "ovw", "technology-facilitated abuse", "technological abuse",
    "abuse", "housing", "confidentiality", "international", "cedaw", "who", "public health"
}

DOCUMENT_ONLY_TERMS = {
    "document", "pdf", "ocr", "image", "screenshot", "handwritten", "handwriting", "signature",
    "tamper", "authentic", "extract", "summarize", "summary", "classify", "facts", "next steps",
    "gaps"
}

DOCUMENT_ACTION_TERMS = {
    "extract", "summarize", "summary", "review", "analyze", "analyse", "classify", "read",
    "detect", "ocr", "what is in", "what does this", "key facts", "facts", "evidence",
    "regions", "layout", "bounding", "boxes", "pages", "document", "file", "pdf", "image"
}

INJECTION_TERMS = {
    "ignore previous", "ignore all", "system prompt", "developer message", "jailbreak",
    "dan mode", "bypass", "override instructions", "forget your rules", "reveal prompt",
    "show your prompt", "print your instructions", "act as unrestricted"
}

US_SCOPE_TERMS = {
    "united states", "u.s.", "us federal", "federal", "vawa", "housing", "34 u.s.c.",
    "covered housing", "self-certification", "ovw"
}

INTERNATIONAL_SCOPE_TERMS = {
    "international", "cedaw", "who", "cdc", "un declaration", "human rights", "public health"
}

TUNISIA_SCOPE_TERMS = {
    "tunisia", "tunisian", "law no. 58", "organic law", "article 1", "article 2",
    "article 3", "article 13", "article 14", "article 30", "article 31", "article 39"
}

LEGAL_CLAIM_TRIGGERS = [
    "organic law", "law no. 58", "article ", "protection request", "family judge", "court",
    "competent authorities", "prosecution", "punishment", "legal protection", "judicial aid",
    "deadline", "procedure", "statute", "legal right", "must file", "can file"
]

CORE_ARTICLES = ["1", "2", "3", "13", "14", "30", "39"]

class QueryGuard:
    def normalize(self, text):
        return " ".join(str(text or "").lower().split())

    def matches(self, text, terms):
        return [term for term in sorted(terms, key=len, reverse=True) if term in text]

    def scope(self, text, has_document=False):
        if self.matches(text, US_SCOPE_TERMS):
            return "us_federal"
        if self.matches(text, INTERNATIONAL_SCOPE_TERMS):
            return "international"
        if self.matches(text, TUNISIA_SCOPE_TERMS):
            return "tunisia"
        if has_document:
            return "document"
        return "general"

    def evaluate(self, question, has_document=False):
        raw = str(question or "").strip()
        text = self.normalize(raw)
        scope = self.scope(text, has_document)
        injection = self.matches(text, INJECTION_TERMS)
        domain = [term for term in self.matches(text, DOMAIN_TERMS) if has_document or term not in DOCUMENT_ONLY_TERMS]
        doc_actions = self.matches(text, DOCUMENT_ACTION_TERMS)
        if not raw:
            return {"in_scope": True, "reason": "default_question", "scope": scope, "matched_terms": [], "prompt_injection_detected": False}
        if injection:
            return {"in_scope": False, "reason": "prompt_injection_or_instruction_override", "scope": "blocked", "matched_terms": injection[:8], "prompt_injection_detected": True}
        if domain:
            return {"in_scope": True, "reason": "domain_terms", "scope": scope, "matched_terms": domain[:12], "prompt_injection_detected": False}
        if has_document and doc_actions:
            return {"in_scope": True, "reason": "uploaded_document_action", "scope": scope, "matched_terms": doc_actions[:12], "prompt_injection_detected": False}
        return {"in_scope": False, "reason": "outside_supported_scope", "scope": "blocked", "matched_terms": [], "prompt_injection_detected": False}

    def refusal(self, has_document=False):
        target = "the uploaded document and Tunisian legal-reference material" if has_document else "the Tunisian legal-reference material"
        return "\n".join([
            "Request Not Covered",
            f"- This system is limited to {target}, evidence triage, OCR/layout findings, and violence-against-women legal reference support.",
            "- The submitted question is outside that scope, so no unrelated answer was generated.",
            "",
            "Supported Requests",
            "- Extract key facts, risks, evidence gaps, and next steps from the document.",
            "- Explain retrieved Tunisian legal references with citations.",
            "- Review evidence quality, OCR findings, layout regions, and model signals from the uploaded file."
        ])

def estimate_tokens(text):
    words = len(str(text or "").split())
    return int(round(words * 1.3)) if words else 0

def provider_name():
    if "11434" in LLM_BASE_URL or LLM_API_KEY == "ollama":
        return "ollama"
    return "openai_compatible"

def answer_citations(answer, retrieved):
    cited = CITATION_RE.findall(answer or "")
    valid_ids = {chunk.get("chunk_id", "") for chunk in retrieved}
    valid = [cid for cid in cited if cid in valid_ids]
    invalid = [cid for cid in cited if cid not in valid_ids]
    total = len(cited)
    return {
        "citations_total": total,
        "citations_unique": len(set(cited)),
        "valid_citations": len(valid),
        "invalid_citations": len(invalid),
        "citation_precision": round(len(valid) / total, 4) if total else 0.0,
        "answer_has_citations": bool(cited),
        "invalid_citation_ids": sorted(set(invalid))[:20],
    }

def legal_claim_lines(answer):
    text = re.split(r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{0,2})?(?:\d+\.\s*)?citations used(?:\*{0,2})?\s*:?\s*$", answer or "", maxsplit=1)[0]
    out = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        low = clean.lower()
        if any(trigger in low for trigger in LEGAL_CLAIM_TRIGGERS):
            out.append(clean)
    return out

def grounding_metrics(answer, retrieved):
    citations = answer_citations(answer, retrieved)
    valid_ids = {chunk.get("chunk_id", "") for chunk in retrieved}
    claims = legal_claim_lines(answer)
    unsupported = []
    for line in claims:
        if not any(cid and f"[{cid}]" in line for cid in valid_ids):
            unsupported.append(line[:240])
    out = dict(citations)
    out.update({
        "legal_claim_lines": len(claims),
        "unsupported_legal_claim_lines": len(unsupported),
        "unsupported_legal_claim_samples": unsupported[:8],
        "placeholder_detected": bool(PLACEHOLDER_RE.search(answer or "")),
    })
    return out

def rag_engine(store):
    semantic = bool(getattr(store, "faiss_index", None) is not None and getattr(store, "embedder", None) is not None)
    lexical = bool(getattr(store, "tfidf_matrix", None) is not None)
    if semantic and lexical:
        return "hybrid_faiss_tfidf"
    if semantic:
        return "faiss"
    if lexical:
        return "tfidf"
    return "none"

def article_coverage(chunks):
    joined = "\n".join(chunk.get("text", "") for chunk in chunks).lower()
    found = []
    for article in CORE_ARTICLES:
        if re.search(rf"\barticle\s+{re.escape(article)}\b", joined):
            found.append(article)
    return {"required": CORE_ARTICLES, "found": found, "coverage": round(len(found) / len(CORE_ARTICLES), 4)}

def retrieval_metrics(store, chunks, retrieved):
    scores = [float(chunk.get("score") or 0.0) for chunk in retrieved]
    kinds = Counter(str(chunk.get("kind") or "unknown") for chunk in retrieved)
    regions = Counter(str(chunk.get("region_type") or "none") for chunk in retrieved)
    law = [chunk for chunk in retrieved if chunk.get("kind") == "legal_knowledge"]
    document = [chunk for chunk in retrieved if chunk.get("kind") != "legal_knowledge"]
    coverage = article_coverage(law)
    return {
        "engine": rag_engine(store),
        "embeddings_enabled": bool(RAG_EMBEDDINGS),
        "chunks_total": len(chunks),
        "retrieved_count": len(retrieved),
        "legal_reference_chunks": len(law),
        "document_chunks": len(document),
        "top_score": round(max(scores), 6) if scores else 0.0,
        "mean_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "kind_counts": dict(kinds),
        "region_type_counts": dict(regions),
        "tunisia_core_article_coverage": coverage,
        "diagnostics": getattr(store, "last_retrieval", {}),
    }

def document_metrics(document):
    if not document:
        return {}
    elements = document.get("elements", [])
    pages = document.get("pages", [])
    kind_counts = Counter(str(item.get("kind") or "unknown") for item in elements)
    region_counts = Counter(str(item.get("region_type") or "none") for item in elements)
    text_sources = Counter(str(item.get("text_source") or "none") for item in elements)
    classified = sum(1 for item in elements if item.get("predictions"))
    text_elements = sum(1 for item in elements if str(item.get("text") or "").strip())
    return {
        "pages": len(pages),
        "elements_total": len(elements),
        "elements_with_text": text_elements,
        "visual_elements_classified": classified,
        "kind_counts": dict(kind_counts),
        "region_type_counts": dict(region_counts),
        "text_source_counts": dict(text_sources),
        "has_model_summary": bool(document.get("model_summary")),
    }

def quality_flags(guard, rag, grounding, llm):
    flags = []
    if guard and not guard.get("in_scope", True):
        flags.append("request_blocked_out_of_scope")
    if rag.get("chunks_total", 0) == 0:
        flags.append("no_indexed_chunks")
    if rag.get("retrieved_count", 0) == 0 and not flags:
        flags.append("no_retrieved_context")
    if rag.get("legal_reference_chunks", 0) == 0 and rag.get("retrieved_count", 0) > 0:
        flags.append("no_legal_reference_retrieved")
    if guard.get("scope") in {"tunisia", "document"} and rag.get("tunisia_core_article_coverage", {}).get("coverage", 0.0) < 0.5 and rag.get("legal_reference_chunks", 0) > 0:
        flags.append("low_tunisia_core_article_coverage")
    if grounding.get("invalid_citations", 0) > 0:
        flags.append("invalid_answer_citation")
    if grounding.get("unsupported_legal_claim_lines", 0) > 0:
        flags.append("unsupported_legal_claim_line")
    if grounding.get("placeholder_detected"):
        flags.append("placeholder_detected")
    if llm.get("attempted") and not llm.get("success"):
        flags.append("llm_generation_failed")
    if llm.get("fallback_used"):
        flags.append("template_fallback_used")
    return flags

def evaluate_run(query, guard, store, chunks, retrieved, answer, document=None, llm=None):
    rag = retrieval_metrics(store, chunks, retrieved) if store is not None else {"engine": "none", "embeddings_enabled": bool(RAG_EMBEDDINGS), "chunks_total": len(chunks or []), "retrieved_count": len(retrieved or [])}
    grounding = grounding_metrics(answer, retrieved or [])
    llm_metrics = dict(llm or {})
    llm_metrics.setdefault("provider", provider_name())
    llm_metrics.setdefault("model", LLM_MODEL)
    out = {
        "guardrail": guard or {},
        "rag": rag,
        "answer_grounding": grounding,
        "llm": llm_metrics,
        "document_extraction": document_metrics(document),
        "quality_flags": quality_flags(guard or {}, rag, grounding, llm_metrics),
        "query": {"chars": len(query or ""), "estimated_tokens": estimate_tokens(query)},
    }
    return out
