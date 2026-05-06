from pathlib import Path
import uuid
import re
from document_io import read_document
from model_loader import ModelManager
from rag import VectorStore, chunks_from_records
from legal_feedback import LegalFeedbackEngine
from evals import QueryGuard, evaluate_run
from utils import sha256_file, write_json
from config import LAW_DIR, CNN_TEXT_REGIONS

VISUAL_REGION_TYPES = {"page", "image", "figure", "picture", "photo", "graphic", "signature", "stamp", "handwriting", "unknown", "pdf_page_overview"}
TEXT_REGION_TYPES = {"text", "title", "list", "table", "form_field", "docx", "xlsx_sheet", "json_text"}

class DocumentLegalPipeline:
    def __init__(self):
        self.models = ModelManager()
        self.feedback = LegalFeedbackEngine()
        self.guard = QueryGuard()

    def model_status(self):
        return self.models.status()

    def law_records(self):
        records = []
        for path in LAW_DIR.rglob("*"):
            if path.is_file():
                try:
                    records.extend(read_document(path))
                except Exception:
                    pass
        for row in records:
            row["kind"] = "legal_knowledge"
        return records

    def should_classify_image(self, record, image):
        if image is None:
            return False
        kind = str(record.get("kind", "")).lower()
        region_type = str(record.get("region_type", "")).lower()
        text = str(record.get("text", "") or "").strip()
        if kind in {"pdf_page_overview", "image"}:
            return True
        if region_type in VISUAL_REGION_TYPES:
            return True
        if region_type in TEXT_REGION_TYPES and CNN_TEXT_REGIONS:
            return True
        if not text:
            return True
        bbox = record.get("bbox") or []
        page_width = float(record.get("page_width") or 0)
        page_height = float(record.get("page_height") or 0)
        if len(bbox) == 4 and page_width > 0 and page_height > 0:
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            if area / max(1.0, page_width * page_height) > 0.45:
                return True
        return False

    def prediction_context(self, predictions):
        if not predictions:
            return ""
        parts = []
        c = predictions.get("content")
        if c:
            parts.append(f"Visual content classification: {c.get('label')} confidence {c.get('confidence')} runner_up {c.get('runner_up')}.")
            if c.get("unclassifiable"):
                parts.append("The visual content classifier marked this element as low-confidence or unclassifiable.")
        e = predictions.get("evidence")
        if e:
            parts.append(f"Evidence weight classification: {e.get('label')} confidence {e.get('confidence')}.")
        q = predictions.get("quality")
        if q:
            parts.append(f"Image quality classification: {q.get('label')} confidence {q.get('confidence')}.")
        t = predictions.get("tamper")
        if t:
            parts.append(f"Tamper classification: {t.get('label')} confidence {t.get('confidence')} risk {t.get('risk_label')}.")
        return " ".join(parts)

    def element_descriptor(self, item):
        parts = []
        if item.get("page") not in [None, ""]:
            parts.append(f"page {item.get('page')}")
        if item.get("id"):
            parts.append(f"element {item.get('id')}")
        if item.get("kind"):
            parts.append(f"kind {item.get('kind')}")
        if item.get("region_type"):
            parts.append(f"region {item.get('region_type')}")
        if item.get("bbox"):
            parts.append(f"bbox {item.get('bbox')}")
        if item.get("text_source"):
            parts.append(f"text_source {item.get('text_source')}")
        if item.get("layout_confidence") not in [None, ""]:
            parts.append(f"layout_confidence {item.get('layout_confidence')}")
        if item.get("reading_order") not in [None, ""]:
            parts.append(f"reading_order {item.get('reading_order')}")
        return "; ".join(parts)

    def document_model_summary(self, elements):
        counts = {}
        covered = set()
        classified = 0
        for item in elements:
            preds = item.get("predictions", {})
            if preds:
                classified += 1
            for key in ["content", "evidence", "quality", "tamper"]:
                label = preds.get(key, {}).get("label")
                if label:
                    covered.add(key)
                    counts[f"{key}:{label}"] = counts.get(f"{key}:{label}", 0) + 1
        if not counts:
            return ""
        models = ", ".join(name for name in ["content", "evidence", "quality", "tamper"] if name in covered)
        total = len(elements)
        return f"Visual review completed with {models} models on {classified} classified visual element{'s' if classified != 1 else ''} across {total} extracted document element{'s' if total != 1 else ''}."

    def extract_fact_lines(self, text, max_lines=18):
        fields = {"jurisdiction", "date prepared", "person requesting help", "person affected", "applicant", "relationship context", "reported person", "location", "incident date", "document type", "legal focus"}
        facts = []
        section = ""
        continuable = ("Reported fact: ", "Available evidence: ", "Point to verify: ")
        for raw in text.splitlines():
            line = " ".join(raw.strip().split())
            if not line:
                continue
            clean = re.sub(r"^[\s\-\*\+\u2022]+", "", line).strip()
            low = clean.lower()
            if clean.endswith(":") and len(clean) < 80:
                section = clean[:-1].lower()
                continue
            if ":" in clean:
                key, value = clean.split(":", 1)
                key_norm = key.strip().lower()
                value = value.strip()
                if key_norm in fields and value:
                    facts.append(f"{key.strip()}: {value}")
            elif raw.lstrip().startswith("-") and clean:
                if section in {"key facts", "incident summary", "summary", "facts to organize"}:
                    facts.append(f"Reported fact: {clean}")
                elif section in {"evidence available", "evidence to bring"}:
                    facts.append(f"Available evidence: {clean}")
                elif section in {"missing facts", "safety points to check", "questions for review", "questions for the lawyer or support service"}:
                    facts.append(f"Point to verify: {clean}")
            elif section in {"key facts", "incident summary", "summary", "facts to organize", "evidence available", "evidence to bring", "missing facts", "safety points to check", "questions for review", "questions for the lawyer or support service"} and facts and facts[-1].startswith(continuable) and not low.endswith(":"):
                facts[-1] = f"{facts[-1]} {clean}"
            if len(facts) >= max_lines:
                break
        if not facts:
            compact = " ".join(text.split())
            if compact:
                facts.append(f"Document text summary: {compact[:500]}")
        return facts

    def document_context(self, document, limit=6000):
        parts = []
        for item in document.get("elements", []):
            descriptor = self.element_descriptor(item)
            text = item.get("text") or ""
            if descriptor:
                parts.append(f"Element context: {descriptor}.")
            for fact in self.extract_fact_lines(text):
                parts.append(fact)
            model_context = " ".join((item.get("model_context") or "").split())
            if model_context:
                parts.append(f"Model assessment: {model_context}")
            if len("\n".join(parts)) >= limit:
                break
        if document.get("model_summary"):
            parts.append(document["model_summary"])
        return "\n".join(parts)[:limit]

    def legal_focus_query(self, query):
        scope = self.guard.evaluate(query, has_document=False).get("scope", "tunisia")
        if scope == "us_federal":
            focus = "United States federal VAWA OVW 34 U.S.C. 12291 domestic violence dating violence stalking sexual assault technology abuse housing confidentiality documentation evidence"
        elif scope == "international":
            focus = "international human rights violence against women UN Declaration CEDAW General Recommendation 35 WHO CDC survivor-centered prevention protection prosecution reparations"
        else:
            focus = "Tunisia Organic Law No. 58 of 2017 violence against women Article 1 Article 2 Article 3 Article 13 Article 14 Article 30 Article 31 Article 39 protection support alert confidentiality evidence"
        return "\n".join(x for x in [query, focus] if x)

    def reference_scope(self, query, has_document=False):
        return self.guard.evaluate(query, has_document=has_document).get("scope", "tunisia")

    def chunk_scope(self, chunk):
        source = str(chunk.get("source_file", "")).lower()
        text = chunk.get("text", "").lower()
        if "us_" in source or "united states" in text or "vawa" in text or "34 u.s.c." in text:
            return "us_federal"
        if "international" in source or "cedaw" in text or "who public health" in text or "un declaration" in text:
            return "international"
        if "tunisia" in source or "law_58" in source or "organic law no. 58" in text or "loi organique" in text:
            return "tunisia"
        return "general"

    def prioritize_reference_scope(self, chunks, scope):
        if scope in {"document", "general", "blocked"}:
            scope = "tunisia"
        primary = [chunk for chunk in chunks if self.chunk_scope(chunk) == scope]
        other = [chunk for chunk in chunks if self.chunk_scope(chunk) != scope]
        return (primary + other)[:len(chunks)]

    def prioritize_tunisia(self, chunks):
        return self.prioritize_reference_scope(chunks, "tunisia")

    def ensure_scope_core(self, retrieved, chunks, scope, limit=8):
        selected = list(retrieved)
        seen = {chunk.get("chunk_id") for chunk in selected}
        if scope == "us_federal":
            core_terms = [("vawa",), ("domestic violence",), ("dating violence",), ("stalking",), ("technology",), ("housing", "confidentiality")]
        elif scope == "international":
            core_terms = [("gender-based violence",), ("cedaw",), ("who",), ("public health",), ("prevention", "protection")]
        else:
            core_terms = [("article 1",), ("article 3",), ("article 13",), ("article 14",), ("article 30",), ("article 39",)]
            scope = "tunisia"
        for terms in core_terms:
            if len(selected) >= limit:
                break
            for chunk in chunks:
                if chunk.get("chunk_id") in seen:
                    continue
                if self.chunk_scope(chunk) == scope and all(term in chunk.get("text", "").lower() for term in terms):
                    row = dict(chunk)
                    row["rank"] = len(selected) + 1
                    row.setdefault("score", 0.0)
                    selected.append(row)
                    seen.add(row.get("chunk_id"))
                    break
        for i, chunk in enumerate(selected, 1):
            chunk["rank"] = i
        return selected[:limit]

    def ensure_tunisia_core(self, retrieved, chunks, limit=8):
        return self.ensure_scope_core(retrieved, chunks, "tunisia", limit)

    def page_summary(self, elements):
        pages = {}
        for item in elements:
            page = item.get("page")
            if page in [None, ""]:
                continue
            pages.setdefault(str(page), {"page": page, "regions": []})
            if item.get("kind") == "layout_region":
                pages[str(page)]["regions"].append({
                    "id": item.get("id"),
                    "region_type": item.get("region_type"),
                    "bbox": item.get("bbox"),
                    "layout_confidence": item.get("layout_confidence"),
                    "text_source": item.get("text_source"),
                    "reading_order": item.get("reading_order"),
                    "has_text": bool(item.get("text")),
                    "has_predictions": bool(item.get("predictions")),
                })
        return [pages[key] for key in sorted(pages, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))]

    def process_document(self, path):
        records = read_document(path)
        elements = []
        for i, record in enumerate(records):
            image = record.pop("image", None)
            predictions = self.models.classify_image(image) if self.should_classify_image(record, image) else {}
            model_context = self.prediction_context(predictions)
            element = {
                "id": record.get("element_id") or f"e{i+1}",
                "source_file": record.get("source_file"),
                "page": record.get("page"),
                "kind": record.get("kind"),
                "region_type": record.get("region_type"),
                "bbox": record.get("bbox"),
                "layout_confidence": record.get("layout_confidence"),
                "text_source": record.get("text_source"),
                "reading_order": record.get("reading_order"),
                "page_width": record.get("page_width"),
                "page_height": record.get("page_height"),
                "text": record.get("text", ""),
                "model_context": model_context,
                "predictions": predictions,
            }
            elements.append(element)
        return {"file": Path(path).name, "file_hash": sha256_file(path), "model_summary": self.document_model_summary(elements), "pages": self.page_summary(elements), "elements": elements}

    def record_text(self, item):
        pieces = []
        descriptor = self.element_descriptor(item)
        if descriptor:
            pieces.append(f"Element context: {descriptor}.")
        if item.get("text"):
            pieces.append(item.get("text", ""))
        if item.get("model_context"):
            pieces.append(f"Model assessment: {item.get('model_context')}")
        return "\n".join(x for x in pieces if x)

    def analyze(self, path, question, case_id=None):
        case_id = case_id or uuid.uuid4().hex[:12]
        query = question or "Extract key facts, legal risks, deadlines, evidence gaps, and recommended next steps."
        guard = self.guard.evaluate(query, has_document=True)
        document = self.process_document(path)
        if not guard.get("in_scope", True):
            store = VectorStore(case_id)
            answer = self.guard.refusal(has_document=True)
            llm = {"enabled": self.feedback.enabled, "attempted": False, "success": False, "fallback_used": False, "response_mode": "blocked_by_guard"}
            metrics = evaluate_run(query, guard, store, [], [], answer, document, llm)
            result = {"case_id": case_id, "file": Path(path).name, "question": query, "answer": answer, "model_status": self.model_status(), "document": document, "retrieved": [], "metrics": metrics, "warnings": ["Educational analysis only. Verify jurisdiction-specific law with a licensed legal professional."]}
            write_json(store.path / "analysis.json", result)
            return result
        records = []
        if document.get("model_summary"):
            records.append({"text": document["model_summary"], "source_file": document["file"], "page": None, "element_id": "model_summary", "kind": "model_summary"})
        for item in document["elements"]:
            text = self.record_text(item)
            records.append({
                "text": text,
                "source_file": item.get("source_file"),
                "page": item.get("page"),
                "element_id": item.get("id"),
                "kind": item.get("kind", "document"),
                "region_type": item.get("region_type"),
                "bbox": item.get("bbox"),
                "text_source": item.get("text_source"),
                "layout_confidence": item.get("layout_confidence"),
                "reading_order": item.get("reading_order"),
            })
        records.extend(self.law_records())
        chunks = chunks_from_records(records, case_id)
        store = VectorStore(case_id)
        store.build(chunks)
        document_context = self.document_context(document)
        retrieval_query = self.legal_focus_query("\n".join(x for x in [query, document_context] if x))
        generation_question = "\n\n".join(x for x in [query, f"Extracted document facts:\n{document_context}" if document_context else ""] if x)
        scope = guard.get("scope", "tunisia")
        retrieved = self.ensure_scope_core(self.prioritize_reference_scope(store.retrieve(retrieval_query), scope), chunks, scope)
        answer = self.feedback.generate(generation_question, retrieved)
        metrics = evaluate_run(query, guard, store, chunks, retrieved, answer, document, self.feedback.last_metrics)
        result = {"case_id": case_id, "file": Path(path).name, "question": query, "answer": answer, "model_status": self.model_status(), "document": document, "retrieved": retrieved, "metrics": metrics, "warnings": ["Educational analysis only. Verify jurisdiction-specific law with a licensed legal professional."]}
        write_json(store.path / "analysis.json", result)
        return result

    def chat(self, question, session_id=None):
        case_id = session_id or f"law_{uuid.uuid4().hex[:12]}"
        query = (question or "").strip() or "Summarize the available legal references and practical next steps."
        guard = self.guard.evaluate(query, has_document=False)
        if not guard.get("in_scope", True):
            store = VectorStore(case_id)
            answer = self.guard.refusal(has_document=False)
            llm = {"enabled": self.feedback.enabled, "attempted": False, "success": False, "fallback_used": False, "response_mode": "blocked_by_guard"}
            metrics = evaluate_run(query, guard, store, [], [], answer, None, llm)
            result = {"case_id": case_id, "question": query, "answer": answer, "retrieved": [], "metrics": metrics, "warnings": ["Educational analysis only. Verify jurisdiction-specific law with a licensed legal professional."]}
            write_json(store.path / "chat.json", result)
            return result
        records = self.law_records()
        chunks = chunks_from_records(records, case_id)
        store = VectorStore(case_id)
        store.build(chunks)
        scope = guard.get("scope", "tunisia")
        retrieved = self.ensure_scope_core(self.prioritize_reference_scope(store.retrieve(self.legal_focus_query(query)), scope), chunks, scope) if chunks else []
        if retrieved:
            answer = self.feedback.generate(query, retrieved)
            llm = self.feedback.last_metrics
        else:
            answer = "No legal reference documents are available. Add files to storage/law_knowledge and try again."
            llm = {"enabled": self.feedback.enabled, "attempted": False, "success": False, "fallback_used": True, "response_mode": "no_retrieved_context"}
        metrics = evaluate_run(query, guard, store, chunks, retrieved, answer, None, llm)
        result = {"case_id": case_id, "question": query, "answer": answer, "retrieved": retrieved, "metrics": metrics, "warnings": ["Educational analysis only. Verify jurisdiction-specific law with a licensed legal professional."]}
        write_json(store.path / "chat.json", result)
        return result
