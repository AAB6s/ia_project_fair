from pathlib import Path
import uuid
import re
from document_io import read_document
from model_loader import ModelManager
from rag import VectorStore, chunks_from_records
from legal_feedback import LegalFeedbackEngine
from utils import sha256_file, write_json
from config import LAW_DIR

class DocumentLegalPipeline:
    def __init__(self):
        self.models = ModelManager()
        self.feedback = LegalFeedbackEngine()
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
    def document_model_summary(self, elements):
        counts = {}
        covered = set()
        for item in elements:
            preds = item.get("predictions", {})
            for key in ["content", "evidence", "quality", "tamper"]:
                label = preds.get(key, {}).get("label")
                if label:
                    covered.add(key)
                    counts[f"{key}:{label}"] = counts.get(f"{key}:{label}", 0) + 1
        if not counts:
            return ""
        models = ", ".join(name for name in ["content", "evidence", "quality", "tamper"] if name in covered)
        total = len(elements)
        return f"Visual review completed with {models} models on {total} document element{'s' if total != 1 else ''}."
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
            text = item.get("text") or ""
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
        focus = "Tunisia Organic Law No. 58 of 2017 violence against women Article 1 Article 2 Article 3 Article 13 Article 14 Article 30 Article 31 Article 39 protection support alert confidentiality evidence"
        return "\n".join(x for x in [query, focus] if x)
    def prioritize_tunisia(self, chunks):
        def is_tunisia(chunk):
            source = str(chunk.get("source_file", "")).lower()
            text = chunk.get("text", "").lower()
            return "tunisia" in source or "law_58" in source or "organic law no. 58" in text or "loi organique" in text
        tunisia = [chunk for chunk in chunks if is_tunisia(chunk)]
        other = [chunk for chunk in chunks if not is_tunisia(chunk)]
        return (tunisia + other)[:len(chunks)]
    def ensure_tunisia_core(self, retrieved, chunks, limit=8):
        selected = list(retrieved)
        seen = {chunk.get("chunk_id") for chunk in selected}
        core_terms = [("article 1",), ("article 3",), ("article 13",), ("article 14",), ("article 30",), ("article 39",)]
        for terms in core_terms:
            if len(selected) >= limit:
                break
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                source = str(chunk.get("source_file", "")).lower()
                if chunk.get("chunk_id") in seen:
                    continue
                if ("tunisia" in source or "law_58" in source or "organic law no. 58" in text) and all(term in text for term in terms):
                    row = dict(chunk)
                    row["rank"] = len(selected) + 1
                    row.setdefault("score", 0.0)
                    selected.append(row)
                    seen.add(row.get("chunk_id"))
                    break
        for i, chunk in enumerate(selected, 1):
            chunk["rank"] = i
        return selected[:limit]
    def process_document(self, path):
        records = read_document(path)
        elements = []
        for i, record in enumerate(records):
            image = record.pop("image", None)
            predictions = self.models.classify_image(image) if image is not None else {}
            model_context = self.prediction_context(predictions)
            element = {"id": f"e{i+1}", "source_file": record.get("source_file"), "page": record.get("page"), "kind": record.get("kind"), "text": record.get("text", ""), "model_context": model_context, "predictions": predictions}
            elements.append(element)
        return {"file": Path(path).name, "file_hash": sha256_file(path), "model_summary": self.document_model_summary(elements), "elements": elements}
    def analyze(self, path, question, case_id=None):
        case_id = case_id or uuid.uuid4().hex[:12]
        document = self.process_document(path)
        records = []
        if document.get("model_summary"):
            records.append({"text": document["model_summary"], "source_file": document["file"], "page": None, "element_id": "model_summary", "kind": "model_summary"})
        for item in document["elements"]:
            text = "\n".join(x for x in [item.get("text", ""), item.get("model_context", "")] if x)
            records.append({"text": text, "source_file": item.get("source_file"), "page": item.get("page"), "element_id": item.get("id"), "kind": item.get("kind", "document")})
        records.extend(self.law_records())
        chunks = chunks_from_records(records, case_id)
        store = VectorStore(case_id)
        store.build(chunks)
        query = question or "Extract key facts, legal risks, deadlines, evidence gaps, and recommended next steps."
        document_context = self.document_context(document)
        retrieval_query = self.legal_focus_query("\n".join(x for x in [query, document_context] if x))
        generation_question = "\n\n".join(x for x in [query, f"Extracted document facts:\n{document_context}" if document_context else ""] if x)
        retrieved = self.ensure_tunisia_core(self.prioritize_tunisia(store.retrieve(retrieval_query)), chunks)
        answer = self.feedback.generate(generation_question, retrieved)
        result = {"case_id": case_id, "file": Path(path).name, "question": query, "answer": answer, "model_status": self.model_status(), "document": document, "retrieved": retrieved, "warnings": ["Educational analysis only. Verify jurisdiction-specific law with a licensed legal professional."]}
        write_json(store.path / "analysis.json", result)
        return result
    def chat(self, question, session_id=None):
        case_id = session_id or f"law_{uuid.uuid4().hex[:12]}"
        query = (question or "").strip() or "Summarize the available legal references and practical next steps."
        records = self.law_records()
        chunks = chunks_from_records(records, case_id)
        store = VectorStore(case_id)
        store.build(chunks)
        retrieved = self.ensure_tunisia_core(self.prioritize_tunisia(store.retrieve(self.legal_focus_query(query))), chunks) if chunks else []
        answer = self.feedback.generate(query, retrieved) if retrieved else "No legal reference documents are available. Add files to storage/law_knowledge and try again."
        result = {"case_id": case_id, "question": query, "answer": answer, "retrieved": retrieved, "warnings": ["Educational analysis only. Verify jurisdiction-specific law with a licensed legal professional."]}
        write_json(store.path / "chat.json", result)
        return result
