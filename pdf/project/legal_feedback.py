import re
import requests
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT, MAX_CONTEXT_CHARS

SYSTEM_PROMPT = """You are a legal document analysis assistant for Tunisian violence-against-women cases.
Use only the cited context and the extracted document facts provided by the user message.
Prioritize Tunisia Organic Law No. 58 of 2017 and Tunisian references when present.
Use international or United States sources only as non-binding background and only if the user asks for comparison.
For direct disclosures of rape, sexual assault, domestic violence, threats, or immediate danger, use a calm survivor-centered tone before legal analysis.
Do not sound detached. Acknowledge seriousness briefly, prioritize safety, medical support, evidence preservation, and professional help.
Never invent article numbers, statutes, parties, dates, deadlines, courts, procedures, claims, or citations.
Never write placeholders such as [insert article number], TBD, unknown article, or similar draft text.
If an article number is not in the cited context, say that the article number is not available in the retrieved sources.
Do not cite the Tunisian Civil Code, Penal Code, or any other code unless the cited context names that code.
Never say there are no urgent safety risks unless the extracted document facts explicitly support that. If facts are missing, say no urgent-risk facts were extracted.
Every legal statement must either cite a retrieved chunk like [chunk_id] or be clearly framed as a fact still needing professional verification.
Never advise confronting the reported abuser, warning the reported abuser, or collecting evidence in a way that increases danger.
This is educational triage, not legal advice or a substitute for a qualified Tunisian lawyer."""

class LegalFeedbackEngine:
    def __init__(self):
        self.enabled = bool(LLM_API_KEY)
        self.last_error = ""
    def citation(self, chunk):
        parts = [chunk.get("chunk_id", "")]
        if chunk.get("source_file"):
            parts.append(str(chunk["source_file"]))
        if chunk.get("page") not in [None, ""]:
            parts.append(f"p.{chunk['page']}")
        if chunk.get("element_id") not in [None, ""]:
            parts.append(str(chunk["element_id"]))
        return " | ".join(parts)
    def context(self, chunks):
        used = 0
        parts = []
        for chunk in chunks:
            block = f"[{chunk['chunk_id']}] {self.citation(chunk)}\n{chunk['text'][:1000]}"
            if used + len(block) > MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)
    def fallback(self, question, chunks):
        return self.grounded_template(question, chunks)
    def source_has(self, chunks, term):
        return term.lower() in " ".join(chunk.get("text", "") for chunk in chunks).lower()
    def has_allowed_citation(self, line, chunks):
        allowed = {chunk.get("chunk_id", "") for chunk in chunks}
        return any(cid and f"[{cid}]" in line for cid in allowed)
    def legal_claim_needs_citation(self, line):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or re.fullmatch(r"\*{0,2}\d+\.\s*[^*]+\*{0,2}", stripped):
            return False
        low = stripped.lower()
        triggers = ["organic law", "law no. 58", "article ", "protection request", "family judge", "court", "competent authorities", "prosecution", "punishment", "legal protection", "judicial aid"]
        return any(trigger in low for trigger in triggers)
    def first_chunk(self, chunks, *terms):
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            if all(term.lower() in text for term in terms):
                return chunk.get("chunk_id", "")
        return ""
    def extracted_lines(self, question):
        facts = []
        models = []
        if "Extracted document facts:" not in question:
            return facts, models
        extracted = question.split("Extracted document facts:", 1)[1].strip()
        for raw in extracted.splitlines():
            line = raw.strip(" -")
            if not line:
                continue
            if line.startswith("Model assessment:") or line.startswith("Visual review completed"):
                models.append(line)
            else:
                facts.append(line)
        return facts, models
    def is_sensitive_disclosure(self, question):
        text = " ".join(str(question or "").lower().split())
        harm_terms = [
            "rape", "raped", "sexual assault", "sexually assaulted", "forced sex", "forced me",
            "my husband raped", "husband raped", "spouse raped", "partner raped",
            "domestic violence", "hit me", "beat me", "strangled", "choked", "threatened to kill",
            "weapon", "kidnap", "locked me", "confined", "stalking", "blackmail", "private photos",
            "viol", "violee", "agression sexuelle", "mon mari", "mari"
        ]
        first_person = [" i ", " i'm", " im ", " me ", " my ", " mine ", " j'", " je ", " moi ", " mon ", " ma "]
        padded = f" {text} "
        return any(term in text for term in harm_terms) and any(token in padded for token in first_person)
    def direct_report_facts(self, question):
        facts, _ = self.extracted_lines(question)
        if facts:
            return facts[:10]
        text = " ".join(str(question or "").split())
        low = text.lower()
        out = ["Jurisdiction: Tunisia is assumed for this demo; the exact city, competent authority, and current procedure still need verification."]
        if any(x in low for x in ["husband", "spouse", "mari", "married"]):
            out.append("Relationship context: the reported person appears to be a husband or spouse.")
        elif any(x in low for x in ["partner", "boyfriend", "former", "ex "]):
            out.append("Relationship context: the reported person appears to be an intimate partner or former partner.")
        if any(x in low for x in ["rape", "raped", "sexual assault", "forced sex", "viol", "agression sexuelle"]):
            out.append("Reported fact: the user reports sexual violence or rape.")
        if any(x in low for x in ["threat", "kill", "weapon", "blackmail", "private photos", "stalking", "locked", "kidnap"]):
            out.append("Reported fact: the user mentions a possible escalation or safety-risk signal.")
        if len(out) == 1:
            out.append(f"Reported statement: {text[:240]}")
        return out[:10]
    def model_review_lines(self, model_lines):
        joined = " ".join(model_lines)
        if not joined:
            return []
        review = []
        content = re.search(r"Visual content classification:\s*([a-z_]+)", joined)
        evidence = re.search(r"Evidence weight classification:\s*([a-z_]+)", joined)
        quality = re.search(r"Image quality classification:\s*([a-z_]+)", joined)
        tamper = re.search(r"Tamper classification:\s*([a-z_]+).*?risk\s*([a-z_]+)", joined)
        if "Visual review completed" in joined:
            review.append("- The local document models all ran on the uploaded page: content, evidence weight, image quality, and authenticity/tamper screening.")
        if content:
            review.append(f"- Content model: classified the page as {content.group(1).replace('_', ' ')}.")
        if evidence:
            review.append(f"- Evidence model: treated the page as {evidence.group(1).replace('_', ' ')} for the legal review.")
        if quality:
            q_label = quality.group(1).replace("_", " ")
            if q_label == "degraded":
                review.append("- Quality model: flagged the rendered page for readability verification, so OCR and visual conclusions should be checked against the original file.")
            else:
                review.append(f"- Quality model: classified the visual quality as {q_label}.")
        if tamper:
            label = tamper.group(1).replace("_", " ")
            risk = tamper.group(2).replace("_", " ")
            if label == "tampered" or risk == "possibly tampered":
                review.append("- Authenticity screen: flagged the page for manual verification; this is a model signal, not proof of alteration.")
            else:
                review.append("- Authenticity screen: did not raise a tamper warning for this page.")
        return review[:6]
    def case_signals(self, lines):
        text = " ".join(lines).lower()
        return {
            "threats": any(x in text for x in ["threat", "intimidation", "pressure", "publish private", "private photos", "blackmail"]),
            "sexual": any(x in text for x in ["rape", "raped", "sexual assault", "forced sex", "sexual violence", "viol", "agression sexuelle"]),
            "physical": any(x in text for x in ["hit", "beat", "injury", "strangled", "choked", "weapon", "physical violence"]),
            "digital": any(x in text for x in ["screenshot", "message", "call log", "phone", "account", "private photos"]),
            "workplace": "workplace" in text or "workplace entrance" in text,
            "no_complaint": "not yet filed" in text or "no police complaint" in text,
            "home": "home" in text or "spouse" in text,
            "children": "child" in text or "children" in text,
        }
    def grounded_template(self, question, chunks):
        c1 = self.first_chunk(chunks, "article 1")
        c2 = self.first_chunk(chunks, "article 2")
        c3 = self.first_chunk(chunks, "article 3")
        c13 = self.first_chunk(chunks, "article 13")
        c14 = self.first_chunk(chunks, "article 14")
        c30 = self.first_chunk(chunks, "article 30")
        c39 = self.first_chunk(chunks, "article 39")
        citations = []
        for cid in [c1, c2, c3, c13, c14, c30, c39]:
            if cid and cid not in citations:
                citations.append(cid)
        q = " ".join(question.split())
        extracted_facts, extracted_models = self.extracted_lines(question)
        facts = [f"- {line[:280]}" for line in extracted_facts[:14]]
        if not facts:
            facts = ["- The submitted facts must be verified from the uploaded document, OCR text, dates, parties, relationship, location, and evidence source.", f"- User question: {q[:300]}"]
        model_review = self.model_review_lines(extracted_models)
        signals = self.case_signals(extracted_facts)
        framework = [
            f"- Tunisia Organic Law No. 58 of 2017 uses a comprehensive approach to eliminating violence against women, including prevention, prosecution, punishment, protection, and victim support [{c1}].",
            f"- The law covers forms of discrimination and violence against women regardless of the author or domain when based on sex discrimination [{c2}].",
            f"- The law defines violence against women broadly, including physical, moral/psychological, sexual, economic, and threat/coercion-related harm in public or private life [{c3}].",
            f"- Victim-support rights include legal protection, information, legal advice, judicial aid, health and psychological follow-up, social support, and emergency accommodation within available means [{c13}].",
            f"- Good-faith alerts to competent authorities are protected, and confidentiality of the reporter identity is required except where legal procedures require disclosure [{c14}].",
            f"- Protection requests may involve the family judge and must be verified against current Tunisian procedure and the specific facts of the case [{c30}].",
            f"- Police, health, social, child-protection, and other competent services should respond promptly to assistance and protection requests, especially where physical, sexual, or psychological safety is threatened [{c39}].",
        ]
        framework = [x for x in framework if "[]" not in x]
        risks = []
        if signals["threats"]:
            risks.append("- The document reports threats, intimidation, or pressure, so escalation, retaliation, coercion, and immediate safety should be checked before any legal strategy.")
        if signals["sexual"]:
            risks.append("- The report involves sexual violence, so immediate safety, urgent health support, confidentiality, and evidence preservation should be prioritized before any procedural step.")
        if signals["physical"]:
            risks.append("- Physical harm or weapons indicators require urgent safety assessment and professional support.")
        if signals["digital"]:
            risks.append("- The document relies on digital evidence, so account attribution, metadata, timestamps, and original-device preservation are important.")
        if signals["workplace"]:
            risks.append("- Workplace contact can create safety, privacy, and witness-preservation issues that should be handled carefully.")
        if signals["children"]:
            risks.append("- Any child-safety issue should be treated as urgent and verified with the competent services.")
        risks.extend(["- Immediate danger, escalating threats, weapons, confinement, stalking escalation, child danger, suicidal or homicidal threats, or protection-order violation require urgent professional help.", "- Low-quality, incomplete, unsigned, or potentially tampered evidence should be treated as weaker until verified."])
        preserve = []
        if signals["digital"]:
            preserve.append("- Preserve the original phone/account data, screenshots with visible date/time/account identifiers, message exports, call logs, and any available metadata.")
        if signals["sexual"] or signals["physical"]:
            preserve.append("- Preserve medical or forensic records, photos of injuries if safely possible, clothing or physical evidence when relevant, and any communication before or after the incident.")
        preserve.extend(["- Preserve witness identities, written statements, photos, medical documents, police complaints, identity documents, and any court or protection documents.", "- Keep originals unchanged and use copies for sharing when possible."])
        gaps = []
        if signals["no_complaint"]:
            gaps.append("- The document says no police complaint has been filed yet, so reporting options, competent authority, and urgent protection route should be verified.")
        gaps.extend(["- Verify exact jurisdiction in Tunisia, relationship status, incident dates, location, ages, injuries, witnesses, prior complaints, existing orders, and whether police/medical/court records exist.", "- Verify current Tunisian procedural deadlines, competent authority, available protection measures, and local court practice with a qualified professional."])
        steps = ["- If there is immediate danger, contact emergency or competent local services before legal strategy.", "- Build a dated evidence table that links each incident to its source: screenshot, call log, witness, medical record, workplace note, or complaint.", "- Ask a qualified Tunisian lawyer or authorized support service to assess filing options, protection measures, confidentiality, and next procedural steps.", "- Do not rely on this output as final legal advice."]
        cited = [f"- [{cid}]" for cid in citations[:8]]
        sections = ["Key Facts From The Document", *facts, ""]
        if model_review:
            sections.extend(["Document Model Review", *model_review, ""])
        sections.extend(["Tunisian Legal Framework To Verify", *framework, "", "Urgent Safety Risks", *risks, "", "Evidence To Preserve", *preserve, "", "Evidence Gaps", *gaps, "", "Recommended Next Steps", *steps, "", "Citations Used", *(cited or ["- No retrieved legal citation was available."])])
        return "\n".join(sections)
    def survivor_template(self, question, chunks):
        c1 = self.first_chunk(chunks, "article 1")
        c3 = self.first_chunk(chunks, "article 3")
        c13 = self.first_chunk(chunks, "article 13")
        c14 = self.first_chunk(chunks, "article 14")
        c30 = self.first_chunk(chunks, "article 30")
        c39 = self.first_chunk(chunks, "article 39")
        citations = []
        for cid in [c1, c3, c13, c14, c30, c39]:
            if cid and cid not in citations:
                citations.append(cid)
        facts = [f"- {line[:280]}" for line in self.direct_report_facts(question)]
        framework = [
            f"- Tunisia Organic Law No. 58 of 2017 uses a comprehensive approach against violence against women, including prevention, prosecution, punishment, protection, and victim support [{c1}].",
            f"- Article 3 defines violence against women broadly, including physical, psychological, sexual, economic, and threat/coercion-related harm in public or private life [{c3}].",
            f"- Article 13 lists victim-support rights, including legal protection, information, legal advice, judicial aid, health and psychological follow-up, social support, and immediate accommodation within available means [{c13}].",
            f"- Article 14 addresses alerts to competent authorities, protects good-faith alerts, and requires confidentiality of the alerting person's identity except where legal procedures require disclosure [{c14}].",
            f"- Article 30 says a protection request may be examined by the family judge following a written petition by the victim or other listed routes; current procedure and fit to the facts must be verified [{c30}].",
            f"- Article 39 requires responsible services to respond without delay to assistance and protection requests, especially where physical, sexual, or psychological safety is threatened [{c39}].",
        ]
        framework = [x for x in framework if "[]" not in x]
        sections = [
            "Urgent Safety Risks",
            "- I'm sorry this happened. This should be treated as serious. If the person is in immediate danger, still near the reported abuser, injured, threatened, or unable to leave safely, contacting emergency or competent local services comes before legal strategy.",
            "- The person should not confront the reported abuser or warn him before getting safety advice.",
            "- If sexual violence was recent, urgent medical support should be considered as soon as safely possible for health, injury care, and potential evidence preservation.",
            "",
            "Key Facts From The Report",
            *facts,
            "",
            "Tunisian Legal Framework To Verify",
            *framework,
            "",
            "Evidence To Preserve",
            "- Preserve the original phone, messages, call logs, photos, clothing or physical evidence where relevant, medical records, witness details, location details, and any police or protection documents.",
            "- Keep originals unchanged. Use copies for sharing and record dates, times, locations, and who created or received each item.",
            "- If there are injuries or health concerns, medical documentation can be important and should be handled privately and safely.",
            "",
            "Evidence Gaps",
            "- Verify the exact date, location, relationship status, whether there were threats or weapons, whether children are at risk, whether there are witnesses, and whether any complaint or medical visit already exists.",
            "- Verify the current Tunisian procedure, competent authority, deadlines, and protection route with a qualified Tunisian lawyer or authorized support service.",
            "",
            "Recommended Next Steps",
            "- First, make a safety plan with a trusted person or competent service if there is any risk of retaliation or further contact.",
            "- Seek medical and psychological support if safe and available, especially after sexual violence or physical harm.",
            "- Organize evidence in a dated table: incident, source, person involved, file name, and reliability.",
            "- Ask a qualified Tunisian lawyer or authorized support service about reporting options, protection measures, confidentiality, and next procedural steps.",
            "- This is educational triage, not final legal advice.",
            "",
            "Citations Used",
            *(f"- [{cid}]" for cid in citations[:8])
        ]
        if not citations:
            sections.append("- No retrieved legal citation was available.")
        return "\n".join(sections)
    def clean_output(self, text, chunks, question):
        bad = [r"\[insert[^\]]*\]", r"insert article number", r"article number\]", r"\bTBD\b", r"\bTODO\b", r"citation needed", r"\[source\]"]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in bad):
            return self.grounded_template(question, chunks)
        if not any(f"[{chunk.get('chunk_id', '')}]" in text for chunk in chunks if chunk.get("chunk_id")):
            return self.grounded_template(question, chunks)
        body_before_citations = re.split(r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{0,2})?(?:\d+\.\s*)?citations used(?:\*{0,2})?\s*:?\s*$", text, maxsplit=1)[0]
        if any(self.legal_claim_needs_citation(line) and not self.has_allowed_citation(line, chunks) for line in body_before_citations.splitlines()):
            return self.grounded_template(question, chunks)
        text = "\n".join(line for line in text.splitlines() if "civil code" not in line.lower())
        cleaned = []
        for line in text.splitlines():
            low = line.lower()
            mentions_penal = "penal code" in low or "code pénal" in low or "code pÃ©nal" in low
            if mentions_penal and (not self.source_has(chunks, "code pénal") and not self.source_has(chunks, "code pÃ©nal") and not self.source_has(chunks, "penal code")):
                continue
            if mentions_penal and not self.has_allowed_citation(line, chunks):
                continue
            cleaned.append(line)
        text = "\n".join(cleaned)
        text = re.sub(r"\(based on the provided sources\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"the document does not mention any urgent safety risks(?: that require immediate attention)?\.?", "No urgent-risk facts were extracted from the available document text.", text, flags=re.IGNORECASE)
        text = re.sub(r"jurisdiction is likely (the )?(United States federal or )?Tunisia", "answer assumes Tunisia because the request asks for Tunisian-law focus", text, flags=re.IGNORECASE)
        allowed = {chunk.get("chunk_id", "") for chunk in chunks}
        if "citations used" in text.lower():
            head = re.split(r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{0,2})?(?:\d+\.\s*)?citations used(?:\*{0,2})?\s*:?\s*$", text, maxsplit=1)[0].rstrip()
            used = []
            for cid in allowed:
                if cid and f"[{cid}]" in text:
                    used.append(cid)
            if not used:
                used = [chunk.get("chunk_id", "") for chunk in chunks[:5] if chunk.get("chunk_id")]
            cites = "\n".join(f"* [{cid}]" for cid in used[:8])
            text = f"{head}\n\n**Citations Used**\n\n{cites}".strip()
        return text.strip()
    def ollama_url(self):
        base = LLM_BASE_URL.rstrip("/")
        return base[:-3] if base.endswith("/v1") else base
    def prompt(self, question, ctx):
        return f"""Question and extracted document facts:
{question}

Cited context:
{ctx}

Write a concise Tunisia-focused violence-against-women legal triage report.
Use plain section headings exactly as listed, without Markdown symbols, bold markers, or numbering.
Required sections:
1. Key Facts From The Document
2. Tunisian Legal Framework To Verify
3. Urgent Safety Risks
4. Evidence To Preserve
5. Evidence Gaps
6. Recommended Next Steps
7. Citations Used

Rules:
- Use only cited context and extracted document facts.
- If the user reports rape, sexual assault, domestic violence, threats, or immediate danger, start with safety and support before legal detail.
- Use direct, humane wording. Avoid detached phrases such as "the document does not mention" when the user personally reports harm.
- Use exact article numbers only when they appear in the cited context.
- Never write placeholders.
- Do not say there are no urgent safety risks unless the extracted document facts explicitly support that.
- Do not advise confronting the reported abuser.
- Do not cite Civil Code, Penal Code, protection orders, deadlines, courts, or procedures unless cited context supports them.
- Cite legal points with [chunk_id].
- If procedure or deadlines are not in the retrieved sources, say they must be verified with current Tunisian procedure and a qualified professional."""
    def generate_ollama(self, prompt):
        payload = {"model": LLM_MODEL, "stream": False, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "options": {"temperature": 0.15, "top_p": 0.9, "num_ctx": 4096, "num_predict": 500}}
        res = requests.post(f"{self.ollama_url()}/api/chat", json=payload, timeout=LLM_TIMEOUT)
        res.raise_for_status()
        return res.json()["message"]["content"].strip()
    def generate_openai_compatible(self, prompt):
        payload = {"model": LLM_MODEL, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.2}
        res = requests.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}, json=payload, timeout=LLM_TIMEOUT)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    def generate(self, question, chunks):
        ctx = self.context(chunks)
        if self.is_sensitive_disclosure(question):
            return self.survivor_template(question, chunks)
        if not self.enabled:
            self.last_error = "LLM disabled"
            return self.grounded_template(question, chunks)
        prompt = self.prompt(question, ctx)
        try:
            self.last_error = ""
            if "11434" in LLM_BASE_URL or LLM_API_KEY == "ollama":
                return self.clean_output(self.generate_ollama(prompt), chunks, question)
            return self.clean_output(self.generate_openai_compatible(prompt), chunks, question)
        except Exception as e:
            self.last_error = str(e)
            return self.grounded_template(question, chunks)
