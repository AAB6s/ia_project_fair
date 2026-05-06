from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from schemas import AnalysisResponse, ChatRequest, ChatResponse, HealthResponse
from pipeline import DocumentLegalPipeline
from config import UPLOAD_DIR
from utils import clean_filename

app = FastAPI(title="Legal Document RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
pipeline = DocumentLegalPipeline()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Legal Workbench</title>
<style>
:root{--ink:#17201d;--muted:#66736d;--line:#d8ded5;--paper:#f2eadc;--card:#fffdf7;--field:#fbfaf5;--accent:#255f4c;--accent2:#b26331;--dark:#111917;--bad:#a43a36;--soft:#e8efe8}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#e8efe8 0,#f7f1e6 42%,#eadfce 100%);color:var(--ink);font-family:"Aptos",Verdana,sans-serif}body:before{content:"";position:fixed;inset:0;background:radial-gradient(circle at 12% 12%,rgba(37,95,76,.18),transparent 28%),radial-gradient(circle at 90% 10%,rgba(178,99,49,.16),transparent 26%),linear-gradient(90deg,rgba(23,32,29,.04) 1px,transparent 1px),linear-gradient(rgba(23,32,29,.035) 1px,transparent 1px);background-size:auto,auto,46px 46px,46px 46px;pointer-events:none}main{position:relative;max-width:1180px;margin:0 auto;padding:34px 18px 56px}.top{display:grid;grid-template-columns:1fr auto;gap:18px;align-items:end;margin-bottom:18px}.brand{max-width:720px}.kicker{font:800 11px/1.2 Verdana,sans-serif;letter-spacing:.18em;text-transform:uppercase;color:var(--accent)}h1{font-family:Georgia,"Times New Roman",serif;font-size:clamp(38px,6.4vw,74px);line-height:.9;margin:10px 0 14px;letter-spacing:-.055em}p{margin:0;color:var(--muted);line-height:1.62}.actions{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}.actions a,.ghost{border:1px solid var(--line);background:rgba(255,253,247,.68);color:var(--accent);border-radius:999px;padding:11px 14px;text-decoration:none;font:800 12px/1 Verdana,sans-serif}.shell{display:grid;grid-template-columns:1.45fr .75fr;gap:18px}.card{background:rgba(255,253,247,.9);border:1px solid var(--line);border-radius:28px;box-shadow:0 22px 70px rgba(30,38,33,.13);padding:22px}.tabs{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}.tab{border:1px solid var(--line);background:var(--field);color:var(--muted);border-radius:18px;padding:14px 12px;font:900 13px/1 Verdana,sans-serif;cursor:pointer}.tab.active{background:var(--dark);color:#fff;border-color:var(--dark)}.pane{display:none}.pane.active{display:grid;gap:14px}.form{display:grid;gap:12px}label{font:900 11px/1.2 Verdana,sans-serif;color:#34423d;text-transform:uppercase;letter-spacing:.11em}input[type=file],textarea{width:100%;border:1px solid var(--line);border-radius:18px;background:#fff;padding:14px;font:14px/1.5 Verdana,sans-serif;color:var(--ink);outline:none}textarea{min-height:126px;resize:vertical}textarea:focus,input[type=file]:focus{border-color:var(--accent);box-shadow:0 0 0 4px rgba(37,95,76,.11)}button{border:0;border-radius:18px;background:var(--accent);color:white;padding:15px 18px;font:900 12px/1 Verdana,sans-serif;letter-spacing:.06em;cursor:pointer}button.secondary{background:#fff;color:var(--accent);border:1px solid var(--accent)}button:disabled{opacity:.55;cursor:not-allowed}.row{display:flex;gap:10px;flex-wrap:wrap}.hint{font:12px/1.55 Verdana,sans-serif;color:var(--muted)}.result{white-space:pre-wrap;background:var(--dark);color:#f8f0df;border-radius:22px;padding:18px;min-height:360px;max-height:680px;overflow:auto;font:13px/1.55 Consolas,monospace}.side{display:grid;gap:18px}.models{display:grid;gap:8px}.model{display:grid;grid-template-columns:1fr auto;gap:12px;border:1px solid var(--line);border-radius:16px;background:var(--field);padding:12px;font:12px/1.35 Verdana,sans-serif}.ok{color:var(--accent);font-weight:900}.miss{color:var(--bad);font-weight:900}.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{border:1px solid var(--line);border-radius:999px;padding:9px 11px;background:var(--field);font:800 11px/1 Verdana,sans-serif;color:#34423d}.smalltitle{font:900 12px/1 Verdana,sans-serif;text-transform:uppercase;letter-spacing:.14em;color:var(--accent);margin-bottom:10px}.miniout{white-space:pre-wrap;border-radius:18px;background:#fff;border:1px solid var(--line);padding:13px;max-height:190px;overflow:auto;font:12px/1.5 Consolas,monospace;color:#31413c}@media(max-width:900px){.top,.shell{grid-template-columns:1fr}.actions{justify-content:flex-start}}
.result{white-space:normal;background:linear-gradient(180deg,#fffdf8,#f8f2e7);border:1px solid var(--line);border-radius:24px;padding:18px;min-height:360px;max-height:720px;overflow:auto;color:var(--ink);font:14px/1.62 Verdana,sans-serif}.result.loading{display:flex;align-items:center;justify-content:center;color:var(--muted);font:900 12px/1 Verdana,sans-serif;text-transform:uppercase;letter-spacing:.14em}.report{display:grid;gap:12px}.report-section{border:1px solid #dce2d7;background:rgba(255,255,250,.84);border-radius:20px;padding:16px 17px;box-shadow:0 10px 30px rgba(30,38,33,.06)}.report-section h3{margin:0 0 10px;font-family:Georgia,"Times New Roman",serif;font-size:22px;line-height:1.05;letter-spacing:-.025em;color:#17332b}.report-section ul{margin:0;padding-left:20px}.report-section li{margin:7px 0;color:#2d3935}.report-section p{margin:7px 0;color:#2d3935}.report-note{border-left:4px solid var(--accent2);padding:10px 12px;background:#fff7ea;border-radius:14px;color:#4d3b2d}.citation{display:inline-flex;align-items:center;max-width:100%;margin:0 3px;padding:2px 7px;border-radius:999px;background:#e8efe8;border:1px solid #cbd8ce;color:#255f4c;font:800 11px/1.5 Verdana,sans-serif;vertical-align:baseline}
</style>
</head>
<body>
<main>
<section class="top">
<div class="brand">
<div class="kicker">Legal Workbench</div>
<h1>Evidence review and legal reference chat</h1>
<p>Review files with local document models, OCR, retrieval, and Ollama. Or ask the legal reference library directly without uploading a document.</p>
</div>
<div class="actions">
<a href="/docs">API docs</a>
<a href="/models">Model status</a>
<a href="/health">Health</a>
</div>
</section>
<section class="shell">
<div class="card">
<div class="tabs">
<button class="tab active" type="button" data-tab="doc">Document review</button>
<button class="tab" type="button" data-tab="chat">Reference chat</button>
</div>
<div class="pane active" id="doc">
<form class="form" id="docForm" method="post" enctype="multipart/form-data">
<label>Document</label>
<input name="file" type="file" required>
<label>Question</label>
<textarea name="question">Extract key facts, legal risks, deadlines, evidence gaps, and recommended next steps.</textarea>
<div class="row">
<button id="docSubmit" type="button">Analyze document</button>
<button class="secondary" id="reloadA" type="button">Reload models</button>
</div>
<div class="hint">Supported: PDF, image, DOCX, TXT, JSON, XLSX.</div>
</form>
</div>
<div class="pane" id="chat">
<form class="form" id="chatForm" method="post">
<label>Legal question</label>
<textarea name="question">Under Tunisian law, especially Organic Law No. 58 of 2017, what evidence should be collected in a violence against women case, and what urgent risks or legal points should be verified?</textarea>
<div class="row">
<button id="chatSubmit" type="button">Ask references</button>
<button class="secondary" type="button" id="sample">Use sample</button>
</div>
<div class="hint">Uses only the files in storage/law_knowledge plus Ollama. No upload required.</div>
</form>
</div>
</div>
<aside class="side">
<div class="card">
<div class="smalltitle">Run status</div>
<div class="models" id="models">Loading...</div>
<div class="row" style="margin-top:12px"><button class="secondary" id="reloadB" type="button">Reload models</button></div>
</div>
<div class="card">
<div class="smalltitle">Coverage</div>
<div class="chips"><span class="chip">content</span><span class="chip">evidence</span><span class="chip">quality</span><span class="chip">tamper</span><span class="chip">OCR</span><span class="chip">law library</span><span class="chip">Ollama</span></div>
</div>
<div class="card">
<div class="smalltitle">Retrieved sources</div>
<div class="miniout" id="sources">No sources yet.</div>
</div>
</aside>
</section>
<section class="card" style="margin-top:18px">
<div class="smalltitle">Result</div>
<div class="result" id="out">Choose a mode and run a request.</div>
</section>
</main>
<script>
const $=s=>document.querySelector(s);
const sources=j=>(j.retrieved||[]).map(x=>`${x.rank}. ${x.source_file||'reference'} ${x.page?`p.${x.page}`:''} ${x.kind||''}\\n${(x.text||'').slice(0,220)}`).join('\\n\\n')||'No sources returned.';
const esc=s=>String(s||'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const clean=s=>esc(String(s||'').replace(/\\*\\*/g,'').replace(/^#+\\s*/,'').replace(/^\\d+\\.\\s*/,'').trim()).replace(/\\[([A-Za-z0-9_]+_C\\d{5})\\]/g,'<span class="citation">[$1]</span>');
function heading(line){let x=line.trim();let m=x.match(/^#{1,6}\\s+(.+)$/)||x.match(/^\\*\\*(?:\\d+\\.\\s*)?([^*]+?)\\*\\*:?$/)||x.match(/^(?:\\d+\\.\\s*)?(Key Facts From The Document|Key Facts From The Report|Document Model Review|Tunisian Legal Framework To Verify|Urgent Safety Risks|Evidence To Preserve|Evidence Gaps|Recommended Next Steps|Citations Used)\\s*:?\\s*$/i);return m?m[1].replace(/\\*\\*/g,'').trim():''}
function renderAnswer(text){
 const lines=String(text||'').replace(/\\r/g,'').split('\\n');
 let html='<div class="report">', title='Summary', items=[], paras=[];
 const flush=()=>{if(!items.length&&!paras.length)return;html+=`<section class="report-section"><h3>${clean(title)}</h3>`;if(items.length)html+=`<ul>${items.map(x=>`<li>${clean(x)}</li>`).join('')}</ul>`;if(paras.length)html+=paras.map(x=>`<p>${clean(x)}</p>`).join('');html+='</section>';items=[];paras=[]};
 for(const raw of lines){let line=raw.trim();if(!line)continue;let h=heading(line);if(h){flush();title=h;continue}line=line.replace(/^[-*+]\\s*/,'').replace(/^\\+\\s*/,'').trim();if(/^note:/i.test(line)){flush();html+=`<div class="report-note">${clean(line)}</div>`;continue}if(raw.trim().match(/^[-*+]\\s+/)||raw.includes('\\t+')||raw.trim().startsWith('+ '))items.push(line);else paras.push(line)}
 flush();return html+'</div>'
}
function setLoading(text){const out=$('#out');out.classList.add('loading');out.textContent=text}
function setReport(text){const out=$('#out');out.classList.remove('loading');out.innerHTML=renderAnswer(text)}
function setPlain(text){const out=$('#out');out.classList.remove('loading');out.textContent=text}
function show(tab){document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===tab));document.querySelectorAll('.pane').forEach(x=>x.classList.toggle('active',x.id===tab))}
document.querySelectorAll('.tab').forEach(x=>x.onclick=()=>show(x.dataset.tab));
async function loadModels(){const r=await fetch('/models');const j=await r.json();$('#models').innerHTML=j.models.map(m=>`<div class="model"><span>${m.name}</span><span class="${m.loaded?'ok':'miss'}">${m.loaded?'loaded':'missing'}</span></div>`).join('')}
async function reloadModels(){await fetch('/reload-models',{method:'POST'});await loadModels()}
$('#reloadA').onclick=reloadModels;$('#reloadB').onclick=reloadModels;
$('#sample').onclick=()=>{$('#chat textarea').value='Under Tunisian law, especially Organic Law No. 58 of 2017, what evidence should be collected in a violence against women case, and what urgent risks or legal points should be verified?'}
$('#docForm').onsubmit=e=>e.preventDefault();
$('#chatForm').onsubmit=e=>e.preventDefault();
$('#docSubmit').onclick=async()=>{const b=$('#docSubmit');b.disabled=true;setLoading('Analyzing document');$('#sources').textContent='Waiting for retrieval...';try{const fd=new FormData($('#docForm'));if(!fd.get('file')||!fd.get('file').name)throw new Error('Choose a document first.');const r=await fetch('/analyze',{method:'POST',body:fd});const j=await r.json();if(!r.ok)throw new Error(j.detail||'Request failed');setReport(j.answer);$('#sources').textContent=sources(j)}catch(err){setPlain(String(err));$('#sources').textContent='No sources.'}finally{b.disabled=false}}
$('#chatSubmit').onclick=async()=>{const b=$('#chatSubmit');b.disabled=true;setLoading('Searching legal references');$('#sources').textContent='Waiting for retrieval...';try{const q=new FormData($('#chatForm')).get('question');const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const j=await r.json();if(!r.ok)throw new Error(j.detail||'Request failed');setReport(j.answer);$('#sources').textContent=sources(j)}catch(err){setPlain(String(err));$('#sources').textContent='No sources.'}finally{b.disabled=false}}
loadModels()
</script>
</body>
</html>
"""

@app.get("/health", response_model=HealthResponse)
def health():
    return {"ok": True, "models": pipeline.model_status()}

@app.get("/models")
def models():
    return {"models": pipeline.model_status()}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), question: str = Form(""), case_id: str = Form("")):
    try:
        name = clean_filename(file.filename or "document")
        path = UPLOAD_DIR / name
        data = await file.read()
        path.write_bytes(data)
        return pipeline.analyze(path, question, case_id or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    try:
        return pipeline.chat(payload.question, payload.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-models")
def reload_models():
    pipeline.models.load_all()
    return {"models": pipeline.model_status()}
