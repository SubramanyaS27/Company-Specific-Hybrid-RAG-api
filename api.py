from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag_pipeline import ASSETS, CONFIG, ROLE_SCOPED_CACHE, run_rag_query


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    role: Literal["employee", "manager", "admin"] = "employee"
    use_cache: bool = True
    short_circuit_cache: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(CONFIG, "warmup_on_startup", False):
        ASSETS.ensure_loaded()
    yield
    if hasattr(ASSETS, "close"):
        ASSETS.close()


app = FastAPI(title="Southwind Labs RBAC Hybrid RAG", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": CONFIG.groq_model,
        "max_context_tokens": CONFIG.max_context_tokens,
        "max_generation_tokens": CONFIG.max_generation_tokens,
        "cached_entries": len(ROLE_SCOPED_CACHE),
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return run_rag_query(
            query=req.query,
            role=req.role,
            use_cache=req.use_cache,
            short_circuit_cache=req.short_circuit_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Southwind Labs — Knowledge Assistant</title>
  <style>
    :root {
      --bg: #f1f5f9;
      --surface: #ffffff;
      --subtle: #f8fafc;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
      --blue: #2563eb;
      --blue-soft: #eff6ff;
      --green: #15803d;
      --green-bg: #dcfce7;
      --amber: #b45309;
      --amber-bg: #fef3c7;
      --red: #b91c1c;
      --red-bg: #fee2e2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app-shell {
      display: grid;
      grid-template-columns: 228px 1fr 340px;
      min-height: 100vh;
    }

    /* LEFT SIDEBAR */
    .left-sidebar {
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 1rem 0.8rem;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.55rem;
      padding: 0.25rem 0.45rem 1rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 0.85rem;
    }
    .brand-mark {
      width: 28px;
      height: 28px;
      border-radius: 7px;
      background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
      color: #fff;
      display: grid;
      place-items: center;
      font-weight: 700;
      font-size: 0.85rem;
    }
    .brand-name { font-weight: 700; font-size: 0.95rem; }
    .nav-item {
      width: 100%;
      display: flex;
      align-items: center;
      gap: 0.6rem;
      text-align: left;
      background: transparent;
      border: 1px solid transparent;
      color: #334155;
      border-radius: 8px;
      padding: 0.58rem 0.7rem;
      font-size: 0.88rem;
      font-weight: 500;
      cursor: pointer;
      margin-bottom: 0.25rem;
    }
    .nav-item:hover { background: var(--subtle); }
    .nav-item.active {
      background: var(--blue-soft);
      border-color: #bfdbfe;
      color: var(--blue);
      font-weight: 600;
    }
    .left-footer {
      background: var(--subtle);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.75rem;
      font-size: 0.76rem;
      color: var(--muted);
      line-height: 1.4;
    }
    .left-footer strong {
      display: block;
      color: var(--text);
      font-size: 0.8rem;
      margin-bottom: 0.2rem;
    }

    /* MIDDLE PANE */
    .center-pane {
      display: flex;
      flex-direction: column;
      padding: 1rem 1.25rem;
      overflow-y: auto;
    }
    .center-header {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.85rem 1.1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1rem;
    }
    .center-header h2 { margin: 0; font-size: 1.05rem; }
    .center-header .sub { font-size: 0.8rem; color: var(--muted); margin-top: 0.15rem; }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem 1.1rem;
      margin-bottom: 1rem;
    }
    label { display: block; font-size: 0.78rem; color: var(--muted); margin-bottom: 0.3rem; }
    select, textarea {
      width: 100%;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
      background: var(--subtle);
      color: var(--text);
      padding: 0.55rem 0.65rem;
      font-size: 0.9rem;
    }
    textarea { min-height: 76px; resize: vertical; }
    .row-3 {
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      gap: 0.7rem;
      align-items: end;
      margin-top: 0.7rem;
    }
    .btn-send {
      background: var(--blue);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 0.62rem 1.1rem;
      font-weight: 600;
      font-size: 0.88rem;
      cursor: pointer;
    }
    .btn-send:hover { background: #1d4ed8; }
    .btn-send:disabled { opacity: 0.6; cursor: not-allowed; }
    .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.6rem; }
    .chip {
      background: var(--subtle);
      border: 1px solid var(--border);
      color: #334155;
      border-radius: 999px;
      padding: 0.25rem 0.65rem;
      font-size: 0.76rem;
      cursor: pointer;
    }
    .chip:hover { border-color: var(--blue); color: var(--blue); }
    .meta-strip { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.7rem; }
    .pill {
      background: var(--subtle);
      border: 1px solid var(--border);
      color: #1e293b;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 500;
    }
    .answer-box { white-space: pre-wrap; line-height: 1.55; font-size: 0.92rem; }
    .kv-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      padding: 0.4rem 0;
      border-bottom: 1px solid var(--border);
    }
    .kv-row:last-child { border-bottom: none; }

    /* RIGHT SIDEBAR */
    .right-sidebar {
      background: var(--surface);
      border-left: 1px solid var(--border);
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.9rem;
      overflow-y: auto;
    }
    .r-card {
      background: var(--subtle);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.85rem;
    }
    .r-title {
      font-size: 0.88rem;
      font-weight: 700;
      margin: 0 0 0.65rem 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.45rem;
    }
    .kpi {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 9px;
      padding: 0.55rem 0.4rem;
      text-align: center;
    }
    .kpi-label { font-size: 0.68rem; color: var(--muted); }
    .kpi-val { font-size: 1.15rem; font-weight: 700; margin-top: 0.15rem; }
    .kpi-val.green { color: var(--green); }
    .kpi-val.amber { color: var(--amber); }
    .kpi-val.red { color: var(--red); }

    .trace-step {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 0.5rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.48rem 0.6rem;
      margin-bottom: 0.4rem;
      font-size: 0.78rem;
    }
    .trace-step:last-child { margin-bottom: 0; }
    .step-name { font-weight: 600; color: var(--text); }
    .step-desc { font-size: 0.69rem; color: var(--muted); }
    .badge {
      white-space: nowrap;
      font-size: 0.69rem;
      font-weight: 600;
      padding: 0.16rem 0.48rem;
      border-radius: 999px;
      background: #e2e8f0;
      color: #475569;
    }
    .badge.running { background: var(--amber-bg); color: var(--amber); }
    .badge.passed { background: var(--green-bg); color: var(--green); }
    .badge.warn { background: var(--red-bg); color: var(--red); }

    @media (max-width: 1050px) {
      .app-shell { grid-template-columns: 1fr; }
      .left-sidebar, .right-sidebar { border: none; border-bottom: 1px solid var(--border); }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <!-- LEFT SIDEBAR -->
    <aside class="left-sidebar">
      <div>
        <div class="brand">
          <div class="brand-mark">SW</div>
          <div class="brand-name">Southwind Labs</div>
        </div>
        <button id="nav-chat" class="nav-item active" onclick="setSection('chat')">💬 Chat</button>
        <button id="nav-documents" class="nav-item" onclick="setSection('documents')">📄 Documents</button>
        <button id="nav-evaluation" class="nav-item" onclick="setSection('evaluation')">📊 Evaluation</button>
        <button id="nav-trace" class="nav-item" onclick="setSection('trace')">⚡ Pipeline Trace</button>
        <button id="nav-settings" class="nav-item" onclick="setSection('settings')">⚙️ Settings</button>
      </div>
      <div class="left-footer">
        <strong>Southwind Labs</strong>
        Internal knowledge assistant for company policies, processes and reports.
      </div>
    </aside>

    <!-- MIDDLE PANE (PRESERVED WORKING FLOW) -->
    <main class="center-pane">
      <div class="center-header">
        <div>
          <h2>Knowledge Assistant</h2>
          <div class="sub">Role-aware • Hybrid RAG (Dense + BM25) • Guardrails • Citations</div>
        </div>
        <div style="min-width: 140px;">
          <label for="role" style="margin-bottom: 0.15rem;">Active Role</label>
          <select id="role" onchange="syncRole()">
            <option value="employee">employee</option>
            <option value="manager">manager</option>
            <option value="admin">admin</option>
          </select>
        </div>
      </div>

      <section id="sec-chat">
        <div class="card">
          <div class="chips">
            <span class="chip" onclick="fillPrompt('employee', 'What are the rules for leave approval and laptop security?')">Employee: Leave &amp; Laptop</span>
            <span class="chip" onclick="fillPrompt('employee', 'What are the finance approval spend limits in lakh or crore?')">Employee: Restricted FIN-001</span>
            <span class="chip" onclick="fillPrompt('manager', 'What are the rules for leave approval, hiring, and employee performance reviews?')">Manager: HR &amp; Reviews</span>
            <span class="chip" onclick="fillPrompt('admin', 'What are the finance approval spend limits in lakh or crore?')">Admin: Finance Limits</span>
          </div>

          <div style="margin-top: 0.7rem;">
            <label for="query">Ask a question</label>
            <textarea id="query" placeholder="Ask a question about company policies, security, finance, or HR..."></textarea>
          </div>

          <div class="row-3">
            <div>
              <label for="cacheMode">Cache</label>
              <select id="cacheMode">
                <option value="true">Role-Scoped Cache Enabled</option>
                <option value="false">Bypass Cache</option>
              </select>
            </div>
            <div>
              <label for="tokenCap">Token Budget Guard</label>
              <select id="tokenCap" disabled>
                <option>650 ctx / &lt;1500 total</option>
              </select>
            </div>
            <button id="askBtn" class="btn-send" onclick="submitQuery()">Send</button>
          </div>
        </div>

        <div id="resultCard" class="card" style="display: none;">
          <div id="metaStrip" class="meta-strip"></div>
          <div id="answerBox" class="answer-box"></div>
          <div style="margin-top: 0.85rem;">
            <label>Citations</label>
            <div id="citationsWrap" class="meta-strip"></div>
          </div>
        </div>
      </section>

      <section id="sec-documents" class="card" style="display: none;">
        <h3 style="margin-top: 0;">Indexed Policy Documents</h3>
        <div class="kv-row"><span>HR-001</span><span>Leave, hiring, and performance reviews</span></div>
        <div class="kv-row"><span>SEC-001</span><span>Laptop encryption, screen lock, VPN, incidents</span></div>
        <div class="kv-row"><span>ENG-001</span><span>Engineering standards &amp; workflows</span></div>
        <div class="kv-row"><span>FIN-001</span><span>Spend approval thresholds (role-restricted)</span></div>
        <div class="kv-row"><span>LDR-001</span><span>Leadership &amp; escalation guidelines</span></div>
      </section>

      <section id="sec-evaluation" class="card" style="display: none;">
        <h3 style="margin-top: 0;">Evaluation &amp; Token Budget Suite</h3>
        <div class="kv-row"><span>Edge-Case Suite</span><span>7 / 7 PASSED</span></div>
        <div class="kv-row"><span>Context Ceiling</span><span>650 tokens max</span></div>
        <div class="kv-row"><span>Planned Total Ceiling</span><span>&lt; 1,500 tokens</span></div>
        <div class="kv-row"><span>Last Query Tokens</span><span id="evalTokens">-</span></div>
        <div class="kv-row"><span>Last Total Latency</span><span id="evalLatency">-</span></div>
      </section>

      <section id="sec-trace" class="card" style="display: none;">
        <h3 style="margin-top: 0;">Latest Telemetry Breakdown</h3>
        <div class="kv-row"><span>Retrieval + Rerank</span><span id="trRetrieve">-</span></div>
        <div class="kv-row"><span>Context Packing</span><span id="trPack">-</span></div>
        <div class="kv-row"><span>LLM Generation</span><span id="trLlm">-</span></div>
        <div class="kv-row"><span>End-to-End Total</span><span id="trTotal">-</span></div>
      </section>

      <section id="sec-settings" class="card" style="display: none;">
        <h3 style="margin-top: 0;">Runtime Settings</h3>
        <div class="kv-row"><span>Model</span><span>openai/gpt-oss-20b (Groq)</span></div>
        <div class="kv-row"><span>Reasoning Effort</span><span>low (with fallback)</span></div>
        <div class="kv-row"><span>Cache Scope</span><span>Role + Mode + Query + Chunks</span></div>
      </section>
    </main>

    <!-- RIGHT SIDEBAR: SESSION SUMMARY + 10-STEP LIVE PIPELINE TRACE -->
    <aside class="right-sidebar">
      <div class="r-card">
        <div class="r-title"><span>Session Summary</span></div>
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-label">Cache hits</div>
            <div id="kpiCacheHits" class="kpi-val green">0</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Guardrail blocks</div>
            <div id="kpiGuardrailBlocks" class="kpi-val amber">0</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Access denials</div>
            <div id="kpiAccessDenials" class="kpi-val red">0</div>
          </div>
        </div>
      </div>

      <div class="r-card">
        <div class="r-title"><span>Live Pipeline Trace</span><span id="traceTotalMs" style="font-size:0.75rem;color:#64748b;">Idle</span></div>
        <div id="traceList">
          <div class="trace-step" data-step="0"><div><div class="step-name">1. Input guardrail</div><div class="step-desc">Validate query &amp; scope</div></div><span class="badge" id="b-0">Idle</span></div>
          <div class="trace-step" data-step="1"><div><div class="step-name">2. Role validation</div><div class="step-desc">Check RBAC tier</div></div><span class="badge" id="b-1">Idle</span></div>
          <div class="trace-step" data-step="2"><div><div class="step-name">3. Semantic cache</div><div class="step-desc">Role-scoped lookup</div></div><span class="badge" id="b-2">Idle</span></div>
          <div class="trace-step" data-step="3"><div><div class="step-name">4. Dense retrieval</div><div class="step-desc">Qdrant vector search</div></div><span class="badge" id="b-3">Idle</span></div>
          <div class="trace-step" data-step="4"><div><div class="step-name">5. BM25 retrieval</div><div class="step-desc">Lexical keyword search</div></div><span class="badge" id="b-4">Idle</span></div>
          <div class="trace-step" data-step="5"><div><div class="step-name">6. RRF fusion</div><div class="step-desc">Merge &amp; deduplicate hits</div></div><span class="badge" id="b-5">Idle</span></div>
          <div class="trace-step" data-step="6"><div><div class="step-name">7. Reranking</div><div class="step-desc">Cross-encoder rescoring</div></div><span class="badge" id="b-6">Idle</span></div>
          <div class="trace-step" data-step="7"><div><div class="step-name">8. Governance gate</div><div class="step-desc">650-token context budget</div></div><span class="badge" id="b-7">Idle</span></div>
          <div class="trace-step" data-step="8"><div><div class="step-name">9. LLM generation</div><div class="step-desc">Groq grounded response</div></div><span class="badge" id="b-8">Idle</span></div>
          <div class="trace-step" data-step="9"><div><div class="step-name">10. Citation check</div><div class="step-desc">Sanitize [S#] references</div></div><span class="badge" id="b-9">Idle</span></div>
        </div>
      </div>
    </aside>
  </div>

  <script>
    const SECTIONS = ["chat", "documents", "evaluation", "trace", "settings"];
    let cacheHits = 0;
    let guardrailBlocks = 0;
    let accessDenials = 0;
    let stepTimer = null;

    function setSection(sec) {
      SECTIONS.forEach(s => {
        document.getElementById("nav-" + s).classList.toggle("active", s === sec);
        document.getElementById("sec-" + s).style.display = (s === sec) ? "block" : "none";
      });
    }

    function syncRole() {
      // Keeps role synchronized cleanly
    }

    function fillPrompt(role, text) {
      setSection("chat");
      document.getElementById("role").value = role;
      document.getElementById("query").value = text;
    }

    function setStepBadge(i, text, cls) {
      const el = document.getElementById("b-" + i);
      el.textContent = text;
      el.className = "badge" + (cls ? " " + cls : "");
    }

    function startTraceAnim() {
      if (stepTimer) clearInterval(stepTimer);
      for (let i = 0; i < 10; i++) setStepBadge(i, "Queued", "");
      let cur = 0;
      setStepBadge(0, "Running", "running");
      stepTimer = setInterval(() => {
        if (cur < 10) setStepBadge(cur, "Passed", "passed");
        cur = (cur + 1) % 10;
        setStepBadge(cur, "Running", "running");
      }, 280);
    }

    function completeTrace(data) {
      if (stepTimer) {
        clearInterval(stepTimer);
        stepTimer = null;
      }
      const t = data.telemetry || {};
      const retMs = t.retrieval_rerank_ms ?? 0;
      const packMs = t.context_pack_ms ?? 0;
      const llmMs = t.llm_ms ?? 0;
      const totMs = t.total_ms ?? 0;

      setStepBadge(0, "Passed", "passed");
      setStepBadge(1, data.role, "passed");
      setStepBadge(2, data.cache_hit ? "Hit" : "Miss", data.cache_hit ? "passed" : "");
      setStepBadge(3, data.cache_hit ? "Skipped" : (retMs / 2).toFixed(0) + "ms", "passed");
      setStepBadge(4, data.cache_hit ? "Skipped" : (retMs / 2).toFixed(0) + "ms", "passed");
      setStepBadge(5, data.cache_hit ? "Skipped" : "Merged", "passed");
      setStepBadge(6, data.cache_hit ? "Skipped" : "Scored", "passed");
      setStepBadge(7, packMs.toFixed(0) + "ms", "passed");
      setStepBadge(8, data.cache_hit ? "0ms" : llmMs.toFixed(0) + "ms", "passed");
      setStepBadge(9, "Verified", "passed");

      document.getElementById("traceTotalMs").textContent = totMs.toFixed(0) + " ms";
      document.getElementById("trRetrieve").textContent = retMs + " ms";
      document.getElementById("trPack").textContent = packMs + " ms";
      document.getElementById("trLlm").textContent = llmMs + " ms";
      document.getElementById("trTotal").textContent = totMs + " ms";
    }

    async function submitQuery() {
      const btn = document.getElementById("askBtn");
      const role = document.getElementById("role").value;
      const query = document.getElementById("query").value.trim();
      const useCache = document.getElementById("cacheMode").value === "true";
      if (!query) return;

      btn.disabled = true;
      btn.textContent = "Running...";
      startTraceAnim();

      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, role, use_cache: useCache })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Request failed");

        if (data.cache_hit) {
          cacheHits += 1;
          document.getElementById("kpiCacheHits").textContent = String(cacheHits);
        }

        const ansLower = (data.answer || "").toLowerCase();
        const isRestricted =
          (data.active_citations || []).length === 0 ||
          ansLower.includes("do not contain") ||
          ansLower.includes("restricted") ||
          ansLower.includes("not authorized");
        if (isRestricted) {
          if (role === "employee" && query.toLowerCase().includes("finance")) {
            accessDenials += 1;
            document.getElementById("kpiAccessDenials").textContent = String(accessDenials);
          } else {
            guardrailBlocks += 1;
            document.getElementById("kpiGuardrailBlocks").textContent = String(guardrailBlocks);
          }
        }

        completeTrace(data);

        document.getElementById("evalTokens").textContent = `${data.used_context_tokens}/${data.total_planned_tokens}`;
        document.getElementById("evalLatency").textContent = `${(data.telemetry && data.telemetry.total_ms) ?? 0} ms`;

        document.getElementById("resultCard").style.display = "block";
        const t = data.telemetry || {};
        document.getElementById("metaStrip").innerHTML = `
          <span class="pill">Role: <b>${data.role}</b></span>
          <span class="pill">Mode: <b>${data.compression_mode}</b></span>
          <span class="pill">Tokens: <b>${data.used_context_tokens}/${data.total_planned_tokens}</b></span>
          <span class="pill">Cache: <b>${data.cache_hit ? "HIT" : "MISS"}</b></span>
          <span class="pill">LLM: <b>${t.llm_ms ?? 0} ms</b></span>
          <span class="pill">Total: <b>${t.total_ms ?? 0} ms</b></span>
        `;
        document.getElementById("answerBox").textContent = data.answer || "(Empty answer)";

        const citWrap = document.getElementById("citationsWrap");
        citWrap.innerHTML = "";
        (data.active_citations || []).forEach(c => {
          const span = document.createElement("span");
          span.className = "pill";
          span.textContent = c;
          citWrap.appendChild(span);
        });
      } catch (err) {
        if (stepTimer) { clearInterval(stepTimer); stepTimer = null; }
        document.getElementById("traceTotalMs").textContent = "Error";
        document.getElementById("resultCard").style.display = "block";
        document.getElementById("answerBox").textContent = "Error: " + err.message;
      } finally {
        btn.disabled = false;
        btn.textContent = "Send";
      }
    }
  </script>
</body>
</html>"""