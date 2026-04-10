"""NeuralGate — AI Routing Intelligence System v5"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

from safety import inspect
from router import route
from validator import validate
from executor import execute
from learner import log, learn, get_learned_weights, get_recent, get_stats
from config import MODELS
import cache
import budget

app = FastAPI(title="NeuralGate")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class HistoryItem(BaseModel):
    user: str
    ai: str

class Query(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    api_keys: dict = Field(default_factory=dict)
    disabled_keys: list = Field(default_factory=list)
    history: List[HistoryItem] = Field(default_factory=list)


@app.post("/route")
async def route_query(req: Query):
    t_start = time.time()
    steps = []
    active_keys = {k: v for k, v in req.api_keys.items() if k not in req.disabled_keys}
    history_dicts = [{"user": h.user, "ai": h.ai} for h in req.history]

    # Cache check (skip if continuing conversation)
    t = time.time()
    cached = None
    if not req.history:
        cached = cache.get(req.query)
    steps.append({"layer": "Cache", "ms": round((time.time()-t)*1000, 1)})
    if cached:
        return {
            "blocked": False, "response": cached["text"],
            "model": cached["model_used"], "model_display": cached["model_display"],
            "model_tier": "cached", "goodness_score": 0,
            "routing_reason": "Cache hit (" + str(int(cached['similarity']*100)) + "% match)",
            "complexity": {"level": "cached", "score": 0, "reasons": ["From cache"]},
            "safety": {"pii_count": 0, "pii_found": [], "domains": [], "must_use_private": False, "risk_score": 0},
            "all_scores": [], "weights_used": {},
            "validation": {"passed": True, "toxicity": {"score": 0}, "hallucination": {"score": 0}},
            "latency_ms": 0, "tokens": 0, "est_cost": 0, "used_fallback": False,
            "steps": steps, "total_ms": round((time.time()-t_start)*1000),
            "learn_result": None, "cached": True,
            "budget_status": budget.get_status(), "cache_stats": cache.stats(),
        }

    # Safety
    t = time.time()
    safety = inspect(req.query)
    steps.append({"layer": "Safety", "ms": round((time.time()-t)*1000, 1)})
    if safety["blocked"]:
        return {"blocked": True, "block_reason": safety["block_reason"], "safety": safety, "steps": steps, "total_ms": round((time.time()-t_start)*1000)}

    force_free = budget.should_force_free()

    # Router
    t = time.time()
    learned = get_learned_weights()
    if force_free:
        free_keys = {k: v for k, v in active_keys.items() if any(MODELS[m]["free"] and MODELS[m]["key_name"] == k for m in MODELS)}
        free_keys[""] = ""
        decision = route(req.query, safety, available_keys=free_keys, learned_weights=learned)
        decision["routing_reason"] += " | BUDGET: free only"
    else:
        decision = route(req.query, safety, available_keys=active_keys, learned_weights=learned)
    steps.append({"layer": "Router", "ms": round((time.time()-t)*1000, 1)})

    # Executor — with conversation history
    t = time.time()
    result = await execute(decision["selected_model"], safety["safe_query"], active_keys, decision["fallback"], history=history_dicts)
    steps.append({"layer": "Executor", "ms": round((time.time()-t)*1000, 1)})

    # Validator
    t = time.time()
    val = validate(result["text"], req.query)
    steps.append({"layer": "Validator", "ms": round((time.time()-t)*1000, 1)})

    tokens_used = result.get("tokens", 0)
    model_meta = MODELS.get(result["model_used"], {})
    est_cost = round(model_meta.get("cost", 0) * tokens_used / 1000, 6)
    budget.add_cost(est_cost)

    if val["passed"] and not result["text"].startswith("[") and not req.history:
        cache.put(req.query, result["text"], result["model_used"], result.get("model_display", ""))

    # Logger
    t = time.time()
    log({"query": req.query[:100], "complexity": decision["complexity"]["level"], "complexity_score": decision["complexity"]["score"], "risk_score": safety["risk_score"], "model": result["model_used"], "goodness_score": decision["goodness_score"], "latency_ms": result["latency_ms"], "tokens": tokens_used, "validation_passed": val["passed"], "toxicity_score": val["toxicity"]["score"], "hallucination_score": val["hallucination"]["score"], "used_fallback": result.get("used_fallback", False), "weights": decision["weights_used"]})
    steps.append({"layer": "Logger", "ms": round((time.time()-t)*1000, 1)})

    stats = get_stats()
    learn_result = None
    if stats["total"] >= 10 and stats["total"] % 10 == 0:
        learn_result = learn()

    return {
        "blocked": False, "response": result["text"],
        "model": result["model_used"], "model_display": result.get("model_display", result["model_used"]),
        "model_tier": decision["tier"], "goodness_score": decision["goodness_score"],
        "routing_reason": decision["routing_reason"], "complexity": decision["complexity"],
        "safety": safety, "all_scores": decision["all_scores"], "weights_used": decision["weights_used"],
        "validation": val, "latency_ms": result["latency_ms"], "tokens": tokens_used,
        "est_cost": est_cost, "used_fallback": result.get("used_fallback", False),
        "steps": steps, "total_ms": round((time.time()-t_start)*1000),
        "learn_result": learn_result, "cached": False,
        "budget_status": budget.get_status(), "cache_stats": cache.stats(),
    }


@app.get("/models")
async def list_models():
    return MODELS

@app.get("/stats")
async def stats():
    s = get_stats()
    s["cache"] = cache.stats()
    s["budget"] = budget.get_status()
    return s

@app.post("/learn")
async def trigger_learn():
    return learn()

@app.post("/budget")
async def set_budget(limit: float = 1.0):
    budget.set_limit(limit)
    return budget.get_status()

@app.post("/cache/clear")
async def clear_cache():
    cache.clear()
    return {"cleared": True}


UI = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuralGate</title>
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments)}gtag('js',new Date());gtag('config','GA_MEASUREMENT_ID');</script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#09090b;--s1:#18181b;--s2:#27272a;--b:#3f3f46;--t:#fafafa;--t2:#a1a1aa;--t3:#71717a;--ac:#8b5cf6;--ac2:#a78bfa;--acbg:rgba(139,92,246,.08);--g:#22c55e;--gbg:rgba(34,197,94,.08);--y:#eab308;--r:#ef4444;--rbg:rgba(239,68,68,.06);--c:#06b6d4;--o:#f97316;--radius:12px}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--t);height:100vh;display:flex;overflow:hidden}
.mono{font-family:'JetBrains Mono',monospace}

/* Sidebar */
.sidebar{width:240px;background:var(--s1);border-right:1px solid var(--s2);display:flex;flex-direction:column;flex-shrink:0}
.sb-hdr{padding:14px;border-bottom:1px solid var(--s2);display:flex;align-items:center;justify-content:space-between}
.sb-logo{font-size:.9rem;font-weight:700}.sb-logo span{color:var(--ac)}
.new-chat{padding:6px 12px;border-radius:6px;border:1px solid var(--ac);background:transparent;color:var(--ac);font-size:.72rem;cursor:pointer;font-family:'Inter',sans-serif;font-weight:500}
.new-chat:hover{background:var(--acbg)}
.sb-list{flex:1;overflow-y:auto;padding:8px}
.sb-item{padding:8px 10px;border-radius:6px;cursor:pointer;font-size:.78rem;color:var(--t2);margin-bottom:2px;display:flex;align-items:center;justify-content:space-between;transition:.1s}
.sb-item:hover{background:var(--s2)}
.sb-item.active{background:var(--acbg);color:var(--ac)}
.sb-item .title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sb-item .del{display:none;color:var(--t3);font-size:.7rem;padding:2px 4px;border-radius:3px;cursor:pointer}
.sb-item:hover .del{display:block}
.sb-item .del:hover{color:var(--r);background:var(--rbg)}

/* Main */
.main-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
.hdr{padding:8px 16px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--s2);flex-shrink:0}
.hdr-right{display:flex;align-items:center;gap:6px}
.hb{font-size:.6rem;padding:3px 7px;border-radius:4px;font-family:'JetBrains Mono',monospace;border:1px solid}
.hb.cost{background:var(--gbg);color:var(--g);border-color:rgba(34,197,94,.12)}
.hb.cache{background:var(--acbg);color:var(--ac);border-color:rgba(139,92,246,.12)}
.hb.q{background:var(--s1);color:var(--t3);border-color:var(--s2)}
.hb.budget{background:rgba(234,179,8,.05);color:var(--y);border-color:rgba(234,179,8,.12)}
.hb.warn{background:var(--rbg);color:var(--r);border-color:rgba(239,68,68,.15)}
.gear{width:30px;height:30px;border-radius:6px;border:1px solid var(--s2);background:transparent;color:var(--t2);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:.9rem}
.gear:hover{background:var(--s1);color:var(--t)}.gear.active{background:var(--acbg);color:var(--ac);border-color:var(--ac)}

/* Settings */
.settings{display:none;position:fixed;top:0;right:0;width:340px;height:100vh;background:var(--s1);border-left:1px solid var(--s2);z-index:100;overflow-y:auto;padding:14px;font-size:.78rem}
.settings.open{display:block}
.set-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.set-hdr h3{font-size:.82rem;font-weight:600}
.xbtn{width:26px;height:26px;border-radius:5px;border:1px solid var(--s2);background:transparent;color:var(--t2);cursor:pointer;font-size:.85rem;display:flex;align-items:center;justify-content:center}
.xbtn:hover{background:var(--s2);color:var(--t)}
.krow{display:flex;align-items:center;gap:5px;margin-bottom:4px}
.klbl{width:80px;font-size:.68rem;color:var(--t3);flex-shrink:0}
.klbl .free{color:var(--g);font-size:.55rem;font-weight:600}.klbl .paid{color:var(--y);font-size:.55rem;font-weight:600}
.kinput{flex:1;padding:5px 7px;background:var(--bg);border:1px solid var(--s2);border-radius:4px;color:var(--t);font-family:'JetBrains Mono',monospace;font-size:.65rem;outline:none}
.kinput:focus{border-color:var(--ac)}
.ktog{width:24px;height:24px;border-radius:4px;border:1px solid var(--s2);background:transparent;cursor:pointer;font-size:.65rem;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ktog.on{background:var(--gbg);color:var(--g);border-color:rgba(34,197,94,.3)}
.ktog.off{background:var(--rbg);color:var(--r);border-color:rgba(239,68,68,.2)}
.khint{font-size:.58rem;color:var(--t3);margin-bottom:7px;padding-left:85px}.khint a{color:var(--ac2)}
.ssec{border-top:1px solid var(--s2);margin-top:8px;padding-top:8px}
.ssec h4{font-size:.65rem;color:var(--t3);margin-bottom:5px;text-transform:uppercase;letter-spacing:.04em}
.sbtn{width:100%;padding:5px;border-radius:4px;border:1px solid rgba(139,92,246,.2);background:var(--acbg);color:var(--ac2);font-family:'Inter',sans-serif;font-size:.7rem;cursor:pointer;margin-bottom:3px}
.sbtn:hover{background:rgba(139,92,246,.15)}
.sbtn.red{border-color:rgba(239,68,68,.2);background:var(--rbg);color:var(--r)}
.sout{font-size:.6rem;color:var(--t3);margin-top:3px;white-space:pre-wrap}
.snote{font-size:.58rem;color:var(--t3);margin-top:6px}

/* Chat */
.chat{flex:1;overflow-y:auto;padding:14px 20px;max-width:820px;width:100%;margin:0 auto}
.msg{margin-bottom:12px}.msg-user{display:flex;justify-content:flex-end}
.msg-user .bubble{background:var(--ac);color:#fff;padding:9px 14px;border-radius:var(--radius) var(--radius) 4px var(--radius);max-width:70%;font-size:.85rem;line-height:1.5}
.msg-ai{display:flex;flex-direction:column;align-items:flex-start}
.ai-meta{display:flex;align-items:center;gap:5px;margin-bottom:3px;flex-wrap:wrap}
.ai-model{font-size:.67rem;color:var(--t2);font-weight:500}
.ai-time{font-size:.62rem;color:var(--t3);font-family:'JetBrains Mono',monospace}
.ai-cost{font-size:.6rem;font-family:'JetBrains Mono',monospace}
.ai-cost.free{color:var(--g)}.ai-cost.paid{color:var(--y)}.ai-cost.cached{color:var(--ac)}
.ai-bubble{background:var(--s1);border:1px solid var(--s2);padding:12px 14px;border-radius:4px var(--radius) var(--radius) var(--radius);max-width:85%;font-size:.85rem;line-height:1.7;white-space:pre-wrap}
.ai-bubble.cache-hit{border-color:rgba(139,92,246,.2);background:rgba(139,92,246,.02)}
.ai-actions{display:flex;gap:4px;margin-top:4px;align-items:center}
.thumb{width:24px;height:24px;border-radius:4px;border:1px solid var(--s2);background:transparent;cursor:pointer;font-size:.75rem;display:flex;align-items:center;justify-content:center;color:var(--t3)}
.thumb:hover{background:var(--s1);color:var(--t)}
.thumb.up-active{background:var(--gbg);border-color:rgba(34,197,94,.3);color:var(--g)}
.thumb.down-active{background:var(--rbg);border-color:rgba(239,68,68,.3);color:var(--r)}
.dbtn{font-size:.62rem;color:var(--t3);background:none;border:none;cursor:pointer;font-family:'Inter',sans-serif;padding:2px 5px;border-radius:3px}
.dbtn:hover{background:var(--s1);color:var(--t2)}
.dpop{display:none;background:var(--s1);border:1px solid var(--s2);border-radius:7px;padding:10px;margin-top:5px;max-width:85%;font-size:.68rem}
.dpop.open{display:block}
.dp-m{display:flex;align-items:center;gap:4px;padding:2px 0;font-size:.65rem}.dp-m.sel{color:var(--g);font-weight:600}.dp-m .sc{margin-left:auto;font-family:'JetBrains Mono',monospace}
.dp-bar{display:flex;align-items:center;gap:4px;margin-top:2px}.dp-bar .lb{font-size:.62rem;width:55px;color:var(--t3)}.dp-bar .bg{flex:1;height:3px;background:var(--bg);border-radius:2px;overflow:hidden}.dp-bar .fill{height:100%;border-radius:2px}.dp-bar .val{font-size:.62rem;width:26px;text-align:right;font-family:'JetBrains Mono',monospace}
.pipe-mini{display:flex;gap:2px;margin-top:5px}.pipe-mini .ps{padding:2px 4px;background:var(--bg);border-radius:3px;font-size:.52rem;text-align:center;flex:1}.pipe-mini .ps .n{color:var(--t3)}.pipe-mini .ps .v{color:var(--t2);font-family:'JetBrains Mono',monospace}
.blocked{background:var(--rbg);border:1px solid rgba(239,68,68,.15);padding:10px;border-radius:var(--radius);color:var(--r);font-size:.82rem}
.ptag{display:inline-block;padding:2px 4px;margin:1px;border-radius:3px;font-size:.62rem;background:var(--rbg);color:var(--r)}

/* Input */
.input-bar{flex-shrink:0;padding:8px 16px;border-top:1px solid var(--s2);background:var(--bg)}
.input-inner{max-width:820px;margin:0 auto;display:flex;gap:6px;align-items:flex-end}
.input-inner textarea{flex:1;min-height:40px;max-height:120px;padding:10px 12px;background:var(--s1);border:1px solid var(--s2);border-radius:8px;color:var(--t);font-family:'Inter',sans-serif;font-size:.85rem;resize:none;outline:none;line-height:1.4}
.input-inner textarea:focus{border-color:var(--ac)}
.input-inner textarea::placeholder{color:var(--t3)}
.send{padding:10px 16px;background:var(--ac);color:#fff;border:none;border-radius:8px;font-family:'Inter',sans-serif;font-weight:600;font-size:.8rem;cursor:pointer;flex-shrink:0}
.send:hover{filter:brightness(1.15)}.send:disabled{opacity:.4}
.pbar{max-width:820px;margin:4px auto 0;display:flex;gap:3px;flex-wrap:wrap}
.pre{padding:2px 8px;border-radius:4px;border:1px solid var(--s2);background:transparent;color:var(--t3);font-size:.64rem;cursor:pointer;font-family:'Inter',sans-serif}
.pre:hover{background:var(--s1);color:var(--t)}

.toast{display:none;position:fixed;bottom:65px;left:50%;transform:translateX(-50%);border-radius:6px;padding:6px 14px;font-size:.7rem;z-index:50}
.toast.show{display:block}
.toast.learn{background:var(--acbg);border:1px solid rgba(139,92,246,.2);color:var(--ac2)}
.toast.cache{background:var(--gbg);border:1px solid rgba(34,197,94,.2);color:var(--g)}

.spin{display:inline-block;width:12px;height:12px;border:2px solid rgba(255,255,255,.15);border-top-color:#fff;border-radius:50%;animation:sp .5s linear infinite;margin-right:4px;vertical-align:middle}
@keyframes sp{to{transform:rotate(360deg)}}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--t3);gap:5px}
.empty .lb{font-size:1.6rem;font-weight:700;color:var(--s2)}.empty .lb span{color:rgba(139,92,246,.25)}
.empty p{font-size:.78rem}
</style></head><body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sb-hdr">
    <div class="sb-logo"><span>N</span>Gate</div>
    <button class="new-chat" onclick="newChat()">+ New</button>
  </div>
  <div class="sb-list" id="sbList"></div>
</div>

<!-- Main -->
<div class="main-area">
  <div class="hdr">
    <div style="font-size:.75rem;color:var(--t2);font-weight:500" id="chatTitle">New Chat</div>
    <div class="hdr-right">
      <div class="hb cost" id="costB">$0.0000</div>
      <div class="hb cache" id="cacheB">0 cached</div>
      <div class="hb q" id="queryB">0 queries</div>
      <button class="gear" id="gearBtn" onclick="toggleS()">&#9881;</button>
    </div>
  </div>

  <!-- Settings -->
  <div class="settings" id="settings">
    <div class="set-hdr"><h3>Settings</h3><button class="xbtn" onclick="toggleS()">&#10005;</button></div>
    <div class="krow"><span class="klbl">Groq <span class="free">FREE</span></span><input class="kinput" id="key_GROQ_API_KEY" type="password" placeholder="gsk_..."><button class="ktog on" id="tog_GROQ_API_KEY" onclick="togKey('GROQ_API_KEY')">&#10003;</button></div>
    <div class="khint"><a href="https://console.groq.com/keys" target="_blank">Get key</a> &mdash; Llama 3.3 70B, 3.1 8B</div>
    <div class="krow"><span class="klbl">Gemini <span class="free">FREE</span></span><input class="kinput" id="key_GEMINI_API_KEY" type="password" placeholder="AIza..."><button class="ktog on" id="tog_GEMINI_API_KEY" onclick="togKey('GEMINI_API_KEY')">&#10003;</button></div>
    <div class="khint"><a href="https://aistudio.google.com/apikey" target="_blank">Get key</a> &mdash; Gemini 2.5 Flash, 3 Flash, 3.1 Pro</div>
    <div class="krow"><span class="klbl">Cohere <span class="free">FREE</span></span><input class="kinput" id="key_COHERE_API_KEY" type="password" placeholder="..."><button class="ktog on" id="tog_COHERE_API_KEY" onclick="togKey('COHERE_API_KEY')">&#10003;</button></div>
    <div class="khint"><a href="https://dashboard.cohere.com/api-keys" target="_blank">Get key</a> &mdash; Command R+, R</div>
    <div class="krow"><span class="klbl">Together <span class="free">FREE</span></span><input class="kinput" id="key_TOGETHER_API_KEY" type="password" placeholder="..."><button class="ktog on" id="tog_TOGETHER_API_KEY" onclick="togKey('TOGETHER_API_KEY')">&#10003;</button></div>
    <div class="khint"><a href="https://api.together.xyz/settings/api-keys" target="_blank">Get key</a> &mdash; Llama, Mixtral</div>
    <div class="krow"><span class="klbl">OpenAI <span class="paid">PAID</span></span><input class="kinput" id="key_OPENAI_API_KEY" type="password" placeholder="sk-..."><button class="ktog on" id="tog_OPENAI_API_KEY" onclick="togKey('OPENAI_API_KEY')">&#10003;</button></div>
    <div class="khint"><a href="https://platform.openai.com/api-keys" target="_blank">Get key</a> &mdash; GPT-5.4, 5.4-mini, nano, 4o</div>
    <div class="krow"><span class="klbl">Anthropic <span class="paid">PAID</span></span><input class="kinput" id="key_ANTHROPIC_API_KEY" type="password" placeholder="sk-ant-..."><button class="ktog on" id="tog_ANTHROPIC_API_KEY" onclick="togKey('ANTHROPIC_API_KEY')">&#10003;</button></div>
    <div class="khint"><a href="https://console.anthropic.com/settings/keys" target="_blank">Get key</a> &mdash; Claude Opus, Sonnet, Haiku</div>
    <div class="snote">Keys in browser only. &#10003;/&#10007; to enable/disable.</div>
    <div class="ssec"><h4>Learning Loop</h4><button class="sbtn" onclick="doLearn()">Run Learning Loop</button><div class="sout" id="learnOut"></div></div>
    <div class="ssec"><h4>Cache</h4><button class="sbtn red" onclick="clearCacheBtn()">Clear Cache</button><div class="sout" id="cacheOut"></div></div>
    <div class="ssec"><h4>Stats</h4><div class="sout" id="statsOut">Loading...</div></div>
  </div>

  <div class="chat" id="chat">
    <div class="empty" id="emptyState"><div class="lb"><span>Neural</span>Gate</div><p>Ask anything. The smartest model is picked for you.</p></div>
  </div>
  <div class="toast" id="toast"></div>
  <div class="input-bar">
    <div class="input-inner">
      <textarea id="q" placeholder="Ask anything..." rows="1"></textarea>
      <button class="send" id="btn" onclick="go()">Send</button>
    </div>
    <div class="pbar">
      <button class="pre" onclick="pre('simple')">Simple</button>
      <button class="pre" onclick="pre('complex')">Complex</button>
      <button class="pre" onclick="pre('pii')">PII Data</button>
      <button class="pre" onclick="pre('health')">Healthcare</button>
      <button class="pre" onclick="pre('code')">Code</button>
    </div>
  </div>
</div>

<script>
const KEYS=['GROQ_API_KEY','GEMINI_API_KEY','COHERE_API_KEY','TOGETHER_API_KEY','OPENAI_API_KEY','ANTHROPIC_API_KEY'];
const P={simple:"What is the capital of Japan?",complex:"Compare microservices vs monolithic architecture for a fintech startup. Step by step with pros and cons.",pii:"My Aadhaar is 1234 5678 9012, PAN is ABCDE1234F, email rajesh@example.com. Check my loan eligibility.",health:"Patient has stage 2 hypertension and elevated creatinine. What treatment adjustments?",code:"Write a Python binary search function with type hints. Explain time complexity."};

let chats = {};        // {id: {title, messages: [{user,ai,model,time,cost,details}]}}
let currentChat = null;
let totalCost=0, queryCount=0, cacheHits=0, mc=0;
let disabledKeys=[];

// --- Storage ---
function saveAll(){localStorage.setItem('ng_chats',JSON.stringify(chats));localStorage.setItem('ng_current',currentChat)}
function loadAll(){
  try{chats=JSON.parse(localStorage.getItem('ng_chats')||'{}');currentChat=localStorage.getItem('ng_current')}catch(e){chats={}}
  if(!currentChat||!chats[currentChat])newChat();
  else renderSidebar();renderChat();
}
function getKeys(){const k={};KEYS.forEach(n=>{const v=document.getElementById('key_'+n).value.trim();if(v)k[n]=v});return k}
function saveKeys(){localStorage.setItem('ng_keys',JSON.stringify(getKeys()))}
function loadKeys(){try{const k=JSON.parse(localStorage.getItem('ng_keys')||'{}');Object.entries(k).forEach(([n,v])=>{const el=document.getElementById('key_'+n);if(el)el.value=v})}catch(e){}}
function loadDisabled(){try{disabledKeys=JSON.parse(localStorage.getItem('ng_disabled')||'[]')}catch(e){disabledKeys=[]}}

// --- Chat Management ---
function newChat(){
  const id='c_'+Date.now();
  chats[id]={title:'New Chat',messages:[]};
  currentChat=id;
  saveAll();renderSidebar();renderChat();
}
function switchChat(id){
  if(!chats[id])return;
  currentChat=id;
  saveAll();renderSidebar();renderChat();
}
function deleteChat(id,e){
  e.stopPropagation();
  delete chats[id];
  if(currentChat===id){const ids=Object.keys(chats);currentChat=ids.length?ids[0]:null;if(!currentChat)newChat();}
  saveAll();renderSidebar();renderChat();
}

function renderSidebar(){
  const list=document.getElementById('sbList');
  list.innerHTML=Object.entries(chats).reverse().map(([id,c])=>'<div class="sb-item'+(id===currentChat?' active':'')+'" onclick="switchChat(\''+id+'\')"><span class="title">'+escHtml(c.title)+'</span><span class="del" onclick="deleteChat(\''+id+'\',event)">&#10005;</span></div>').join('');
}

function renderChat(){
  const chat=document.getElementById('chat');
  const c=chats[currentChat];
  document.getElementById('chatTitle').textContent=c?c.title:'New Chat';
  if(!c||!c.messages.length){
    chat.innerHTML='<div class="empty" id="emptyState"><div class="lb"><span>Neural</span>Gate</div><p>Ask anything. The smartest model is picked for you.</p></div>';
    return;
  }
  let html='';
  c.messages.forEach((m,i)=>{
    const mid='m-'+currentChat+'-'+i;
    const costStr=m.cached?'cached':m.cost>0?'$'+m.cost.toFixed(5):'free';
    const costCls=m.cached?'cached':m.cost>0?'paid':'free';
    html+='<div class="msg msg-user"><div class="bubble">'+escHtml(m.user)+'</div></div>';
    html+='<div class="msg msg-ai"><div class="ai-meta"><span class="ai-model">'+escHtml(m.model)+'</span><span class="ai-time">'+m.time+'ms</span><span class="ai-cost '+costCls+'">'+costStr+'</span></div><div class="ai-bubble'+(m.cached?' cache-hit':'')+'">'+escHtml(m.ai)+'</div>';
    html+='<div class="ai-actions"><button class="thumb" id="up-'+mid+'" onclick="feedback(\''+mid+'\',\'up\')">&#128077;</button><button class="thumb" id="dn-'+mid+'" onclick="feedback(\''+mid+'\',\'down\')">&#128078;</button>';
    if(m.details)html+='<button class="dbtn" onclick="toggleDet(\''+mid+'\')">Details</button>';
    html+='</div>';
    if(m.details)html+='<div class="dpop" id="det-'+mid+'">'+m.details+'</div>';
    html+='</div>';
  });
  chat.innerHTML=html;
  chat.scrollTop=chat.scrollHeight;
}

// --- Helpers ---
function pre(k){document.getElementById('q').value=P[k];document.getElementById('q').focus()}
function toggleS(){document.getElementById('settings').classList.toggle('open');document.getElementById('gearBtn').classList.toggle('active');loadStats()}
function escHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function showToast(msg,type){const t=document.getElementById('toast');t.textContent=msg;t.className='toast show '+type;setTimeout(()=>t.classList.remove('show'),2500)}
function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px'}
function feedback(id,type){const up=document.getElementById('up-'+id),dn=document.getElementById('dn-'+id);if(!up||!dn)return;if(type==='up'){up.classList.toggle('up-active');dn.classList.remove('down-active')}else{dn.classList.toggle('down-active');up.classList.remove('up-active')}}
function toggleDet(id){const el=document.getElementById('det-'+id);if(el)el.classList.toggle('open')}
function togKey(k){const i=disabledKeys.indexOf(k);if(i>=0)disabledKeys.splice(i,1);else disabledKeys.push(k);localStorage.setItem('ng_disabled',JSON.stringify(disabledKeys));renderToggles()}
function renderToggles(){KEYS.forEach(k=>{const b=document.getElementById('tog_'+k);if(!b)return;if(disabledKeys.includes(k)){b.className='ktog off';b.innerHTML='&#10007;'}else{b.className='ktog on';b.innerHTML='&#10003;'}})}
function updBadges(){document.getElementById('costB').textContent='$'+totalCost.toFixed(4);document.getElementById('queryB').textContent=queryCount+' queries';document.getElementById('cacheB').textContent=cacheHits+' cached'}

async function doLearn(){const r=await fetch('/learn',{method:'POST'});const d=await r.json();let h='Status: '+d.status+'\n';if(d.adjustments){Object.entries(d.adjustments).forEach(([l,i])=>{h+=l.toUpperCase()+': q='+i.weights.quality+' c='+i.weights.cost+'\n'})}else{h+='Need '+(d.min_required||10)+' samples'}document.getElementById('learnOut').textContent=h}
async function clearCacheBtn(){await fetch('/cache/clear',{method:'POST'});cacheHits=0;updBadges();document.getElementById('cacheOut').textContent='Cleared';showToast('Cache cleared','cache')}
async function loadStats(){try{const r=await fetch('/stats');const d=await r.json();let h='Queries: '+(d.total||0)+' | Latency: '+(d.avg_latency_ms||0)+'ms';if(d.cache)h+='\nCache: '+d.cache.entries+'/'+d.cache.max_entries;if(d.model_distribution){h+='\n\nModels:';Object.entries(d.model_distribution).forEach(([m,c])=>{h+='\n '+m+': '+c})}document.getElementById('statsOut').textContent=h}catch(e){}}

// --- Send ---
async function go(){
  const q=document.getElementById('q').value.trim();if(!q)return;
  const c=chats[currentChat];if(!c)return;

  // Update title on first message
  if(!c.messages.length)c.title=q.slice(0,40)+(q.length>40?'...':'');

  // Build history for API
  const history=c.messages.map(m=>({user:m.user,ai:m.ai}));

  // Show user msg immediately
  const chat=document.getElementById('chat');
  const es=document.getElementById('emptyState');if(es)es.style.display='none';
  chat.innerHTML+='<div class="msg msg-user"><div class="bubble">'+escHtml(q)+'</div></div>';
  const lid='ld-'+Date.now();
  chat.innerHTML+='<div class="msg msg-ai" id="'+lid+'"><div class="ai-meta"><span class="ai-model"><span class="spin"></span> Routing...</span></div></div>';
  chat.scrollTop=chat.scrollHeight;

  document.getElementById('q').value='';autoResize(document.getElementById('q'));
  const b=document.getElementById('btn');b.disabled=true;

  try{
    const r=await fetch('/route',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,api_keys:getKeys(),disabled_keys:disabledKeys,history:history})});
    const d=await r.json();
    const le=document.getElementById(lid);if(le)le.remove();

    if(d.blocked){
      c.messages.push({user:q,ai:'[BLOCKED] '+d.block_reason,model:'Safety',time:d.total_ms,cost:0,cached:false,details:''});
    } else {
      queryCount++;totalCost+=(d.est_cost||0);if(d.cached)cacheHits++;updBadges();

      // Build details HTML
      let det='<div class="pipe-mini">';(d.steps||[]).forEach(s=>{det+='<div class="ps"><div class="n">'+s.layer+'</div><div class="v">'+s.ms+'ms</div></div>'});det+='</div>';
      det+='<div style="margin-top:4px;font-size:.65rem;color:var(--t3);white-space:pre-wrap">'+escHtml(d.routing_reason).split(' | ').join('\n')+'</div>';
      if(d.all_scores&&d.all_scores.length){det+='<div style="margin-top:4px">';d.all_scores.forEach(m=>{det+='<div class="dp-m'+(m.model_id===d.model?' sel':'')+'">'+m.display+'<span class="sc">'+m.goodness_score.toFixed(3)+'</span></div>'});det+='</div>'}

      c.messages.push({user:q,ai:d.response,model:d.model_display||d.model,time:d.total_ms,cost:d.est_cost||0,cached:d.cached||false,details:det});

      if(d.learn_result&&d.learn_result.adjustments)showToast('Learning loop triggered','learn');
      if(d.cached)showToast('From cache - zero cost','cache');
    }
    saveAll();renderSidebar();renderChat();
  }catch(e){alert('Error: '+e.message);const le=document.getElementById(lid);if(le)le.remove()}
  finally{b.disabled=false}
}

// --- Init ---
document.addEventListener('DOMContentLoaded',()=>{
  loadKeys();loadDisabled();renderToggles();loadAll();
  const q=document.getElementById('q');
  q.addEventListener('input',()=>autoResize(q));
  q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();go()}});
  document.querySelectorAll('.kinput').forEach(el=>el.addEventListener('change',saveKeys));
});
</script>
</body></html>"""

@app.get("/", response_class=HTMLResponse)
async def ui():
    return UI

if __name__ == "__main__":
    import uvicorn
    print("\n  NeuralGate - AI Routing Intelligence")
    print("  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
