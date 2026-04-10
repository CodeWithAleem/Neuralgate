"""NeuralGate — AI Routing Intelligence System"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import json as jsonlib

from safety import inspect
from router import route
from validator import validate
from executor import execute, execute_stream
from learner import log, learn, get_learned_weights, get_recent, get_stats, save_feedback
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

class FeedbackReq(BaseModel):
    model: str
    complexity: str = "simple"
    vote: str
    query: str = ""

# ---- Regular endpoint (for cache/blocked) ----
@app.post("/route")
async def route_query(req: Query):
    t_start = time.time()
    steps = []
    active_keys = {k: v for k, v in req.api_keys.items() if k not in req.disabled_keys}
    history_dicts = [{"user": h.user, "ai": h.ai} for h in req.history]

    t = time.time()
    cached_result = None
    if not req.history:
        cached_result = cache.get(req.query)
    steps.append({"layer": "Cache", "ms": round((time.time()-t)*1000, 1)})
    if cached_result:
        return {"blocked": False, "response": cached_result["text"], "model": cached_result["model_used"], "model_display": cached_result["model_display"], "model_tier": "cached", "goodness_score": 0, "routing_reason": "Cache hit", "complexity": {"level": "cached", "score": 0, "reasons": ["From cache"]}, "safety": {"pii_count": 0, "pii_found": [], "domains": [], "must_use_private": False, "risk_score": 0}, "all_scores": [], "weights_used": {}, "validation": {"passed": True, "toxicity": {"score": 0}, "hallucination": {"score": 0}}, "latency_ms": 0, "tokens": 0, "est_cost": 0, "used_fallback": False, "steps": steps, "total_ms": round((time.time()-t_start)*1000), "learn_result": None, "cached": True, "budget_status": budget.get_status(), "cache_stats": cache.stats()}

    t = time.time()
    safety = inspect(req.query)
    steps.append({"layer": "Safety", "ms": round((time.time()-t)*1000, 1)})
    if safety["blocked"]:
        return {"blocked": True, "block_reason": safety["block_reason"], "safety": safety, "steps": steps, "total_ms": round((time.time()-t_start)*1000)}

    force_free = budget.should_force_free()
    t = time.time()
    learned = get_learned_weights()
    if force_free:
        free_keys = {k: v for k, v in active_keys.items() if any(MODELS[m]["free"] and MODELS[m]["key_name"] == k for m in MODELS)}
        free_keys[""] = ""
        decision = route(req.query, safety, available_keys=free_keys, learned_weights=learned)
    else:
        decision = route(req.query, safety, available_keys=active_keys, learned_weights=learned)
    steps.append({"layer": "Router", "ms": round((time.time()-t)*1000, 1)})

    t = time.time()
    result = await execute(decision["selected_model"], safety["safe_query"], active_keys, decision["fallback"], history=history_dicts)
    steps.append({"layer": "Executor", "ms": round((time.time()-t)*1000, 1)})

    t = time.time()
    val = validate(result["text"], req.query)
    steps.append({"layer": "Validator", "ms": round((time.time()-t)*1000, 1)})

    tokens_used = result.get("tokens", 0)
    model_meta = MODELS.get(result["model_used"], {})
    est_cost = round(model_meta.get("cost", 0) * tokens_used / 1000, 6)
    budget.add_cost(est_cost)

    if val["passed"] and not result["text"].startswith("[") and not req.history:
        cache.put(req.query, result["text"], result["model_used"], result.get("model_display", ""))

    t = time.time()
    log({"query": req.query[:100], "complexity": decision["complexity"]["level"], "complexity_score": decision["complexity"]["score"], "risk_score": safety["risk_score"], "model": result["model_used"], "goodness_score": decision["goodness_score"], "latency_ms": result["latency_ms"], "tokens": tokens_used, "validation_passed": val["passed"], "toxicity_score": val["toxicity"]["score"], "hallucination_score": val["hallucination"]["score"], "used_fallback": result.get("used_fallback", False), "weights": decision["weights_used"]})
    steps.append({"layer": "Logger", "ms": round((time.time()-t)*1000, 1)})

    stats_data = get_stats()
    learn_result = None
    if stats_data["total"] >= 10 and stats_data["total"] % 10 == 0:
        learn_result = learn()

    return {"blocked": False, "response": result["text"], "model": result["model_used"], "model_display": result.get("model_display", result["model_used"]), "model_tier": decision["tier"], "goodness_score": decision["goodness_score"], "routing_reason": decision["routing_reason"], "complexity": decision["complexity"], "safety": safety, "all_scores": decision["all_scores"], "weights_used": decision["weights_used"], "validation": val, "latency_ms": result["latency_ms"], "tokens": tokens_used, "est_cost": est_cost, "used_fallback": result.get("used_fallback", False), "steps": steps, "total_ms": round((time.time()-t_start)*1000), "learn_result": learn_result, "cached": False, "budget_status": budget.get_status(), "cache_stats": cache.stats()}

# ---- SSE Streaming endpoint ----
@app.post("/route/stream")
async def route_stream(req: Query):
    active_keys = {k: v for k, v in req.api_keys.items() if k not in req.disabled_keys}
    history_dicts = [{"user": h.user, "ai": h.ai} for h in req.history]
    safety = inspect(req.query)

    if safety["blocked"]:
        async def bg():
            yield f"data: {jsonlib.dumps({'blocked': True, 'block_reason': safety['block_reason'], 'safety': safety})}\n\n"
        return StreamingResponse(bg(), media_type="text/event-stream")

    if not req.history:
        cached_result = cache.get(req.query)
        if cached_result:
            async def cg():
                yield f"data: {jsonlib.dumps({'cached': True, 'text': cached_result['text'], 'model_display': cached_result['model_display'], 'model': cached_result['model_used']})}\n\n"
            return StreamingResponse(cg(), media_type="text/event-stream")

    force_free = budget.should_force_free()
    learned = get_learned_weights()
    if force_free:
        free_keys = {k: v for k, v in active_keys.items() if any(MODELS[m]["free"] and MODELS[m]["key_name"] == k for m in MODELS)}
        free_keys[""] = ""
        decision = route(req.query, safety, available_keys=free_keys, learned_weights=learned)
    else:
        decision = route(req.query, safety, available_keys=active_keys, learned_weights=learned)

    model_id = decision["selected_model"]
    model_meta = MODELS.get(model_id, {})

    async def sg():
        yield f"data: {jsonlib.dumps({'type': 'meta', 'model': model_id, 'model_display': decision.get('display', model_id), 'model_tier': decision['tier'], 'routing_reason': decision['routing_reason'], 'complexity': decision['complexity'], 'safety': safety, 'all_scores': decision['all_scores'], 'weights_used': decision['weights_used'], 'goodness_score': decision['goodness_score']})}\n\n"
        full_text = ""
        start = time.time()
        try:
            async for chunk in execute_stream(model_id, safety["safe_query"] if safety["pii_found"] else req.query, active_keys, history=history_dicts):
                full_text += chunk
                yield f"data: {jsonlib.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        except Exception as e:
            yield f"data: {jsonlib.dumps({'type': 'chunk', 'text': '[Error: ' + str(e)[:80] + ']'})}\n\n"
        latency = round((time.time() - start) * 1000)
        tokens = len(full_text.split())
        est_cost = round(model_meta.get("cost", 0) * tokens / 1000, 6)
        budget.add_cost(est_cost)
        val = validate(full_text, req.query)
        if val["passed"] and not full_text.startswith("[") and not req.history:
            cache.put(req.query, full_text, model_id, decision.get("display", ""))
        log({"query": req.query[:100], "complexity": decision["complexity"]["level"], "complexity_score": decision["complexity"]["score"], "risk_score": safety["risk_score"], "model": model_id, "goodness_score": decision["goodness_score"], "latency_ms": latency, "tokens": tokens, "validation_passed": val["passed"], "toxicity_score": val["toxicity"]["score"], "hallucination_score": val["hallucination"]["score"], "used_fallback": False, "weights": decision["weights_used"]})
        stats_data = get_stats()
        learn_result = None
        if stats_data["total"] >= 10 and stats_data["total"] % 10 == 0:
            learn_result = learn()
        yield f"data: {jsonlib.dumps({'type': 'done', 'latency_ms': latency, 'tokens': tokens, 'est_cost': est_cost, 'validation': val, 'learn_result': learn_result, 'budget_status': budget.get_status(), 'cache_stats': cache.stats()})}\n\n"
    return StreamingResponse(sg(), media_type="text/event-stream")

@app.get("/models")
async def list_models(): return MODELS
@app.get("/stats")
async def stats():
    s = get_stats(); s["cache"] = cache.stats(); s["budget"] = budget.get_status(); return s
@app.post("/learn")
async def trigger_learn(): return learn()
@app.post("/budget")
async def set_budget(limit: float = 1.0): budget.set_limit(limit); return budget.get_status()
@app.post("/cache/clear")
async def clear_cache(): cache.clear(); return {"cleared": True}
@app.post("/feedback")
async def submit_feedback(req: FeedbackReq): return save_feedback(req.model, req.complexity, req.vote, req.query)

# ---- Premium UI ----
UI = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuralGate</title>
<script async src="https://www.googletagmanager.com/gtag/js?id=G-05CFTDWVS3"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments)}gtag('js',new Date());gtag('config','G-05CFTDWVS3');</script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0a0f;--bg2:#101018;--s1:#16161e;--s2:#1e1e28;--s3:#26263a;
  --b1:#2a2a3d;--b2:#35354d;
  --t1:#eeeef0;--t2:#9d9daa;--t3:#6b6b7a;--t4:#4a4a58;
  --ac:#7c6aef;--ac2:#9b8af7;--acbg:rgba(124,106,239,.06);--acb:rgba(124,106,239,.12);
  --g:#34d399;--gbg:rgba(52,211,153,.06);--gb:rgba(52,211,153,.15);
  --am:#fbbf24;--ambg:rgba(251,191,36,.06);--amb:rgba(251,191,36,.15);
  --r:#f87171;--rbg:rgba(248,113,113,.06);--rb:rgba(248,113,113,.15);
  --cy:#22d3ee;
  --radius:10px;--radius2:14px
}
body{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:var(--bg);color:var(--t1);height:100vh;display:flex;overflow:hidden;-webkit-font-smoothing:antialiased}
.mono{font-family:'JetBrains Mono',monospace}

/* Sidebar */
.sb{width:260px;background:var(--bg2);border-right:1px solid var(--b1);display:flex;flex-direction:column;flex-shrink:0}
.sb-top{padding:16px 16px 12px;border-bottom:1px solid var(--b1)}
.sb-brand{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.sb-icon{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--ac),#5b4bc7);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.8rem;color:#fff}
.sb-name{font-size:.95rem;font-weight:700;letter-spacing:-.02em}
.sb-name span{color:var(--ac2)}
.new-btn{width:100%;padding:9px;border-radius:8px;border:1px dashed var(--b2);background:transparent;color:var(--t2);font-family:inherit;font-size:.78rem;font-weight:500;cursor:pointer;transition:.2s;display:flex;align-items:center;justify-content:center;gap:6px}
.new-btn:hover{background:var(--acbg);border-color:var(--ac);color:var(--ac2)}
.sb-list{flex:1;overflow-y:auto;padding:8px}
.sb-list::-webkit-scrollbar{width:3px}.sb-list::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}
.sb-item{padding:9px 12px;border-radius:8px;cursor:pointer;font-size:.78rem;color:var(--t3);margin-bottom:2px;display:flex;align-items:center;gap:8px;transition:.15s;position:relative}
.sb-item:hover{background:var(--s2);color:var(--t2)}
.sb-item.active{background:var(--acbg);color:var(--ac2);border:1px solid var(--acb)}
.sb-item .ico{font-size:.7rem;opacity:.5}.sb-item .title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sb-item .del{display:none;position:absolute;right:8px;color:var(--t4);font-size:.65rem;padding:2px 5px;border-radius:4px;cursor:pointer}
.sb-item:hover .del{display:block}.sb-item .del:hover{color:var(--r);background:var(--rbg)}
.sb-stats{padding:12px 16px;border-top:1px solid var(--b1);font-size:.65rem;color:var(--t4);line-height:1.6}

/* Main */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative}

/* Header */
.hdr{padding:10px 20px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--b1);backdrop-filter:blur(12px);background:rgba(10,10,15,.8);flex-shrink:0}
.hdr-title{font-size:.82rem;font-weight:600;color:var(--t2)}
.hdr-pills{display:flex;gap:6px;align-items:center}
.pill{font-size:.6rem;padding:4px 10px;border-radius:6px;font-family:'JetBrains Mono',monospace;font-weight:500;letter-spacing:-.01em;border:1px solid;transition:.2s}
.pill.cost{background:var(--gbg);color:var(--g);border-color:var(--gb)}
.pill.cache{background:var(--acbg);color:var(--ac2);border-color:var(--acb)}
.pill.q{background:var(--s1);color:var(--t3);border-color:var(--b1)}
.gear-btn{width:32px;height:32px;border-radius:8px;border:1px solid var(--b1);background:transparent;color:var(--t3);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:.9rem;transition:.2s}
.gear-btn:hover{background:var(--s2);color:var(--t1);border-color:var(--b2)}
.gear-btn.active{background:var(--acbg);color:var(--ac);border-color:var(--ac)}

/* Settings Panel */
.settings{display:none;position:fixed;top:0;right:0;width:360px;height:100vh;background:var(--bg2);border-left:1px solid var(--b1);z-index:100;overflow-y:auto;padding:20px;animation:slideIn .2s ease}
.settings.open{display:block}
@keyframes slideIn{from{transform:translateX(20px);opacity:0}to{transform:translateX(0);opacity:1}}
.settings::-webkit-scrollbar{width:3px}.settings::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}
.s-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.s-hdr h3{font-size:.9rem;font-weight:700;letter-spacing:-.02em}
.s-close{width:28px;height:28px;border-radius:6px;border:1px solid var(--b1);background:transparent;color:var(--t3);cursor:pointer;font-size:.85rem;display:flex;align-items:center;justify-content:center;transition:.15s}
.s-close:hover{background:var(--s2);color:var(--t1)}
.s-group{margin-bottom:16px}
.s-group-title{font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:var(--t4);font-weight:600;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid var(--b1)}
.krow{display:flex;align-items:center;gap:6px;margin-bottom:6px}
.klbl{width:80px;font-size:.72rem;color:var(--t3);flex-shrink:0;font-weight:500}
.klbl .tag{font-size:.5rem;font-weight:700;padding:1px 4px;border-radius:3px;vertical-align:middle;margin-left:2px}
.klbl .tag.free{background:var(--gbg);color:var(--g);border:1px solid var(--gb)}
.klbl .tag.paid{background:var(--ambg);color:var(--am);border:1px solid var(--amb)}
.kinput{flex:1;padding:7px 10px;background:var(--s1);border:1px solid var(--b1);border-radius:6px;color:var(--t1);font-family:'JetBrains Mono',monospace;font-size:.65rem;outline:none;transition:.2s}
.kinput:focus{border-color:var(--ac);box-shadow:0 0 0 2px var(--acb)}
.ktog{width:26px;height:26px;border-radius:6px;border:1px solid var(--b1);background:transparent;cursor:pointer;font-size:.6rem;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:.2s;font-weight:700}
.ktog.on{background:var(--gbg);color:var(--g);border-color:var(--gb)}
.ktog.off{background:var(--rbg);color:var(--r);border-color:var(--rb)}
.khint{font-size:.58rem;color:var(--t4);margin-bottom:8px;padding-left:86px}.khint a{color:var(--ac2);text-decoration:none}.khint a:hover{text-decoration:underline}
.s-btn{width:100%;padding:8px;border-radius:6px;border:1px solid var(--acb);background:var(--acbg);color:var(--ac2);font-family:inherit;font-size:.72rem;font-weight:600;cursor:pointer;transition:.2s;margin-bottom:4px}
.s-btn:hover{background:var(--ac);color:#fff}
.s-btn.danger{border-color:var(--rb);background:var(--rbg);color:var(--r)}.s-btn.danger:hover{background:var(--r);color:#fff}
.s-out{font-size:.6rem;color:var(--t4);margin-top:4px;white-space:pre-wrap;line-height:1.5}
.s-note{font-size:.58rem;color:var(--t4);margin-top:8px;line-height:1.4;padding:8px;background:var(--s1);border-radius:6px}

/* Chat */
.chat{flex:1;overflow-y:auto;padding:20px 24px;max-width:780px;width:100%;margin:0 auto}
.chat::-webkit-scrollbar{width:4px}.chat::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}
.msg{margin-bottom:18px;animation:fadeUp .3s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

/* User message */
.msg-user{display:flex;justify-content:flex-end}
.msg-user .bubble{background:linear-gradient(135deg,var(--ac),#6352d9);color:#fff;padding:10px 16px;border-radius:16px 16px 4px 16px;max-width:65%;font-size:.85rem;line-height:1.5;font-weight:400;box-shadow:0 2px 12px rgba(124,106,239,.15)}

/* AI message */
.msg-ai{display:flex;flex-direction:column;align-items:flex-start;gap:4px}

/* Routing pill */
.route-pill{display:inline-flex;align-items:center;gap:8px;padding:5px 12px;background:var(--s1);border:1px solid var(--b1);border-radius:8px;font-size:.68rem;color:var(--t3)}
.route-pill .dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.route-pill .dot.free{background:var(--g)}.route-pill .dot.paid{background:var(--am)}.route-pill .dot.cached{background:var(--ac)}
.route-pill .model{color:var(--t1);font-weight:600}
.route-pill .meta{color:var(--t4);font-family:'JetBrains Mono',monospace;font-size:.6rem}
.route-pill .cost-tag{padding:1px 6px;border-radius:4px;font-size:.55rem;font-weight:600}
.route-pill .cost-tag.free{background:var(--gbg);color:var(--g)}.route-pill .cost-tag.paid{background:var(--ambg);color:var(--am)}.route-pill .cost-tag.cached{background:var(--acbg);color:var(--ac2)}

/* AI bubble */
.ai-bubble{background:var(--s1);border:1px solid var(--b1);padding:16px 20px;border-radius:4px 16px 16px 16px;max-width:85%;font-size:.85rem;line-height:1.8;white-space:pre-wrap;color:var(--t1);position:relative}
.ai-bubble.cache-hit{border-color:var(--acb);background:linear-gradient(135deg,var(--acbg),var(--s1))}

/* Response actions */
.ai-actions{display:flex;gap:4px;align-items:center;padding-left:4px}
.act-btn{width:26px;height:26px;border-radius:6px;border:1px solid transparent;background:transparent;cursor:pointer;font-size:.75rem;display:flex;align-items:center;justify-content:center;color:var(--t4);transition:.2s}
.act-btn:hover{background:var(--s2);color:var(--t2);border-color:var(--b1)}
.act-btn.up-active{background:var(--gbg);border-color:var(--gb);color:var(--g)}
.act-btn.down-active{background:var(--rbg);border-color:var(--rb);color:var(--r)}
.act-btn.copy-ok{background:var(--gbg);color:var(--g)}
.det-btn{font-size:.62rem;color:var(--t4);background:none;border:none;cursor:pointer;font-family:inherit;padding:4px 8px;border-radius:4px;font-weight:500;transition:.15s}
.det-btn:hover{background:var(--s2);color:var(--t2)}

/* Detail popup */
.dpop{display:none;background:var(--s1);border:1px solid var(--b1);border-radius:10px;padding:14px;margin-top:4px;max-width:85%;font-size:.68rem;animation:fadeUp .2s ease}
.dpop.open{display:block}
.dp-m{display:flex;align-items:center;gap:6px;padding:3px 0;font-size:.65rem;color:var(--t3)}.dp-m.sel{color:var(--g);font-weight:600}.dp-m .sc{margin-left:auto;font-family:'JetBrains Mono',monospace}

/* Blocked */
.blocked-msg{background:var(--rbg);border:1px solid var(--rb);padding:14px 18px;border-radius:var(--radius2);color:var(--r);font-size:.82rem;line-height:1.5}
.ptag{display:inline-block;padding:2px 6px;margin:2px;border-radius:4px;font-size:.62rem;background:var(--rbg);color:var(--r);border:1px solid var(--rb)}

/* Input */
.input-bar{flex-shrink:0;padding:12px 20px 16px;border-top:1px solid var(--b1);background:var(--bg)}
.input-wrap{max-width:780px;margin:0 auto}
.input-box{display:flex;gap:8px;align-items:flex-end;background:var(--s1);border:1px solid var(--b1);border-radius:14px;padding:6px 6px 6px 16px;transition:.2s}
.input-box:focus-within{border-color:var(--ac);box-shadow:0 0 0 3px var(--acb)}
.input-box textarea{flex:1;min-height:36px;max-height:120px;padding:8px 0;background:transparent;border:none;color:var(--t1);font-family:inherit;font-size:.88rem;resize:none;outline:none;line-height:1.4}
.input-box textarea::placeholder{color:var(--t4)}
.send-btn{padding:8px 16px;background:var(--ac);color:#fff;border:none;border-radius:10px;font-family:inherit;font-weight:700;font-size:.8rem;cursor:pointer;transition:.2s;flex-shrink:0}
.send-btn:hover{background:var(--ac2);transform:translateY(-1px)}.send-btn:disabled{opacity:.3;transform:none}
.presets{display:flex;gap:4px;margin-top:8px;flex-wrap:wrap}
.pre-btn{padding:4px 12px;border-radius:20px;border:1px solid var(--b1);background:transparent;color:var(--t3);font-size:.65rem;cursor:pointer;font-family:inherit;font-weight:500;transition:.2s}
.pre-btn:hover{background:var(--s2);color:var(--t1);border-color:var(--b2)}
.clr-btn{padding:4px 12px;border-radius:20px;border:1px solid var(--rb);background:transparent;color:var(--r);font-size:.65rem;cursor:pointer;font-family:inherit;font-weight:500;margin-left:auto;transition:.2s;opacity:.5}
.clr-btn:hover{background:var(--rbg);opacity:1}

/* Toast */
.toast{display:none;position:fixed;bottom:80px;left:50%;transform:translateX(-50%);border-radius:8px;padding:8px 18px;font-size:.72rem;z-index:50;font-weight:500;box-shadow:0 8px 24px rgba(0,0,0,.3);animation:fadeUp .2s ease}
.toast.show{display:block}
.toast.learn{background:var(--acbg);border:1px solid var(--acb);color:var(--ac2)}
.toast.cache{background:var(--gbg);border:1px solid var(--gb);color:var(--g)}

/* Empty state */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:16px;padding:40px}
.empty-icon{width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,var(--ac),#5b4bc7);display:flex;align-items:center;justify-content:center;font-size:1.4rem;color:#fff;box-shadow:0 8px 32px rgba(124,106,239,.2)}
.empty h2{font-size:1.1rem;font-weight:700;color:var(--t2);letter-spacing:-.02em}
.empty p{font-size:.8rem;color:var(--t4);max-width:360px;text-align:center;line-height:1.5}
.empty-prompts{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-top:4px}
.ep{padding:8px 16px;border-radius:10px;border:1px solid var(--b1);background:var(--s1);color:var(--t3);font-size:.75rem;cursor:pointer;font-family:inherit;transition:.2s;max-width:220px;text-align:left;line-height:1.3}
.ep:hover{background:var(--s2);color:var(--t1);border-color:var(--b2);transform:translateY(-1px)}

.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--b2);border-top-color:var(--ac);border-radius:50%;animation:sp .5s linear infinite;margin-right:6px;vertical-align:middle}
@keyframes sp{to{transform:rotate(360deg)}}
@media(max-width:768px){.sb{display:none}.input-box{border-radius:10px}}
</style></head><body>

<!-- Sidebar -->
<div class="sb">
  <div class="sb-top">
    <div class="sb-brand">
      <div class="sb-icon">N</div>
      <div class="sb-name"><span>Neural</span>Gate</div>
    </div>
    <button class="new-btn" onclick="newChat()">+ New conversation</button>
  </div>
  <div class="sb-list" id="sbList"></div>
  <div class="sb-stats" id="sbStats"></div>
</div>

<!-- Main -->
<div class="main">
  <div class="hdr">
    <div class="hdr-title" id="chatTitle">New conversation</div>
    <div class="hdr-pills">
      <div class="pill cost" id="costB">$0.00</div>
      <div class="pill cache" id="cacheB">0 cached</div>
      <div class="pill q" id="queryB">0 queries</div>
      <button class="gear-btn" id="gearBtn" onclick="toggleS()">&#9881;</button>
    </div>
  </div>

  <!-- Settings -->
  <div class="settings" id="settings">
    <div class="s-hdr"><h3>Settings</h3><button class="s-close" onclick="toggleS()">&#10005;</button></div>

    <div class="s-group"><div class="s-group-title">Free Providers</div>
      <div class="krow"><span class="klbl">Groq <span class="tag free">FREE</span></span><input class="kinput" id="key_GROQ_API_KEY" type="password" placeholder="gsk_..."><button class="ktog on" id="tog_GROQ_API_KEY" onclick="togKey('GROQ_API_KEY')">&#10003;</button></div>
      <div class="khint"><a href="https://console.groq.com/keys" target="_blank">Get key</a> &mdash; Llama 3.3 70B, 3.1 8B</div>
      <div class="krow"><span class="klbl">Gemini <span class="tag free">FREE</span></span><input class="kinput" id="key_GEMINI_API_KEY" type="password" placeholder="AIza..."><button class="ktog on" id="tog_GEMINI_API_KEY" onclick="togKey('GEMINI_API_KEY')">&#10003;</button></div>
      <div class="khint"><a href="https://aistudio.google.com/apikey" target="_blank">Get key</a> &mdash; Gemini 2.5 Flash, 3 Flash, 3.1 Pro</div>
      <div class="krow"><span class="klbl">Cohere <span class="tag free">FREE</span></span><input class="kinput" id="key_COHERE_API_KEY" type="password" placeholder="..."><button class="ktog on" id="tog_COHERE_API_KEY" onclick="togKey('COHERE_API_KEY')">&#10003;</button></div>
      <div class="khint"><a href="https://dashboard.cohere.com/api-keys" target="_blank">Get key</a> &mdash; Command A, R+</div>
      <div class="krow"><span class="klbl">Together <span class="tag free">FREE</span></span><input class="kinput" id="key_TOGETHER_API_KEY" type="password" placeholder="..."><button class="ktog on" id="tog_TOGETHER_API_KEY" onclick="togKey('TOGETHER_API_KEY')">&#10003;</button></div>
      <div class="khint"><a href="https://api.together.xyz/settings/api-keys" target="_blank">Get key</a> &mdash; Llama, Mixtral</div>
    </div>

    <div class="s-group"><div class="s-group-title">Paid Providers</div>
      <div class="krow"><span class="klbl">OpenAI <span class="tag paid">PAID</span></span><input class="kinput" id="key_OPENAI_API_KEY" type="password" placeholder="sk-..."><button class="ktog on" id="tog_OPENAI_API_KEY" onclick="togKey('OPENAI_API_KEY')">&#10003;</button></div>
      <div class="khint"><a href="https://platform.openai.com/api-keys" target="_blank">Get key</a> &mdash; GPT-5.4, mini, nano, 4o</div>
      <div class="krow"><span class="klbl">Anthropic <span class="tag paid">PAID</span></span><input class="kinput" id="key_ANTHROPIC_API_KEY" type="password" placeholder="sk-ant-..."><button class="ktog on" id="tog_ANTHROPIC_API_KEY" onclick="togKey('ANTHROPIC_API_KEY')">&#10003;</button></div>
      <div class="khint"><a href="https://console.anthropic.com/settings/keys" target="_blank">Get key</a> &mdash; Claude Opus, Sonnet, Haiku</div>
    </div>

    <div class="s-note">Keys stored in your browser only. Sent directly to each provider. Toggle &#10003;/&#10007; to enable/disable.</div>

    <div class="s-group"><div class="s-group-title">Intelligence</div>
      <button class="s-btn" onclick="doLearn()">Run Learning Loop</button>
      <div class="s-out" id="learnOut"></div>
    </div>

    <div class="s-group"><div class="s-group-title">Cache</div>
      <button class="s-btn danger" onclick="clearCacheBtn()">Clear Semantic Cache</button>
      <div class="s-out" id="cacheOut"></div>
    </div>

    <div class="s-group"><div class="s-group-title">System Stats</div>
      <div class="s-out" id="statsOut">Loading...</div>
    </div>
  </div>

  <!-- Chat -->
  <div class="chat" id="chat">
    <div class="empty" id="emptyState">
      <div class="empty-icon">N</div>
      <h2>What can I help you with?</h2>
      <p>NeuralGate intelligently routes your query to the best AI model based on complexity, cost, and speed.</p>
      <div class="empty-prompts">
        <button class="ep" onclick="pre('complex')">Compare microservices vs monolithic architecture</button>
        <button class="ep" onclick="pre('code')">Write a Python binary search function</button>
        <button class="ep" onclick="pre('simple')">What is the capital of Japan?</button>
        <button class="ep" onclick="pre('health')">Patient has stage 2 hypertension...</button>
      </div>
    </div>
  </div>
  <div class="toast" id="toast"></div>

  <!-- Input -->
  <div class="input-bar">
    <div class="input-wrap">
      <div class="input-box">
        <textarea id="q" placeholder="Ask anything..." rows="1"></textarea>
        <button class="send-btn" id="btn" onclick="go()">Send</button>
      </div>
      <div class="presets">
        <button class="pre-btn" onclick="pre('simple')">Simple</button>
        <button class="pre-btn" onclick="pre('complex')">Complex</button>
        <button class="pre-btn" onclick="pre('pii')">PII Data</button>
        <button class="pre-btn" onclick="pre('health')">Healthcare</button>
        <button class="pre-btn" onclick="pre('code')">Code</button>
        <button class="pre-btn" onclick="window.print()">Export PDF</button>
      </div>
    </div>
  </div>
</div>

<script>
const KEYS=['GROQ_API_KEY','GEMINI_API_KEY','COHERE_API_KEY','TOGETHER_API_KEY','OPENAI_API_KEY','ANTHROPIC_API_KEY'];
const P={simple:"What is the capital of Japan?",complex:"Compare microservices vs monolithic architecture for a fintech startup. Evaluate scalability, team size, deployment complexity. Step by step with pros and cons.",pii:"My Aadhaar is 1234 5678 9012, PAN is ABCDE1234F, email rajesh@example.com. Check my loan eligibility.",health:"Patient has stage 2 hypertension and elevated creatinine. What treatment adjustments?",code:"Write a Python binary search function with type hints. Explain time complexity."};
let chats={},currentChat=null,totalCost=0,queryCount=0,cacheHits=0,mc=0,disabledKeys=[];

function pre(k){document.getElementById('q').value=P[k];document.getElementById('q').focus()}
function toggleS(){document.getElementById('settings').classList.toggle('open');document.getElementById('gearBtn').classList.toggle('active');loadStats()}
function clearChat(){document.getElementById('chat').innerHTML='<div class="empty" id="emptyState"><div class="empty-icon">N</div><h2>What can I help you with?</h2><p>NeuralGate routes your query to the best AI model.</p><div class="empty-prompts"><button class="ep" onclick="pre(\'complex\')">Compare microservices vs monolithic</button><button class="ep" onclick="pre(\'code\')">Write a Python binary search</button><button class="ep" onclick="pre(\'simple\')">Capital of Japan?</button></div></div>';if(chats[currentChat])chats[currentChat].messages=[];totalCost=0;queryCount=0;cacheHits=0;updB();saveAll();renderSB()}
function escH(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function showToast(m,t){const el=document.getElementById('toast');el.textContent=m;el.className='toast show '+t;setTimeout(()=>el.classList.remove('show'),2500)}
function autoR(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px'}

// Storage
function saveAll(){localStorage.setItem('ng_chats',JSON.stringify(chats));localStorage.setItem('ng_current',currentChat)}
function loadAll(){try{chats=JSON.parse(localStorage.getItem('ng_chats')||'{}');currentChat=localStorage.getItem('ng_current')}catch(e){chats={}}if(!currentChat||!chats[currentChat])newChat();else{renderSB();renderChat()}}
function getKeys(){const k={};KEYS.forEach(n=>{const v=document.getElementById('key_'+n).value.trim();if(v)k[n]=v});return k}
function saveKeys(){localStorage.setItem('ng_keys',JSON.stringify(getKeys()))}
function loadKeys(){try{const k=JSON.parse(localStorage.getItem('ng_keys')||'{}');Object.entries(k).forEach(([n,v])=>{const el=document.getElementById('key_'+n);if(el)el.value=v})}catch(e){}}
function loadDK(){try{disabledKeys=JSON.parse(localStorage.getItem('ng_disabled')||'[]')}catch(e){disabledKeys=[]}}
function togKey(k){const i=disabledKeys.indexOf(k);if(i>=0)disabledKeys.splice(i,1);else disabledKeys.push(k);localStorage.setItem('ng_disabled',JSON.stringify(disabledKeys));renderTog()}
function renderTog(){KEYS.forEach(k=>{const b=document.getElementById('tog_'+k);if(!b)return;if(disabledKeys.includes(k)){b.className='ktog off';b.innerHTML='&#10007;'}else{b.className='ktog on';b.innerHTML='&#10003;'}})}
function updB(){document.getElementById('costB').textContent='$'+totalCost.toFixed(2);document.getElementById('queryB').textContent=queryCount+' queries';document.getElementById('cacheB').textContent=cacheHits+' cached'}

// Chat management
function newChat(){const id='c_'+Date.now();chats[id]={title:'New conversation',messages:[]};currentChat=id;saveAll();renderSB();renderChat()}
function switchChat(id){if(!chats[id])return;currentChat=id;saveAll();renderSB();renderChat()}
function deleteChat(id,e){e.stopPropagation();delete chats[id];if(currentChat===id){const ids=Object.keys(chats);currentChat=ids.length?ids[0]:null;if(!currentChat)newChat()}saveAll();renderSB();renderChat()}
function renderSB(){
  const list=document.getElementById('sbList');
  list.innerHTML=Object.entries(chats).reverse().map(([id,c])=>'<div class="sb-item'+(id===currentChat?' active':'')+'" onclick="switchChat(\''+id+'\')"><span class="ico">&#9679;</span><span class="title">'+escH(c.title)+'</span><span class="del" onclick="deleteChat(\''+id+'\',event)">&#10005;</span></div>').join('');
  document.getElementById('sbStats').innerHTML='<div>'+Object.keys(chats).length+' conversations</div>';
}
function renderChat(){
  const chat=document.getElementById('chat');const c=chats[currentChat];
  document.getElementById('chatTitle').textContent=c?c.title:'New conversation';
  if(!c||!c.messages.length){chat.innerHTML='<div class="empty" id="emptyState"><div class="empty-icon">N</div><h2>What can I help you with?</h2><p>NeuralGate routes your query to the best AI model.</p><div class="empty-prompts"><button class="ep" onclick="pre(\'complex\')">Compare microservices vs monolithic</button><button class="ep" onclick="pre(\'code\')">Write a Python binary search</button><button class="ep" onclick="pre(\'simple\')">Capital of Japan?</button></div></div>';return}
  let html='';
  c.messages.forEach((m,i)=>{
    const mid='m-'+currentChat.slice(2)+'-'+i;
    const costStr=m.cached?'cached':m.cost>0?'$'+m.cost.toFixed(5):'free';
    const dotCls=m.cached?'cached':m.cost>0?'paid':'free';
    html+='<div class="msg msg-user"><div class="bubble">'+escH(m.user)+'</div></div>';
    html+='<div class="msg msg-ai">';
    html+='<div class="route-pill"><span class="dot '+dotCls+'"></span><span class="model">'+escH(m.model)+'</span><span class="meta">'+m.time+'ms</span><span class="cost-tag '+dotCls+'">'+costStr+'</span></div>';
    html+='<div class="ai-bubble'+(m.cached?' cache-hit':'')+'">'+escH(m.ai)+'</div>';
    html+='<div class="ai-actions"><button class="act-btn" id="up-'+mid+'" onclick="fb(\''+mid+'\',\'up\',\''+escH(m.model_id||m.model)+'\',\''+escH(m.complexity||'')+'\')">&#128077;</button><button class="act-btn" id="dn-'+mid+'" onclick="fb(\''+mid+'\',\'down\',\''+escH(m.model_id||m.model)+'\',\''+escH(m.complexity||'')+'\')">&#128078;</button><button class="act-btn" onclick="copyMsg(this,'+i+')">Copy</button>';
    if(m.details)html+='<button class="det-btn" onclick="togDet(\''+mid+'\')">Details</button>';
    html+='</div>';
    if(m.details)html+='<div class="dpop" id="det-'+mid+'">'+m.details+'</div>';
    html+='</div>';
  });
  chat.innerHTML=html;chat.scrollTop=chat.scrollHeight;
}

function fb(id,type,model,complexity){
  const up=document.getElementById('up-'+id),dn=document.getElementById('dn-'+id);if(!up||!dn)return;
  if(type==='up'){up.classList.toggle('up-active');dn.classList.remove('down-active')}else{dn.classList.toggle('down-active');up.classList.remove('up-active')}
  fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:model,complexity:complexity,vote:type,query:''})}).catch(()=>{});
}
function togDet(id){const el=document.getElementById('det-'+id);if(el)el.classList.toggle('open')}
function copyMsg(btn,i){const c=chats[currentChat];if(!c||!c.messages[i])return;navigator.clipboard.writeText(c.messages[i].ai);btn.classList.add('copy-ok');setTimeout(()=>btn.classList.remove('copy-ok'),1000)}

async function doLearn(){const r=await fetch('/learn',{method:'POST'});const d=await r.json();let h='Status: '+d.status+'\n';if(d.adjustments){Object.entries(d.adjustments).forEach(([l,i])=>{h+=l+': q='+i.weights.quality+' c='+i.weights.cost+' sat='+i.user_satisfaction+'\n'})}else{h+='Need '+(d.min_required||10)+' samples'}document.getElementById('learnOut').textContent=h}
async function clearCacheBtn(){await fetch('/cache/clear',{method:'POST'});cacheHits=0;updB();showToast('Cache cleared','cache')}
async function loadStats(){try{const r=await fetch('/stats');const d=await r.json();let h='Queries: '+(d.total||0)+' | Latency: '+(d.avg_latency_ms||0)+'ms | Pass: '+(d.validation_pass_rate||0)+'%';if(d.cache)h+='\nCache: '+d.cache.entries+'/'+d.cache.max_entries;if(d.total_feedback)h+='\nFeedback: '+d.total_feedback+' votes';if(d.model_distribution){h+='\n\nModel usage:';Object.entries(d.model_distribution).forEach(([m,c])=>{h+='\n  '+m+': '+c})}document.getElementById('statsOut').textContent=h}catch(e){}}

// ---- STREAMING SEND ----
async function go(){
  const q=document.getElementById('q').value.trim();if(!q)return;
  const c=chats[currentChat];if(!c)return;
  if(!c.messages.length)c.title=q.slice(0,40)+(q.length>40?'...':'');
  const history=c.messages.map(m=>({user:m.user,ai:m.ai}));
  const chat=document.getElementById('chat');
  const es=document.getElementById('emptyState');if(es)es.style.display='none';
  chat.innerHTML+='<div class="msg msg-user"><div class="bubble">'+escH(q)+'</div></div>';
  mc++;const mid='m-'+mc;
  const aiDiv=document.createElement('div');aiDiv.className='msg msg-ai';aiDiv.id='ai-'+mid;
  aiDiv.innerHTML='<div class="route-pill" id="pill-'+mid+'"><span class="dot free"></span><span class="model"><span class="spin"></span>Routing...</span></div><div class="ai-bubble" id="bub-'+mid+'"></div><div class="ai-actions" id="act-'+mid+'" style="display:none"></div><div class="dpop" id="det-'+mid+'"></div>';
  chat.appendChild(aiDiv);chat.scrollTop=chat.scrollHeight;
  document.getElementById('q').value='';autoR(document.getElementById('q'));
  const b=document.getElementById('btn');b.disabled=true;
  let fullText='',modelId='',modelDisplay='',modelTier='',complexity='',routingReason='',allScores=[],goodness=0;
  try{
    const r=await fetch('/route/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,api_keys:getKeys(),disabled_keys:disabledKeys,history:history})});
    const reader=r.body.getReader();const decoder=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=decoder.decode(value,{stream:true});const lines=buf.split('\n');buf=lines.pop()||'';
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        try{const d=JSON.parse(line.slice(6));
          if(d.blocked){document.getElementById('bub-'+mid).innerHTML='<div class="blocked-msg">&#9940; '+escH(d.block_reason)+'</div>';c.messages.push({user:q,ai:'[BLOCKED]',model:'Safety',time:0,cost:0,cached:false,details:''});saveAll();renderSB();break}
          if(d.cached){fullText=d.text;modelDisplay=d.model_display;modelId=d.model;document.getElementById('pill-'+mid).innerHTML='<span class="dot cached"></span><span class="model">'+escH(modelDisplay)+'</span><span class="cost-tag cached">cached</span>';document.getElementById('bub-'+mid).textContent=fullText;document.getElementById('bub-'+mid).classList.add('cache-hit');cacheHits++;updB();showToast('Cache hit','cache');c.messages.push({user:q,ai:fullText,model:modelDisplay,model_id:modelId,complexity:'cached',time:0,cost:0,cached:true,details:''});saveAll();renderSB();break}
          if(d.type==='meta'){modelId=d.model;modelDisplay=d.model_display;modelTier=d.model_tier;routingReason=d.routing_reason||'';allScores=d.all_scores||[];goodness=d.goodness_score||0;complexity=d.complexity?d.complexity.level:'';const dotCls=modelTier==='free'?'free':'paid';document.getElementById('pill-'+mid).innerHTML='<span class="dot '+dotCls+'"></span><span class="model">'+escH(modelDisplay)+'</span><span class="meta" id="latm-'+mid+'"></span>'}
          if(d.type==='chunk'){fullText+=d.text;document.getElementById('bub-'+mid).textContent=fullText;chat.scrollTop=chat.scrollHeight}
          if(d.type==='done'){
            queryCount++;totalCost+=(d.est_cost||0);updB();
            document.getElementById('latm-'+mid).textContent=d.latency_ms+'ms';
            const costStr=d.est_cost>0?'$'+d.est_cost.toFixed(5):'free';const dotCls=d.est_cost>0?'paid':'free';
            document.getElementById('pill-'+mid).innerHTML+='<span class="cost-tag '+dotCls+'">'+costStr+'</span>';
            const actEl=document.getElementById('act-'+mid);actEl.style.display='flex';
            actEl.innerHTML='<button class="act-btn" id="up-'+mid+'" onclick="fb(\''+mid+'\',\'up\',\''+escH(modelId)+'\',\''+escH(complexity)+'\')">&#128077;</button><button class="act-btn" id="dn-'+mid+'" onclick="fb(\''+mid+'\',\'down\',\''+escH(modelId)+'\',\''+escH(complexity)+'\')">&#128078;</button><button class="act-btn" onclick="navigator.clipboard.writeText(document.getElementById(\'bub-'+mid+'\').textContent);this.classList.add(\'copy-ok\');setTimeout(()=>this.classList.remove(\'copy-ok\'),1000)">Copy</button><button class="det-btn" onclick="togDet(\''+mid+'\')">Details</button>';
            let det='<div style="color:var(--t4);font-size:.65rem;white-space:pre-wrap;margin-bottom:6px">'+escH(routingReason).split(' | ').join('\n')+'</div>';
            if(allScores.length){allScores.forEach(m=>{det+='<div class="dp-m'+(m.model_id===modelId?' sel':'')+'">'+m.display+'<span class="sc">'+m.goodness_score.toFixed(3)+'</span></div>'})}
            document.getElementById('det-'+mid).innerHTML=det;
            c.messages.push({user:q,ai:fullText,model:modelDisplay,model_id:modelId,complexity:complexity,time:d.latency_ms,cost:d.est_cost||0,cached:false,details:det});
            saveAll();renderSB();
            if(d.learn_result&&d.learn_result.adjustments)showToast('Learning loop triggered','learn');
          }
        }catch(pe){}
      }
    }
  }catch(e){document.getElementById('bub-'+mid).textContent='[Error: '+e.message+']'}
  finally{b.disabled=false}
}

document.addEventListener('DOMContentLoaded',()=>{
  loadKeys();loadDK();renderTog();loadAll();
  const q=document.getElementById('q');
  q.addEventListener('input',()=>autoR(q));
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
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  NeuralGate - AI Routing Intelligence")
    print(f"  http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
