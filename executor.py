"""NeuralGate — Model Execution Layer (with streaming + conversation history)"""

import asyncio
import random
import time
import httpx
import json
from config import MODELS


def _build_messages(query, history=None):
    msgs = []
    if history:
        for h in history[-10:]:
            msgs.append({"role": "user", "content": h["user"]})
            msgs.append({"role": "assistant", "content": h["ai"]})
    msgs.append({"role": "user", "content": query})
    return msgs


# ==================== REGULAR (non-streaming) ====================

async def call_groq(query, model_id, api_key, history=None):
    model_name = MODELS[model_id]["api_model"]
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": _build_messages(query, history), "max_tokens": 1024, "temperature": 0.7})
            d = r.json()
            if "error" in d: return {"text": f"[Groq error: {d['error'].get('message','')}]", "tokens": 0, "finish": "error"}
            return {"text": d["choices"][0]["message"]["content"], "tokens": d.get("usage",{}).get("total_tokens",0), "finish": "stop"}
    except Exception as e:
        return {"text": f"[Groq error: {str(e)[:100]}]", "tokens": 0, "finish": "error"}


async def call_gemini(query, model_id, api_key, history=None):
    model_name = MODELS[model_id]["api_model"]
    try:
        contents = []
        if history:
            for h in history[-10:]:
                contents.append({"role": "user", "parts": [{"text": h["user"]}]})
                contents.append({"role": "model", "parts": [{"text": h["ai"]}]})
        contents.append({"role": "user", "parts": [{"text": query}]})
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": contents, "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7}})
            d = r.json()
            if "error" in d: return {"text": f"[Gemini error: {d['error'].get('message','')}]", "tokens": 0, "finish": "error"}
            text = d["candidates"][0]["content"]["parts"][0]["text"]
            tokens = d.get("usageMetadata",{}).get("totalTokenCount",0)
            return {"text": text, "tokens": tokens, "finish": "stop"}
    except Exception as e:
        return {"text": f"[Gemini error: {str(e)[:100]}]", "tokens": 0, "finish": "error"}


async def call_cohere(query, model_id, api_key, history=None):
    model_name = MODELS[model_id]["api_model"]
    try:
        messages = _build_messages(query, history)
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post("https://api.cohere.com/v2/chat",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": messages, "max_tokens": 1024, "temperature": 0.7})
            d = r.json()
            if isinstance(d, dict) and "message" in d and isinstance(d["message"], str) and "error" in d["message"].lower():
                return {"text": f"[Cohere error: {d['message']}]", "tokens": 0, "finish": "error"}
            text = ""
            msg = d.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", [])
                if isinstance(content, list) and len(content) > 0:
                    c_item = content[0]
                    text = c_item.get("text", "") if isinstance(c_item, dict) else str(c_item)
            if not text:
                text = d.get("text", str(d)[:300])
            usage = d.get("usage", {})
            billed = usage.get("billed_units", {}) if isinstance(usage, dict) else {}
            tokens = (billed.get("input_tokens", 0) or 0) + (billed.get("output_tokens", 0) or 0)
            return {"text": text, "tokens": tokens, "finish": "stop"}
    except Exception as e:
        return {"text": f"[Cohere error: {str(e)[:100]}]", "tokens": 0, "finish": "error"}


async def call_together(query, model_id, api_key, history=None):
    model_name = MODELS[model_id]["api_model"]
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post("https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": _build_messages(query, history), "max_tokens": 1024, "temperature": 0.7})
            d = r.json()
            if "error" in d: return {"text": f"[Together error: {d['error'].get('message','')}]", "tokens": 0, "finish": "error"}
            return {"text": d["choices"][0]["message"]["content"], "tokens": d.get("usage",{}).get("total_tokens",0), "finish": "stop"}
    except Exception as e:
        return {"text": f"[Together error: {str(e)[:100]}]", "tokens": 0, "finish": "error"}


async def call_openai(query, model_id, api_key, history=None):
    model_name = MODELS[model_id]["api_model"]
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": _build_messages(query, history), "max_tokens": 1024, "temperature": 0.7})
            d = r.json()
            if "error" in d: return {"text": f"[OpenAI error: {d['error'].get('message','')}]", "tokens": 0, "finish": "error"}
            return {"text": d["choices"][0]["message"]["content"], "tokens": d.get("usage",{}).get("total_tokens",0), "finish": "stop"}
    except Exception as e:
        return {"text": f"[OpenAI error: {str(e)[:100]}]", "tokens": 0, "finish": "error"}


async def call_anthropic(query, model_id, api_key, history=None):
    model_name = MODELS[model_id]["api_model"]
    try:
        messages = _build_messages(query, history)
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": model_name, "max_tokens": 1024, "messages": messages})
            d = r.json()
            if "error" in d: return {"text": f"[Claude error: {d['error'].get('message','')}]", "tokens": 0, "finish": "error"}
            text = d["content"][0]["text"]
            tokens = d.get("usage",{}).get("input_tokens",0) + d.get("usage",{}).get("output_tokens",0)
            return {"text": text, "tokens": tokens, "finish": "stop"}
    except Exception as e:
        return {"text": f"[Claude error: {str(e)[:100]}]", "tokens": 0, "finish": "error"}


async def call_mock(query, model_id, api_key, history=None):
    await asyncio.sleep(random.uniform(0.1, 0.3))
    if history:
        text = f"Continuing conversation (turn {len(history)+1}). Regarding your question about: {query[:80]}"
    elif len(query.split()) > 20:
        text = "This is a detailed analysis.\n\n1. Multiple considerations involved.\n2. Balanced approach recommended.\n3. Start focused, iterate.\n\nCareful evaluation of constraints is key."
    else:
        text = "Based on available information: this is well-understood. Focus on fundamentals and build from there."
    return {"text": text, "tokens": len(text.split()), "finish": "stop"}


PROVIDERS = {
    "groq": call_groq, "gemini": call_gemini, "cohere": call_cohere,
    "together": call_together, "openai": call_openai, "anthropic": call_anthropic, "mock": call_mock,
}


async def execute(model_id, query, api_keys, fallback_id="", history=None):
    meta = MODELS.get(model_id)
    if not meta:
        return {"text": f"Unknown model: {model_id}", "model_used": model_id, "model_display": model_id, "latency_ms": 0, "tokens": 0, "used_fallback": False}
    provider = PROVIDERS.get(meta["provider"])
    key = api_keys.get(meta["key_name"], "") if meta["key_name"] else ""
    start = time.time()
    for attempt in range(2):
        result = await provider(query, model_id, key, history)
        if result["finish"] != "error":
            return {"text": result["text"], "model_used": model_id, "model_display": meta["display"],
                    "latency_ms": round((time.time()-start)*1000), "tokens": result["tokens"], "used_fallback": False}
    if fallback_id and fallback_id in MODELS:
        fb = MODELS[fallback_id]
        fb_key = api_keys.get(fb["key_name"], "") if fb["key_name"] else ""
        result = await PROVIDERS[fb["provider"]](query, fallback_id, fb_key, history)
        return {"text": result["text"], "model_used": fallback_id, "model_display": fb["display"],
                "latency_ms": round((time.time()-start)*1000), "tokens": result["tokens"], "used_fallback": True}
    return {"text": "[All models failed. Check API keys in Settings.]",
            "model_used": model_id, "model_display": meta["display"],
            "latency_ms": round((time.time()-start)*1000), "tokens": 0, "used_fallback": False}


# ==================== STREAMING ====================

async def stream_openai_compatible(url, model_name, api_key, query, history=None, auth_header="Authorization", auth_prefix="Bearer "):
    """Stream from any OpenAI-compatible API (Groq, Together, OpenAI)."""
    headers = {auth_header: f"{auth_prefix}{api_key}", "Content-Type": "application/json"}
    body = {"model": model_name, "messages": _build_messages(query, history), "max_tokens": 1024, "temperature": 0.7, "stream": True}
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            async with c.stream("POST", url, headers=headers, json=body) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            text = delta.get("content", "")
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        yield f"[Stream error: {str(e)[:80]}]"


async def stream_gemini(model_name, api_key, query, history=None):
    """Stream from Gemini API."""
    contents = []
    if history:
        for h in history[-10:]:
            contents.append({"role": "user", "parts": [{"text": h["user"]}]})
            contents.append({"role": "model", "parts": [{"text": h["ai"]}]})
    contents.append({"role": "user", "parts": [{"text": query}]})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse&key={api_key}"
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            async with c.stream("POST", url, headers={"Content-Type": "application/json"},
                json={"contents": contents, "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7}}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            parts = chunk.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            for p in parts:
                                t = p.get("text", "")
                                if t:
                                    yield t
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        yield f"[Gemini stream error: {str(e)[:80]}]"


async def stream_anthropic(model_name, api_key, query, history=None):
    """Stream from Anthropic API."""
    messages = _build_messages(query, history)
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            async with c.stream("POST", "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": model_name, "max_tokens": 1024, "messages": messages, "stream": True}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            if chunk.get("type") == "content_block_delta":
                                text = chunk.get("delta", {}).get("text", "")
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        yield f"[Claude stream error: {str(e)[:80]}]"


async def stream_mock(query, history=None):
    """Simulate streaming for mock provider."""
    if history:
        text = f"Continuing conversation (turn {len(history)+1}). Regarding: {query[:60]}"
    else:
        text = "Based on available information, this is a well-understood topic. The key insight is to focus on fundamentals, evaluate trade-offs carefully, and iterate based on real-world feedback."
    for word in text.split():
        yield word + " "
        await asyncio.sleep(0.05)


async def execute_stream(model_id, query, api_keys, history=None):
    """Stream response from the selected model. Yields text chunks."""
    meta = MODELS.get(model_id)
    if not meta:
        yield "[Unknown model]"
        return

    key = api_keys.get(meta["key_name"], "") if meta["key_name"] else ""
    provider = meta["provider"]

    if provider == "groq":
        async for chunk in stream_openai_compatible(
            "https://api.groq.com/openai/v1/chat/completions",
            meta["api_model"], key, query, history):
            yield chunk
    elif provider == "together":
        async for chunk in stream_openai_compatible(
            "https://api.together.xyz/v1/chat/completions",
            meta["api_model"], key, query, history):
            yield chunk
    elif provider == "openai":
        async for chunk in stream_openai_compatible(
            "https://api.openai.com/v1/chat/completions",
            meta["api_model"], key, query, history):
            yield chunk
    elif provider == "gemini":
        async for chunk in stream_gemini(meta["api_model"], key, query, history):
            yield chunk
    elif provider == "anthropic":
        async for chunk in stream_anthropic(meta["api_model"], key, query, history):
            yield chunk
    elif provider == "cohere":
        # Cohere streaming is complex, fall back to regular call
        result = await call_cohere(query, model_id, key, history)
        yield result["text"]
    elif provider == "mock":
        async for chunk in stream_mock(query, history):
            yield chunk
    else:
        yield "[No streaming support for this provider]"
