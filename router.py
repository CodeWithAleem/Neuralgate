"""NeuralGate — Routing Intelligence Engine"""

from config import MODELS, DEFAULT_WEIGHTS

COMPLEX_WORDS = [
    "compare", "evaluate", "analyze", "trade-off", "pros and cons",
    "step by step", "explain why", "design", "architecture", "strategy",
    "comprehensive", "critique", "synthesize", "implications", "algorithm",
    "implement", "multi-step", "reasoning",
]
SIMPLE_WORDS = [
    "what is", "define", "translate", "who is", "when did",
    "yes or no", "how many", "convert", "calculate", "list",
]


def classify_complexity(query):
    q = query.lower()
    score = 0.0
    reasons = []
    words = len(query.split())
    if words > 80:
        score += 0.3; reasons.append(f"Long ({words} words)")
    elif words > 30:
        score += 0.15; reasons.append(f"Medium length ({words} words)")
    else:
        reasons.append(f"Short ({words} words)")
    hits = sum(1 for w in COMPLEX_WORDS if w in q)
    if hits >= 2:
        score += 0.35; reasons.append(f"{hits} complex keywords")
    elif hits == 1:
        score += 0.15; reasons.append("1 complex keyword")
    simple = sum(1 for w in SIMPLE_WORDS if w in q)
    if simple >= 1:
        score -= 0.15; reasons.append(f"{simple} simple keywords")
    if query.count("?") > 2:
        score += 0.2; reasons.append(f"{query.count('?')} sub-questions")
    if any(kw in q for kw in ["code", "function", "```", "implement"]):
        score += 0.2; reasons.append("Code/technical content")
    score = round(max(0.0, min(1.0, score)), 2)
    level = "complex" if score >= 0.45 else "medium" if score >= 0.2 else "simple"
    return {"level": level, "score": score, "reasons": reasons}


def get_weights(complexity_level, risk_score, learned_weights=None):
    if learned_weights and learned_weights.get(complexity_level):
        return learned_weights[complexity_level]
    if risk_score >= 0.5:
        return {"quality": 0.25, "cost": 0.10, "latency": 0.10, "risk": 0.55}
    elif complexity_level == "simple":
        return {"quality": 0.20, "cost": 0.40, "latency": 0.30, "risk": 0.10}
    elif complexity_level == "complex":
        return {"quality": 0.50, "cost": 0.15, "latency": 0.15, "risk": 0.20}
    else:
        return DEFAULT_WEIGHTS.copy()


def score_models(complexity_level, risk_score, must_use_private, available_keys, learned_weights=None):
    """Score models. Only considers models whose API keys are available."""
    weights = get_weights(complexity_level, risk_score, learned_weights)
    results = []

    for model_id, m in MODELS.items():
        # Check if we have the API key for this model
        has_key = m["free"] or m["key_name"] == "" or available_keys.get(m["key_name"], "")
        if not has_key:
            continue  # Skip models we can't call

        eligible = True
        if must_use_private and not m["handles_sensitive"]:
            eligible = False

        quality_score = m["quality"]
        cost_score = 1.0 - m["cost"]
        latency_score = 1.0 - m["latency"]
        risk_fit = 1.0 if (not must_use_private or m["handles_sensitive"]) else 0.1

        goodness = (
            weights["quality"] * quality_score +
            weights["cost"] * cost_score +
            weights["latency"] * latency_score +
            weights["risk"] * risk_fit
        )
        results.append({
            "model_id": model_id,
            "display": m["display"],
            "tier": m["tier"],
            "goodness_score": round(goodness, 3),
            "eligible": eligible,
            "breakdown": {
                "quality": round(quality_score, 2),
                "cost_efficiency": round(cost_score, 2),
                "speed": round(latency_score, 2),
                "risk_fit": round(risk_fit, 2),
            },
            "weights": weights,
        })

    results.sort(key=lambda x: (x["eligible"], x["goodness_score"]), reverse=True)
    return results


def route(query, safety, available_keys=None, learned_weights=None):
    if available_keys is None:
        available_keys = {}
    complexity = classify_complexity(query)
    scores = score_models(
        complexity["level"], safety["risk_score"],
        safety["must_use_private"], available_keys, learned_weights,
    )
    eligible = [s for s in scores if s["eligible"]]
    best = eligible[0] if eligible else (scores[0] if scores else None)

    if not best:
        return {
            "selected_model": "local-mock", "display": "Local Mock",
            "tier": "mock", "goodness_score": 0, "complexity": complexity,
            "all_scores": [], "weights_used": {}, "routing_reason": "No models available",
            "fallback": "",
        }

    reason_parts = [
        f"Complexity: {complexity['level']} ({complexity['score']})",
        f"Risk: {safety['risk_score']}",
    ]
    if safety["risk_score"] >= 0.5:
        reason_parts.append("HIGH RISK → trusted models only")
    elif complexity["level"] == "simple":
        reason_parts.append("SIMPLE → optimized for cost + speed")
    elif complexity["level"] == "complex":
        reason_parts.append("COMPLEX → optimized for quality")
    else:
        reason_parts.append("BALANCED routing")
    reason_parts.append(f"Selected: {best['display']} (score: {best['goodness_score']})")

    return {
        "selected_model": best["model_id"],
        "display": best["display"],
        "tier": best["tier"],
        "goodness_score": best["goodness_score"],
        "complexity": complexity,
        "all_scores": scores,
        "weights_used": best["weights"],
        "routing_reason": " | ".join(reason_parts),
        "fallback": eligible[1]["model_id"] if len(eligible) > 1 else "",
    }
