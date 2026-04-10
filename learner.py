"""
NeuralGate — Self-Learning Loop + Feedback System

Logs every routing decision, collects user feedback (thumbs up/down),
and adjusts routing weights based on both validation outcomes AND human feedback.

The feedback loop:
  1. User thumbs-down a Groq response → model penalty
  2. User thumbs-up a Gemini response → model boost
  3. Learn function analyzes: "For simple queries, Gemini gets more thumbs-up than Groq"
  4. Next routing: quality weight increases for simple queries → Gemini picked more often
"""

import sqlite3
import json
import time
from config import DB_PATH, DEFAULT_WEIGHTS


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL, query_preview TEXT, complexity TEXT,
            complexity_score REAL, risk_score REAL, selected_model TEXT,
            goodness_score REAL, latency_ms INTEGER, tokens INTEGER,
            validation_passed INTEGER, toxicity_score REAL,
            hallucination_score REAL, used_fallback INTEGER, weights_used TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learned_weights (
            complexity TEXT PRIMARY KEY,
            w_quality REAL, w_cost REAL, w_latency REAL, w_risk REAL,
            updated_at REAL, sample_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            model TEXT,
            complexity TEXT,
            vote TEXT,
            query_preview TEXT
        )
    """)
    conn.commit()
    conn.close()


def log(data):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO logs (
            timestamp, query_preview, complexity, complexity_score,
            risk_score, selected_model, goodness_score, latency_ms,
            tokens, validation_passed, toxicity_score, hallucination_score,
            used_fallback, weights_used
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        time.time(), data.get("query", "")[:100],
        data.get("complexity", ""), data.get("complexity_score", 0),
        data.get("risk_score", 0), data.get("model", ""),
        data.get("goodness_score", 0), data.get("latency_ms", 0),
        data.get("tokens", 0), 1 if data.get("validation_passed") else 0,
        data.get("toxicity_score", 0), data.get("hallucination_score", 0),
        1 if data.get("used_fallback") else 0,
        json.dumps(data.get("weights", {})),
    ))
    conn.commit()
    conn.close()


def save_feedback(model, complexity, vote, query=""):
    """Save thumbs up/down feedback."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO feedback (timestamp, model, complexity, vote, query_preview)
        VALUES (?, ?, ?, ?, ?)
    """, (time.time(), model, complexity, vote, query[:100]))
    conn.commit()
    conn.close()
    return {"saved": True, "model": model, "vote": vote}


def learn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    if total < 10:
        conn.close()
        return {"status": "need_more_data", "total_samples": total, "min_required": 10}

    # Get feedback stats per model
    feedback_scores = {}
    try:
        fb_rows = conn.execute("""
            SELECT model, vote, COUNT(*) as cnt FROM feedback GROUP BY model, vote
        """).fetchall()
        for r in fb_rows:
            m = r["model"]
            if m not in feedback_scores:
                feedback_scores[m] = {"up": 0, "down": 0}
            feedback_scores[m][r["vote"]] = r["cnt"]
    except Exception:
        pass

    # Compute feedback ratio (0 to 1, where 1 = all thumbs up)
    model_satisfaction = {}
    for m, scores in feedback_scores.items():
        total_fb = scores["up"] + scores["down"]
        if total_fb > 0:
            model_satisfaction[m] = scores["up"] / total_fb
        else:
            model_satisfaction[m] = 0.5  # neutral

    learned = {}
    for level in ("simple", "medium", "complex"):
        rows = conn.execute("""
            SELECT selected_model, validation_passed, latency_ms,
                   risk_score, weights_used
            FROM logs WHERE complexity = ?
            ORDER BY timestamp DESC LIMIT 100
        """, (level,)).fetchall()

        if len(rows) < 3:
            continue

        pass_rate = sum(r["validation_passed"] for r in rows) / len(rows)
        avg_latency = sum(r["latency_ms"] for r in rows) / len(rows)
        avg_risk = sum(r["risk_score"] for r in rows) / len(rows)

        # Check if users are happy with model choices for this complexity
        models_used = {}
        for r in rows:
            m = r["selected_model"]
            if m not in models_used:
                models_used[m] = 0
            models_used[m] += 1
        
        # Average satisfaction for models used at this complexity level
        avg_satisfaction = 0.5
        sat_count = 0
        for m, count in models_used.items():
            if m in model_satisfaction:
                avg_satisfaction += model_satisfaction[m] * count
                sat_count += count
        if sat_count > 0:
            avg_satisfaction /= sat_count

        w = DEFAULT_WEIGHTS.copy()

        # Validation-based adjustments
        if pass_rate > 0.9:
            w["quality"] = max(0.15, w["quality"] - 0.05)
            w["cost"] = min(0.40, w["cost"] + 0.05)
        elif pass_rate < 0.7:
            w["quality"] = min(0.55, w["quality"] + 0.10)
            w["cost"] = max(0.10, w["cost"] - 0.05)

        # Feedback-based adjustments
        if avg_satisfaction < 0.4:
            # Users unhappy → need better models → boost quality weight
            w["quality"] = min(0.55, w["quality"] + 0.08)
            w["cost"] = max(0.10, w["cost"] - 0.04)
        elif avg_satisfaction > 0.8:
            # Users happy → current routing is good → optimize cost
            w["cost"] = min(0.40, w["cost"] + 0.03)

        if avg_latency > 500:
            w["latency"] = min(0.35, w["latency"] + 0.05)
            w["cost"] = max(0.10, w["cost"] - 0.03)

        if avg_risk > 0.4:
            w["risk"] = min(0.40, w["risk"] + 0.05)

        # Normalize
        total_w = sum(w.values())
        w = {k: round(v / total_w, 3) for k, v in w.items()}

        conn.execute("""
            INSERT OR REPLACE INTO learned_weights
            (complexity, w_quality, w_cost, w_latency, w_risk, updated_at, sample_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (level, w["quality"], w["cost"], w["latency"], w["risk"], time.time(), len(rows)))

        learned[level] = {
            "weights": w,
            "based_on": len(rows),
            "pass_rate": round(pass_rate, 2),
            "avg_latency_ms": round(avg_latency),
            "user_satisfaction": round(avg_satisfaction, 2),
        }

    conn.commit()
    conn.close()
    return {"status": "learned", "total_samples": total, "adjustments": learned, "feedback_data": feedback_scores}


def get_learned_weights():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM learned_weights").fetchall()
    conn.close()
    if not rows:
        return {}
    return {r["complexity"]: {"quality": r["w_quality"], "cost": r["w_cost"], "latency": r["w_latency"], "risk": r["w_risk"]} for r in rows}


def get_recent(limit=20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    if total == 0:
        conn.close()
        return {"total": 0}
    model_dist = {r[0]: r[1] for r in conn.execute("SELECT selected_model, COUNT(*) FROM logs GROUP BY selected_model").fetchall()}
    avgs = conn.execute("SELECT AVG(latency_ms), AVG(risk_score), AVG(CAST(validation_passed AS FLOAT)), AVG(goodness_score) FROM logs").fetchone()
    # Feedback stats
    fb_total = 0
    try:
        fb_total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    except Exception:
        pass
    conn.close()
    return {
        "total": total, "model_distribution": model_dist,
        "avg_latency_ms": round(avgs[0] or 0),
        "avg_risk": round(avgs[1] or 0, 2),
        "validation_pass_rate": round((avgs[2] or 0) * 100, 1),
        "avg_goodness": round(avgs[3] or 0, 3),
        "total_feedback": fb_total,
    }


init_db()
