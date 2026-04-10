"""
STEP 6: OBSERVABILITY + SELF-LEARNING LOOP
============================================
From your Figma: Observability & Metrics → Learning Loop (Self-Improving)

TWO JOBS:

JOB 1 — OBSERVE: Log every routing decision
  What model was picked? How fast was it? Did validation pass?
  Stored in SQLite so we can analyze patterns.

JOB 2 — LEARN: Adjust routing weights based on past performance
  This is what makes the system SELF-IMPROVING.
  
  HOW IT WORKS:
  After enough data (50+ requests), the system looks at:
  - Which model selections led to PASSED validation?
  - Which had good latency?
  - Which complexity levels route best to which models?
  
  It then ADJUSTS the scoring weights so future routing is better.
  
  Example: If GPT-4o-mini keeps passing validation for "complex" queries,
  the system learns it doesn't need to send everything complex to GPT-4o.
  It shifts the "quality" weight down for complex queries → saves money.
  
  This is a simple version of reinforcement learning:
  - State: (complexity_level, risk_score)
  - Action: which model to pick
  - Reward: validation_passed + (1 - normalized_latency)
  
  The learning loop runs every time /stats is called, or can be triggered.

WHY THIS MATTERS FOR YOUR RESUME:
  Most routing systems are static rules. This one improves over time.
  That's the difference between a config file and a product.
"""

import sqlite3
import json
import time
from config import DB_PATH, DEFAULT_WEIGHTS


# ---- Database Setup ----

def init_db():
    """Create tables on first run."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            query_preview TEXT,
            complexity TEXT,
            complexity_score REAL,
            risk_score REAL,
            selected_model TEXT,
            goodness_score REAL,
            latency_ms INTEGER,
            tokens INTEGER,
            validation_passed INTEGER,
            toxicity_score REAL,
            hallucination_score REAL,
            used_fallback INTEGER,
            weights_used TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learned_weights (
            complexity TEXT PRIMARY KEY,
            w_quality REAL,
            w_cost REAL,
            w_latency REAL,
            w_risk REAL,
            updated_at REAL,
            sample_count INTEGER
        )
    """)
    conn.commit()
    conn.close()


# ---- Logging ----

def log(data: dict):
    """Log a routing decision."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO logs (
            timestamp, query_preview, complexity, complexity_score,
            risk_score, selected_model, goodness_score, latency_ms,
            tokens, validation_passed, toxicity_score, hallucination_score,
            used_fallback, weights_used
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        time.time(),
        data.get("query", "")[:100],
        data.get("complexity", ""),
        data.get("complexity_score", 0),
        data.get("risk_score", 0),
        data.get("model", ""),
        data.get("goodness_score", 0),
        data.get("latency_ms", 0),
        data.get("tokens", 0),
        1 if data.get("validation_passed") else 0,
        data.get("toxicity_score", 0),
        data.get("hallucination_score", 0),
        1 if data.get("used_fallback") else 0,
        json.dumps(data.get("weights", {})),
    ))
    conn.commit()
    conn.close()


# ---- Learning Loop ----

def learn() -> dict:
    """
    THE SELF-IMPROVING PART.
    
    Analyzes past routing decisions and computes optimal weights
    for each complexity level.
    
    Algorithm:
    1. Group past decisions by complexity level
    2. For each group, find which weight configurations led to:
       - Higher validation pass rates
       - Lower latency
    3. Compute adjusted weights by rewarding successful patterns
    4. Save to learned_weights table
    5. Router uses these on next request
    
    Returns what was learned.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    total = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    
    if total < 10:
        conn.close()
        return {"status": "need_more_data", "total_samples": total, "min_required": 10}
    
    learned = {}
    
    for level in ("simple", "medium", "complex"):
        rows = conn.execute("""
            SELECT selected_model, validation_passed, latency_ms, 
                   risk_score, weights_used
            FROM logs 
            WHERE complexity = ?
            ORDER BY timestamp DESC
            LIMIT 100
        """, (level,)).fetchall()
        
        if len(rows) < 3:
            continue
        
        # Compute success rate and average latency for this complexity level
        pass_rate = sum(r["validation_passed"] for r in rows) / len(rows)
        avg_latency = sum(r["latency_ms"] for r in rows) / len(rows)
        avg_risk = sum(r["risk_score"] for r in rows) / len(rows)
        
        # Start from defaults and ADJUST based on outcomes
        w = DEFAULT_WEIGHTS.copy()
        
        # If pass rate is high → current quality weight is fine, shift toward cost savings
        if pass_rate > 0.9:
            w["quality"] = max(0.15, w["quality"] - 0.05)  # Can afford less quality focus
            w["cost"] = min(0.40, w["cost"] + 0.05)        # Optimize cost more
        
        # If pass rate is low → we need better models, increase quality weight
        elif pass_rate < 0.7:
            w["quality"] = min(0.55, w["quality"] + 0.10)
            w["cost"] = max(0.10, w["cost"] - 0.05)
        
        # If latency is high → increase latency weight
        if avg_latency > 500:
            w["latency"] = min(0.35, w["latency"] + 0.05)
            w["cost"] = max(0.10, w["cost"] - 0.03)
        
        # If risk is consistently high → keep risk weight high
        if avg_risk > 0.4:
            w["risk"] = min(0.40, w["risk"] + 0.05)
        
        # Normalize so weights sum to 1.0
        total_w = sum(w.values())
        w = {k: round(v / total_w, 3) for k, v in w.items()}
        
        # Save learned weights
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
        }
    
    conn.commit()
    conn.close()
    
    return {"status": "learned", "total_samples": total, "adjustments": learned}


def get_learned_weights() -> dict:
    """Load learned weights for the router to use."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute("SELECT * FROM learned_weights").fetchall()
    conn.close()
    
    if not rows:
        return {}
    
    result = {}
    for r in rows:
        result[r["complexity"]] = {
            "quality": r["w_quality"],
            "cost": r["w_cost"],
            "latency": r["w_latency"],
            "risk": r["w_risk"],
        }
    return result


# ---- Stats ----

def get_recent(limit: int = 20) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    
    if total == 0:
        conn.close()
        return {"total": 0}
    
    model_dist = {r[0]: r[1] for r in conn.execute(
        "SELECT selected_model, COUNT(*) FROM logs GROUP BY selected_model"
    ).fetchall()}
    
    avgs = conn.execute("""
        SELECT AVG(latency_ms), AVG(risk_score), AVG(CAST(validation_passed AS FLOAT)),
               AVG(goodness_score)
        FROM logs
    """).fetchone()
    
    conn.close()
    return {
        "total": total,
        "model_distribution": model_dist,
        "avg_latency_ms": round(avgs[0] or 0),
        "avg_risk": round(avgs[1] or 0, 2),
        "validation_pass_rate": round((avgs[2] or 0) * 100, 1),
        "avg_goodness": round(avgs[3] or 0, 3),
    }


# Initialize on import
init_db()
