"""
NeuralGate — Budget Manager

Auto-downgrades to free models when spending exceeds a threshold.

HOW IT WORKS:
  - Tracks cumulative estimated cost across all queries
  - When budget limit is hit, forces router to only use free models
  - Resets daily (or manually)
"""

import time

_state = {
    "total_cost": 0.0,
    "query_count": 0,
    "budget_limit": 1.0,  # Default $1/day
    "last_reset": time.time(),
    "force_free": False,
}


def add_cost(cost):
    """Track a query's cost."""
    # Auto-reset daily
    if time.time() - _state["last_reset"] > 86400:
        _state["total_cost"] = 0.0
        _state["query_count"] = 0
        _state["last_reset"] = time.time()
        _state["force_free"] = False
    
    _state["total_cost"] += cost
    _state["query_count"] += 1
    
    if _state["total_cost"] >= _state["budget_limit"]:
        _state["force_free"] = True


def should_force_free():
    """Should the router only use free models?"""
    # Auto-reset daily
    if time.time() - _state["last_reset"] > 86400:
        _state["total_cost"] = 0.0
        _state["force_free"] = False
        _state["last_reset"] = time.time()
    return _state["force_free"]


def set_limit(amount):
    """Set daily budget limit."""
    _state["budget_limit"] = max(0.01, amount)


def get_status():
    return {
        "total_cost": round(_state["total_cost"], 4),
        "budget_limit": _state["budget_limit"],
        "remaining": round(max(0, _state["budget_limit"] - _state["total_cost"]), 4),
        "force_free": _state["force_free"],
        "queries_today": _state["query_count"],
    }


def reset():
    _state["total_cost"] = 0.0
    _state["query_count"] = 0
    _state["force_free"] = False
    _state["last_reset"] = time.time()
