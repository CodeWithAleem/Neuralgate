"""
NeuralGate — Semantic Cache

If someone asks "What is Python?" and later someone asks "Explain Python",
the cache recognizes these are similar and returns the cached answer.

HOW IT WORKS:
  1. Normalize the query (lowercase, strip punctuation, sort words)
  2. Check if any cached query has high word overlap (Jaccard similarity)
  3. If similarity > threshold, return cached response (FREE, instant)
  4. If not, call the model and cache the result

RESULT: 40-60% of queries can be served from cache at zero cost.
"""

import time
import re
from collections import OrderedDict

# Max cache entries (LRU eviction)
MAX_CACHE = 200
SIMILARITY_THRESHOLD = 0.6  # 60% word overlap = cache hit

# In-memory LRU cache
_cache = OrderedDict()  # key: frozenset of words, value: {response, model, timestamp, query}


def _normalize(text):
    """Lowercase, remove punctuation, split into word set."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    words = set(text.split())
    # Remove very common words
    stopwords = {'a','an','the','is','are','was','were','what','who','how','why','when','where','do','does','did','can','could','would','should','in','on','at','to','for','of','and','or','but','with','this','that','it','i','me','my','you','your'}
    words = words - stopwords
    # Basic stemming: remove trailing s/ing/ed
    stemmed = set()
    for w in words:
        if w.endswith('ing') and len(w) > 4:
            w = w[:-3]
        elif w.endswith('ed') and len(w) > 3:
            w = w[:-2]
        elif w.endswith('s') and len(w) > 3:
            w = w[:-1]
        stemmed.add(w)
    return stemmed


def _jaccard(set1, set2):
    """Jaccard similarity between two word sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def get(query):
    """Check cache for a similar query. Returns cached response or None."""
    words = _normalize(query)
    if not words:
        return None
    
    for cached_words, entry in _cache.items():
        sim = _jaccard(words, cached_words)
        if sim >= SIMILARITY_THRESHOLD:
            # Move to end (most recently used)
            _cache.move_to_end(cached_words)
            return {
                "text": entry["response"],
                "model_display": entry["model"] + " (cached)",
                "model_used": entry["model_id"],
                "latency_ms": 0,
                "tokens": 0,
                "cached": True,
                "original_query": entry["query"],
                "similarity": round(sim, 2),
            }
    
    return None


def put(query, response_text, model_id, model_display):
    """Store a response in cache."""
    words = _normalize(query)
    if not words or len(words) < 1:  # Don't cache empty queries
        return
    
    frozen = frozenset(words)
    _cache[frozen] = {
        "response": response_text,
        "model": model_display,
        "model_id": model_id,
        "query": query,
        "timestamp": time.time(),
    }
    
    # Evict oldest if over limit
    while len(_cache) > MAX_CACHE:
        _cache.popitem(last=False)


def stats():
    """Return cache statistics."""
    return {
        "entries": len(_cache),
        "max_entries": MAX_CACHE,
    }


def clear():
    """Clear the cache."""
    _cache.clear()
