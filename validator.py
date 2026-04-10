"""
STEP 5: OUTPUT VALIDATION
==========================
From your Figma: Model Execution → Output Validation

After the model responds, we CHECK the output before returning it.

THREE CHECKS:
  1. TOXICITY: Does the response contain harmful content?
  2. HALLUCINATION: Does it make suspiciously confident claims?
  3. FORMAT: Is the response non-empty and well-formed?

WHY: A model might return toxic content, hallucinated stats, or
garbage. Catching it here prevents bad outputs reaching the user.
"""

import re


def check_toxicity(text: str) -> dict:
    """Flag harmful content using keyword patterns."""
    BAD_PATTERNS = [
        r"\b(kill|murder|bomb|weapon|attack)\b",
        r"\b(hate\s+speech|racist|sexist)\b",
        r"\b(hack|exploit|steal|illegal)\b",
    ]
    flags = []
    for pattern in BAD_PATTERNS:
        flags.extend(re.findall(pattern, text.lower()))
    
    score = round(min(1.0, len(flags) * 0.25), 2)
    return {"score": score, "flagged": list(set(flags)), "toxic": score >= 0.5}


def check_hallucination(text: str, query: str) -> dict:
    """Detect signs of hallucination."""
    signals = []
    score = 0.0

    # Suspiciously many specific statistics
    stats = re.findall(r"\b\d{2,3}(\.\d+)?%\b", text)
    if len(stats) > 3:
        signals.append(f"{len(stats)} specific percentages — possible fabrication")
        score += 0.3

    # Absolute confidence language
    for phrase in ["100% guaranteed", "it is certain", "there is no doubt", "always true"]:
        if phrase in text.lower():
            signals.append(f"Overconfident: '{phrase}'")
            score += 0.2

    # Response way too long for a short query
    if len(query.split()) < 10 and len(text.split()) > 300:
        signals.append("Disproportionately long response")
        score += 0.15

    return {"score": round(min(1.0, score), 2), "signals": signals, "suspect": score >= 0.5}


def validate(text: str, query: str) -> dict:
    """Run all checks. Returns overall pass/fail + details."""
    tox = check_toxicity(text)
    hal = check_hallucination(text, query)
    
    # Format check: non-empty?
    format_ok = len(text.strip()) >= 5
    
    passed = not tox["toxic"] and not hal["suspect"] and format_ok
    
    return {
        "passed": passed,
        "toxicity": tox,
        "hallucination": hal,
        "format_ok": format_ok,
    }
