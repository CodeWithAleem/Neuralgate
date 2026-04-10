"""
STEP 2: SAFETY GATEWAY
=======================
From your Figma: Client App → API Gateway → Safety Decision

This layer inspects EVERY request BEFORE it reaches the router.

WHAT IT DOES:
  1. Scans for PII (Aadhaar, PAN, credit cards, emails, phone numbers)
  2. Detects sensitive domains (healthcare, finance, legal)
  3. Computes a RISK SCORE (0 to 1)
  4. Makes a SAFETY DECISION:
     - Low risk  → route normally
     - High risk → restrict to models that handle sensitive data
     - Too much PII → block the request entirely

WHY IT EXISTS:
  You don't want someone's Aadhaar number sent to an open-source model
  hosted on a random server. Sensitive data must only go to trusted,
  private models. This layer enforces that BEFORE routing happens.

YOUR TWO MODES (from your deck):
  - Manual Mode: predefined policy rules (if finance → private only)
  - Automatic Mode: AI risk scoring based on PII count + domain
"""

import re


# ---- PII Detection Patterns ----

PII_PATTERNS = {
    "aadhaar":     r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "pan_card":    r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone":       r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "ssn":         r"\b\d{3}-?\d{2}-?\d{4}\b",
}

# ---- Sensitive Domain Keywords ----

SENSITIVE_DOMAINS = {
    "healthcare": ["patient", "diagnosis", "medical record", "prescription", "symptoms", "treatment"],
    "finance":    ["bank account", "credit score", "loan", "salary", "tax return", "investment"],
    "legal":      ["lawsuit", "court order", "attorney", "settlement", "contract"],
}


def detect_pii(text: str) -> list:
    """Find all PII in the text. Returns list of {type, masked_value}."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append({
                "type": pii_type,
                "masked": match.group()[:4] + "****",
                "start": match.start(),
                "end": match.end(),
            })
    return found


def detect_domains(text: str) -> list:
    """Which sensitive domains does this query touch?"""
    text_lower = text.lower()
    return [
        domain
        for domain, keywords in SENSITIVE_DOMAINS.items()
        if any(kw in text_lower for kw in keywords)
    ]


def redact(text: str, pii_list: list) -> str:
    """Replace PII with [REDACTED] so models never see raw PII."""
    result = text
    for pii in sorted(pii_list, key=lambda x: x["start"], reverse=True):
        tag = f"[REDACTED_{pii['type'].upper()}]"
        result = result[:pii["start"]] + tag + result[pii["end"]:]
    return result


def compute_risk(pii_count: int, domains: list) -> float:
    """
    Risk score formula:
      risk = pii_factor + domain_factor
      
    Each PII entity adds 0.15 (capped at 0.6)
    Healthcare/finance adds 0.3 each, others 0.2 (capped at 0.6)
    Total capped at 1.0
    """
    pii_factor = min(0.6, pii_count * 0.15)
    
    domain_factor = 0.0
    for d in domains:
        domain_factor += 0.3 if d in ("healthcare", "finance") else 0.2
    domain_factor = min(0.6, domain_factor)
    
    return round(min(1.0, pii_factor + domain_factor), 2)


def inspect(query: str) -> dict:
    """
    Full safety inspection. This is the API Gateway + Safety Decision.
    
    Returns everything the router needs to make a safe decision:
    - pii_found: what PII was detected
    - domains: what sensitive domains
    - risk_score: 0-1
    - safe_query: redacted version to send to models
    - blocked: True if too much PII (threshold: 4+)
    - must_use_private: True if sensitive data requires private models
    """
    pii = detect_pii(query)
    domains = detect_domains(query)
    risk = compute_risk(len(pii), domains)
    safe_query = redact(query, pii) if pii else query
    
    blocked = len(pii) >= 4  # Too many PII entities → block
    must_use_private = risk >= 0.5 or len(domains) > 0
    
    return {
        "pii_found": pii,
        "pii_count": len(pii),
        "domains": domains,
        "risk_score": risk,
        "safe_query": safe_query,
        "blocked": blocked,
        "block_reason": f"Blocked: {len(pii)} PII entities detected (limit: 4)" if blocked else "",
        "must_use_private": must_use_private,
    }
