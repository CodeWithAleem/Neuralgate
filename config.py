"""NeuralGate — Configuration & Model Registry"""

import os

DB_PATH = "neuralgate.db"

MODELS = {
    # --- FREE: Groq ---
    "groq-llama-3.3-70b": {
        "provider": "groq",
        "api_model": "llama-3.3-70b-versatile",
        "display": "Llama 3.3 70B (Groq)",
        "key_name": "GROQ_API_KEY",
        "cost": 0.0, "latency": 0.15, "quality": 0.88,
        "handles_sensitive": False, "tier": "free", "free": True,
    },
    "groq-llama-3.1-8b": {
        "provider": "groq",
        "api_model": "llama-3.1-8b-instant",
        "display": "Llama 3.1 8B (Groq)",
        "key_name": "GROQ_API_KEY",
        "cost": 0.0, "latency": 0.05, "quality": 0.70,
        "handles_sensitive": False, "tier": "free", "free": True,
    },
    # --- FREE: Google Gemini ---
    "gemini-2.0-flash": {
        "provider": "gemini",
        "api_model": "gemini-2.0-flash",
        "display": "Gemini 2.0 Flash (Google)",
        "key_name": "GEMINI_API_KEY",
        "cost": 0.0, "latency": 0.3, "quality": 0.85,
        "handles_sensitive": True, "tier": "free", "free": True,
    },
   # --- FREE: Cohere ---
    "cohere-command-a": {
        "provider": "cohere",
        "api_model": "command-a-03-2025",
        "display": "Command A (Cohere)",
        "key_name": "COHERE_API_KEY",
        "cost": 0.0, "latency": 0.35, "quality": 0.90,
        "handles_sensitive": True, "tier": "free", "free": True,
    },
    "cohere-command-r-plus": {
        "provider": "cohere",
        "api_model": "command-r-plus-08-2024",
        "display": "Command R+ (Cohere)",
        "key_name": "COHERE_API_KEY",
        "cost": 0.0, "latency": 0.4, "quality": 0.86,
        "handles_sensitive": True, "tier": "free", "free": True,
    },
    # --- FREE CREDITS: Together AI ---
    "together-llama-3.3-70b": {
        "provider": "together",
        "api_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "display": "Llama 3.3 70B (Together)",
        "key_name": "TOGETHER_API_KEY",
        "cost": 0.05, "latency": 0.3, "quality": 0.88,
        "handles_sensitive": False, "tier": "free", "free": True,
    },
    "together-mixtral-8x7b": {
        "provider": "together",
        "api_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "display": "Mixtral 8x7B (Together)",
        "key_name": "TOGETHER_API_KEY",
        "cost": 0.03, "latency": 0.2, "quality": 0.80,
        "handles_sensitive": False, "tier": "free", "free": True,
    },
    # --- PAID: OpenAI ---
    "gpt-4o-mini": {
        "provider": "openai",
        "api_model": "gpt-4o-mini",
        "display": "GPT-4o Mini (OpenAI)",
        "key_name": "OPENAI_API_KEY",
        "cost": 0.3, "latency": 0.4, "quality": 0.85,
        "handles_sensitive": True, "tier": "paid", "free": False,
    },
    "gpt-4o": {
        "provider": "openai",
        "api_model": "gpt-4o",
        "display": "GPT-4o (OpenAI)",
        "key_name": "OPENAI_API_KEY",
        "cost": 0.9, "latency": 0.7, "quality": 0.95,
        "handles_sensitive": True, "tier": "premium", "free": False,
    },
    # --- PAID: Anthropic ---
    "claude-sonnet": {
        "provider": "anthropic",
        "api_model": "claude-sonnet-4-20250514",
        "display": "Claude Sonnet (Anthropic)",
        "key_name": "ANTHROPIC_API_KEY",
        "cost": 0.5, "latency": 0.5, "quality": 0.92,
        "handles_sensitive": True, "tier": "premium", "free": False,
    },
    # --- MOCK ---
    "local-mock": {
        "provider": "mock",
        "api_model": "mock",
        "display": "Local Mock (Testing)",
        "key_name": "",
        "cost": 0.0, "latency": 0.1, "quality": 0.40,
        "handles_sensitive": True, "tier": "mock", "free": True,
    },
}

DEFAULT_WEIGHTS = {
    "quality": 0.35,
    "cost": 0.25,
    "latency": 0.20,
    "risk": 0.20,
}
