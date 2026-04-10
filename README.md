# NeuralGate — AI Routing Intelligence System

An intelligent routing layer that sits between your application and multiple LLM providers. It decides which model to use for each request based on cost, quality, latency, and risk — then learns from outcomes to get smarter over time.

**18 models. 6 providers. 1 API.**

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Architecture

```
User Query
    |
    v
[Semantic Cache] -----> Cache Hit? Return instantly (0ms, $0)
    |
    v (cache miss)
[Safety Gateway] -----> PII Detection (Aadhaar, PAN, CC, SSN, email, phone)
    |                    Domain Detection (healthcare, finance, legal)
    |                    Risk Scoring + PII Redaction
    v
[Routing Engine] -----> Complexity Classification (simple/medium/complex)
    |                    Dynamic Weight Selection
    |                    Goodness Score: w_q*Quality + w_c*(1-Cost) + w_l*(1-Latency) + w_r*RiskFit
    |                    Budget Check (auto-downgrade to free if over limit)
    v
[Model Executor] -----> 6 providers: Groq, Gemini, Cohere, Together AI, OpenAI, Anthropic
    |                    Retry (2 attempts) + Fallback to next best model
    |                    Conversation history for multi-turn chats
    v
[Validator] ----------> Toxicity scan + Hallucination heuristics + Format check
    |
    v
[Self-Learning Loop] -> Logs every decision to SQLite
                         Auto-analyzes every 10 queries
                         Adjusts routing weights based on outcomes
```

**[View full architecture diagram on Figma](https://www.figma.com/board/2keVVVXKRTqufWombhhSnX/AI-Model-Routing-Platform-Architecture)**

## Models

| Provider | Models | Cost | Key |
|----------|--------|------|-----|
| **Groq** | Llama 3.3 70B, Llama 3.1 8B | Free | [Get key](https://console.groq.com/keys) |
| **Google** | Gemini 2.5 Flash, 3 Flash, 3.1 Pro | Free/Paid | [Get key](https://aistudio.google.com/apikey) |
| **Cohere** | Command R+, Command R | Free | [Get key](https://dashboard.cohere.com/api-keys) |
| **Together AI** | Llama 3.3 70B, Mixtral 8x7B | Free | [Get key](https://api.together.xyz/settings/api-keys) |
| **OpenAI** | GPT-5.4, 5.4-mini, 5.4-nano, 4o | Paid | [Get key](https://platform.openai.com/api-keys) |
| **Anthropic** | Claude Opus 4, Sonnet 4, Haiku 4.5 | Paid | [Get key](https://console.anthropic.com/settings/keys) |

## Features

- **Intelligent Routing** — Goodness Score engine picks the optimal model per query
- **Safety Gateway** — PII detection + redaction for Aadhaar, PAN, credit cards, SSN, email, phone
- **Semantic Caching** — Similar questions return cached answers at zero cost
- **Self-Learning** — Routing weights auto-adjust every 10 queries based on outcomes
- **Budget Auto-Downgrade** — Set daily spending limit; auto-switches to free models when exceeded
- **Multi-Turn Chat** — Conversation history sent to models for contextual responses
- **Chat Sessions** — Multiple conversations with sidebar, persisted in localStorage
- **Thumbs Up/Down** — Rate every response
- **Cost Tracker** — Per-message and running total cost in real-time
- **Provider Toggles** — Enable/disable any provider without deleting keys

## Quick Start

```bash
git clone https://github.com/CodeWithAleem/neuralgate.git
cd neuralgate
pip install fastapi uvicorn pydantic httpx
python main.py
```

Open `http://localhost:8000` → Click ⚙ → Add at least one free API key (Groq recommended) → Ask anything.

## How Routing Works

The **Goodness Score** formula:

```
Score = w_quality * Quality + w_cost * (1-Cost) + w_latency * (1-Latency) + w_risk * RiskFit
```

Weights shift dynamically based on context:

| Context | Quality | Cost | Latency | Risk |
|---------|---------|------|---------|------|
| Simple query | 0.20 | 0.40 | 0.30 | 0.10 |
| Complex query | 0.50 | 0.15 | 0.15 | 0.20 |
| High risk (PII/sensitive) | 0.25 | 0.10 | 0.10 | 0.55 |

The self-learning loop analyzes past routing decisions and adjusts these weights. If simple queries consistently pass validation, cost weight increases (use cheaper models). If complex queries fail, quality weight increases.

## Project Structure

```
neuralgate/
  config.py       — Model registry (18 models, 6 providers)
  safety.py       — PII detection, domain detection, risk scoring, redaction
  router.py       — Complexity classifier + Goodness Score engine
  executor.py     — LLM API calls (all raw HTTP, no SDKs)
  validator.py    — Toxicity + hallucination checks
  learner.py      — SQLite logging + self-learning weight optimization
  cache.py        — Semantic caching with Jaccard similarity
  budget.py       — Daily budget tracking + auto-downgrade
  main.py         — FastAPI server + chat UI
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/route` | POST | Main pipeline — routes query through all layers |
| `/models` | GET | List all available models |
| `/stats` | GET | Usage statistics, cache info, budget status |
| `/learn` | POST | Manually trigger learning loop |
| `/learned-weights` | GET | View current learned routing weights |
| `/budget?limit=N` | POST | Set daily budget limit |
| `/cache/clear` | POST | Clear semantic cache |

## Roadmap

- [ ] Response streaming (Server-Sent Events for word-by-word display)
- [ ] Wire thumbs feedback into learning loop
- [ ] Go/Rust rewrite for sub-millisecond overhead at scale
- [ ] Agent workflow support (multi-step tasks with shared budget)
- [ ] Multi-modal routing (image/audio/video)
- [ ] Edge + cloud hybrid (Ollama for local models)
- [ ] Live benchmarking (A/B testing models on same query)
- [ ] Google sign-in for user accounts
- [ ] Pro version with Stripe payments

## Tech Stack

- **Backend**: Python, FastAPI, SQLite
- **Frontend**: Vanilla HTML/CSS/JS (no framework, single-file)
- **LLM Calls**: Raw HTTP via httpx (no SDKs)
- **Styling**: Custom dark theme, Inter + JetBrains Mono fonts

## License

MIT

---

Built by [Aleem Rayeen](https://github.com/CodeWithAleem) | IIT Bombay
