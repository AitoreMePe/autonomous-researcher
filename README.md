# 🔬 AI Research Agent - Local Edition

An autonomous AI research agent that runs **100% locally** using [Ollama](https://ollama.ai/).

No cloud APIs, no subscriptions - just your local GPU and open-source models.

## 🙏 Credits

This project is a **fork** of the original [AI Research Agent](https://github.com/plowsai/researcher) by **plowsai**.

The original project used cloud APIs (Gemini, Claude, Modal). This fork has been modified to run **100% locally** using Ollama.

## 👤 About This Fork

**Fork Author:** Aitoremepe

This is my testing framework for evaluating LLM models and AI agents. I use the **Casas-Alvero Conjecture** as my benchmark problem because:

1. It's a real unsolved mathematical problem (open since 2001)
2. It requires both symbolic reasoning and computational verification
3. It tests the agent's ability to decompose complex problems
4. It reveals the limitations of current AI in mathematical discovery

The conjecture states: *If a monic polynomial P(x) of degree n satisfies gcd(P, P^(k)) ≠ 1 for all k = 1, ..., n-1, then P(x) = (x-r)^n.*

See `papers/casas_alvero_counterexample_search.md` for the latest research results.

---

## ✨ Features

- **100% Local**: Runs entirely on your machine with Ollama
- **Multi-Agent Orchestration**: Coordinates multiple research agents to tackle complex problems  
- **Code Execution**: Automatically writes and executes Python code to test hypotheses
- **Scientific Computing**: Built-in support for NumPy, Pandas, SymPy, SciPy, Matplotlib
- **CUDA Acceleration**: GPU-accelerated polynomial search (RTX 4070 tested)

## 📋 Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- A local LLM model (recommended: `ministral-3:8b` or `qwen2.5-coder:7b`)
- ~8GB VRAM for 7B-8B models
- NVIDIA GPU with CUDA (optional, for accelerated search)

## 🚀 Quick Start

### 1. Install Ollama

Download from https://ollama.ai/ and install.

### 2. Pull a Model

```bash
# Recommended for research tasks
ollama pull ministral-3:8b

# Alternative: good for code generation
ollama pull qwen2.5-coder:7b
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama

```bash
ollama serve
```

### 5. Run the Agent

```bash
# Single agent mode
python main.py "Verify that the quadratic formula correctly solves x^2 - 5x + 6 = 0"

# Orchestrator mode
python main.py "Investigate the Casas-Alvero conjecture for degree 5 polynomials" --mode orchestrator
```

## 🧪 Testing with Casas-Alvero

This is my standard benchmark for testing AI agents:

```bash
# Test counterexample search
python main.py "Search for counterexamples to Casas-Alvero conjecture" --mode orchestrator

# GPU-accelerated massive search
python cuda_search.py
```

## ⚙️ Command Line Options

```
--mode {single,orchestrator}  Execution mode (default: single)
--num-agents N                Number of initial hypotheses (default: 3)
--max-rounds N                Maximum orchestration rounds (default: 3)
--max-parallel N              Maximum parallel experiments (default: 2)
--test-mode                   Run with mock data (no LLM)
```

## 📁 Project Structure

```
gdv/
├── main.py              # CLI entry point
├── agent.py             # Single researcher agent
├── orchestrator.py      # Multi-agent coordinator
├── logger.py            # Rich console output
├── cuda_search.py       # GPU-accelerated polynomial search
├── requirements.txt     # Python dependencies
└── papers/              # Generated research papers
```

## 📊 Current Results (January 2025)

### 🏆 BILLION-SCALE VERIFICATION

GPU-accelerated search on RTX 4070:

| Metric | Value |
|--------|-------|
| **Polynomials tested** | **1,000,000,000** |
| **Degrees tested** | 4, 5, 6, 7 |
| **Coefficient range** | [-100, 100] |
| **Time** | ~4.6 hours |
| **Counterexamples found** | **0** |

This is likely the **largest empirical verification** of the Casas-Alvero Conjecture ever conducted.

See `papers/casas_alvero_counterexample_search.md` for full details.

## 📄 License

MIT License

---

*Original project by [plowsai](https://github.com/plowsai/researcher)*  
*Fork modified for local execution by Aitoremepe - January 2025*
