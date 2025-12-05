"""
Orchestrator for AI Research Agent - Ollama Local Version

This module coordinates multiple research agents to investigate complex research tasks.
Runs 100% locally using Ollama with models like qwen2.5-coder or ministral.
"""

import os
import sys
import json
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any

import requests

from logger import print_panel, print_status, log_step, logger


# Global orchestrator state
_default_gpu: Optional[str] = None
_default_model: str = "ollama-local"
_experiment_counter: int = 0

# Regex for stripping ANSI escape sequences
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI colour / style escape sequences from terminal output."""
    return ANSI_ESCAPE_RE.sub("", text)


def _clean_transcript_for_llm(transcript: str) -> str:
    """Produce a cleaned transcript suitable for LLM consumption."""
    clean = _strip_ansi(transcript)
    cleaned_lines: List[str] = []
    for line in clean.splitlines():
        if line.startswith("::EVENT::"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def emit_event(event_type: str, data: Dict[str, Any]) -> None:
    """Emit a structured event for the frontend."""
    if not os.environ.get("AI_RESEARCHER_ENABLE_EVENTS"):
        return
    payload = {
        "type": event_type,
        "timestamp": 0,
        "data": data,
    }
    print(f"::EVENT::{json.dumps(payload)}")
    sys.stdout.flush()


def run_orchestrator_loop(
    research_task: str,
    num_initial_agents: int = 3,
    max_rounds: int = 3,
    default_gpu: Optional[str] = None,
    max_parallel_experiments: int = 2,
    test_mode: bool = False,
    model: str = "ollama-local",
) -> None:
    """
    Main orchestrator loop using Ollama locally.

    Args:
        research_task: High-level research question or task to investigate.
        num_initial_agents: How many distinct hypotheses to target in the first wave.
        max_rounds: Soft cap on how many orchestration steps/waves to perform.
        default_gpu: Default GPU string for experiments (informational only).
        max_parallel_experiments: Maximum number of experiments to run in parallel.
        test_mode: If True, runs in test mode with mock data.
        model: LLM model to use (default: "ollama-local").
    """
    global _default_gpu, _default_model
    _default_gpu = default_gpu
    _default_model = model

    print_panel(
        f"Research Task:\n{research_task}",
        "Orchestrator: Starting Research Project",
        "bold magenta",
    )
    log_step("ORCH_START", f"Research Task: {research_task}")
    print_status(
        f"Orchestrator configuration: {num_initial_agents} initial agents, "
        f"up to {max_rounds} rounds.",
        "info",
    )
    print_status(
        f"Default GPU for experiments: {default_gpu or 'CPU-only / none'}",
        "info",
    )
    print_status(
        f"Maximum parallel experiments per wave: {max_parallel_experiments}",
        "info",
    )
    
    if test_mode:
        print_status("TEST MODE ENABLED: Using mock data.", "bold yellow")
        _run_test_mode(research_task, default_gpu, max_parallel_experiments)
        return

    print_status(f"Model: {model}", "info")
    
    _run_ollama_orchestrator_loop(
        research_task=research_task,
        num_initial_agents=num_initial_agents,
        max_rounds=max_rounds,
        default_gpu=default_gpu,
        max_parallel_experiments=max_parallel_experiments,
    )


def _run_test_mode(research_task: str, default_gpu: Optional[str], max_parallel: int):
    """Run in test mode with mock data."""
    import time
    
    mock_hypotheses = [
        f"Hypothesis A: {research_task} can be solved by method X.",
        f"Hypothesis B: {research_task} requires method Y optimization.",
    ]
    
    thought = (
        "I need to decompose the research task into testable hypotheses.\n"
        "I will investigate two main directions."
    )
    print_panel(thought, "Orchestrator Thinking", "thought")
    time.sleep(1)
    
    final_paper = (
        "# Research Report: " + research_task + "\n\n"
        "## Abstract\n"
        "We investigated the research task using a multi-agent approach.\n\n"
        "## Conclusion\n"
        "This is a mock paper generated in Test Mode.\n\n"
        "[DONE]"
    )
    print_panel(final_paper, "Final Paper", "bold green")


def _run_ollama_orchestrator_loop(
    research_task: str,
    num_initial_agents: int,
    max_rounds: int,
    default_gpu: Optional[str],
    max_parallel_experiments: int,
) -> None:
    """Run the orchestrator loop using Ollama locally."""
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "ministral-3:8b")
    
    print_status(f"Using Ollama model: {OLLAMA_MODEL}", "info")
    print_status("Running 100% locally on your GPU", "info")
    
    # Verify Ollama is running
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in response.json().get("models", [])]
        if not models:
            print_status("No models found in Ollama. Run: ollama pull ministral-3:8b", "error")
            return
        if not any(OLLAMA_MODEL in m for m in models):
            OLLAMA_MODEL = models[0]
            print_status(f"Using available model: {OLLAMA_MODEL}", "warning")
    except requests.exceptions.ConnectionError:
        print_status("Ollama is not running. Start it with: ollama serve", "error")
        return
    
    system_prompt = f"""You are a research orchestrator AI that coordinates multiple research agents.

Your task is to decompose a high-level research question into specific, testable hypotheses,
and then coordinate experiments to test each hypothesis.

IMPORTANT - Tool Usage:
When you want to run an experiment, output in this EXACT format:
[RUN_EXPERIMENT]
hypothesis: <the specific hypothesis to test>
[/RUN_EXPERIMENT]

You can launch up to {max_parallel_experiments} experiments at once.
Each experiment will be run by an independent agent with access to Python.

Process:
1. Analyze the research task
2. Decompose into {num_initial_agents} specific hypotheses  
3. Launch experiments using [RUN_EXPERIMENT] blocks
4. Analyze results when they come back
5. Decide if more experiments are needed (up to {max_rounds} rounds)
6. Write a final research paper summarizing findings

When you're done, end with [DONE]

Default compute: {default_gpu or 'local GPU'}
"""

    def chat_ollama(messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream": True,
            "options": {"num_ctx": 16384, "temperature": 0.7}
        }
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat", 
            json=payload, 
            stream=True, 
            timeout=1800
        )
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    full_response += content
                    emit_event("ORCH_THOUGHT_STREAM", {"chunk": content})
        return full_response

    def extract_experiments(text: str) -> List[str]:
        """Extract experiment hypotheses from the text."""
        patterns = [
            r'\[RUN_EXPERIMENT\]\s*hypothesis:\s*(.*?)\s*\[/RUN_EXPERIMENT\]',
            r'\[RUN_EXPERIMENT\]\s*([^[]+?)\s*\[/RUN_EXPERIMENT\]',
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            all_matches.extend(matches)
        
        # Clean and deduplicate
        cleaned = []
        seen = set()
        for m in all_matches:
            clean = ' '.join(m.split())
            if clean and clean not in seen:
                seen.add(clean)
                cleaned.append(clean)
        
        if cleaned:
            print_status(f"Extracted {len(cleaned)} experiments from response", "info")
        return cleaned

    def run_single_experiment(hypothesis: str, exp_id: int) -> Dict[str, Any]:
        """Run a single experiment via subprocess."""
        log_step("ORCH_TOOL", f"Launching experiment {exp_id}: {hypothesis[:80]}...")
        emit_event("ORCH_SPAWN", {"experiment_id": exp_id, "hypothesis": hypothesis})
        
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "main.py"),
            hypothesis,
            "--mode", "single",
            "--model", "ollama-local",
        ]
        
        try:
            env = os.environ.copy()
            env["AI_RESEARCHER_ENABLE_EVENTS"] = "1"
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )
            
            transcript = proc.stdout + proc.stderr
            clean_transcript = _clean_transcript_for_llm(transcript)
            
            return {
                "experiment_id": exp_id,
                "hypothesis": hypothesis,
                "exit_code": proc.returncode,
                "transcript": clean_transcript[:8000],
                "success": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "experiment_id": exp_id,
                "hypothesis": hypothesis,
                "exit_code": -1,
                "transcript": "Experiment timed out after 30 minutes",
                "success": False,
            }
        except Exception as e:
            return {
                "experiment_id": exp_id,
                "hypothesis": hypothesis,
                "exit_code": -1,
                "transcript": f"Error: {str(e)}",
                "success": False,
            }

    # Initial message
    messages = [{
        "role": "user",
        "content": f"""High-level research task:
{research_task}

Begin by analyzing this task and decomposing it into {num_initial_agents} specific, testable hypotheses.
For each hypothesis that needs empirical validation, use [RUN_EXPERIMENT] blocks.
"""
    }]
    
    all_experiments: List[Dict[str, Any]] = []
    max_steps = max(8, max_rounds * 3)
    experiment_id = 0
    
    for step in range(1, max_steps + 1):
        print_status(f"Orchestrator step {step}...", "dim")
        
        try:
            response_text = chat_ollama(messages)
        except Exception as e:
            print_status(f"Ollama Error: {e}", "error")
            logger.error(f"Ollama Error: {e}")
            break
        
        print_panel(response_text, "Orchestrator Response", "info")
        log_step("ORCH_MODEL", response_text)
        
        messages.append({"role": "assistant", "content": response_text})
        
        # Extract experiments
        hypotheses = extract_experiments(response_text)
        
        if hypotheses:
            print_status(f"Found {len(hypotheses)} experiments to run", "info")
            emit_event("ORCH_AGENTS", {"agent_ids": list(range(experiment_id, experiment_id + len(hypotheses)))})
            
            results = []
            with ThreadPoolExecutor(max_workers=max_parallel_experiments) as executor:
                futures = []
                for hyp in hypotheses:
                    experiment_id += 1
                    futures.append(executor.submit(run_single_experiment, hyp, experiment_id))
                
                for future in futures:
                    result = future.result()
                    results.append(result)
                    all_experiments.append(result)
                    
                    print_panel(
                        f"Hypothesis: {result['hypothesis'][:100]}...\n\n"
                        f"Exit code: {result['exit_code']}\n\n"
                        f"Summary:\n{result['transcript'][:500]}...",
                        f"Experiment {result['experiment_id']} Result",
                        "result"
                    )
            
            # Feed results back
            results_summary = "\n\n".join([
                f"**Experiment {r['experiment_id']}:**\n"
                f"Hypothesis: {r['hypothesis']}\n"
                f"Status: {'SUCCESS' if r['success'] else 'FAILED'}\n"
                f"Findings:\n{r['transcript'][:2000]}"
                for r in results
            ])
            
            messages.append({
                "role": "user",
                "content": f"""Experiment results:

{results_summary}

Analyze these results. You can:
1. Run more experiments if needed using [RUN_EXPERIMENT] blocks
2. If you have enough evidence, write your final research paper and end with [DONE]

Current round: {step}/{max_rounds}
"""
            })
            continue
        
        # Check for completion
        if "[DONE]" in response_text and "[RUN_EXPERIMENT]" not in response_text:
            paper_content = response_text.split("[DONE]")[0].strip()
            if paper_content:
                print_panel(paper_content, "Final Research Paper", "bold green")
                emit_event("ORCH_PAPER", {"content": paper_content})
            print_status("Orchestrator completed.", "success")
            return
        
        messages.append({
            "role": "user",
            "content": "Continue with your analysis. If you need to run experiments, use [RUN_EXPERIMENT] blocks. If you're done, write your final paper and end with [DONE]."
        })
    
    # Force final paper
    print_status("Reached maximum steps, generating final paper...", "warning")
    messages.append({
        "role": "user",
        "content": "You've reached the maximum rounds. Please write your final research paper summarizing all findings and end with [DONE]."
    })
    
    try:
        final_response = chat_ollama(messages)
        paper_content = final_response.replace("[DONE]", "").strip()
        print_panel(paper_content, "Final Research Paper", "bold green")
        emit_event("ORCH_PAPER", {"content": paper_content})
    except Exception as e:
        print_status(f"Error generating final paper: {e}", "error")
