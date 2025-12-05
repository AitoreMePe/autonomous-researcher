"""
AI Research Agent - Ollama Local Version

This module implements a single research agent that can run experiments
and verify hypotheses using local Python execution and Ollama LLMs.
"""

import os
import sys
import json
import subprocess
import tempfile
import re
from typing import Optional, List

import requests

from logger import print_panel, print_status, log_step, logger


def extract_code_blocks(text: str) -> list:
    """Extract valid Python code blocks from markdown-style text."""
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    valid_blocks = []
    for m in matches:
        code = m.strip()
        if not code:
            continue
        first_line = code.split('\n')[0].strip()
        if (first_line.startswith(('import ', 'from ', 'def ', 'class ', '#', 'x ', 'a ', 'P ')) or 
            '=' in first_line or 
            first_line.startswith('print(') or
            first_line.startswith('sp.') or
            first_line.startswith('sympy.')):
            valid_blocks.append(code)
    return valid_blocks


def emit_event(event_type: str, data: dict) -> None:
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


def execute_code_locally(code: str) -> str:
    """Execute Python code locally and return the output."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        proc = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        output = proc.stdout + proc.stderr
        exit_code = proc.returncode
        
        os.unlink(temp_file)
        
        return f"Exit Code: {exit_code}\nOutput:\n{output}"
    except subprocess.TimeoutExpired:
        return "Execution timed out after 5 minutes"
    except Exception as e:
        return f"Execution error: {e}"


def _build_system_prompt(gpu_hint: str) -> str:
    """System-level instructions for the agent."""
    return f"""You are an autonomous research scientist.
Your job is to rigorously verify the user's hypothesis using experiments
run in Python.

You can write and execute Python code to test hypotheses.
Common libraries available: numpy, pandas, sympy, matplotlib, scipy.

Working loop:
1. **Think before acting.** Plan your next step in natural language.
2. **Act with code.** When you need computation, write Python code in ```python blocks.
3. **Observe and update.** Interpret results and decide what to do next.
4. **Finish clearly.** When you have verified or falsified the hypothesis,
   write a conclusion and end with [DONE].

Compute: {gpu_hint}
"""


def run_experiment_loop(hypothesis: str, test_mode: bool = False, model: str = "ollama-local"):
    """Main agent loop using Ollama locally."""
    gpu_hint = "Local GPU/CPU"

    print_panel(f"Hypothesis: {hypothesis}", "Starting Experiment", "bold green")
    log_step("START", f"Hypothesis: {hypothesis}")
    print_status(f"Model: {model}", "info")

    if test_mode:
        _run_test_mode(hypothesis)
        return

    _run_ollama_experiment_loop(hypothesis, gpu_hint)


def _run_test_mode(hypothesis: str):
    """Run in test mode with mock data."""
    import time
    
    print_status("TEST MODE ENABLED: Using mock data.", "bold yellow")
    
    thought = (
        "I need to verify this hypothesis using a Python script.\n"
        "I will create a simple test."
    )
    print_panel(thought, "Agent Thinking", "thought")
    time.sleep(1)
    
    final_report = (
        "## Experiment Report\n\n"
        "We tested the hypothesis: " + hypothesis + "\n\n"
        "### Conclusion\n"
        "The hypothesis was tested in a mock environment.\n"
        "[DONE]"
    )
    print_panel(final_report, "Final Report", "bold green")


def _run_ollama_experiment_loop(hypothesis: str, gpu_hint: str):
    """Run the experiment loop using local Ollama."""
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "ministral-3:8b")
    
    print_status(f"Using Ollama model: {OLLAMA_MODEL}", "info")
    print_status("Running 100% locally", "info")
    
    # Verify Ollama is running
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in response.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            if models:
                OLLAMA_MODEL = models[0]
                print_status(f"Using available model: {OLLAMA_MODEL}", "warning")
            else:
                print_status("No models found. Run: ollama pull ministral-3:8b", "error")
                return
    except requests.exceptions.ConnectionError:
        print_status("Ollama is not running. Start it with: ollama serve", "error")
        return
    
    system_prompt = _build_system_prompt(gpu_hint)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Hypothesis: {hypothesis}"}
    ]
    
    max_steps = 10
    
    for step in range(1, max_steps + 1):
        print_status(f"Step {step}...", "dim")
        
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": True
                },
                timeout=600,
                stream=True
            )
            response.raise_for_status()
            
            response_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            response_text += content
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
            print()
            
        except Exception as e:
            print_status(f"Ollama Error: {e}", "error")
            logger.error(f"Ollama Error: {e}")
            break
        
        print_panel(response_text, "Agent Response", "info")
        log_step("MODEL", response_text)
        emit_event("AGENT_THOUGHT", {"thought": response_text})
        
        messages.append({"role": "assistant", "content": response_text})
        
        # Extract code blocks
        code_blocks = extract_code_blocks(response_text)
        
        if "[DONE]" in response_text and not code_blocks:
            print_status("Agent completed experiment.", "success")
            break
        
        if code_blocks:
            for code in code_blocks:
                print_status("Executing code locally...", "info")
                print_panel(code, "Code", "code")
                emit_event("AGENT_TOOL_CALL", {"tool": "execute_code", "code": code[:200]})
                
                output = execute_code_locally(code)
                
                print_panel(output[:2000], "Execution Result", "result")
                log_step("TOOL_RESULT", output)
                emit_event("AGENT_TOOL_RESULT", {"output": output[:1000]})
                
                messages.append({
                    "role": "user",
                    "content": f"Code execution result:\n```\n{output}\n```\nContinue your analysis."
                })
    
    print_status("Experiment complete.", "success")
    emit_event("AGENT_REPORT", {"status": "complete"})
