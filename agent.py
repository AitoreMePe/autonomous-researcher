import os
import sys
import threading
import json
from typing import Optional, List

from logger import print_panel, print_status, log_step, logger

# Lazy imports for different backends
_genai_client = None
_anthropic_client = None
_ollama_client = None
_use_local_sandbox = False

# Cache a single sandbox per run so the agent can keep state across tool calls.
_shared_sandbox = None  # For Modal sandbox (when using remote)
_shared_gpu: Optional[str] = None  # Track which GPU the sandbox was created with
_selected_gpu: Optional[str] = None  # User-selected GPU for this run


def _init_ollama(model: str):
    """Initialize Ollama client."""
    global _ollama_client, _use_local_sandbox
    from ollama_client import OllamaClient, test_ollama_connection
    
    if not test_ollama_connection():
        raise ConnectionError(
            "Cannot connect to Ollama. Make sure Ollama is running:\n"
            "  1. Install Ollama: https://ollama.ai\n"
            "  2. Start Ollama: ollama serve\n"
            "  3. Pull a model: ollama pull qwen2.5-coder:14b"
        )
    
    _ollama_client = OllamaClient(model=model)
    _use_local_sandbox = True
    
    # Check if model exists
    if not _ollama_client.check_model_exists(model):
        available = [m.get("name", "") for m in _ollama_client.list_models()]
        raise ValueError(
            f"Model '{model}' not found in Ollama.\n"
            f"Available models: {', '.join(available) if available else 'none'}\n"
            f"Pull it with: ollama pull {model}"
        )
    
    return _ollama_client


def _init_gemini():
    """Initialize Gemini client."""
    global _genai_client
    from google import genai
    _genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _genai_client


def _init_anthropic():
    """Initialize Anthropic client."""
    global _anthropic_client
    import anthropic
    _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def emit_event(event_type: str, data: dict) -> None:
    """Emit a structured event for the frontend."""
    # Only emit structured events when explicitly enabled (e.g. from the web API).
    # This keeps the CLI output clean while still allowing rich UIs to subscribe.
    if not os.environ.get("AI_RESEARCHER_ENABLE_EVENTS"):
        return

    import json
    payload = {
        "type": event_type,
        "timestamp": 0,
        "data": data,
    }
    print(f"::EVENT::{json.dumps(payload)}")
    sys.stdout.flush()


def _build_generation_config(
    *,
    tools: Optional[list] = None,
    system_instruction: Optional[str] = None,
    disable_autofc: bool = False,
):
    """
    Build a GenerateContentConfig that:

    - Enables Gemini "thinking mode" with visible thought summaries.
    - Sets thinking_level=HIGH (recommended for Gemini 3 Pro).
    - Optionally disables automatic function calling so we can control
      when tools run and show thoughts before actions.
    """
    from google.genai import types
    
    thinking_config = types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.HIGH,
        include_thoughts=True,
    )

    config_kwargs = {
        "tools": tools,
        "system_instruction": system_instruction,
        "thinking_config": thinking_config,
    }

    if disable_autofc:
        # Turn off automatic Python function calling so we get function_call
        # parts back and can execute tools manually in our loop.
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )

    return types.GenerateContentConfig(**config_kwargs)


def _get_shared_sandbox(gpu: Optional[str]):
    """Create (once) and return a persistent sandbox for this run."""
    import modal
    
    global _shared_sandbox, _shared_gpu
    if _shared_sandbox is not None:
        # Reuse only if GPU selection matches
        if gpu == _shared_gpu:
            return _shared_sandbox
        _close_shared_sandbox()

    log_step("EXECUTION", "Initializing shared Sandbox...")

    # Define a robust image with common dependencies (built once).
    image = (
        modal.Image.debian_slim()
        .pip_install("numpy", "pandas", "torch", "scikit-learn", "matplotlib")
    )

    # Create a Modal App to associate with the Sandbox
    log_step("EXECUTION", "Looking up Modal App 'agent-sandbox-app'...")
    app = modal.App.lookup("agent-sandbox-app", create_if_missing=True)
    log_step("EXECUTION", "Modal App found/created.")

    # Keep the sandbox alive by running an inert loop; subcommands run via sandbox.exec.
    gpu_msg = f"gpu={gpu}" if gpu else "cpu-only"
    log_step("EXECUTION", f"Creating persistent Sandbox (keep-alive loop, {gpu_msg})...")
    _shared_sandbox = modal.Sandbox.create(
        "bash",
        "-lc",
        "while true; do sleep 3600; done",
        app=app,
        image=image,
        timeout=7200,
        gpu=gpu,
    )
    _shared_gpu = gpu
    log_step("EXECUTION", "Persistent Sandbox ready.")
    return _shared_sandbox


def _close_shared_sandbox():
    """Terminate the shared sandbox if it exists."""
    global _shared_sandbox
    if _shared_sandbox is not None:
        try:
            _shared_sandbox.terminate()
            log_step("EXECUTION", "Persistent Sandbox terminated.")
        except Exception as e:
            log_step("WARNING", f"Failed to terminate sandbox cleanly: {e}")
        _shared_sandbox = None


def execute_in_sandbox(code: str):
    """
    Executes Python code in a sandbox environment.

    When using Ollama (local mode):
    - Executes code locally using the system's Python and GPU
    
    When using Gemini/Claude (remote mode):
    - Uses Modal's remote sandboxes with cloud GPUs

    Behavior:
    - Streams both STDOUT and STDERR to your local CLI *as they are produced*
    - Captures full STDOUT/STDERR buffers and returns them as a string
    """
    global _use_local_sandbox
    
    if _use_local_sandbox:
        return _execute_in_local_sandbox(code)
    else:
        return _execute_in_modal_sandbox(code)


def _execute_in_local_sandbox(code: str):
    """Execute code locally using the local sandbox."""
    from local_sandbox import execute_code_locally
    
    def on_stdout(chunk: str):
        try:
            emit_event("AGENT_STREAM", {"stream": "stdout", "chunk": chunk})
        except Exception:
            pass
    
    def on_stderr(chunk: str):
        try:
            emit_event("AGENT_STREAM", {"stream": "stderr", "chunk": chunk})
        except Exception:
            pass
    
    try:
        result = execute_code_locally(
            code,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )
        return result
    except Exception as e:
        log_step("ERROR", f"Local Execution Failed: {str(e)}")
        return f"Local Execution Failed: {str(e)}"


def _execute_in_modal_sandbox(code: str):
    """
    Executes Python code inside a persistent Modal Sandbox using sandbox.exec.

    Behavior:
    - Starts a long-lived `python -u -` process in the sandbox.
    - Streams both STDOUT and STDERR to your local CLI *as they are produced*,
      similar to running a long training job in Colab.
    - Captures full STDOUT/STDERR buffers and returns them as a string so the
      agent can inspect logs after the run finishes.
    """
    # Lazy import Modal only when needed
    import modal
    from modal.stream_type import StreamType
    
    try:
        sandbox = _get_shared_sandbox(_selected_gpu)

        log_step("EXECUTION", "Launching python exec inside Sandbox...")
        print_panel(code, "Sandbox Code", "code")

        # Use PIPE on both streams so we can capture and stream them ourselves.
        proc = sandbox.exec(
            "python",
            "-u",
            "-",
            stdout=StreamType.PIPE,
            stderr=StreamType.PIPE,
        )

        # Send the code into the sandboxed Python process.
        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.write_eof()
        proc.stdin.drain()  # Flush buffered stdin

        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []

        log_step("EXECUTION", "Streaming stdout/stderr from Sandbox...")

        def _drain_stream(reader, buffer: List[str], is_stderr: bool):
            """Continuously read from a StreamReader and mirror to local stdout/stderr."""
            try:
                for chunk in reader:
                    # Modal returns text lines (with trailing newline preserved).
                    buffer.append(chunk)
                    if is_stderr:
                        print(chunk, end="", file=sys.stderr, flush=True)
                    else:
                        print(chunk, end="", flush=True)

                    # Also emit a structured streaming event for the web UI so it can
                    # render progress bars and logs as they happen, without waiting
                    # for the entire sandbox run to complete.
                    try:
                        emit_event(
                            "AGENT_STREAM",
                            {
                                "stream": "stderr" if is_stderr else "stdout",
                                "chunk": chunk,
                            },
                        )
                    except Exception as e:
                        # Structured events are best-effort only; don't break execution.
                        log_step("WARNING", f"Failed to emit AGENT_STREAM event: {e}")
            except Exception as e:
                # Don't crash the whole tool if streaming fails; just log.
                stream_name = "stderr" if is_stderr else "stdout"
                log_step("WARNING", f"Error while streaming {stream_name}: {e}")

        # Read stdout and stderr concurrently so training logs / progress bars
        # appear in real time regardless of which stream they use.
        stdout_thread = threading.Thread(
            target=_drain_stream, args=(proc.stdout, stdout_chunks, False), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_drain_stream, args=(proc.stderr, stderr_chunks, True), daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to finish.
        log_step("EXECUTION", "Waiting for process exit...")
        exit_code = proc.wait()

        # Make sure we've drained any remaining output.
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

        log_step("EXECUTION", f"Process exited with code {exit_code}")

        stdout_str = "".join(stdout_chunks)
        stderr_str = "".join(stderr_chunks)

        return f"Exit Code: {exit_code}\nSTDOUT:\n{stdout_str}\nSTDERR:\n{stderr_str}"

    except Exception as e:
        log_step("ERROR", f"Sandbox Execution Failed: {str(e)}")
        return f"Sandbox Execution Failed: {str(e)}"


def _build_claude_tool_definition() -> dict:
    """Build the tool definition for Claude's format."""
    return {
        "name": "execute_in_sandbox",
        "description": (
            "Executes Python code inside a persistent Modal Sandbox. "
            "The sandbox has numpy, pandas, torch, scikit-learn, and matplotlib installed. "
            "Returns the exit code, stdout, and stderr from the execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute in the sandbox."
                }
            },
            "required": ["code"]
        }
    }


def _build_system_prompt(gpu_hint: str) -> str:
    """System-level instructions for the Gemini agent."""
    return f"""You are an autonomous research scientist.
Your job is to rigorously verify the user's hypothesis using experiments
run in a Python sandbox.

Tool:
- `execute_in_sandbox(code: str)`: Runs a Python script in a persistent Modal Sandbox.
  - Preinstalled: numpy, pandas, torch, scikit-learn, matplotlib.
  - Compute: Sandbox GPU request for this run: {gpu_hint}.
  - The code runs as a normal Python script; no need to import `modal`.

Working loop:
1. **Think before acting.** Plan your next step in natural language.
   We will show these thoughts in the CLI, so keep them understandable.
2. **Act with tools.** When you need computation, call `execute_in_sandbox`
   with a complete, self-contained script.
3. **Observe and update.** Interpret tool results and decide what to do next.
4. **Finish clearly.** When you have confidently verified or falsified
   the hypothesis, write a short natural-language conclusion and then a
   final line that contains only `[DONE]`.
"""


def run_experiment_loop(hypothesis: str, test_mode: bool = False, model: str = "gemini-3-pro-preview"):
    """Main agent loop using Gemini 3 Pro, Claude Opus 4.5, or Ollama with thinking + manual tool calling."""
    global _use_local_sandbox
    
    gpu_hint = _selected_gpu or "CPU"
    
    # Determine if using Ollama (local mode)
    is_ollama = model.startswith("ollama:")
    
    if is_ollama:
        # Extract model name after "ollama:"
        ollama_model = model[7:]  # Remove "ollama:" prefix
        from local_sandbox import get_gpu_info, get_environment_info
        gpu_hint = get_gpu_info()
        _use_local_sandbox = True
    
    print_panel(f"Hypothesis: {hypothesis}", "Starting Experiment", "bold green")
    log_step("START", f"Hypothesis: {hypothesis}")
    
    # Emit AGENT_START for the frontend to track this agent
    emit_event("AGENT_START", {
        "agent_id": "1",  # Single agent mode uses ID "1"
        "hypothesis": hypothesis,
        "gpu": gpu_hint if is_ollama else _selected_gpu or "any",
    })
    
    if is_ollama:
        print_status(f"Local execution: {gpu_hint}", "info")
        print_status(f"Model: Ollama ({ollama_model})", "info")
    else:
        print_status(f"Sandbox GPU request: {gpu_hint}", "info")
        print_status(f"Model: {model}", "info")

    if test_mode:
        print_status("TEST MODE ENABLED: Using mock data and skipping LLM calls.", "bold yellow")
        import time
        
        # Mock Agent Loop
        
        # Step 1: Thinking
        thought = (
            "I need to verify this hypothesis using a Python script.\n"
            "I will create a synthetic dataset and run a simple regression model.\n"
            "Then I will analyze the coefficients to check the relationship."
        )
        print_panel(thought, "Agent Thinking", "thought")
        log_step("THOUGHT", thought)
        emit_event("AGENT_THOUGHT", {"thought": thought})
        time.sleep(1.5)
        
        # Step 2: Tool Call
        code = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "print('Generating synthetic data...')\n"
            "data = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)})\n"
            "print('Data shape:', data.shape)\n"
            "print('Correlation:', data.corr().iloc[0,1])"
        )
        fn_name = "execute_in_sandbox"
        fn_args = {"code": code}
        
        print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
        log_step("TOOL_CALL", f"{fn_name}({fn_args})")
        emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})
        time.sleep(1)
        
        # Step 3: Tool Result
        result = (
            "Exit Code: 0\n"
            "STDOUT:\n"
            "Generating synthetic data...\n"
            "Data shape: (100, 2)\n"
            "Correlation: 0.042\n"
            "STDERR:\n"
        )
        print_panel(result, "Tool Result", "result")
        log_step("TOOL_RESULT", "Executed")
        emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})
        time.sleep(1.5)
        
        # Step 4: Analysis
        message = (
            "The correlation is very low, which suggests no strong linear relationship.\n"
            "However, since this is mock data, I will conclude based on the hypothesis."
        )
        print_panel(message, "Agent Message", "info")
        log_step("MODEL", message)
        time.sleep(1)
        
        # Step 5: Final Report
        print_status("Generating Final Report...", "bold green")
        final_report = (
            "## Experiment Report\n\n"
            "We tested the hypothesis: " + hypothesis + "\n\n"
            "### Methodology\n"
            "We ran a simulation using synthetic data.\n\n"
            "### Conclusion\n"
            "The hypothesis was tested in a mock environment.\n"
            "[DONE]"
        )
        print_panel(final_report, "Final Report", "bold green")
        emit_event("AGENT_COMPLETE", {"agent_id": "1", "exit_code": 0})
        return

    # Branch based on model selection
    if is_ollama:
        _run_ollama_experiment_loop(hypothesis, ollama_model, gpu_hint)
    elif model == "claude-opus-4-5":
        _run_claude_experiment_loop(hypothesis, gpu_hint)
    else:
        _run_gemini_experiment_loop(hypothesis, gpu_hint)


def _run_claude_experiment_loop(hypothesis: str, gpu_hint: str):
    """Run the experiment loop using Claude Opus 4.5 with extended thinking."""
    import anthropic
    
    print_status("Claude extended thinking enabled", "info")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system_prompt = _build_system_prompt(gpu_hint)
    tool_def = _build_claude_tool_definition()

    # Initial conversation with hypothesis
    messages = [
        {"role": "user", "content": f"Hypothesis: {hypothesis}"}
    ]

    max_steps = 10

    for step in range(1, max_steps + 1):
        print_status(f"Step {step}...", "dim")

        try:
            # Use streaming for Claude with extended thinking enabled
            # We need to track thinking blocks with their signatures for proper history
            thinking_blocks = []  # List of {"thinking": str, "signature": str}
            text_content = []
            tool_use_blocks = []

            with client.messages.stream(
                model="claude-opus-4-5-20251101",
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000
                },
                system=system_prompt,
                tools=[tool_def],
                messages=messages,
            ) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event, 'content_block'):
                                block = event.content_block
                                if hasattr(block, 'type'):
                                    if block.type == 'thinking':
                                        thinking_blocks.append({"thinking": "", "signature": None})
                                    elif block.type == 'text':
                                        text_content.append("")
                                    elif block.type == 'tool_use':
                                        tool_use_blocks.append({
                                            "id": block.id,
                                            "name": block.name,
                                            "input": ""
                                        })
                        elif event.type == 'content_block_delta':
                            if hasattr(event, 'delta'):
                                delta = event.delta
                                if hasattr(delta, 'type'):
                                    if delta.type == 'thinking_delta' and hasattr(delta, 'thinking'):
                                        if thinking_blocks:
                                            thinking_blocks[-1]["thinking"] += delta.thinking
                                            emit_event("AGENT_THOUGHT_STREAM", {"chunk": delta.thinking})
                                    elif delta.type == 'text_delta' and hasattr(delta, 'text'):
                                        if text_content:
                                            text_content[-1] += delta.text
                                    elif delta.type == 'input_json_delta' and hasattr(delta, 'partial_json'):
                                        if tool_use_blocks:
                                            tool_use_blocks[-1]["input"] += delta.partial_json
                                    elif delta.type == 'signature_delta' and hasattr(delta, 'signature'):
                                        # Capture signature for thinking blocks
                                        if thinking_blocks:
                                            if thinking_blocks[-1]["signature"] is None:
                                                thinking_blocks[-1]["signature"] = ""
                                            thinking_blocks[-1]["signature"] += delta.signature

        except Exception as e:
            print_status(f"API Error: {e}", "error")
            logger.error(f"API Error: {e}")
            break

        # Process thinking content
        thinking_texts = [tb["thinking"] for tb in thinking_blocks if tb["thinking"]]
        if thinking_texts:
            joined_thinking = "\n\n".join(thinking_texts)
            if joined_thinking:
                print_panel(joined_thinking, "Agent Thinking", "thought")
                log_step("THOUGHT", joined_thinking)

        # Process text content
        if text_content:
            joined_text = "\n\n".join(t for t in text_content if t)
            if joined_text:
                print_panel(joined_text, "Agent Message", "info")
                log_step("MODEL", joined_text)

        # Check for completion
        combined_text = "\n".join(thinking_texts + text_content)
        if "[DONE]" in combined_text:
            print_status("Agent signaled completion.", "success")
            break

        # Build assistant message for history - include signature for thinking blocks
        assistant_content = []
        for tb in thinking_blocks:
            if tb["thinking"]:
                thinking_block = {"type": "thinking", "thinking": tb["thinking"]}
                if tb["signature"]:
                    thinking_block["signature"] = tb["signature"]
                assistant_content.append(thinking_block)
        for t in text_content:
            if t:
                assistant_content.append({"type": "text", "text": t})

        # Process tool calls
        if not tool_use_blocks:
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            print_status(
                "No tool calls in this step; assuming experiment is complete.", "info"
            )
            break

        # Execute tool calls
        tool_results = []
        for tool_block in tool_use_blocks:
            fn_name = tool_block["name"]
            try:
                fn_args = json.loads(tool_block["input"]) if tool_block["input"] else {}
            except json.JSONDecodeError:
                fn_args = {}

            print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
            log_step("TOOL_CALL", f"{fn_name}({fn_args})")
            emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})

            # Add tool_use to assistant content
            assistant_content.append({
                "type": "tool_use",
                "id": tool_block["id"],
                "name": fn_name,
                "input": fn_args
            })

            if fn_name == "execute_in_sandbox":
                # Check if code contains [DONE] - treat as completion signal
                code_content = fn_args.get("code", "")
                if "[DONE]" in code_content:
                    print_status("Agent signaled completion via tool call.", "success")
                    result = "Task completed."
                else:
                    result = execute_in_sandbox(**fn_args)
                    
                    # Check for execution errors and prompt for correction
                    if "Exit Code: 1" in result or "Error" in result or "Traceback" in result:
                        print_status("Execution error detected - agent will auto-correct", "warning")
                        emit_event("AGENT_ERROR", {"error": result[:500]})
                        result = (
                            f"EXECUTION ERROR:\n{result}\n\n"
                            "Please analyze the error above and fix your code. "
                            "Common issues: syntax errors, missing imports, incorrect indentation. "
                            "Provide corrected code in a new execute_in_sandbox call."
                        )
            else:
                result = (
                    f"Unsupported tool '{fn_name}'. "
                    "Only 'execute_in_sandbox' is available."
                )

            # Truncate long outputs
            if isinstance(result, str) and len(result) > 20000:
                result = (
                    result[:10000]
                    + "\n...[TRUNCATED]...\n"
                    + result[-10000:]
                )

            print_panel(result, "Tool Result", "result")
            log_step("TOOL_RESULT", "Executed")
            emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block["id"],
                "content": result
            })

        # Add assistant message and tool results to history
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    # Final report generation
    try:
        print_status("Generating Final Report...", "bold green")
        messages.append({
            "role": "user",
            "content": (
                "Generate a concise, information-dense report that explains "
                "how you tested the hypothesis, what you observed, and your "
                "final conclusion."
            )
        })

        final_thinking = []
        final_text = []

        with client.messages.stream(
            model="claude-opus-4-5-20251101",
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            system=system_prompt,
            messages=messages,
        ) as stream:
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        if hasattr(event, 'content_block'):
                            block = event.content_block
                            if hasattr(block, 'type'):
                                if block.type == 'thinking':
                                    final_thinking.append("")
                                elif block.type == 'text':
                                    final_text.append("")
                    elif event.type == 'content_block_delta':
                        if hasattr(event, 'delta'):
                            delta = event.delta
                            if hasattr(delta, 'type'):
                                if delta.type == 'thinking_delta' and hasattr(delta, 'thinking'):
                                    if final_thinking:
                                        final_thinking[-1] += delta.thinking
                                        emit_event("AGENT_THOUGHT_STREAM", {"chunk": delta.thinking})
                                elif delta.type == 'text_delta' and hasattr(delta, 'text'):
                                    if final_text:
                                        final_text[-1] += delta.text

        final_report = "\n\n".join(t for t in final_text if t)
        print_panel(final_report, "Final Report", "bold green")
        emit_event("AGENT_COMPLETE", {"agent_id": "1", "exit_code": 0})
    finally:
        _close_shared_sandbox()


def _run_ollama_experiment_loop(hypothesis: str, model: str, gpu_hint: str):
    """Run the experiment loop using a local Ollama model."""
    from ollama_client import OllamaClient, OllamaMessage
    from local_sandbox import cleanup as cleanup_sandbox
    
    print_status(f"Ollama model: {model}", "info")
    print_status(f"Local GPU: {gpu_hint}", "info")
    
    # Initialize Ollama client
    try:
        client = _init_ollama(model)
    except Exception as e:
        print_status(f"Failed to initialize Ollama: {e}", "error")
        logger.error(f"Ollama initialization failed: {e}")
        return
    
    # Build tool definition
    tool_def = {
        "name": "execute_in_sandbox",
        "description": (
            "Executes Python code locally on the user's machine. "
            "The environment has numpy, pandas, torch, scikit-learn, and matplotlib. "
            f"GPU: {gpu_hint}. "
            "Returns the exit code, stdout, and stderr from the execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute."
                }
            },
            "required": ["code"]
        }
    }
    
    system_prompt = _build_system_prompt(gpu_hint)
    
    # Conversation history
    messages = [
        OllamaMessage(role="user", content=f"Hypothesis: {hypothesis}")
    ]
    
    max_steps = 10
    
    try:
        for step in range(1, max_steps + 1):
            print_status(f"Step {step}...", "dim")
            
            try:
                # Accumulate streaming response
                full_content = ""
                full_thinking = ""
                
                for chunk in client.chat(
                    messages=messages,
                    tools=[tool_def],
                    system_prompt=system_prompt,
                    stream=True,
                ):
                    if chunk.content and not chunk.done:
                        # Stream partial content
                        emit_event("AGENT_THOUGHT_STREAM", {"chunk": chunk.content})
                    
                    if chunk.done:
                        full_content = chunk.content
                        full_thinking = chunk.thinking
                        tool_calls = chunk.tool_calls
                        break
                
            except Exception as e:
                print_status(f"Ollama API Error: {e}", "error")
                logger.error(f"Ollama API Error: {e}")
                break
            
            # Show thinking if present
            if full_thinking:
                print_panel(full_thinking, "Agent Thinking", "thought")
                log_step("THOUGHT", full_thinking)
                emit_event("AGENT_THOUGHT", {"thought": full_thinking})
            
            # Extract text content (remove tool_call blocks for display)
            import re
            display_content = re.sub(r'```tool_call\s*\n?.*?\n?```', '', full_content, flags=re.DOTALL).strip()
            
            if display_content:
                print_panel(display_content, "Agent Message", "info")
                log_step("MODEL", display_content)
            
            # Check for completion
            if "[DONE]" in full_content:
                print_status("Agent signaled completion.", "success")
                break
            
            # Add assistant message to history
            messages.append(OllamaMessage(role="assistant", content=full_content))
            
            # Process tool calls
            if not tool_calls:
                print_status(
                    "No tool calls in this step; assuming experiment is complete.", "info"
                )
                break
            
            # Execute each tool call
            for tc in tool_calls:
                fn_name = tc.name
                fn_args = tc.arguments
                
                print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
                log_step("TOOL_CALL", f"{fn_name}({fn_args})")
                emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})
                
                if fn_name == "execute_in_sandbox":
                    # Check if code contains [DONE] - treat as completion signal
                    code_content = fn_args.get("code", "")
                    if "[DONE]" in code_content:
                        print_status("Agent signaled completion via tool call.", "success")
                        result = "Task completed."
                    else:
                        result = execute_in_sandbox(**fn_args)
                        
                        # Check for execution errors and prompt for correction
                        if "Exit Code: 1" in result or "Error" in result or "Traceback" in result:
                            print_status("Execution error detected - agent will auto-correct", "warning")
                            emit_event("AGENT_ERROR", {"error": result[:500]})
                            # Add error context to help the model fix it
                            result = (
                                f"EXECUTION ERROR:\n{result}\n\n"
                                "Please analyze the error above and fix your code. "
                                "Common issues: syntax errors, missing imports, incorrect indentation. "
                                "Provide corrected code in a new execute_in_sandbox call."
                            )
                else:
                    result = f"Unsupported tool '{fn_name}'. Only 'execute_in_sandbox' is available."
                
                # Truncate long outputs
                if isinstance(result, str) and len(result) > 20000:
                    result = result[:10000] + "\n...[TRUNCATED]...\n" + result[-10000:]
                
                print_panel(result, "Tool Result", "result")
                log_step("TOOL_RESULT", "Executed")
                emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})
                
                # Add tool result to history
                messages.append(OllamaMessage(
                    role="user",  # Ollama uses "user" role for tool results
                    content=f"Tool result for {fn_name}:\n{result}"
                ))
        
        # Final report generation
        print_status("Generating Final Report...", "bold green")
        messages.append(OllamaMessage(
            role="user",
            content=(
                "Generate a concise, information-dense report that explains "
                "how you tested the hypothesis, what you observed, and your "
                "final conclusion."
            )
        ))
        
        try:
            final_response = client.chat(
                messages=messages,
                tools=None,  # No tools for final report
                system_prompt=system_prompt,
                stream=False,
            )
            
            final_report = final_response.content
            print_panel(final_report, "Final Report", "bold green")
            emit_event("AGENT_COMPLETE", {"agent_id": "1", "exit_code": 0})
            
        except Exception as e:
            print_status(f"Failed to generate final report: {e}", "error")
            logger.error(f"Failed to generate final report: {e}")
            emit_event("AGENT_COMPLETE", {"agent_id": "1", "exit_code": 1})
    
    finally:
        cleanup_sandbox()


def _run_gemini_experiment_loop(hypothesis: str, gpu_hint: str):
    """Run the experiment loop using Gemini 3 Pro with thinking mode."""
    from google import genai
    from google.genai import types
    
    print_status("Gemini thinking: HIGH (thought summaries visible)", "info")

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    # Expose the sandbox executor as a tool.
    tools = [execute_in_sandbox]
    system_prompt = _build_system_prompt(gpu_hint)

    # Initial conversation: just the hypothesis as a user message.
    history: List[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"Hypothesis: {hypothesis}")],
        )
    ]

    max_steps = 10

    for step in range(1, max_steps + 1):
        print_status(f"Step {step}...", "dim")

        try:
            # Stream the model's response so we can surface thinking and tool calls in real time.
            response_stream = client.models.generate_content_stream(
                model="gemini-3-pro-preview",
                contents=history,
                config=_build_generation_config(
                    tools=tools,
                    system_instruction=system_prompt,
                    disable_autofc=True,  # manual tool loop
                ),
            )
        except Exception as e:
            print_status(f"API Error: {e}", "error")
            logger.error(f"API Error: {e}")
            break

        # Accumulate full response for history and logic
        accumulated_parts = []

        # Track chunks
        for chunk in response_stream:
             if not chunk.candidates:
                 continue
             
             candidate = chunk.candidates[0]
             if not candidate.content or not candidate.content.parts:
                 continue
             
             for part in candidate.content.parts:
                 # 1. Streaming thoughts
                 if getattr(part, "thought", False) and part.text:
                     emit_event("AGENT_THOUGHT_STREAM", {"chunk": part.text})
                 
                 # Add to accumulator
                 accumulated_parts.append(part)

        # Reconstruct the full Content object (merge logic similar to orchestrator)
        merged_parts = []
        current_text_part = None
        current_thought_part = None
        
        for part in accumulated_parts:
            # Handle Function Calls
            if part.function_call:
                if current_text_part:
                    merged_parts.append(current_text_part)
                    current_text_part = None
                if current_thought_part:
                    merged_parts.append(current_thought_part)
                    current_thought_part = None
                merged_parts.append(part)
                continue
                
            # Handle Thoughts
            if getattr(part, "thought", False):
                if current_text_part:
                    merged_parts.append(current_text_part)
                    current_text_part = None
                
                if current_thought_part:
                    current_thought_part.text += part.text
                else:
                    current_thought_part = part
                continue

            # Handle Text
            if part.text:
                if current_thought_part:
                    merged_parts.append(current_thought_part)
                    current_thought_part = None
                    
                if current_text_part:
                    current_text_part.text += part.text
                else:
                    current_text_part = part
                continue
        
        if current_text_part:
            merged_parts.append(current_text_part)
        if current_thought_part:
            merged_parts.append(current_thought_part)

        if not merged_parts:
            print_status("Empty content from model.", "warning")
            break
            
        model_content = types.Content(role="model", parts=merged_parts)

        # IMPORTANT: append the full model message (including thought signatures
        # and function call parts) so the SDK can preserve reasoning state.
        history.append(model_content)

        thoughts: List[str] = []
        messages: List[str] = []
        function_calls = []

        for part in model_content.parts:
            # Thought summaries from thinking mode.
            if getattr(part, "thought", False) and part.text:
                thoughts.append(part.text)

            # Function/tool call parts.
            if part.function_call:
                function_calls.append(part.function_call)

            # Regular assistant text (exclude thought parts so we don't double-print).
            if part.text and not getattr(part, "thought", False):
                messages.append(part.text)

        # 1. Show reasoning before any action.
        if thoughts:
            joined_thoughts = "\n\n".join(thoughts)
            print_panel(joined_thoughts, "Agent Thinking", "thought")
            log_step("THOUGHT", joined_thoughts)

        # 2. Show natural-language messages (plans, explanations, etc.).
        if messages:
            joined_messages = "\n\n".join(messages)
            print_panel(joined_messages, "Agent Message", "info")
            log_step("MODEL", joined_messages)

        combined_text = "\n".join(thoughts + messages)
        if "[DONE]" in combined_text:
            print_status("Agent signaled completion.", "success")
            break

        # If the model didn't call any tools this turn, assume we're done.
        if not function_calls:
            print_status(
                "No tool calls in this step; assuming experiment is complete.", "info"
            )
            break

        # 3. Execute requested tools (currently just execute_in_sandbox).
        for fn_call in function_calls:
            fn_name = fn_call.name
            fn_args = dict(fn_call.args or {})

            print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
            log_step("TOOL_CALL", f"{fn_name}({fn_args})")
            emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})

            if fn_name == "execute_in_sandbox":
                # Check if code contains [DONE] - treat as completion signal
                code_content = fn_args.get("code", "")
                if "[DONE]" in code_content:
                    print_status("Agent signaled completion via tool call.", "success")
                    result = "Task completed."
                else:
                    result = execute_in_sandbox(**fn_args)
                    
                    # Check for execution errors and prompt for correction
                    if "Exit Code: 1" in result or "Error" in result or "Traceback" in result:
                        print_status("Execution error detected - agent will auto-correct", "warning")
                        emit_event("AGENT_ERROR", {"error": result[:500]})
                        result = (
                            f"EXECUTION ERROR:\n{result}\n\n"
                            "Please analyze the error above and fix your code. "
                            "Common issues: syntax errors, missing imports, incorrect indentation. "
                            "Provide corrected code in a new execute_in_sandbox call."
                        )
            else:
                result = (
                    f"Unsupported tool '{fn_name}'. "
                    "Only 'execute_in_sandbox' is available."
                )

            # Truncate long outputs to keep console readable.
            if isinstance(result, str) and len(result) > 20000:
                result = (
                    result[:10000]
                    + "\n...[TRUNCATED]...\n"
                    + result[-10000:]
                )

            print_panel(result, "Tool Result", "result")
            log_step("TOOL_RESULT", "Executed")
            emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})

            # Feed the tool response back as a TOOL message with a functionResponse part.
            history.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": result},
                        )
                    ],
                )
            )

    # Final report generation.
    try:
        print_status("Generating Final Report...", "bold green")
        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Generate a concise, information-dense report that explains "
                            "how you tested the hypothesis, what you observed, and your "
                            "final conclusion."
                        )
                    )
                ],
            )
        )

        final_response_stream = client.models.generate_content_stream(
            model="gemini-3-pro-preview",
            contents=history,
            # Still use thinking so the model can reason about its own trace,
            # but tools are not needed here.
            config=_build_generation_config(
                tools=None,
                system_instruction=system_prompt,
                disable_autofc=True,
            ),
        )

        final_parts = []
        for chunk in final_response_stream:
            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False) and part.text:
                        emit_event("AGENT_THOUGHT_STREAM", {"chunk": part.text})
                    final_parts.append(part)

        # Basic merge for final text extraction
        final_text = ""
        for part in final_parts:
            if part.text and not getattr(part, "thought", False):
                final_text += part.text
        
        print_panel(final_text, "Final Report", "bold green")
        emit_event("AGENT_COMPLETE", {"agent_id": "1", "exit_code": 0})
    finally:
        _close_shared_sandbox()
