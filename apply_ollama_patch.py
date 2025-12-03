"""
Script to apply Ollama support patches to the AI Researcher codebase.
Run this once to enable local Ollama execution.
"""

import re
import sys

def patch_agent():
    """Patch agent.py to support Ollama and local execution."""
    
    with open('agent.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the imports at the top
    old_imports = '''import os
import sys
import threading
import json
from typing import Optional, List

from google import genai
from google.genai import types

import anthropic

from logger import print_panel, print_status, log_step, logger

import modal
from modal.stream_type import StreamType'''

    new_imports = '''import os
import sys
import threading
import json
from typing import Optional, List

from logger import print_panel, print_status, log_step, logger

# Lazy imports for different backends
_genai_client = None
_anthropic_client = None
_ollama_client = None
_use_local_sandbox = False'''

    if old_imports in content:
        content = content.replace(old_imports, new_imports)
        print("✓ Updated imports to use lazy loading")
    else:
        print("⚠ Import section already modified or different")
    
    # Replace the sandbox cache section
    old_cache = '''# Cache a single sandbox per run so the agent can keep state across tool calls.
_shared_sandbox: Optional[modal.Sandbox] = None
_shared_gpu: Optional[str] = None  # Track which GPU the sandbox was created with
_selected_gpu: Optional[str] = None  # User-selected GPU for this run'''

    new_cache = '''# Cache a single sandbox per run so the agent can keep state across tool calls.
_shared_sandbox = None  # For Modal sandbox (when using remote)
_shared_gpu: Optional[str] = None  # Track which GPU the sandbox was created with
_selected_gpu: Optional[str] = None  # User-selected GPU for this run


def _init_ollama(model: str):
    """Initialize Ollama client."""
    global _ollama_client, _use_local_sandbox
    from ollama_client import OllamaClient, test_ollama_connection
    
    if not test_ollama_connection():
        raise ConnectionError(
            "Cannot connect to Ollama. Make sure Ollama is running:\\n"
            "  1. Install Ollama: https://ollama.ai\\n"
            "  2. Start Ollama: ollama serve\\n"
            "  3. Pull a model: ollama pull qwen2.5-coder:14b"
        )
    
    _ollama_client = OllamaClient(model=model)
    _use_local_sandbox = True
    
    # Check if model exists
    if not _ollama_client.check_model_exists(model):
        available = [m.get("name", "") for m in _ollama_client.list_models()]
        raise ValueError(
            f"Model '{model}' not found in Ollama.\\n"
            f"Available models: {', '.join(available) if available else 'none'}\\n"
            f"Pull it with: ollama pull {model}"
        )
    
    return _ollama_client'''

    if old_cache in content:
        content = content.replace(old_cache, new_cache)
        print("✓ Added Ollama initialization function")
    else:
        print("⚠ Cache section already modified or different")
    
    with open('agent.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ agent.py patched successfully")


def patch_main():
    """Patch main.py to support Ollama model selection."""
    
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_model_arg = '''    parser.add_argument(
        "--model",
        type=str,
        choices=["gemini-3-pro-preview", "claude-opus-4-5"],
        default="gemini-3-pro-preview",
        help=(
            "LLM model to use: "
            "'gemini-3-pro-preview' (default) or 'claude-opus-4-5'."
        ),
    )'''

    new_model_arg = '''    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-pro-preview",
        help=(
            "LLM model to use. Options:\\n"
            "  - 'gemini-3-pro-preview' (default) - Google Gemini 3 Pro\\n"
            "  - 'claude-opus-4-5' - Anthropic Claude Opus 4.5\\n"
            "  - 'ollama:<model>' - Local Ollama model (e.g., 'ollama:qwen2.5-coder:14b')\\n"
            "For Ollama, code runs locally using your GPU instead of Modal."
        ),
    )'''

    if old_model_arg in content:
        content = content.replace(old_model_arg, new_model_arg)
        print("✓ Updated model argument in main.py")
    else:
        print("⚠ Model argument already modified or different")
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ main.py patched successfully")


if __name__ == "__main__":
    print("Applying Ollama support patches...")
    print()
    
    try:
        patch_agent()
        patch_main()
        print()
        print("Done! You can now use Ollama models with:")
        print("  python main.py 'Your hypothesis' --model ollama:qwen3:8b")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)







