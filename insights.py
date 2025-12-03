"""Lightweight sidebar summarizer for streaming agent thoughts.

This helper stays **separate** from the main agents/orchestrator logic.
It only consumes the recent public transcript (last ~5 steps) and asks a
cheaper model to condense it into a tiny finding plus an optional chart spec
the frontend can render.

Supports both Gemini (cloud) and Ollama (local) backends.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_gemini_client = None
_ollama_client = None
_backend: Optional[str] = None  # "gemini", "ollama", or None


def _get_backend() -> str:
    """Determine which backend to use for summarization."""
    global _backend
    
    if _backend is not None:
        return _backend
    
    # Prefer Gemini if API key is available
    if os.environ.get("GOOGLE_API_KEY"):
        _backend = "gemini"
        return _backend
    
    # Fall back to Ollama if available
    try:
        from ollama_client import test_ollama_connection
        if test_ollama_connection():
            _backend = "ollama"
            return _backend
    except ImportError:
        pass
    
    # No backend available
    _backend = "none"
    return _backend


def _get_gemini_client():
    """Lazily create a single Gemini client (re-used across requests)."""
    global _gemini_client
    
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        _gemini_client = genai.Client(api_key=api_key)
    
    return _gemini_client


def _get_ollama_client():
    """Lazily create a single Ollama client (re-used across requests)."""
    global _ollama_client
    
    if _ollama_client is None:
        from ollama_client import OllamaClient
        # Use a smaller/faster model for summarization
        _ollama_client = OllamaClient(model="qwen3:8b", temperature=0.2)
    
    return _ollama_client


def _build_prompt(history: List[Dict[str, str]]) -> str:
    """Format the last few steps into a compact textual context."""

    lines: List[str] = []
    for item in history[-5:]:  # hard cap: last 5 turns only
        role = (item.get("type") or "text").upper()
        content = (item.get("content") or "").strip()
        # Trim individual snippets to keep context small and cheap
        if len(content) > 1600:
            content = content[:1600] + "\n...[truncated]"
        lines.append(f"[{role}]\n{content}")

    return "\n\n".join(lines)


def summarize_agent_findings(
    agent_id: str,
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Return a JSON-friendly finding + optional chart for a single agent.

    Args:
        agent_id: Identifier of the sub-agent (for logging only).
        history: List of dicts with at least ``type`` and ``content`` keys.
                 Only the 5 most recent entries are used.

    Returns:
        {"summary": str, "chart": Optional[dict]}
    """
    prompt = _build_prompt(history)

    if not prompt.strip():
        return {"summary": "Waiting for agent output...", "chart": None}

    system_instruction = (
        "You distill an autonomous research agent's most recent scratch notes "
        "into crisp sidebar findings. Keep it short (<=120 words), prefer "
        "bullets, surface concrete numbers, and call out the next action.\n"
        "If you can see numeric progressions (loss/accuracy/score vs step), "
        "add a compact chart spec. Use simple types only: line or bar.\n"
        "Respond as JSON with keys: summary (markdown-safe string) and optional "
        "chart. Chart shape: {\"title\": str, \"type\": \"line\"|\"bar\", "
        "\"labels\": [str], \"series\":[{\"name\": str, \"values\": [number]}]}. "
        "Omit chart if no numeric series are present."
    )

    backend = _get_backend()
    
    if backend == "none":
        # No LLM available - return a simple extraction
        return {"summary": "LLM not available for summarization", "chart": None}
    
    raw_text = ""
    
    if backend == "ollama":
        # Use Ollama for summarization
        try:
            from ollama_client import OllamaMessage
            client = _get_ollama_client()
            
            response = client.chat(
                messages=[OllamaMessage(role="user", content=prompt)],
                system_prompt=system_instruction,
                stream=False,
            )
            raw_text = response.content.strip()
        except Exception as e:
            logger.error("Ollama summarize failed for agent %s: %s", agent_id, e)
            raise
    else:
        # Use Gemini for summarization
        from google.genai import types
        
        client = _get_gemini_client()

        try:
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    max_output_tokens=4000,
                ),
            )
        except Exception as e:
            logger.error("Gemini summarize failed for agent %s: %s", agent_id, e)
            raise

        try:
            # Prefer the convenience accessor if available
            raw_text = getattr(response, "text", "") or ""
            if not raw_text:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if getattr(part, "text", None):
                            raw_text += part.text
                        elif getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                            try:
                                raw_text += part.inline_data.data.decode("utf-8", errors="ignore")
                            except Exception:
                                pass
            raw_text = raw_text.strip()
        except Exception as e:
            logger.warning("Failed to extract text for agent %s: %s", agent_id, e)

    result: Dict[str, Any]
    try:
        result = json.loads(raw_text)
    except Exception as json_err:
        logger.debug(
            "summarize_agent: json decode failed for agent=%s err=%s raw_sample=%s",
            agent_id,
            json_err,
            (raw_text[:200] + ("..." if len(raw_text) > 200 else "")),
        )
        # Heuristic: try to salvage a JSON-ish blob between the first { and last }
        salvaged = None
        if "{" in raw_text and "}" in raw_text:
            candidate_blob = raw_text[raw_text.find("{") : raw_text.rfind("}") + 1]
            try:
                salvaged = json.loads(candidate_blob)
            except Exception:
                pass

        if salvaged and isinstance(salvaged, dict):
            result = salvaged
        else:
            # Fallback: treat the raw text as the summary string.
            result = {"summary": raw_text or "No summary produced", "chart": None}

    # Ensure required fields exist and are JSON-serializable
    if "summary" not in result or not isinstance(result.get("summary"), str):
        result["summary"] = raw_text or "No summary produced"
    if "chart" in result and result["chart"] is not None:
        if not isinstance(result["chart"], dict):
            result["chart"] = None

    # Trim overly verbose summaries so the rail stays tight
    if result.get("summary") and len(result["summary"]) > 800:
        result["summary"] = result["summary"][:800] + "..."

    return result
