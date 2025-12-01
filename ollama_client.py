"""
Ollama client wrapper for AI Researcher.

Provides a unified interface for interacting with local Ollama models,
including support for tool/function calling and streaming responses.
"""

import json
import requests
from typing import Optional, List, Dict, Any, Generator, Callable
from dataclasses import dataclass, field

# Default Ollama server URL
OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class OllamaMessage:
    """A message in the conversation history."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool responses


@dataclass
class OllamaToolCall:
    """Represents a tool/function call from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class OllamaResponse:
    """Response from Ollama API."""
    content: str = ""
    thinking: str = ""  # For models that support thinking/reasoning
    tool_calls: List[OllamaToolCall] = field(default_factory=list)
    done: bool = False
    model: str = ""
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None


class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    Supports:
    - Chat completions with streaming
    - Tool/function calling (for compatible models)
    - Conversation history management
    """
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.7,
        num_ctx: int = 32768,  # Context window size
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.session = requests.Session()
    
    def _build_tools_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Build a system prompt section that describes available tools.
        
        Since not all Ollama models support native tool calling,
        we use a prompt-based approach that works universally.
        """
        if not tools:
            return ""
        
        tools_desc = []
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description")
            params = tool.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])
            
            param_lines = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                req_marker = " (required)" if param_name in required else " (optional)"
                param_lines.append(f"    - {param_name}: {param_type}{req_marker} - {param_desc}")
            
            tools_desc.append(f"""
**{name}**
{desc}
Parameters:
{chr(10).join(param_lines) if param_lines else '    (none)'}
""")
        
        return f"""
## Available Tools

You have access to the following tools. To use a tool, respond with a JSON block in this exact format:

```tool_call
{{
  "tool": "<tool_name>",
  "arguments": {{
    "<param1>": <value1>,
    "<param2>": <value2>
  }}
}}
```

{"".join(tools_desc)}

IMPORTANT:
- Only use the ```tool_call``` format when you want to execute a tool
- You can include thinking/reasoning before the tool call
- After receiving tool results, analyze them and continue your research
- When finished, include [DONE] in your response
"""
    
    def _parse_tool_calls(self, content: str) -> List[OllamaToolCall]:
        """
        Parse tool calls from model response.
        
        Looks for tool calls in multiple formats:
        1. ```tool_call ... ``` blocks (preferred)
        2. ```json { "tool": ... } ``` blocks  
        3. Standalone JSON objects with "tool" key
        4. Any JSON-like structure with balanced braces containing "tool"
        """
        tool_calls = []
        import re
        
        # Preprocess: Convert Python triple-quoted strings to JSON-escaped strings
        # This handles cases where the model uses """ instead of \n
        def fix_triple_quotes(text: str) -> str:
            """Convert Python triple-quoted strings in JSON to proper escaped strings."""
            # Pattern to find "code": """ ... """ or 'code': """ ... """
            pattern = r'("code"\s*:\s*)"""(.*?)"""'
            
            def replace_triple_quotes(match):
                prefix = match.group(1)
                code_content = match.group(2)
                # Escape the content for JSON
                escaped = code_content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                return f'{prefix}"{escaped}"'
            
            return re.sub(pattern, replace_triple_quotes, text, flags=re.DOTALL)
        
        def remove_json_comments(text: str) -> str:
            """Remove JavaScript-style comments from JSON (// comments)."""
            # Remove // comments but not inside strings
            # Simple approach: remove lines that have // outside of quotes
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                # Find // that's not inside a string
                in_string = False
                escape_next = False
                comment_start = -1
                for i, char in enumerate(line):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"':
                        in_string = not in_string
                        continue
                    if not in_string and i < len(line) - 1 and line[i:i+2] == '//':
                        comment_start = i
                        break
                if comment_start >= 0:
                    line = line[:comment_start].rstrip()
                cleaned_lines.append(line)
            return '\n'.join(cleaned_lines)
        
        # Apply fixes
        content = fix_triple_quotes(content)
        content = remove_json_comments(content)
        
        # Pattern 1: ```tool_call ... ``` blocks (preferred format)
        pattern1 = r'```tool_call\s*\n?(.*?)\n?```'
        matches = re.findall(pattern1, content, re.DOTALL)
        
        # Pattern 2: ```json ... ``` or ``` ... ``` blocks with tool key
        pattern2 = r'```(?:json)?\s*\n?(\{[^`]*?"tool"[^`]*?\})\n?```'
        matches += re.findall(pattern2, content, re.DOTALL)
        
        # Pattern 3: Find JSON objects by balanced brace matching
        # This handles complex nested JSON with strings containing special chars
        def extract_json_objects(text: str) -> List[str]:
            """Extract JSON objects by finding balanced braces."""
            objects = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    # Check if this might be a tool call
                    start = i
                    brace_count = 1
                    in_string = False
                    escape_next = False
                    i += 1
                    
                    while i < len(text) and brace_count > 0:
                        char = text[i]
                        if escape_next:
                            escape_next = False
                        elif char == '\\':
                            escape_next = True
                        elif char == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                        i += 1
                    
                    if brace_count == 0:
                        json_str = text[start:i]
                        if '"tool"' in json_str:
                            objects.append(json_str)
                else:
                    i += 1
            return objects
        
        # Add any balanced JSON objects found
        matches += extract_json_objects(content)
        
        seen_tools = set()  # Avoid duplicates
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                tool_name = data.get("tool", "")
                arguments = data.get("arguments", {})
                
                # Create unique key to avoid duplicates
                tool_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                
                if tool_name and tool_key not in seen_tools:
                    seen_tools.add(tool_key)
                    tool_calls.append(OllamaToolCall(
                        id=f"call_{len(tool_calls)}",
                        name=tool_name,
                        arguments=arguments,
                    ))
            except json.JSONDecodeError:
                # Invalid JSON, skip this block
                continue
        
        return tool_calls
    
    def _extract_thinking(self, content: str) -> tuple[str, str]:
        """
        Extract thinking/reasoning from content if present.
        
        Some models (like DeepSeek) use <think>...</think> tags.
        Returns (thinking, remaining_content).
        """
        import re
        
        # Check for <think> tags
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, content, re.DOTALL)
        
        if think_matches:
            thinking = "\n\n".join(think_matches)
            # Remove thinking blocks from content
            remaining = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
            return thinking, remaining
        
        return "", content
    
    def chat(
        self,
        messages: List[OllamaMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> Generator[OllamaResponse, None, None] | OllamaResponse:
        """
        Send a chat completion request to Ollama.
        
        Args:
            messages: Conversation history
            tools: Optional list of tool definitions
            system_prompt: Optional system prompt (prepended to messages)
            stream: Whether to stream the response
        
        Returns:
            OllamaResponse or generator of OllamaResponse chunks if streaming
        """
        # Build the messages list for the API
        api_messages = []
        
        # Add system prompt with tools if provided
        full_system = system_prompt or ""
        if tools:
            full_system += self._build_tools_prompt(tools)
        
        if full_system:
            api_messages.append({
                "role": "system",
                "content": full_system,
            })
        
        # Add conversation messages
        for msg in messages:
            api_msg = {
                "role": msg.role,
                "content": msg.content,
            }
            api_messages.append(api_msg)
        
        # Build request payload
        payload = {
            "model": self.model,
            "messages": api_messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        
        url = f"{self.base_url}/api/chat"
        
        if stream:
            return self._stream_chat(url, payload)
        else:
            return self._sync_chat(url, payload)
    
    def _sync_chat(self, url: str, payload: Dict[str, Any]) -> OllamaResponse:
        """Synchronous chat completion."""
        try:
            response = self.session.post(url, json=payload, timeout=3600)  # 1 hour for slow models
            response.raise_for_status()
            data = response.json()
            
            content = data.get("message", {}).get("content", "")
            thinking, content = self._extract_thinking(content)
            tool_calls = self._parse_tool_calls(content)
            
            return OllamaResponse(
                content=content,
                thinking=thinking,
                tool_calls=tool_calls,
                done=True,
                model=data.get("model", self.model),
                total_duration=data.get("total_duration"),
                eval_count=data.get("eval_count"),
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}")
    
    def _stream_chat(
        self, 
        url: str, 
        payload: Dict[str, Any]
    ) -> Generator[OllamaResponse, None, None]:
        """Streaming chat completion."""
        try:
            with self.session.post(url, json=payload, stream=True, timeout=3600) as response:  # 1 hour for slow models
                response.raise_for_status()
                
                full_content = ""
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    chunk_content = data.get("message", {}).get("content", "")
                    full_content += chunk_content
                    is_done = data.get("done", False)
                    
                    # For intermediate chunks, just yield the content
                    if not is_done:
                        yield OllamaResponse(
                            content=chunk_content,
                            done=False,
                            model=data.get("model", self.model),
                        )
                    else:
                        # Final chunk - parse everything
                        thinking, remaining = self._extract_thinking(full_content)
                        tool_calls = self._parse_tool_calls(full_content)
                        
                        yield OllamaResponse(
                            content=remaining,
                            thinking=thinking,
                            tool_calls=tool_calls,
                            done=True,
                            model=data.get("model", self.model),
                            total_duration=data.get("total_duration"),
                            eval_count=data.get("eval_count"),
                        )
                        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to list Ollama models: {e}")
    
    def check_model_exists(self, model: str) -> bool:
        """Check if a model is available locally."""
        try:
            models = self.list_models()
            model_names = [m.get("name", "") for m in models]
            # Check both exact match and without tag
            return model in model_names or any(
                m.startswith(model.split(":")[0]) for m in model_names
            )
        except ConnectionError:
            return False


def get_recommended_models() -> Dict[str, Dict[str, Any]]:
    """
    Return a dictionary of recommended models for AI Researcher.
    
    Keys are model identifiers, values contain metadata.
    """
    return {
        "qwen2.5-coder:14b": {
            "name": "Qwen 2.5 Coder 14B",
            "description": "Excellent for code generation and analysis. Good balance of speed and quality.",
            "vram_required": "~10GB",
            "best_for": ["code", "analysis"],
        },
        "qwen2.5-coder:32b": {
            "name": "Qwen 2.5 Coder 32B",
            "description": "High-quality code model. Requires more VRAM but better results.",
            "vram_required": "~20GB",
            "best_for": ["code", "complex-reasoning"],
        },
        "deepseek-coder-v2:16b": {
            "name": "DeepSeek Coder V2 16B",
            "description": "Strong reasoning with native thinking support. Good for research.",
            "vram_required": "~12GB",
            "best_for": ["reasoning", "code"],
        },
        "llama3.1:8b": {
            "name": "Llama 3.1 8B",
            "description": "Fast and efficient. Good for quick experiments.",
            "vram_required": "~6GB",
            "best_for": ["general", "fast"],
        },
        "llama3.1:70b": {
            "name": "Llama 3.1 70B",
            "description": "High quality but requires significant VRAM or CPU offloading.",
            "vram_required": "~40GB",
            "best_for": ["quality", "complex-tasks"],
        },
        "codellama:34b": {
            "name": "Code Llama 34B",
            "description": "Specialized for code. Good alternative to Qwen.",
            "vram_required": "~20GB",
            "best_for": ["code"],
        },
        "mistral:7b": {
            "name": "Mistral 7B",
            "description": "Lightweight but capable. Good for systems with limited VRAM.",
            "vram_required": "~5GB",
            "best_for": ["fast", "efficient"],
        },
        "mixtral:8x7b": {
            "name": "Mixtral 8x7B (MoE)",
            "description": "Mixture of Experts. High quality with efficient inference.",
            "vram_required": "~26GB",
            "best_for": ["quality", "efficiency"],
        },
    }


# Convenience function to test Ollama connection
def test_ollama_connection(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Test if Ollama is running and accessible."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


