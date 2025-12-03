"""
Local sandbox for executing Python code with GPU support.

Replaces Modal's remote sandboxes with local subprocess execution,
using the system's GPU if available.
"""

import os
import sys
import subprocess
import threading
import tempfile
import shutil
from typing import Optional, List, Callable
from pathlib import Path

from logger import log_step, print_panel


# Global state for persistent sandbox
_sandbox_process: Optional[subprocess.Popen] = None
_sandbox_dir: Optional[Path] = None


def _check_gpu_available() -> dict:
    """
    Check if CUDA GPU is available on the local system.
    
    Returns dict with:
        - available: bool
        - device_name: str or None
        - cuda_version: str or None
    """
    result = {
        "available": False,
        "device_name": None,
        "cuda_version": None,
        "vram_total": None,
        "vram_free": None,
    }
    
    try:
        # Try to import torch and check CUDA
        import torch
        if torch.cuda.is_available():
            result["available"] = True
            result["device_name"] = torch.cuda.get_device_name(0)
            result["cuda_version"] = torch.version.cuda
            
            # Get VRAM info
            total = torch.cuda.get_device_properties(0).total_memory
            free = total - torch.cuda.memory_allocated(0)
            result["vram_total"] = f"{total / 1024**3:.1f}GB"
            result["vram_free"] = f"{free / 1024**3:.1f}GB"
    except ImportError:
        pass
    except Exception as e:
        log_step("WARNING", f"Error checking GPU: {e}")
    
    return result


def get_gpu_info() -> str:
    """Get a human-readable GPU info string."""
    info = _check_gpu_available()
    if info["available"]:
        return (
            f"GPU: {info['device_name']} | "
            f"CUDA {info['cuda_version']} | "
            f"VRAM: {info['vram_free']} free / {info['vram_total']} total"
        )
    return "GPU: Not available (running on CPU)"


def _get_sandbox_dir() -> Path:
    """Get or create the sandbox working directory."""
    global _sandbox_dir
    
    if _sandbox_dir is None or not _sandbox_dir.exists():
        # Create a temporary directory for the sandbox
        _sandbox_dir = Path(tempfile.mkdtemp(prefix="ai_researcher_sandbox_"))
        log_step("SANDBOX", f"Created sandbox directory: {_sandbox_dir}")
    
    return _sandbox_dir


def _cleanup_sandbox_dir():
    """Clean up the sandbox directory."""
    global _sandbox_dir
    
    if _sandbox_dir is not None and _sandbox_dir.exists():
        try:
            shutil.rmtree(_sandbox_dir)
            log_step("SANDBOX", "Cleaned up sandbox directory")
        except Exception as e:
            log_step("WARNING", f"Failed to cleanup sandbox: {e}")
        _sandbox_dir = None


def execute_code_locally(
    code: str,
    timeout: int = 3600,
    on_stdout: Optional[Callable[[str], None]] = None,
    on_stderr: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Execute Python code locally in a subprocess.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 1 hour)
        on_stdout: Optional callback for stdout lines (for streaming)
        on_stderr: Optional callback for stderr lines (for streaming)
    
    Returns:
        String with exit code, stdout, and stderr
    """
    sandbox_dir = _get_sandbox_dir()
    
    # Write code to a temporary file
    script_path = sandbox_dir / "script.py"
    script_path.write_text(code, encoding="utf-8")
    
    log_step("SANDBOX", "Executing code locally...")
    print_panel(code, "Local Sandbox Code", "code")
    
    # Prepare environment
    env = os.environ.copy()
    
    # Ensure CUDA is visible if available
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU by default
    
    # Set working directory to sandbox
    env["PYTHONPATH"] = str(sandbox_dir) + ":" + env.get("PYTHONPATH", "")
    
    # Start the subprocess
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(sandbox_dir),
        env=env,
    )
    
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    
    def _read_stream(stream, chunks: List[str], callback: Optional[Callable], is_stderr: bool):
        """Read from a stream, optionally calling a callback for each line."""
        try:
            for line in stream:
                chunks.append(line)
                
                # Print to console
                target = sys.stderr if is_stderr else sys.stdout
                print(line, end="", file=target, flush=True)
                
                # Call callback if provided
                if callback:
                    try:
                        callback(line)
                    except Exception:
                        pass
        except Exception as e:
            log_step("WARNING", f"Error reading {'stderr' if is_stderr else 'stdout'}: {e}")
    
    # Read stdout and stderr concurrently
    stdout_thread = threading.Thread(
        target=_read_stream,
        args=(proc.stdout, stdout_chunks, on_stdout, False),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_stream,
        args=(proc.stderr, stderr_chunks, on_stderr, True),
        daemon=True,
    )
    
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for completion with timeout
    try:
        exit_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log_step("WARNING", f"Code execution timed out after {timeout}s")
        proc.kill()
        exit_code = -1
    
    # Wait for streams to finish
    stdout_thread.join(timeout=5.0)
    stderr_thread.join(timeout=5.0)
    
    log_step("SANDBOX", f"Execution completed with exit code {exit_code}")
    
    stdout_str = "".join(stdout_chunks)
    stderr_str = "".join(stderr_chunks)
    
    return f"Exit Code: {exit_code}\nSTDOUT:\n{stdout_str}\nSTDERR:\n{stderr_str}"


def install_package(package: str) -> bool:
    """
    Install a Python package using pip.
    
    Args:
        package: Package name (e.g., "numpy", "torch>=2.0")
    
    Returns:
        True if installation succeeded
    """
    log_step("SANDBOX", f"Installing package: {package}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", package],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode == 0:
            log_step("SANDBOX", f"Successfully installed {package}")
            return True
        else:
            log_step("WARNING", f"Failed to install {package}: {result.stderr}")
            return False
    except Exception as e:
        log_step("ERROR", f"Error installing {package}: {e}")
        return False


def check_packages_available(packages: List[str]) -> dict:
    """
    Check which packages are available.
    
    Args:
        packages: List of package names to check
    
    Returns:
        Dict mapping package names to availability (True/False)
    """
    result = {}
    
    for package in packages:
        # Handle package names with version specifiers
        pkg_name = package.split(">=")[0].split("==")[0].split("<")[0].strip()
        
        try:
            __import__(pkg_name)
            result[package] = True
        except ImportError:
            result[package] = False
    
    return result


def ensure_packages(packages: List[str]) -> bool:
    """
    Ensure all required packages are available, installing if needed.
    
    Args:
        packages: List of package names
    
    Returns:
        True if all packages are available
    """
    availability = check_packages_available(packages)
    
    missing = [pkg for pkg, available in availability.items() if not available]
    
    if not missing:
        return True
    
    log_step("SANDBOX", f"Installing missing packages: {', '.join(missing)}")
    
    for package in missing:
        if not install_package(package):
            return False
    
    return True


def cleanup():
    """Clean up sandbox resources."""
    _cleanup_sandbox_dir()


# Pre-check common ML packages
COMMON_PACKAGES = [
    "numpy",
    "pandas",
    "torch",
    "scikit-learn",
    "matplotlib",
]


def get_environment_info() -> str:
    """Get a summary of the local environment for the agent."""
    gpu_info = get_gpu_info()
    packages = check_packages_available(COMMON_PACKAGES)
    
    pkg_status = []
    for pkg, available in packages.items():
        status = "✓" if available else "✗"
        pkg_status.append(f"  {status} {pkg}")
    
    return f"""Local Environment:
{gpu_info}

Python: {sys.version.split()[0]}
Packages:
{chr(10).join(pkg_status)}
"""


# Test function
def test_local_execution():
    """Test that local execution works."""
    print("Testing local sandbox execution...")
    print(get_environment_info())
    
    test_code = """
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed")

print("Local sandbox test completed!")
"""
    
    result = execute_code_locally(test_code)
    print("\n" + "="*50)
    print("Result:")
    print(result)
    
    cleanup()


if __name__ == "__main__":
    test_local_execution()







