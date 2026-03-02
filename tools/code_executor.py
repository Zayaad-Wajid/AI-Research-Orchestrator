"""
AI Research Orchestrator - Code Execution Tools
Provides safe code execution with sandboxing support.
"""

import asyncio
import subprocess
import tempfile
import os
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path
import structlog
from tenacity import retry, stop_after_attempt, wait_fixed

from models.task import CodeExecutionResult
from config.settings import settings

logger = structlog.get_logger()


class RestrictedExecutor:
    """
    Executes code in a restricted environment.
    Uses subprocess with timeout and resource limits.
    """
    
    ALLOWED_MODULES = {
        'math', 'statistics', 'random', 'datetime', 'json',
        'collections', 'itertools', 'functools', 'operator',
        're', 'string', 'textwrap', 'unicodedata',
        'typing', 'dataclasses', 'enum'
    }
    
    BLOCKED_BUILTINS = {
        'exec', 'eval', 'compile', 'open', 'input',
        '__import__', 'globals', 'locals', 'vars',
        'exit', 'quit'
    }
    
    def __init__(self, timeout: float = 30.0, max_output: int = 10000):
        self.timeout = timeout
        self.max_output = max_output
    
    def _create_safe_code(self, code: str) -> str:
        """Wrap code with safety checks."""
        # Add import restrictions
        safe_wrapper = f'''
import sys
import builtins

# Restrict dangerous builtins
_blocked = {self.BLOCKED_BUILTINS!r}
_original_import = builtins.__import__
_allowed_modules = {self.ALLOWED_MODULES!r}

def _safe_import(name, *args, **kwargs):
    base_module = name.split('.')[0]
    if base_module not in _allowed_modules:
        raise ImportError(f"Import of '{{name}}' is not allowed")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _safe_import

for blocked in _blocked:
    if hasattr(builtins, blocked):
        delattr(builtins, blocked)

# User code starts here
{code}
'''
        return safe_wrapper
    
    async def execute(
        self, 
        code: str, 
        language: str = "python"
    ) -> CodeExecutionResult:
        """
        Execute code in a restricted subprocess.
        
        Args:
            code: Code to execute
            language: Programming language (currently only Python supported)
            
        Returns:
            CodeExecutionResult with output or error
        """
        if language.lower() != "python":
            return CodeExecutionResult(
                success=False,
                error=f"Language '{language}' not supported. Only Python is available.",
                language=language
            )
        
        import time
        start_time = time.time()
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False,
                encoding='utf-8'
            ) as f:
                safe_code = self._create_safe_code(code)
                f.write(safe_code)
                temp_path = f.name
            
            try:
                # Execute in subprocess
                result = await asyncio.wait_for(
                    self._run_subprocess(temp_path),
                    timeout=self.timeout
                )
                
                execution_time = time.time() - start_time
                
                return CodeExecutionResult(
                    success=result["returncode"] == 0,
                    output=result["stdout"][:self.max_output] if result["stdout"] else None,
                    error=result["stderr"][:self.max_output] if result["stderr"] and result["returncode"] != 0 else None,
                    execution_time=execution_time,
                    language=language
                )
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except asyncio.TimeoutError:
            return CodeExecutionResult(
                success=False,
                error=f"Execution timed out after {self.timeout} seconds",
                execution_time=self.timeout,
                language=language
            )
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                language=language
            )
    
    async def _run_subprocess(self, script_path: str) -> Dict[str, Any]:
        """Run Python script in subprocess."""
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tempfile.gettempdir()
        )
        
        stdout, stderr = await process.communicate()
        
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode('utf-8', errors='replace'),
            "stderr": stderr.decode('utf-8', errors='replace')
        }


class DockerExecutor:
    """
    Executes code in Docker containers for maximum isolation.
    Requires Docker to be installed and running.
    """
    
    DOCKER_IMAGE = "python:3.11-slim"
    
    def __init__(self, timeout: float = 60.0, max_output: int = 10000):
        self.timeout = timeout
        self.max_output = max_output
        self._docker_available = None
    
    async def is_available(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available
        
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            self._docker_available = process.returncode == 0
        except:
            self._docker_available = False
        
        return self._docker_available
    
    async def execute(
        self, 
        code: str, 
        language: str = "python"
    ) -> CodeExecutionResult:
        """
        Execute code in a Docker container.
        """
        if not await self.is_available():
            return CodeExecutionResult(
                success=False,
                error="Docker is not available. Falling back to restricted execution.",
                language=language
            )
        
        import time
        start_time = time.time()
        
        try:
            # Create temp directory for code
            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = Path(temp_dir) / "script.py"
                script_path.write_text(code, encoding='utf-8')
                
                # Run in Docker
                cmd = [
                    "docker", "run", "--rm",
                    "--network", "none",  # No network access
                    "--memory", "256m",    # Memory limit
                    "--cpus", "0.5",       # CPU limit
                    "-v", f"{temp_dir}:/code:ro",
                    "-w", "/code",
                    self.DOCKER_IMAGE,
                    "python", "script.py"
                ]
                
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    ),
                    timeout=self.timeout
                )
                
                stdout, stderr = await process.communicate()
                execution_time = time.time() - start_time
                
                return CodeExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode('utf-8', errors='replace')[:self.max_output] if stdout else None,
                    error=stderr.decode('utf-8', errors='replace')[:self.max_output] if stderr and process.returncode != 0 else None,
                    execution_time=execution_time,
                    language=language
                )
                
        except asyncio.TimeoutError:
            # Kill the container
            await self._kill_containers()
            return CodeExecutionResult(
                success=False,
                error=f"Execution timed out after {self.timeout} seconds",
                execution_time=self.timeout,
                language=language
            )
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                language=language
            )
    
    async def _kill_containers(self):
        """Kill any hanging containers."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "ps", "-q", "--filter", "ancestor=" + self.DOCKER_IMAGE,
                stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            container_ids = stdout.decode().strip().split()
            
            for cid in container_ids:
                await asyncio.create_subprocess_exec("docker", "kill", cid)
        except:
            pass


class CodeExecutionTool:
    """
    Unified code execution tool with fallback support.
    Tries Docker first, falls back to restricted subprocess.
    """
    
    def __init__(self):
        self.docker_executor = DockerExecutor()
        self.restricted_executor = RestrictedExecutor()
    
    async def execute(
        self, 
        code: str, 
        language: str = "python",
        prefer_docker: bool = True
    ) -> CodeExecutionResult:
        """
        Execute code with automatic fallback.
        
        Args:
            code: Code to execute
            language: Programming language
            prefer_docker: Whether to prefer Docker execution
            
        Returns:
            CodeExecutionResult with output or error
        """
        logger.info("Executing code", language=language, code_length=len(code))
        
        if prefer_docker and await self.docker_executor.is_available():
            logger.info("Using Docker executor")
            result = await self.docker_executor.execute(code, language)
            if result.success or "not available" not in (result.error or ""):
                return result
        
        # Fall back to restricted execution
        logger.info("Using restricted executor")
        return await self.restricted_executor.execute(code, language)
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def execute_with_retry(
        self, 
        code: str, 
        language: str = "python"
    ) -> CodeExecutionResult:
        """Execute code with automatic retries."""
        result = await self.execute(code, language)
        if not result.success:
            logger.warning("Code execution failed, may retry", error=result.error)
            raise RuntimeError(result.error)
        return result


# Singleton instance
code_execution_tool = CodeExecutionTool()


# Function-based tools for OpenAI Agents SDK
async def execute_code(code: str, language: str = "python") -> str:
    """
    Execute code and return the result.
    
    Args:
        code: The code to execute
        language: Programming language (default: python)
        
    Returns:
        Execution output or error message
    """
    result = await code_execution_tool.execute(code, language)
    
    if result.success:
        output = result.output or "Code executed successfully (no output)"
        return f"Execution successful ({result.execution_time:.2f}s):\n{output}"
    else:
        return f"Execution failed ({result.execution_time:.2f}s):\n{result.error}"


async def validate_code(code: str) -> str:
    """
    Validate Python code syntax without executing it.
    
    Args:
        code: The Python code to validate
        
    Returns:
        Validation result message
    """
    try:
        compile(code, '<string>', 'exec')
        return "Code syntax is valid."
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return f"Validation error: {str(e)}"
