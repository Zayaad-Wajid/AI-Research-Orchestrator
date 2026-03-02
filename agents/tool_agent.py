"""
AI Research Orchestrator - Tool Agent
Handles tool-based operations including code execution.
"""

from typing import Dict, Any, Optional
import structlog
import json

from .base_agent import BaseAgent
from models.task import SubTask, CodeExecutionResult
from tools.code_executor import code_execution_tool

logger = structlog.get_logger()


class ToolAgent(BaseAgent):
    """
    Tool Agent responsible for:
    - Generating code to solve problems
    - Executing code safely
    - Analyzing execution results
    - Debugging and fixing errors
    """
    
    def __init__(self):
        super().__init__(
            name="tool_agent",
            description="Executes tools and code for analysis"
        )
        self.code_executor = code_execution_tool
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert coding agent. Your role is to write and execute Python code to solve problems, perform calculations, and analyze data.

Your capabilities:
1. Write clean, efficient Python code
2. Perform mathematical calculations and data analysis
3. Process and transform data
4. Debug and fix code errors

Guidelines:
- Write self-contained code that doesn't require external files
- Use only standard library modules when possible
- Include print statements to show results
- Handle edge cases and errors gracefully
- Keep code concise but readable

Allowed modules: math, statistics, random, datetime, json, collections, itertools, functools, re, string, typing, dataclasses, enum

When generating code:
1. Clearly state what the code will do
2. Provide complete, runnable code
3. Explain the expected output

Output format for code generation:
```python
# Your code here
```

Expected output: [description of what the code will produce]"""
    
    async def execute(
        self, 
        task: SubTask, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a code-related task.
        
        Args:
            task: Task description (what to compute/analyze)
            context: Additional context and data
            
        Returns:
            Execution results
        """
        objective = task.description
        data = context.get("data", {})
        previous_code = context.get("previous_code")
        previous_error = context.get("previous_error")
        
        try:
            # Generate code if not provided
            if previous_error and previous_code:
                code = await self._fix_code(previous_code, previous_error)
            else:
                code = await self._generate_code(objective, data)
            
            if not code:
                return {
                    "success": False,
                    "error": "Failed to generate code",
                    "agent": self.name,
                    "retry_recommended": False
                }
            
            # Execute the code
            result = await self.code_executor.execute(code, "python")
            
            if result.success:
                # Analyze the output
                analysis = await self._analyze_output(objective, result.output)
                
                logger.info(
                    f"Code execution successful",
                    agent=self.name,
                    execution_time=result.execution_time
                )
                
                return {
                    "success": True,
                    "code": code,
                    "output": result.output,
                    "analysis": analysis,
                    "execution_time": result.execution_time,
                    "agent": self.name
                }
            else:
                logger.warning(
                    f"Code execution failed",
                    agent=self.name,
                    error=result.error
                )
                
                return {
                    "success": False,
                    "code": code,
                    "error": result.error,
                    "agent": self.name,
                    "retry_recommended": True,
                    "retry_context": {
                        "previous_code": code,
                        "previous_error": result.error
                    }
                }
                
        except Exception as e:
            return await self.handle_error(e, task, context)
    
    async def _generate_code(
        self, 
        objective: str, 
        data: Dict[str, Any]
    ) -> Optional[str]:
        """Generate Python code for the given objective."""
        
        data_context = ""
        if data:
            data_context = f"\nAvailable data:\n```json\n{json.dumps(data, indent=2)}\n```"
        
        prompt = f"""Write Python code to accomplish this objective:

Objective: {objective}
{data_context}

Requirements:
- Code must be complete and self-contained
- Use print() to output results
- Handle potential errors
- Only use standard library modules (math, statistics, json, collections, itertools, re, datetime)

Provide ONLY the Python code wrapped in ```python ... ``` markers. No explanations needed."""

        response = await self.generate_response(prompt, temperature=0.2)
        
        # Extract code from response
        code = self._extract_code(response)
        return code
    
    async def _fix_code(
        self, 
        original_code: str, 
        error: str
    ) -> Optional[str]:
        """Fix code that produced an error."""
        
        prompt = f"""The following Python code produced an error. Fix it:

Original code:
```python
{original_code}
```

Error:
{error}

Provide the corrected code wrapped in ```python ... ``` markers. Only output the fixed code."""

        response = await self.generate_response(prompt, temperature=0.2)
        code = self._extract_code(response)
        return code
    
    async def _analyze_output(
        self, 
        objective: str, 
        output: str
    ) -> str:
        """Analyze the code execution output."""
        
        prompt = f"""Analyze this code execution output:

Objective: {objective}

Output:
{output}

Provide a brief analysis of what the output means in the context of the original objective. Be concise."""

        analysis = await self.generate_response(prompt, temperature=0.3)
        return analysis.strip()
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # Try to identify code by looking for common patterns
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Detect code-like lines
            if any(keyword in line for keyword in ['import ', 'def ', 'class ', 'print(', '=', 'for ', 'if ', 'while ']):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return None
    
    async def execute_code_directly(
        self, 
        code: str
    ) -> CodeExecutionResult:
        """Execute provided code directly without generation."""
        return await self.code_executor.execute(code, "python")
