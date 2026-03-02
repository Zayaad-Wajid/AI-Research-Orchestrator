"""
AI Research Orchestrator - Retry Agent
Handles failed tasks with fallback strategies.
"""

from typing import Dict, Any, List, Optional
import structlog
import asyncio

from .base_agent import BaseAgent
from models.task import SubTask, TaskStatus, TaskType
from config.settings import settings

logger = structlog.get_logger()


class RetryAgent(BaseAgent):
    """
    Retry Agent responsible for:
    - Analyzing failed tasks
    - Determining appropriate retry strategies
    - Executing fallback approaches
    - Improving task execution based on errors
    """
    
    def __init__(self):
        super().__init__(
            name="retry_agent",
            description="Handles failed tasks and implements fallback strategies"
        )
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert problem-solving agent specializing in error recovery and retry strategies.

Your responsibilities:
1. Analyze why a task failed
2. Determine the best retry or fallback strategy
3. Modify the approach to increase chances of success
4. Know when to give up and report failure

Strategies available:
- RETRY_SAME: Retry the exact same approach (for transient errors)
- MODIFY_QUERY: Modify the search query or prompt
- SIMPLIFY: Break down into simpler sub-tasks
- ALTERNATIVE: Try a completely different approach
- SKIP: Skip this task if non-critical
- FAIL: Mark as failed if unrecoverable

When analyzing errors, consider:
- Is this a transient error (network, rate limit)?
- Is the task description unclear or too complex?
- Are there missing dependencies or context?
- Would different search terms help?
- Can we work around the issue?

Provide your strategy as JSON:
{
    "strategy": "RETRY_SAME|MODIFY_QUERY|SIMPLIFY|ALTERNATIVE|SKIP|FAIL",
    "reason": "Why this strategy",
    "modifications": {"key": "value"},  // Any modifications to apply
    "new_tasks": [{"description": "...", "task_type": "..."}]  // For SIMPLIFY
}"""
    
    async def execute(
        self, 
        task: SubTask, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle a failed task.
        
        Args:
            task: The failed task
            context: Error context and history
            
        Returns:
            Retry strategy and any new tasks
        """
        error = context.get("error", "Unknown error")
        retry_count = context.get("retry_count", 0)
        original_context = context.get("original_context", {})
        
        try:
            # Analyze the failure
            strategy = await self._analyze_failure(task, error, retry_count)
            
            logger.info(
                f"Retry strategy determined",
                agent=self.name,
                task_id=task.id,
                strategy=strategy.get("strategy")
            )
            
            # Execute the strategy
            result = await self._execute_strategy(
                task, 
                strategy, 
                original_context
            )
            
            return {
                "success": True,
                "strategy_used": strategy.get("strategy"),
                "reason": strategy.get("reason"),
                "modifications": strategy.get("modifications", {}),
                "new_tasks": result.get("new_tasks", []),
                "should_retry": result.get("should_retry", False),
                "modified_task": result.get("modified_task"),
                "agent": self.name
            }
            
        except Exception as e:
            return await self.handle_error(e, task, context)
    
    async def _analyze_failure(
        self, 
        task: SubTask, 
        error: str, 
        retry_count: int
    ) -> Dict[str, Any]:
        """Analyze why the task failed and determine strategy."""
        
        # Check for common error patterns
        error_lower = error.lower()
        
        # Transient errors - simple retry
        if any(pattern in error_lower for pattern in [
            "timeout", "rate limit", "connection", "temporary", "503", "502", "429"
        ]):
            if retry_count < self.max_retries:
                return {
                    "strategy": "RETRY_SAME",
                    "reason": "Transient error detected, simple retry may succeed",
                    "modifications": {}
                }
        
        # Search-related errors
        if task.task_type == TaskType.WEB_SEARCH:
            if "no results" in error_lower or "not found" in error_lower:
                return {
                    "strategy": "MODIFY_QUERY",
                    "reason": "Search returned no results, need different query",
                    "modifications": {"broaden_search": True}
                }
        
        # Code execution errors
        if task.task_type == TaskType.CODE_EXECUTION:
            if "syntax" in error_lower:
                return {
                    "strategy": "MODIFY_QUERY",
                    "reason": "Code syntax error, need to regenerate code",
                    "modifications": {"fix_syntax": True, "error_details": error}
                }
            if "timeout" in error_lower:
                return {
                    "strategy": "SIMPLIFY",
                    "reason": "Code execution timed out, need simpler approach",
                    "modifications": {"simplify_code": True}
                }
        
        # Complex task - might need simplification
        if retry_count >= 2:
            if "complex" in error_lower or "multiple" in error_lower or len(task.description) > 200:
                return {
                    "strategy": "SIMPLIFY",
                    "reason": "Task may be too complex, breaking down",
                    "modifications": {}
                }
        
        # Use LLM for deeper analysis
        return await self._llm_analyze_failure(task, error, retry_count)
    
    async def _llm_analyze_failure(
        self, 
        task: SubTask, 
        error: str, 
        retry_count: int
    ) -> Dict[str, Any]:
        """Use LLM to analyze complex failures."""
        
        prompt = f"""Analyze this task failure and recommend a retry strategy:

TASK:
Type: {task.task_type.value}
Description: {task.description}

ERROR:
{error}

RETRY COUNT: {retry_count}/{self.max_retries}

Choose the best strategy:
- RETRY_SAME: If this seems like a transient error
- MODIFY_QUERY: If the approach needs adjustment
- SIMPLIFY: If the task should be broken down
- ALTERNATIVE: If a different approach is needed
- SKIP: If this task is non-critical and blocking progress
- FAIL: If this is unrecoverable

Respond with JSON containing:
- strategy: one of the above
- reason: why you chose this strategy
- modifications: any specific changes to make"""

        response = await self.generate_response(prompt, temperature=0.2)
        
        # Parse the response
        import json
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                json_str = response
            
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            # Default to simple retry if parsing fails
            if retry_count < self.max_retries:
                return {
                    "strategy": "RETRY_SAME",
                    "reason": "Could not determine specific strategy, attempting retry",
                    "modifications": {}
                }
            return {
                "strategy": "FAIL",
                "reason": "Max retries exceeded",
                "modifications": {}
            }
    
    async def _execute_strategy(
        self,
        task: SubTask,
        strategy: Dict[str, Any],
        original_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the determined retry strategy."""
        
        strategy_type = strategy.get("strategy", "FAIL")
        
        if strategy_type == "RETRY_SAME":
            # Simple retry - wait and signal retry
            await asyncio.sleep(self.retry_delay)
            return {
                "should_retry": True,
                "modified_task": task
            }
        
        elif strategy_type == "MODIFY_QUERY":
            # Modify the task description
            modifications = strategy.get("modifications", {})
            modified_task = await self._modify_task(task, modifications)
            return {
                "should_retry": True,
                "modified_task": modified_task
            }
        
        elif strategy_type == "SIMPLIFY":
            # Break into smaller tasks
            new_tasks = await self._simplify_task(task)
            return {
                "should_retry": False,
                "new_tasks": new_tasks
            }
        
        elif strategy_type == "ALTERNATIVE":
            # Create alternative approach
            alt_task = await self._create_alternative(task)
            return {
                "should_retry": True,
                "modified_task": alt_task
            }
        
        elif strategy_type == "SKIP":
            # Mark as skipped
            task.status = TaskStatus.CANCELLED
            return {
                "should_retry": False,
                "skipped": True
            }
        
        else:  # FAIL
            task.status = TaskStatus.FAILED
            return {
                "should_retry": False,
                "failed": True
            }
    
    async def _modify_task(
        self, 
        task: SubTask, 
        modifications: Dict[str, Any]
    ) -> SubTask:
        """Modify task based on specified modifications."""
        
        prompt = f"""Improve this task description based on the error context:

Original task: {task.description}
Task type: {task.task_type.value}
Modifications needed: {modifications}

Provide an improved, clearer task description that is more likely to succeed.
Output only the new task description, nothing else."""

        new_description = await self.generate_response(prompt, temperature=0.3)
        
        # Create new task with modified description
        modified = SubTask(
            description=new_description.strip(),
            task_type=task.task_type,
            dependencies=task.dependencies,
            status=TaskStatus.PENDING,
            retry_count=task.retry_count + 1
        )
        
        return modified
    
    async def _simplify_task(self, task: SubTask) -> List[SubTask]:
        """Break a complex task into simpler sub-tasks."""
        
        prompt = f"""Break this task into 2-3 simpler, more focused sub-tasks:

Task: {task.description}
Type: {task.task_type.value}

For each sub-task, provide:
1. A clear, specific description
2. The appropriate task type (web_search, code_execution, research, validation, synthesis)

Format as:
1. [TASK_TYPE] Description
2. [TASK_TYPE] Description
..."""

        response = await self.generate_response(prompt, temperature=0.3)
        
        # Parse the response into sub-tasks
        new_tasks = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Extract task type and description
            task_type = TaskType.RESEARCH  # Default
            description = line
            
            for tt in TaskType:
                if tt.value.upper() in line.upper():
                    task_type = tt
                    # Remove the task type indicator from description
                    description = line.replace(f"[{tt.value.upper()}]", "").strip()
                    description = description.lstrip("0123456789.-) ").strip()
                    break
            
            if description and len(description) > 5:
                new_tasks.append(SubTask(
                    description=description,
                    task_type=task_type,
                    status=TaskStatus.PENDING
                ))
        
        return new_tasks if new_tasks else [task]  # Return original if parsing failed
    
    async def _create_alternative(self, task: SubTask) -> SubTask:
        """Create an alternative approach for the task."""
        
        prompt = f"""Suggest an alternative approach for this task:

Original task: {task.description}
Type: {task.task_type.value}

The original approach failed. Provide a different way to accomplish the same goal.
Consider using a different task type or methodology.

Output only the new task description."""

        new_description = await self.generate_response(prompt, temperature=0.5)
        
        return SubTask(
            description=new_description.strip(),
            task_type=task.task_type,
            dependencies=task.dependencies,
            status=TaskStatus.PENDING,
            retry_count=task.retry_count + 1
        )
