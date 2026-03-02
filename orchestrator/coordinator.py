"""
AI Research Orchestrator - Main Coordinator
Orchestrates all agents and manages the research workflow.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

from models.task import (
    ResearchTask, SubTask, TaskStatus, TaskType,
    OrchestratorState, AgentMessage
)
from agents import (
    PlannerAgent, ResearcherAgent, ToolAgent,
    ValidatorAgent, SynthesizerAgent, RetryAgent
)
from config.settings import settings

logger = structlog.get_logger()
console = Console()


class ResearchOrchestrator:
    """
    Main orchestrator that coordinates all agents to complete research tasks.
    
    Workflow:
    1. Receive research query
    2. Plan: Decompose query into sub-tasks
    3. Execute: Run sub-tasks with appropriate agents
    4. Validate: Verify findings
    5. Synthesize: Combine results into final output
    6. Handle failures with retry logic
    """
    
    def __init__(self):
        # Initialize all agents
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.tool_agent = ToolAgent()
        self.validator = ValidatorAgent()
        self.synthesizer = SynthesizerAgent()
        self.retry_agent = RetryAgent()
        
        # Agent registry for dynamic dispatch
        self.agents = {
            "planner": self.planner,
            "researcher": self.researcher,
            "tool_agent": self.tool_agent,
            "validator": self.validator,
            "synthesizer": self.synthesizer,
            "retry_agent": self.retry_agent
        }
        
        # State management
        self.state = OrchestratorState()
        self.max_iterations = 20
        self.max_parallel_tasks = settings.max_concurrent_agents
        
        logger.info("Orchestrator initialized", agents=list(self.agents.keys()))
    
    async def research(
        self, 
        query: str, 
        context: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a complete research workflow.
        
        Args:
            query: The research query
            context: Optional additional context
            verbose: Whether to show progress output
            
        Returns:
            Complete research results
        """
        start_time = datetime.now()
        
        # Create research task
        task = ResearchTask(
            query=query,
            context=context,
            status=TaskStatus.IN_PROGRESS
        )
        self.state.current_task = task
        
        if verbose:
            console.print(Panel(
                f"[bold blue]Research Query:[/bold blue]\n{query}",
                title="🔬 AI Research Orchestrator",
                expand=False
            ))
        
        try:
            # Phase 1: Planning
            if verbose:
                console.print("\n[bold cyan]Phase 1: Planning...[/bold cyan]")
            
            plan_result = await self._execute_planning(task)
            if not plan_result.get("success"):
                return self._create_error_result(task, "Planning failed", plan_result)
            
            # Add sub-tasks to the research task
            for subtask_dict in plan_result.get("sub_tasks", []):
                subtask = SubTask(**subtask_dict) if isinstance(subtask_dict, dict) else subtask_dict
                task.add_subtask(subtask)
            
            if verbose:
                console.print(f"  Created {len(task.sub_tasks)} sub-tasks")
            
            # Phase 2: Execution
            if verbose:
                console.print("\n[bold cyan]Phase 2: Executing sub-tasks...[/bold cyan]")
            
            execution_results = await self._execute_subtasks(task, verbose)
            
            # Phase 3: Validation
            if verbose:
                console.print("\n[bold cyan]Phase 3: Validating findings...[/bold cyan]")
            
            validation_result = await self._execute_validation(
                execution_results.get("findings", [])
            )
            
            # Phase 4: Synthesis
            if verbose:
                console.print("\n[bold cyan]Phase 4: Synthesizing results...[/bold cyan]")
            
            synthesis_result = await self._execute_synthesis(
                task,
                execution_results,
                validation_result
            )
            
            # Finalize
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.final_output = synthesis_result.get("report", "")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if verbose:
                console.print(Panel(
                    Markdown(task.final_output[:2000] + "..." if len(task.final_output) > 2000 else task.final_output),
                    title="📋 Research Results",
                    expand=False
                ))
                console.print(f"\n[green]✓ Research completed in {duration:.2f} seconds[/green]")
            
            return {
                "success": True,
                "query": query,
                "report": task.final_output,
                "summary": synthesis_result.get("summary", ""),
                "key_findings": synthesis_result.get("key_findings", []),
                "sources": synthesis_result.get("sources", []),
                "validation": validation_result,
                "duration_seconds": duration,
                "sub_tasks_completed": len(task.get_completed_subtasks()),
                "total_retries": self.state.total_retries
            }
            
        except Exception as e:
            logger.error("Research failed", error=str(e))
            task.status = TaskStatus.FAILED
            return self._create_error_result(task, str(e), {})
    
    async def _execute_planning(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute the planning phase."""
        planning_subtask = SubTask(
            description="Create research plan",
            task_type=TaskType.PLANNING
        )
        
        result = await self.planner.execute(
            planning_subtask,
            {"query": task.query, "additional_context": task.context}
        )
        
        return result
    
    async def _execute_subtasks(
        self, 
        task: ResearchTask,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Execute all sub-tasks with parallel processing where possible."""
        
        findings = []
        code_results = []
        completed = set()
        iteration = 0
        
        while not task.all_subtasks_complete() and iteration < self.max_iterations:
            iteration += 1
            self.state.iteration_count = iteration
            
            # Get executable tasks (dependencies satisfied)
            executable = self._get_executable_tasks(task, completed)
            
            if not executable:
                # Check for deadlock
                pending = [st for st in task.sub_tasks 
                          if st.status == TaskStatus.PENDING]
                if pending:
                    logger.warning("Potential deadlock, forcing execution")
                    executable = pending[:1]
                else:
                    break
            
            # Execute in parallel batches
            batch_size = min(len(executable), self.max_parallel_tasks)
            batch = executable[:batch_size]
            
            if verbose:
                for st in batch:
                    console.print(f"  → Executing: {st.description[:50]}...")
            
            results = await asyncio.gather(
                *[self._execute_single_task(st, task) for st in batch],
                return_exceptions=True
            )
            
            # Process results
            for subtask, result in zip(batch, results):
                if isinstance(result, Exception):
                    await self._handle_task_failure(subtask, str(result), task)
                elif result.get("success"):
                    subtask.status = TaskStatus.COMPLETED
                    subtask.result = result
                    subtask.completed_at = datetime.now()
                    completed.add(subtask.id)
                    
                    # Collect findings
                    if "findings" in result:
                        findings.append({
                            "source": subtask.description,
                            "findings": result["findings"],
                            "confidence": result.get("confidence", "medium")
                        })
                    if "output" in result and subtask.task_type == TaskType.CODE_EXECUTION:
                        code_results.append({
                            "task": subtask.description,
                            "output": result["output"],
                            "analysis": result.get("analysis", "")
                        })
                else:
                    await self._handle_task_failure(
                        subtask, 
                        result.get("error", "Unknown error"),
                        task,
                        result.get("retry_context")
                    )
        
        return {
            "findings": findings,
            "code_results": code_results,
            "completed_count": len(completed)
        }
    
    async def _execute_single_task(
        self, 
        subtask: SubTask,
        parent_task: ResearchTask
    ) -> Dict[str, Any]:
        """Execute a single sub-task with the appropriate agent."""
        
        subtask.status = TaskStatus.IN_PROGRESS
        
        # Select agent based on task type
        agent = self._select_agent(subtask.task_type)
        
        # Build context
        context = {
            "query": parent_task.query,
            "previous_results": [
                st.result for st in parent_task.sub_tasks 
                if st.status == TaskStatus.COMPLETED and st.result
            ]
        }
        
        try:
            result = await agent.execute(subtask, context)
            return result
            
        except Exception as e:
            logger.error(f"Task execution error", task_id=subtask.id, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "retry_recommended": True
            }
    
    def _select_agent(self, task_type: TaskType):
        """Select the appropriate agent for a task type."""
        agent_map = {
            TaskType.PLANNING: self.planner,
            TaskType.WEB_SEARCH: self.researcher,
            TaskType.RESEARCH: self.researcher,
            TaskType.CODE_EXECUTION: self.tool_agent,
            TaskType.VALIDATION: self.validator,
            TaskType.SYNTHESIS: self.synthesizer
        }
        return agent_map.get(task_type, self.researcher)
    
    def _get_executable_tasks(
        self, 
        task: ResearchTask,
        completed: set
    ) -> List[SubTask]:
        """Get tasks whose dependencies are satisfied."""
        executable = []
        
        for subtask in task.sub_tasks:
            if subtask.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_satisfied = all(
                dep in completed for dep in subtask.dependencies
            )
            
            if deps_satisfied:
                executable.append(subtask)
        
        return executable
    
    async def _handle_task_failure(
        self,
        subtask: SubTask,
        error: str,
        parent_task: ResearchTask,
        retry_context: Optional[Dict] = None
    ):
        """Handle a failed task with retry logic."""
        
        subtask.error = error
        subtask.retry_count += 1
        self.state.total_retries += 1
        
        if subtask.retry_count > settings.max_retries:
            logger.warning(f"Max retries exceeded", task_id=subtask.id)
            subtask.status = TaskStatus.FAILED
            return
        
        # Use retry agent to determine strategy
        retry_result = await self.retry_agent.execute(
            subtask,
            {
                "error": error,
                "retry_count": subtask.retry_count,
                "original_context": retry_context or {}
            }
        )
        
        if retry_result.get("should_retry"):
            # Update task with modifications
            modified = retry_result.get("modified_task")
            if modified:
                subtask.description = modified.description
            subtask.status = TaskStatus.PENDING
            
        elif retry_result.get("new_tasks"):
            # Add new simplified tasks
            for new_task in retry_result["new_tasks"]:
                if isinstance(new_task, dict):
                    new_task = SubTask(**new_task)
                parent_task.add_subtask(new_task)
            subtask.status = TaskStatus.CANCELLED
            
        else:
            subtask.status = TaskStatus.FAILED
    
    async def _execute_validation(
        self, 
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute validation on all findings."""
        
        if not findings:
            return {"validation_status": "skip", "reason": "No findings to validate"}
        
        # Combine findings for validation
        combined_findings = "\n\n".join([
            f"Source: {f.get('source', 'Unknown')}\n{f.get('findings', '')}"
            for f in findings
        ])
        
        sources = []
        for f in findings:
            if "sources" in f:
                sources.extend(f["sources"])
        
        validation_task = SubTask(
            description="Validate research findings",
            task_type=TaskType.VALIDATION
        )
        
        result = await self.validator.execute(
            validation_task,
            {"findings": combined_findings, "sources": sources}
        )
        
        return result
    
    async def _execute_synthesis(
        self,
        task: ResearchTask,
        execution_results: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute synthesis to create final output."""
        
        synthesis_task = SubTask(
            description="Synthesize research findings",
            task_type=TaskType.SYNTHESIS
        )
        
        # Prepare code results summary
        code_summary = None
        if execution_results.get("code_results"):
            code_parts = []
            for cr in execution_results["code_results"]:
                code_parts.append(f"Task: {cr['task']}\nOutput: {cr['output']}\nAnalysis: {cr.get('analysis', '')}")
            code_summary = "\n\n".join(code_parts)
        
        result = await self.synthesizer.execute(
            synthesis_task,
            {
                "query": task.query,
                "findings": execution_results.get("findings", []),
                "validation_results": validation_result,
                "code_results": code_summary
            }
        )
        
        return result
    
    def _create_error_result(
        self, 
        task: ResearchTask, 
        error: str,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an error result object."""
        return {
            "success": False,
            "query": task.query,
            "error": error,
            "details": details,
            "partial_results": {
                "completed_subtasks": [
                    st.model_dump() for st in task.get_completed_subtasks()
                ]
            }
        }
    
    async def quick_search(self, query: str) -> Dict[str, Any]:
        """
        Quick search without full orchestration.
        Useful for simple queries.
        """
        task = SubTask(description=query, task_type=TaskType.WEB_SEARCH)
        result = await self.researcher.execute(task, {"max_searches": 2})
        return result
    
    async def execute_code(self, code: str) -> Dict[str, Any]:
        """
        Direct code execution without orchestration.
        """
        task = SubTask(description="Execute code", task_type=TaskType.CODE_EXECUTION)
        result = await self.tool_agent.execute(task, {"code": code})
        return result
