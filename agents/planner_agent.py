"""
AI Research Orchestrator - Planner Agent
Decomposes complex queries into actionable sub-tasks.
"""

import json
from typing import Dict, Any, List
import structlog

from .base_agent import BaseAgent
from models.task import SubTask, TaskType, TaskStatus, ResearchTask

logger = structlog.get_logger()


class PlannerAgent(BaseAgent):
    """
    Planner Agent responsible for:
    - Analyzing complex research queries
    - Decomposing them into structured sub-tasks
    - Identifying dependencies between tasks
    - Assigning appropriate task types
    """
    
    def __init__(self):
        super().__init__(
            name="planner",
            description="Analyzes queries and creates execution plans"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert research planner agent. Your role is to analyze complex research queries and decompose them into structured, actionable sub-tasks.

For each query, you must:
1. Understand the core research objective
2. Identify what information needs to be gathered
3. Determine if code execution or analysis is needed
4. Create a logical sequence of sub-tasks with clear dependencies

Available task types:
- RESEARCH: General research and information synthesis
- WEB_SEARCH: Search the web for specific information
- CODE_EXECUTION: Write and execute code for analysis/computation
- VALIDATION: Verify and validate information or results
- SYNTHESIS: Combine multiple sources into coherent output

Output your plan as a JSON array of sub-tasks with this structure:
{
    "analysis": "Brief analysis of the query",
    "sub_tasks": [
        {
            "description": "Clear description of what to do",
            "task_type": "WEB_SEARCH|CODE_EXECUTION|RESEARCH|VALIDATION|SYNTHESIS",
            "dependencies": ["task_id_1", "task_id_2"],  // IDs of tasks this depends on
            "priority": "high|medium|low"
        }
    ]
}

Guidelines:
- Keep sub-tasks focused and atomic
- Minimize dependencies where possible for parallel execution
- Include validation steps for critical information
- Always end with a synthesis task to combine results"""
    
    async def execute(
        self, 
        task: SubTask, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute planning for a research query.
        
        Args:
            task: Planning task containing the query
            context: Context including the main research task
            
        Returns:
            Plan with decomposed sub-tasks
        """
        query = context.get("query", task.description)
        additional_context = context.get("additional_context", "")
        
        prompt = f"""Analyze and decompose this research query into sub-tasks:

Query: {query}

{f'Additional Context: {additional_context}' if additional_context else ''}

Create a comprehensive plan to address this query. Consider what information needs to be searched, what computations might be needed, and how to synthesize the final output.

Output your plan as valid JSON."""

        try:
            response = await self.generate_response(prompt, temperature=0.3)
            
            # Parse the JSON response
            plan = self._parse_plan(response)
            
            # Create SubTask objects
            sub_tasks = self._create_subtasks(plan)
            
            logger.info(
                f"Plan created",
                agent=self.name,
                num_subtasks=len(sub_tasks)
            )
            
            return {
                "success": True,
                "analysis": plan.get("analysis", ""),
                "sub_tasks": [st.model_dump() for st in sub_tasks],
                "agent": self.name
            }
            
        except Exception as e:
            logger.error(f"Planning failed", error=str(e))
            return await self.handle_error(e, task, context)
    
    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse the JSON plan from the LLM response."""
        # Try to extract JSON from the response
        try:
            # Look for JSON block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # Try to find JSON object directly
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse plan JSON", error=str(e))
            # Return a default minimal plan
            return {
                "analysis": "Failed to parse plan, using default approach",
                "sub_tasks": [
                    {
                        "description": "Search for information about the query",
                        "task_type": "WEB_SEARCH",
                        "dependencies": [],
                        "priority": "high"
                    },
                    {
                        "description": "Synthesize findings into a coherent response",
                        "task_type": "SYNTHESIS",
                        "dependencies": [],
                        "priority": "medium"
                    }
                ]
            }
    
    def _create_subtasks(self, plan: Dict[str, Any]) -> List[SubTask]:
        """Create SubTask objects from the plan."""
        sub_tasks = []
        task_id_map = {}  # Map temporary IDs to actual IDs
        
        for i, task_def in enumerate(plan.get("sub_tasks", [])):
            task_type_str = task_def.get("task_type", "RESEARCH").upper()
            
            try:
                task_type = TaskType(task_type_str.lower())
            except ValueError:
                task_type = TaskType.RESEARCH
            
            subtask = SubTask(
                description=task_def.get("description", f"Task {i+1}"),
                task_type=task_type,
                dependencies=[],  # Will be resolved after all tasks are created
                status=TaskStatus.PENDING
            )
            
            task_id_map[str(i)] = subtask.id
            sub_tasks.append(subtask)
        
        # Resolve dependencies
        for i, task_def in enumerate(plan.get("sub_tasks", [])):
            deps = task_def.get("dependencies", [])
            if deps and i < len(sub_tasks):
                resolved_deps = []
                for dep in deps:
                    if str(dep) in task_id_map:
                        resolved_deps.append(task_id_map[str(dep)])
                sub_tasks[i].dependencies = resolved_deps
        
        return sub_tasks
    
    async def refine_plan(
        self, 
        original_plan: Dict[str, Any],
        feedback: str
    ) -> Dict[str, Any]:
        """
        Refine an existing plan based on feedback.
        
        Args:
            original_plan: The original plan
            feedback: Feedback about what needs to change
            
        Returns:
            Updated plan
        """
        prompt = f"""You previously created this research plan:

{json.dumps(original_plan, indent=2)}

However, there is feedback that requires changes:
{feedback}

Please provide an updated plan that addresses this feedback. Output as valid JSON."""

        response = await self.generate_response(prompt, temperature=0.3)
        return self._parse_plan(response)
