"""
AI Research Orchestrator - Task Models
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority level of a task."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    """Type of task to be executed."""
    RESEARCH = "research"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    PLANNING = "planning"


class SubTask(BaseModel):
    """A sub-task decomposed from the main query."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    task_type: TaskType
    dependencies: List[str] = Field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class ResearchTask(BaseModel):
    """Main research task container."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    context: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    sub_tasks: List[SubTask] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    final_output: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def add_subtask(self, subtask: SubTask) -> None:
        """Add a sub-task to this research task."""
        self.sub_tasks.append(subtask)
    
    def get_pending_subtasks(self) -> List[SubTask]:
        """Get all pending sub-tasks."""
        return [st for st in self.sub_tasks if st.status == TaskStatus.PENDING]
    
    def get_completed_subtasks(self) -> List[SubTask]:
        """Get all completed sub-tasks."""
        return [st for st in self.sub_tasks if st.status == TaskStatus.COMPLETED]
    
    def all_subtasks_complete(self) -> bool:
        """Check if all sub-tasks are complete."""
        return all(st.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED] 
                   for st in self.sub_tasks)


class SearchResult(BaseModel):
    """Web search result."""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    relevance_score: float = 0.0


class CodeExecutionResult(BaseModel):
    """Result from code execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    language: str = "python"


class AgentMessage(BaseModel):
    """Message passed between agents."""
    sender: str
    receiver: str
    content: str
    message_type: str = "info"
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorState(BaseModel):
    """State of the orchestrator."""
    current_task: Optional[ResearchTask] = None
    active_agents: List[str] = Field(default_factory=list)
    message_queue: List[AgentMessage] = Field(default_factory=list)
    iteration_count: int = 0
    total_retries: int = 0
