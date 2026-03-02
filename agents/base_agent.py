"""
AI Research Orchestrator - Base Agent
Abstract base class for all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import structlog
from datetime import datetime
import google.generativeai as genai

from config.settings import settings
from models.task import SubTask, TaskStatus, AgentMessage

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Provides common functionality for LLM interaction and message handling.
    """
    
    def __init__(
        self, 
        name: str,
        description: str,
        model: str = None
    ):
        self.name = name
        self.description = description
        self.model = model or settings.gemini_model
        self._initialized = False
        self._gemini_model = None
        
    def initialize(self):
        """Initialize the agent's LLM connection."""
        if self._initialized:
            return
            
        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
            self._gemini_model = genai.GenerativeModel(self.model)
            self._initialized = True
            logger.info(f"Agent initialized", agent=self.name, model=self.model)
        else:
            logger.warning(f"No Google API key configured for agent", agent=self.name)
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt that defines the agent's behavior."""
        pass
    
    @abstractmethod
    async def execute(self, task: SubTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task.
        
        Args:
            task: The sub-task to execute
            context: Additional context from the orchestrator
            
        Returns:
            Result dictionary with output and metadata
        """
        pass
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response using the configured LLM.
        
        Args:
            prompt: The prompt to send
            context: Optional additional context
            temperature: Generation temperature
            
        Returns:
            Generated response text
        """
        self.initialize()
        
        full_prompt = self.system_prompt + "\n\n"
        if context:
            full_prompt += f"Context:\n{context}\n\n"
        full_prompt += f"Task:\n{prompt}"
        
        try:
            response = await self._gemini_model.generate_content_async(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=4096
                )
            )
            
            if response.text:
                logger.info(f"Generated response", agent=self.name, length=len(response.text))
                return response.text
            else:
                logger.warning(f"Empty response from LLM", agent=self.name)
                return ""
                
        except Exception as e:
            logger.error(f"LLM generation failed", agent=self.name, error=str(e))
            raise
    
    def create_message(
        self, 
        receiver: str, 
        content: str, 
        message_type: str = "info"
    ) -> AgentMessage:
        """Create a message to send to another agent."""
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=message_type,
            timestamp=datetime.now()
        )
    
    async def handle_error(
        self, 
        error: Exception, 
        task: SubTask,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle errors during task execution.
        Can be overridden by subclasses for custom error handling.
        """
        logger.error(
            f"Agent error",
            agent=self.name,
            task_id=task.id,
            error=str(error)
        )
        
        return {
            "success": False,
            "error": str(error),
            "agent": self.name,
            "task_id": task.id,
            "retry_recommended": True
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
