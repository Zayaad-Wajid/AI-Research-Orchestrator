"""
AI Research Orchestrator - Researcher Agent
Handles research tasks and information gathering.
"""

from typing import Dict, Any, List, Optional
import structlog
import asyncio

from .base_agent import BaseAgent
from models.task import SubTask, SearchResult
from tools.web_search import web_search_tool

logger = structlog.get_logger()


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent responsible for:
    - Conducting web searches
    - Analyzing search results
    - Extracting relevant information
    - Summarizing findings
    """
    
    def __init__(self):
        super().__init__(
            name="researcher",
            description="Conducts research and gathers information"
        )
        self.search_tool = web_search_tool
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert research agent. Your role is to gather, analyze, and summarize information from various sources.

Your responsibilities:
1. Formulate effective search queries based on research objectives
2. Analyze search results for relevance and reliability
3. Extract key information and insights
4. Identify gaps in information that require additional searches
5. Synthesize findings into clear, well-structured summaries

When analyzing information:
- Prioritize authoritative and recent sources
- Cross-reference information across multiple sources
- Note any conflicting information or uncertainties
- Cite sources appropriately

Output format:
Provide your findings in a structured format with:
- Key findings (bullet points)
- Supporting details
- Source references
- Confidence level (high/medium/low)
- Any additional searches recommended"""
    
    async def execute(
        self, 
        task: SubTask, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute research task.
        
        Args:
            task: Research task with description
            context: Additional context from orchestrator
            
        Returns:
            Research findings
        """
        query = task.description
        previous_results = context.get("previous_results", [])
        max_searches = context.get("max_searches", 3)
        
        try:
            # Generate search queries
            search_queries = await self._generate_search_queries(
                query, 
                previous_results
            )
            
            # Execute searches
            all_results = []
            for search_query in search_queries[:max_searches]:
                results = await self.search_tool.search(
                    search_query, 
                    num_results=5,
                    fetch_content=True
                )
                all_results.extend(results)
            
            if not all_results:
                return {
                    "success": False,
                    "error": "No search results found",
                    "agent": self.name,
                    "retry_recommended": True
                }
            
            # Analyze and synthesize results
            analysis = await self._analyze_results(query, all_results)
            
            logger.info(
                f"Research completed",
                agent=self.name,
                num_results=len(all_results),
                num_queries=len(search_queries)
            )
            
            return {
                "success": True,
                "findings": analysis["findings"],
                "sources": [r.url for r in all_results if r.url],
                "confidence": analysis.get("confidence", "medium"),
                "additional_queries": analysis.get("additional_queries", []),
                "agent": self.name
            }
            
        except Exception as e:
            return await self.handle_error(e, task, context)
    
    async def _generate_search_queries(
        self, 
        research_objective: str,
        previous_results: List[str]
    ) -> List[str]:
        """Generate effective search queries for the research objective."""
        
        context = ""
        if previous_results:
            context = f"Previous research has covered: {', '.join(previous_results[:5])}"
        
        prompt = f"""Generate 3 effective web search queries to research this topic:

Topic: {research_objective}

{context}

Requirements:
- Make queries specific and targeted
- Cover different aspects of the topic
- Avoid redundancy with previous research
- Use search operators where helpful (quotes for exact phrases)

Output only the queries, one per line, no numbering or formatting."""

        response = await self.generate_response(prompt, temperature=0.4)
        
        # Parse queries from response
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        
        # Ensure we have at least the original query
        if not queries:
            queries = [research_objective]
        
        return queries
    
    async def _analyze_results(
        self, 
        query: str,
        results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Analyze search results and extract key findings."""
        
        # Format results for analysis
        results_text = ""
        for i, r in enumerate(results[:10], 1):
            results_text += f"\n### Source {i}: {r.title}\n"
            results_text += f"URL: {r.url}\n"
            results_text += f"Snippet: {r.snippet}\n"
            if r.content:
                # Include first 1000 chars of content
                content = r.content[:1000] + "..." if len(r.content) > 1000 else r.content
                results_text += f"Content: {content}\n"
        
        prompt = f"""Analyze these search results for the research query:

Query: {query}

Search Results:
{results_text}

Provide:
1. Key findings (5-10 bullet points)
2. Confidence level (high/medium/low) based on source quality and consistency
3. Any gaps or additional queries needed

Format as:
## Key Findings
- Finding 1
- Finding 2
...

## Confidence
[level]: [explanation]

## Additional Queries Needed
- Query 1 (if any)"""

        response = await self.generate_response(prompt, temperature=0.3)
        
        # Parse the response
        findings = response
        confidence = "medium"
        additional_queries = []
        
        if "## Confidence" in response:
            parts = response.split("## Confidence")
            findings = parts[0].replace("## Key Findings", "").strip()
            
            if len(parts) > 1:
                confidence_section = parts[1]
                if "high" in confidence_section.lower():
                    confidence = "high"
                elif "low" in confidence_section.lower():
                    confidence = "low"
                
                if "## Additional Queries" in confidence_section:
                    queries_part = confidence_section.split("## Additional Queries")[1]
                    additional_queries = [
                        q.strip().lstrip("- ").strip() 
                        for q in queries_part.split("\n") 
                        if q.strip() and q.strip() != "-"
                    ]
        
        return {
            "findings": findings,
            "confidence": confidence,
            "additional_queries": additional_queries
        }
    
    async def deep_dive(
        self, 
        topic: str, 
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Conduct a deep dive research on a topic.
        
        Args:
            topic: Topic to research
            depth: How many levels of follow-up research to conduct
            
        Returns:
            Comprehensive research findings
        """
        all_findings = []
        covered_queries = set()
        pending_queries = [topic]
        
        for level in range(depth):
            if not pending_queries:
                break
            
            current_query = pending_queries.pop(0)
            if current_query in covered_queries:
                continue
            
            covered_queries.add(current_query)
            
            # Create a temporary subtask for this query
            temp_task = SubTask(
                description=current_query,
                task_type="web_search"
            )
            
            result = await self.execute(temp_task, {
                "previous_results": list(covered_queries)
            })
            
            if result.get("success"):
                all_findings.append({
                    "query": current_query,
                    "level": level,
                    "findings": result.get("findings"),
                    "sources": result.get("sources", [])
                })
                
                # Add additional queries for next level
                for q in result.get("additional_queries", []):
                    if q not in covered_queries:
                        pending_queries.append(q)
        
        return {
            "topic": topic,
            "depth_reached": len(all_findings),
            "findings": all_findings,
            "queries_covered": list(covered_queries)
        }
