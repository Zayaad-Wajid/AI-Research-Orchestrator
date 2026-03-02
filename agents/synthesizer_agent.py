"""
AI Research Orchestrator - Synthesizer Agent
Synthesizes findings from multiple sources into coherent output.
"""

from typing import Dict, Any, List, Optional
import structlog

from .base_agent import BaseAgent
from models.task import SubTask

logger = structlog.get_logger()


class SynthesizerAgent(BaseAgent):
    """
    Synthesizer Agent responsible for:
    - Combining findings from multiple sources
    - Creating coherent, structured outputs
    - Resolving conflicting information
    - Generating final research reports
    """
    
    def __init__(self):
        super().__init__(
            name="synthesizer",
            description="Synthesizes research findings into coherent output"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert research synthesizer. Your role is to combine findings from multiple sources into clear, coherent, and comprehensive outputs.

Your responsibilities:
1. Integrate information from multiple research sources
2. Identify key themes and insights
3. Resolve conflicting information
4. Create well-structured, readable outputs
5. Ensure completeness and accuracy

Synthesis principles:
- Maintain objectivity and balance
- Highlight consensus and disagreements
- Organize information logically
- Use clear, accessible language
- Cite sources appropriately

Output should include:
1. Executive summary (2-3 sentences)
2. Key findings organized by theme
3. Detailed analysis
4. Conclusions and insights
5. Recommendations (if applicable)
6. Sources and references"""
    
    async def execute(
        self, 
        task: SubTask, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize research findings.
        
        Args:
            task: Synthesis task
            context: Contains all findings to synthesize
            
        Returns:
            Synthesized output
        """
        original_query = context.get("query", task.description)
        findings_list = context.get("findings", [])
        validation_results = context.get("validation_results", {})
        include_code_results = context.get("code_results")
        
        try:
            synthesis = await self._synthesize(
                original_query,
                findings_list,
                validation_results,
                include_code_results
            )
            
            logger.info(
                f"Synthesis completed",
                agent=self.name,
                output_length=len(synthesis.get("report", ""))
            )
            
            return {
                "success": True,
                "report": synthesis.get("report", ""),
                "summary": synthesis.get("summary", ""),
                "key_findings": synthesis.get("key_findings", []),
                "sources": synthesis.get("sources", []),
                "confidence": synthesis.get("confidence", "medium"),
                "agent": self.name
            }
            
        except Exception as e:
            return await self.handle_error(e, task, context)
    
    async def _synthesize(
        self,
        query: str,
        findings: List[Any],
        validation_results: Dict[str, Any],
        code_results: Optional[str]
    ) -> Dict[str, Any]:
        """Perform synthesis of all gathered information."""
        
        # Format findings for synthesis
        findings_text = self._format_findings(findings)
        
        validation_text = ""
        if validation_results:
            validation_text = f"""
VALIDATION STATUS: {validation_results.get('validation_status', 'N/A')}
Confidence Score: {validation_results.get('confidence_score', 'N/A')}
Verified Claims: {', '.join(validation_results.get('verified_claims', [])[:5])}
Issues: {', '.join(validation_results.get('inconsistencies', [])[:3])}
"""
        
        code_text = ""
        if code_results:
            code_text = f"\n\nCODE ANALYSIS RESULTS:\n{code_results}"
        
        prompt = f"""Synthesize the following research findings into a comprehensive report.

ORIGINAL QUERY: {query}

RESEARCH FINDINGS:
{findings_text}
{validation_text}
{code_text}

Create a well-structured report with:
1. **Executive Summary** (2-3 sentences capturing the essence)
2. **Key Findings** (bullet points of main discoveries)
3. **Detailed Analysis** (organized by themes/topics)
4. **Conclusions** (what we can conclude from this research)
5. **Limitations** (any gaps or uncertainties)
6. **Sources** (list of sources used)

Write in a clear, professional style. Be comprehensive but concise."""

        response = await self.generate_response(prompt, temperature=0.4)
        
        # Extract structured components from response
        summary = self._extract_summary(response)
        key_findings = self._extract_key_findings(response)
        sources = self._extract_sources(response, findings)
        
        return {
            "report": response,
            "summary": summary,
            "key_findings": key_findings,
            "sources": sources,
            "confidence": validation_results.get("validation_status", "medium") if validation_results else "medium"
        }
    
    def _format_findings(self, findings: List[Any]) -> str:
        """Format findings list into readable text."""
        if not findings:
            return "No findings provided."
        
        formatted_parts = []
        for i, finding in enumerate(findings, 1):
            if isinstance(finding, dict):
                content = finding.get("findings") or finding.get("content") or str(finding)
                source = finding.get("source") or finding.get("agent", f"Source {i}")
                formatted_parts.append(f"### {source}\n{content}\n")
            elif isinstance(finding, str):
                formatted_parts.append(f"### Finding {i}\n{finding}\n")
            else:
                formatted_parts.append(f"### Finding {i}\n{str(finding)}\n")
        
        return "\n".join(formatted_parts)
    
    def _extract_summary(self, report: str) -> str:
        """Extract executive summary from the report."""
        lines = report.split('\n')
        in_summary = False
        summary_lines = []
        
        for line in lines:
            if "executive summary" in line.lower() or "summary" in line.lower()[:20]:
                in_summary = True
                continue
            if in_summary:
                if line.startswith('#') or line.startswith('**') and 'finding' in line.lower():
                    break
                if line.strip():
                    summary_lines.append(line.strip())
                if len(summary_lines) >= 3:
                    break
        
        if summary_lines:
            return ' '.join(summary_lines)
        
        # Fallback: use first paragraph
        paragraphs = report.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 50 and not para.startswith('#'):
                return para.strip()[:500]
        
        return report[:500]
    
    def _extract_key_findings(self, report: str) -> List[str]:
        """Extract key findings from the report."""
        findings = []
        lines = report.split('\n')
        in_findings = False
        
        for line in lines:
            if "key finding" in line.lower():
                in_findings = True
                continue
            if in_findings:
                if line.startswith('#') and 'finding' not in line.lower():
                    break
                line = line.strip()
                if line.startswith(('-', '*', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    # Clean up the bullet point
                    finding = line.lstrip('-*•0123456789. ').strip()
                    if finding and len(finding) > 10:
                        findings.append(finding)
        
        return findings[:10]  # Limit to 10 findings
    
    def _extract_sources(
        self, 
        report: str, 
        findings: List[Any]
    ) -> List[str]:
        """Extract sources from findings and report."""
        sources = set()
        
        # Extract from findings
        for finding in findings:
            if isinstance(finding, dict):
                if "sources" in finding:
                    for s in finding["sources"]:
                        if isinstance(s, str) and s.startswith('http'):
                            sources.add(s)
                if "url" in finding:
                    sources.add(finding["url"])
        
        return list(sources)[:20]  # Limit to 20 sources
    
    async def create_brief(
        self, 
        content: str, 
        max_words: int = 200
    ) -> str:
        """
        Create a brief summary of content.
        
        Args:
            content: Content to summarize
            max_words: Maximum words in brief
            
        Returns:
            Brief summary
        """
        prompt = f"""Create a brief summary of the following content in {max_words} words or less:

{content}

Be concise, capture the key points, and write in a clear professional style."""

        return await self.generate_response(prompt, temperature=0.3)
