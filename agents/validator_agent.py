"""
AI Research Orchestrator - Validator Agent
Validates and verifies research findings.
"""

from typing import Dict, Any, List
import structlog

from .base_agent import BaseAgent
from models.task import SubTask

logger = structlog.get_logger()


class ValidatorAgent(BaseAgent):
    """
    Validator Agent responsible for:
    - Verifying accuracy of research findings
    - Cross-referencing information
    - Identifying inconsistencies or errors
    - Assessing source credibility
    """
    
    def __init__(self):
        super().__init__(
            name="validator",
            description="Validates and verifies research findings"
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert validation agent. Your role is to critically evaluate research findings and ensure accuracy.

Your responsibilities:
1. Verify factual accuracy of claims
2. Check for logical consistency
3. Identify potential biases or errors
4. Assess source credibility
5. Cross-reference information across sources
6. Flag uncertain or unverified claims

Validation criteria:
- Factual accuracy: Are claims supported by evidence?
- Source quality: Are sources authoritative and recent?
- Consistency: Do findings align with each other?
- Completeness: Are there gaps in the information?
- Bias: Are there any apparent biases?

Output format:
{
    "validation_status": "pass|partial|fail",
    "confidence_score": 0.0-1.0,
    "verified_claims": ["claim 1", "claim 2"],
    "unverified_claims": ["claim 3"],
    "inconsistencies": ["inconsistency 1"],
    "recommendations": ["recommendation 1"],
    "summary": "Overall validation summary"
}"""
    
    async def execute(
        self, 
        task: SubTask, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate research findings.
        
        Args:
            task: Validation task
            context: Contains findings to validate
            
        Returns:
            Validation results
        """
        findings = context.get("findings", task.description)
        sources = context.get("sources", [])
        
        try:
            validation_result = await self._validate_findings(findings, sources)
            
            logger.info(
                f"Validation completed",
                agent=self.name,
                status=validation_result.get("validation_status")
            )
            
            return {
                "success": True,
                "validation_status": validation_result.get("validation_status", "partial"),
                "confidence_score": validation_result.get("confidence_score", 0.5),
                "verified_claims": validation_result.get("verified_claims", []),
                "unverified_claims": validation_result.get("unverified_claims", []),
                "inconsistencies": validation_result.get("inconsistencies", []),
                "recommendations": validation_result.get("recommendations", []),
                "summary": validation_result.get("summary", ""),
                "agent": self.name
            }
            
        except Exception as e:
            return await self.handle_error(e, task, context)
    
    async def _validate_findings(
        self, 
        findings: str, 
        sources: List[str]
    ) -> Dict[str, Any]:
        """Perform validation analysis on findings."""
        
        sources_text = "\n".join(f"- {s}" for s in sources[:10]) if sources else "No sources provided"
        
        prompt = f"""Critically validate these research findings:

FINDINGS:
{findings}

SOURCES:
{sources_text}

Analyze for:
1. Factual accuracy - Are claims verifiable?
2. Logical consistency - Do findings make sense together?
3. Source credibility - Are sources trustworthy?
4. Completeness - Any important gaps?
5. Potential biases - Any apparent biases?

Provide your validation as JSON with:
- validation_status: "pass" (all verified), "partial" (some issues), or "fail" (major issues)
- confidence_score: 0.0 to 1.0
- verified_claims: list of verified claims
- unverified_claims: list of claims that couldn't be verified
- inconsistencies: list of any inconsistencies found
- recommendations: list of recommendations for improvement
- summary: brief overall summary"""

        response = await self.generate_response(prompt, temperature=0.2)
        
        # Parse JSON response
        import json
        try:
            # Extract JSON from response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            
            result = json.loads(json_str)
            return result
            
        except json.JSONDecodeError:
            # Parse manually if JSON fails
            return self._parse_validation_text(response)
    
    def _parse_validation_text(self, text: str) -> Dict[str, Any]:
        """Parse validation response when JSON parsing fails."""
        result = {
            "validation_status": "partial",
            "confidence_score": 0.5,
            "verified_claims": [],
            "unverified_claims": [],
            "inconsistencies": [],
            "recommendations": [],
            "summary": text[:500] if len(text) > 500 else text
        }
        
        text_lower = text.lower()
        
        # Determine status based on keywords
        if any(word in text_lower for word in ["all verified", "fully validated", "accurate", "pass"]):
            result["validation_status"] = "pass"
            result["confidence_score"] = 0.8
        elif any(word in text_lower for word in ["major issues", "fail", "inaccurate", "unreliable"]):
            result["validation_status"] = "fail"
            result["confidence_score"] = 0.3
        
        return result
    
    async def quick_check(
        self, 
        claim: str
    ) -> Dict[str, Any]:
        """
        Quick validation check for a single claim.
        
        Args:
            claim: The claim to validate
            
        Returns:
            Quick validation result
        """
        prompt = f"""Quick validation check for this claim:

Claim: {claim}

Provide:
1. Is this claim likely accurate? (yes/no/uncertain)
2. Confidence (0-100%)
3. Brief reasoning (1-2 sentences)

Format: [yes/no/uncertain] - [confidence]% - [reasoning]"""

        response = await self.generate_response(prompt, temperature=0.2)
        
        # Parse simple response
        is_valid = "yes" in response.lower()[:20]
        is_uncertain = "uncertain" in response.lower()[:30]
        
        return {
            "claim": claim,
            "valid": is_valid,
            "uncertain": is_uncertain,
            "response": response.strip()
        }
