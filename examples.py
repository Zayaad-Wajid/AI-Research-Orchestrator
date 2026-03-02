"""
AI Research Orchestrator - Example Usage
Demonstrates various ways to use the orchestrator.
"""

import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from orchestrator import ResearchOrchestrator
from agents import PlannerAgent, ResearcherAgent, ToolAgent


async def example_basic_research():
    """Basic research example."""
    print("\n" + "="*60)
    print("Example 1: Basic Research Query")
    print("="*60)
    
    orchestrator = ResearchOrchestrator()
    
    result = await orchestrator.research(
        "What are the key differences between transformers and RNNs in deep learning?",
        verbose=True
    )
    
    if result["success"]:
        print("\n✓ Research completed successfully!")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
        print(f"Key findings: {len(result.get('key_findings', []))} points")
    else:
        print(f"✗ Research failed: {result.get('error')}")


async def example_quick_search():
    """Quick search example without full orchestration."""
    print("\n" + "="*60)
    print("Example 2: Quick Search")
    print("="*60)
    
    orchestrator = ResearchOrchestrator()
    
    result = await orchestrator.quick_search(
        "latest Python 3.12 features"
    )
    
    if result["success"]:
        print("\n✓ Search completed!")
        print(f"Findings:\n{result.get('findings', 'No findings')[:500]}...")
    else:
        print(f"✗ Search failed: {result.get('error')}")


async def example_code_execution():
    """Code execution example."""
    print("\n" + "="*60)
    print("Example 3: Code Execution")
    print("="*60)
    
    orchestrator = ResearchOrchestrator()
    
    code = """
import math

# Calculate some statistics
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean = sum(numbers) / len(numbers)
variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
std_dev = math.sqrt(variance)

print(f"Numbers: {numbers}")
print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev:.4f}")
"""
    
    result = await orchestrator.execute_code(code)
    
    if result.get("success"):
        print("\n✓ Code executed successfully!")
        print(f"Output:\n{result.get('output', 'No output')}")
    else:
        print(f"✗ Execution failed: {result.get('error')}")


async def example_using_agents_directly():
    """Example of using individual agents directly."""
    print("\n" + "="*60)
    print("Example 4: Using Agents Directly")
    print("="*60)
    
    # Use the planner agent to decompose a query
    planner = PlannerAgent()
    from models.task import SubTask, TaskType
    
    planning_task = SubTask(
        description="Plan research",
        task_type=TaskType.PLANNING
    )
    
    plan_result = await planner.execute(
        planning_task,
        {"query": "How does CRISPR gene editing work and what are its applications?"}
    )
    
    if plan_result["success"]:
        print("\n✓ Planning completed!")
        print(f"Analysis: {plan_result.get('analysis', 'N/A')}")
        print(f"Sub-tasks created: {len(plan_result.get('sub_tasks', []))}")
        for i, st in enumerate(plan_result.get("sub_tasks", [])[:5], 1):
            desc = st.get("description", st) if isinstance(st, dict) else str(st)
            print(f"  {i}. {desc[:60]}...")


async def example_complex_research():
    """Complex research with multiple aspects."""
    print("\n" + "="*60)
    print("Example 5: Complex Multi-Aspect Research")
    print("="*60)
    
    orchestrator = ResearchOrchestrator()
    
    # This will trigger multiple sub-tasks including web search and synthesis
    result = await orchestrator.research(
        """
        Compare the environmental impact of electric vehicles vs traditional 
        gasoline vehicles. Consider:
        1. Manufacturing carbon footprint
        2. Operational emissions
        3. Battery production and disposal
        4. Current market trends
        Include specific data and statistics where possible.
        """,
        verbose=True
    )
    
    if result["success"]:
        print("\n✓ Complex research completed!")
        print(f"Report length: {len(result.get('report', ''))} characters")
        print(f"Sources used: {len(result.get('sources', []))}")
        print(f"Total sub-tasks: {result.get('sub_tasks_completed', 0)}")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("AI RESEARCH ORCHESTRATOR - EXAMPLES")
    print("="*60)
    
    # Check configuration
    from config.settings import settings
    if not settings.google_api_key:
        print("\n⚠ Warning: GOOGLE_API_KEY not set. Some examples may fail.")
        print("Set your API key in .env file to run all examples.\n")
    
    examples = [
        ("Quick Search", example_quick_search),
        ("Code Execution", example_code_execution),
        ("Using Agents Directly", example_using_agents_directly),
        # Uncomment below for full research examples (requires API key)
        # ("Basic Research", example_basic_research),
        # ("Complex Research", example_complex_research),
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")
        
        print("\n")
    
    print("="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
