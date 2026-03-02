"""
AI Research Orchestrator - Main Entry Point
A multi-agent system for orchestrated research and analysis.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

from orchestrator import ResearchOrchestrator
from config.settings import settings, get_llm_config

# Load environment variables
load_dotenv()

console = Console()


def print_banner():
    """Print the application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║           AI RESEARCH ORCHESTRATOR                    ║
    ║   Multi-Agent System for Intelligent Research         ║
    ╚═══════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def check_configuration():
    """Check if required configuration is present."""
    llm_config = get_llm_config()
    
    if not llm_config["gemini"]["available"]:
        console.print("[yellow]Warning: Google Gemini API key not configured.[/yellow]")
        console.print("Set GOOGLE_API_KEY in your .env file or environment.")
        return False
    
    console.print("[green]✓ Configuration valid[/green]")
    return True


async def run_research(query: str, verbose: bool = True) -> dict:
    """
    Run a research query through the orchestrator.
    
    Args:
        query: The research query
        verbose: Whether to show detailed progress
        
    Returns:
        Research results dictionary
    """
    orchestrator = ResearchOrchestrator()
    result = await orchestrator.research(query, verbose=verbose)
    return result


async def interactive_mode():
    """Run in interactive mode."""
    print_banner()
    
    if not check_configuration():
        console.print("\n[red]Please configure your API keys before continuing.[/red]")
        console.print("Copy .env.example to .env and add your keys.")
        return
    
    console.print("\nEnter your research queries (type 'exit' to quit, 'help' for commands):\n")
    
    orchestrator = ResearchOrchestrator()
    
    while True:
        try:
            query = console.input("[bold cyan]Research Query > [/bold cyan]").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if query.lower() == 'help':
                print_help()
                continue
            
            if query.lower().startswith('quick:'):
                # Quick search mode
                search_query = query[6:].strip()
                console.print(f"\n[dim]Quick search: {search_query}[/dim]\n")
                result = await orchestrator.quick_search(search_query)
                if result.get("success"):
                    console.print(Panel(result.get("findings", "No findings"), 
                                       title="Quick Search Results"))
                else:
                    console.print(f"[red]Search failed: {result.get('error')}[/red]")
                continue
            
            if query.lower().startswith('code:'):
                # Code execution mode
                code = query[5:].strip()
                console.print(f"\n[dim]Executing code...[/dim]\n")
                result = await orchestrator.execute_code(code)
                console.print(Panel(
                    f"Output:\n{result.get('output', result.get('error', 'No output'))}",
                    title="Code Execution"
                ))
                continue
            
            # Full research mode
            console.print()
            result = await orchestrator.research(query)
            
            if result.get("success"):
                # Save results
                save_results(result)
            else:
                console.print(f"\n[red]Research failed: {result.get('error')}[/red]")
            
            console.print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def print_help():
    """Print help information."""
    help_text = """
## Available Commands

- **Regular query**: Just type your research question
- **quick:<query>**: Quick web search without full orchestration  
- **code:<python_code>**: Execute Python code directly
- **exit**: Quit the application
- **help**: Show this help message

## Examples

```
Research Query > What are the latest advances in quantum computing?
Research Query > quick: Python web frameworks comparison 2024
Research Query > code: print(sum(range(100)))
```

## Tips

- Be specific with your research queries for best results
- Complex queries will be automatically decomposed into sub-tasks
- The orchestrator will validate and synthesize findings automatically
    """
    console.print(Markdown(help_text))


def save_results(result: dict, filename: Optional[str] = None):
    """Save research results to a file."""
    import json
    from datetime import datetime
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_result_{timestamp}.json"
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    
    # Create a serializable version
    serializable = {
        "query": result.get("query", ""),
        "success": result.get("success", False),
        "report": result.get("report", ""),
        "summary": result.get("summary", ""),
        "key_findings": result.get("key_findings", []),
        "sources": result.get("sources", []),
        "duration_seconds": result.get("duration_seconds", 0),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    # Also save markdown version
    md_path = output_dir / filename.replace('.json', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Research: {result.get('query', 'Unknown')}\n\n")
        f.write(f"*Generated: {datetime.now().isoformat()}*\n\n")
        f.write(result.get("report", "No report generated"))
    
    console.print(f"\n[dim]Results saved to: {output_path}[/dim]")
    console.print(f"[dim]Markdown saved to: {md_path}[/dim]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Research Orchestrator - Multi-Agent Research System"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query (omit for interactive mode)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick search mode (no full orchestration)"
    )
    
    args = parser.parse_args()
    
    if args.interactive or not args.query:
        # Interactive mode
        asyncio.run(interactive_mode())
    else:
        # Single query mode
        print_banner()
        
        if not check_configuration():
            sys.exit(1)
        
        async def single_query():
            orchestrator = ResearchOrchestrator()
            
            if args.quick:
                result = await orchestrator.quick_search(args.query)
            else:
                result = await orchestrator.research(args.query, verbose=not args.quiet)
            
            if result.get("success"):
                if args.output:
                    save_results(result, args.output)
                else:
                    save_results(result)
                return 0
            else:
                console.print(f"[red]Failed: {result.get('error')}[/red]")
                return 1
        
        exit_code = asyncio.run(single_query())
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
