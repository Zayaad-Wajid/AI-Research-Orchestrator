# AI Research Orchestrator

A sophisticated multi-agent system that orchestrates specialized AI agents to decompose complex research queries, execute sub-tasks, and synthesize coherent research outputs.

## Features

- **Multi-Agent Architecture**: Specialized agents for planning, research, code execution, validation, and synthesis
- **Intelligent Task Decomposition**: Automatically breaks complex queries into manageable sub-tasks
- **Web Search Integration**: Multiple search backends with automatic fallback (DuckDuckGo, Tavily)
- **Code Execution**: Safe sandboxed Python execution with Docker support
- **Retry & Fallback Logic**: Automatic error handling and task recovery
- **Gemini LLM Integration**: Powered by Google's Gemini models

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Orchestrator                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Planner  │→ │ Researcher │→ │Validator │→ │Synthesizer│ │
│  │  Agent   │  │   Agent    │  │  Agent   │  │   Agent   │ │
│  └──────────┘  └────────────┘  └──────────┘  └───────────┘ │
│       ↓              ↓                                       │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────────────┐│
│  │  Tool    │  │   Retry    │  │     Tools Library        ││
│  │  Agent   │  │   Agent    │  │ • Web Search             ││
│  └──────────┘  └────────────┘  │ • Code Execution         ││
│                                │ • Content Fetching       ││
│                                └──────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. Clone the repository:
```bash
cd AI-Research-orchestrator
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Configuration

Create a `.env` file with the following:

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional
TAVILY_API_KEY=your_tavily_key  # For enhanced web search
MAX_RETRIES=3
MAX_CONCURRENT_AGENTS=5
```

## Usage

### Interactive Mode
```bash
python main.py
```

### Single Query
```bash
python main.py "What are the latest advances in quantum computing?"
```

### Quick Search
```bash
python main.py --quick "Python 3.12 new features"
```

### Programmatic Usage

```python
import asyncio
from orchestrator import ResearchOrchestrator

async def main():
    orchestrator = ResearchOrchestrator()
    
    result = await orchestrator.research(
        "Explain the impact of AI on healthcare",
        verbose=True
    )
    
    if result["success"]:
        print(result["report"])
        print(f"Key findings: {result['key_findings']}")

asyncio.run(main())
```

## Agent Details

### Planner Agent
Analyzes complex queries and decomposes them into structured sub-tasks with dependencies.

### Researcher Agent
Conducts web searches, fetches content, and extracts relevant information from multiple sources.

### Tool Agent
Generates and executes Python code for calculations, data analysis, and computations.

### Validator Agent
Verifies accuracy of findings, checks source credibility, and identifies inconsistencies.

### Synthesizer Agent
Combines findings from multiple sources into coherent, well-structured reports.

### Retry Agent
Handles failed tasks with intelligent retry strategies and fallback approaches.

## Project Structure

```
AI-Research-orchestrator/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py        # Abstract base class
│   ├── planner_agent.py     # Query decomposition
│   ├── researcher_agent.py  # Web research
│   ├── tool_agent.py        # Code execution
│   ├── validator_agent.py   # Result validation
│   ├── synthesizer_agent.py # Output synthesis
│   └── retry_agent.py       # Error recovery
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── models/
│   ├── __init__.py
│   └── task.py              # Data models
├── orchestrator/
│   ├── __init__.py
│   └── coordinator.py       # Main orchestration logic
├── tools/
│   ├── __init__.py
│   ├── web_search.py        # Web search tools
│   └── code_executor.py     # Code execution tools
├── outputs/                  # Generated reports
├── .env.example             # Environment template
├── requirements.txt         # Dependencies
├── main.py                  # Entry point
├── examples.py              # Usage examples
└── README.md
```

## Workflow

1. **Query Input**: User provides a research query
2. **Planning**: Planner agent decomposes query into sub-tasks
3. **Execution**: Sub-tasks are executed in parallel where possible
   - Web searches for information gathering
   - Code execution for analysis
4. **Validation**: Findings are verified for accuracy
5. **Synthesis**: Results are combined into a coherent report
6. **Error Handling**: Failed tasks are retried with fallback strategies

## Error Handling

The orchestrator includes robust error handling:

- **Transient Errors**: Automatic retry with exponential backoff
- **Search Failures**: Fallback to alternative search providers
- **Code Errors**: Automatic code fixing and re-execution
- **Complex Tasks**: Automatic task simplification

## Output

Research results are saved in the `outputs/` directory:
- JSON format with full metadata
- Markdown format for readable reports

## Requirements

- Python 3.10+
- Google Gemini API key
- Optional: Docker (for sandboxed code execution)
- Optional: Tavily API key (for enhanced search)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
