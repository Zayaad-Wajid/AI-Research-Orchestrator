"""
AI Research Orchestrator - Web Search Tools
Provides multiple search backends with fallback support.
"""

import asyncio
import httpx
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup
import trafilatura

from models.task import SearchResult
from config.settings import settings

logger = structlog.get_logger()


class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Execute a search query."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass


class DuckDuckGoSearch(SearchProvider):
    """DuckDuckGo search provider (no API key required)."""
    
    @property
    def name(self) -> str:
        return "duckduckgo"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=num_results))
                
                for item in search_results:
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("href", ""),
                        snippet=item.get("body", ""),
                        relevance_score=0.8
                    ))
            
            logger.info(f"DuckDuckGo search completed", query=query, results=len(results))
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed", error=str(e))
            raise


class TavilySearch(SearchProvider):
    """Tavily AI search provider (requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.tavily_api_key
        self.base_url = "https://api.tavily.com/search"
    
    @property
    def name(self) -> str:
        return "tavily"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using Tavily API."""
        if not self.api_key:
            raise ValueError("Tavily API key not configured")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": num_results,
                    "include_answer": True
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    relevance_score=item.get("score", 0.5)
                ))
            
            logger.info(f"Tavily search completed", query=query, results=len(results))
            return results


class WebContentFetcher:
    """Fetches and extracts content from web pages."""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def fetch_content(self, url: str) -> Optional[str]:
        """Fetch and extract main content from a URL."""
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(
                    url, 
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                html = response.text
                
                # Use trafilatura for content extraction
                content = trafilatura.extract(
                    html,
                    include_links=False,
                    include_images=False,
                    include_tables=True
                )
                
                if not content:
                    # Fallback to BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                    content = soup.get_text(separator='\n', strip=True)
                
                # Limit content length
                if content and len(content) > 10000:
                    content = content[:10000] + "..."
                
                logger.info(f"Fetched content from URL", url=url, length=len(content) if content else 0)
                return content
                
        except Exception as e:
            logger.error(f"Failed to fetch content", url=url, error=str(e))
            return None


class WebSearchTool:
    """
    Unified web search tool with multiple provider support and fallback.
    """
    
    def __init__(self):
        self.providers: List[SearchProvider] = []
        self.content_fetcher = WebContentFetcher()
        self._init_providers()
    
    def _init_providers(self):
        """Initialize available search providers."""
        # DuckDuckGo is always available (no API key needed)
        self.providers.append(DuckDuckGoSearch())
        
        # Add Tavily if API key is available
        if settings.tavily_api_key:
            self.providers.insert(0, TavilySearch())  # Prefer Tavily if available
        
        logger.info(f"Initialized search providers", providers=[p.name for p in self.providers])
    
    async def search(
        self, 
        query: str, 
        num_results: int = 5,
        fetch_content: bool = False
    ) -> List[SearchResult]:
        """
        Execute search with automatic fallback between providers.
        
        Args:
            query: Search query
            num_results: Number of results to return
            fetch_content: Whether to fetch full content from URLs
            
        Returns:
            List of search results
        """
        last_error = None
        
        for provider in self.providers:
            try:
                logger.info(f"Attempting search with provider", provider=provider.name)
                results = await provider.search(query, num_results)
                
                if results:
                    # Optionally fetch full content
                    if fetch_content:
                        results = await self._enrich_with_content(results)
                    
                    return results
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Provider failed, trying next", 
                             provider=provider.name, error=str(e))
                continue
        
        # All providers failed
        logger.error("All search providers failed", last_error=str(last_error))
        return []
    
    async def _enrich_with_content(
        self, 
        results: List[SearchResult],
        max_concurrent: int = 3
    ) -> List[SearchResult]:
        """Fetch full content for search results."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(result: SearchResult) -> SearchResult:
            async with semaphore:
                content = await self.content_fetcher.fetch_content(result.url)
                result.content = content
                return result
        
        tasks = [fetch_with_semaphore(r) for r in results]
        enriched = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        return [r for r in enriched if isinstance(r, SearchResult)]
    
    async def search_and_summarize(
        self, 
        query: str, 
        num_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search and return structured results with metadata.
        """
        results = await self.search(query, num_results, fetch_content=True)
        
        return {
            "query": query,
            "num_results": len(results),
            "results": [r.model_dump() for r in results],
            "has_content": any(r.content for r in results)
        }


# Singleton instance
web_search_tool = WebSearchTool()


# Function-based tools for OpenAI Agents SDK
async def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for information.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
        
    Returns:
        Formatted search results as a string
    """
    results = await web_search_tool.search(query, num_results, fetch_content=True)
    
    if not results:
        return f"No results found for: {query}"
    
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"## Result {i}: {r.title}")
        formatted.append(f"URL: {r.url}")
        formatted.append(f"Summary: {r.snippet}")
        if r.content:
            # Include first 500 chars of content
            content_preview = r.content[:500] + "..." if len(r.content) > 500 else r.content
            formatted.append(f"Content: {content_preview}")
        formatted.append("")
    
    return "\n".join(formatted)


async def fetch_webpage(url: str) -> str:
    """
    Fetch and extract content from a specific webpage.
    
    Args:
        url: The URL to fetch
        
    Returns:
        Extracted content from the page
    """
    fetcher = WebContentFetcher()
    content = await fetcher.fetch_content(url)
    
    if content:
        return content
    return f"Failed to fetch content from: {url}"
