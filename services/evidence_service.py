import sys
import types
from pathlib import Path
import os

# ==================================================
# MOCK TIKTOKEN TO BYPASS DLL APPLICATION CONTROL BLOCK
# ==================================================

class DummyEncoding:
    """Fallback dummy encoder to prevent tiktoken C-extension DLL execution errors."""
    def encode(self, text, *args, **kwargs):
        return text.split()
    def decode(self, tokens, *args, **kwargs):
        return " ".join(tokens)

dummy_module = types.ModuleType("tiktoken")
dummy_module.get_encoding = lambda *args, **kwargs: DummyEncoding()
dummy_module.encoding_for_model = lambda *args, **kwargs: DummyEncoding()
dummy_module.Encoding = DummyEncoding

# Inject the dummy module into Python's sys.modules before tavily imports tiktoken
sys.modules["tiktoken"] = dummy_module
sys.modules["tiktoken.core"] = dummy_module
sys.modules["_tiktoken"] = dummy_module

# ==================================================
# IMPORTS & ENVIRONMENT CONFIGURATION
# ==================================================

from dotenv import load_dotenv
from tavily import TavilyClient

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


# ==================================================
# SEARCH EVIDENCE
# ==================================================

def search_evidence(query, max_results=5):
    """
    Search the web using Tavily and return
    relevant evidence sources.
    """
    query = query.strip()

    if not query:
        return []

    # ==============================================
    # GET API KEY
    # ==============================================
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. "
            "Please add it to the .env file."
        )

    # ==============================================
    # CREATE TAVILY CLIENT
    # ==============================================
    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False
        )

        results = []

        for item in response.get("results", []):
            results.append({
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": item.get("content", "")
            })

        return results

    except Exception as error:
        print("Tavily search error:", error)
        return []