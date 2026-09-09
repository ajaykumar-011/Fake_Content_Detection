import os
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient


# ==================================================
# LOAD .ENV FILE
# ==================================================

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

    api_key = os.getenv(
        "TAVILY_API_KEY"
    )


    if not api_key:

        raise RuntimeError(
            "TAVILY_API_KEY is not set. "
            "Please add it to the .env file."
        )


    # ==============================================
    # CREATE TAVILY CLIENT
    # ==============================================

    client = TavilyClient(
        api_key=api_key
    )


    try:

        response = client.search(

            query=query,

            search_depth="advanced",

            max_results=max_results,

            include_answer=False

        )


        results = []


        for item in response.get(
            "results",
            []
        ):

            results.append({

                "title": item.get(
                    "title",
                    "Untitled"
                ),

                "url": item.get(
                    "url",
                    ""
                ),

                "content": item.get(
                    "content",
                    ""
                )

            })


        return results


    except Exception as error:

        print(
            "Tavily search error:",
            error
        )

        return []