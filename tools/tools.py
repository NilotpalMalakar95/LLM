from langchain.serpapi import SerpAPIWrapper


def get_wikipedia_data(text: str) -> str:
    """Searches for content from Wikipedia"""
    search = SerpAPIWrapper()
    result = search.run(f"{text}")
    return result
