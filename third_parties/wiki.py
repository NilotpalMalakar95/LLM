import wikipedia


def scrape_wiki_data(keyword: str):
    """Manually scrape wikipedia using the keyword provided"""

    # Search for our keyword
    search_results = wikipedia.search(keyword)
    # Most relevant search result
    page_title = search_results[0]

    # Fetch the page data from wikipedia
    page = wikipedia.page(page_title)
    # Fetch the page data
    page_data = page.summary

    return page_data
