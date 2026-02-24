import requests
from bs4 import BeautifulSoup

def web_search_tool(query) -> dict:
    """It takes the user query (text) as input.
    Outputs result of the search 
    """

    q = query.strip()
    if not q: 
        return {"query": q, "result": "The query is empty"}
    
    #web tool does search with DuckDuckGo

    url = f"https://html.duckduckgo.com/html/?q={q}"
    res = requests.get(url)
    
    #extract the data from the html
    soup = BeautifulSoup(res.text, "html.parser")

    #get the top result 
    results = soup.find_all("div", class_="result")

    if not results: 
        return {"query":q, "result": "No results found"}
    
    top = results[0]
    title = (top.find("a", class_="result__a")).get_text()

    snippet_tag = top.find("a", class_="result__snippet") or top.find("div", class_="result__snippet")
    snippet = snippet_tag.get_text() if snippet_tag else "No snippet available"

    result = f"{title}\n{snippet}"


    return {"query":q, "result":result}