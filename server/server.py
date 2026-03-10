import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mcp.server.fastmcp import FastMCP
from tool.web_search_tool import WebSearchTool

mcp = FastMCP("web-search") #create mcp server

@mcp.tool()
def call_web_search_tool(query : str) -> str : 
    """
    calls the web search tool and returns a response

    Args : 
        query (str) : sentence in natural language for the search tool
    
    Returns :
        str : the response collected in the web
    """
    response_dic = WebSearchTool.web_search_tool(query)
    return response_dic["result"]




def main():
    #mcp.run()
    #we can also use 
    mcp.run()
    #if we want to use http and if is only going to run locally


if __name__ == "__main__":
    print("starting server")
    main()