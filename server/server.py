from mcp.server.fastmcp import FastMCP
from tool.web_search_tool import web_search_tool

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
    response_dic = web_search_tool(query)
    return response_dic["result"]




def main():
    mcp.run()
    #we can also use 
    #mcp.run(transport="http", host="127.0.0.1", port=8000)
    #if we want to use http and if is only going to run locally


if __name__ == "__main__":
    print("starting server")
    main()
