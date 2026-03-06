import os
from serpapi import GoogleSearch
from dotenv import load_dotenv


from preprocessor.image_processor import ImagePreprocessor
from preprocessor.speech_processor import SpeechToText


# Load variables from .env into environment
load_dotenv()

class WebSearchTool:
    """tool for performing web searches using SerpApi"""

    @staticmethod
    def process_input(input: str) -> str:
        """
        Detects the type of input and uses the preprocessing
        functions to convert it to text if needed
        """
        #image url
        if input.startswith("http") and (input.endswith(".png")
                                         or input.endswith(".jpg")
                                         or input.endswith(".jpeg")
                                         ) :
            return ImagePreprocessor.image_to_base64(input)
        
        #audio file
        if input.endswith(".wav"):
            return SpeechToText.speech_to_text()

        #text 
        return input
    
    @staticmethod
    def web_search_tool(query) -> dict:
        """It takes the user query (text) as input.
        Outputs result of the search 
        """
        #converts multimodal input to text
        text_query = WebSearchTool.process_input(query)
        q = text_query.strip()

        #if query is empty
        if not q: 
            return {"query": q, "result": "The query is empty"}
        

        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            return {"query": q, "result": "SerpAPI key not configured."}
        
        #SerpAPI request parameters
        params = {
            "engine": "google",
            "q": q,
            "api_key": api_key,
            "num": 1
        }

        #executes search 
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
        except Exception as e:
            return {"query": q, "result": f"Request failed: {str(e)}"}
        
        #extract top organic search result
        organic_results = results.get("organic_results")

        if not organic_results: 
            return {"query":q, "result": "No results found"}
        
        top = organic_results[0]

        title = top.get("title", "No title")
        snippet = top.get("snippet", "no snippet")

        result = f"{title}\n{snippet}"


        return {"query":q, "result":result}