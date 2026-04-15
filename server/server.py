import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mcp.server.fastmcp import FastMCP
from tool.web_search_tool import WebSearchTool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from preprocessor.image_processor import ImagePreprocessor
from tool.harmoni_tool import HarmoniTool
import asyncio
import cv2

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

vision_model = ChatOllama(model= "llama3.2-vision", temperature=0)

@mcp.tool()
async def call_image_processing_tool(source: str, question: str)->str:
    """Analyzes the image and returns a detailed description of its content.
    Call this tool whenever the user provides an image URL or local file path.
    Use the description returned to answer the user's question.
    """
    data_uri = ImagePreprocessor.image_to_base64(source)


    message = HumanMessage(content=[
        {"type": "text", "text": "Describe this image in as much detail as possible."},
        {"type": "image_url", "image_url": data_uri},
    ])
    response = await vision_model.ainvoke([message])
    return response.content

@mcp.tool()
def call_harmoni_tool(video_path: str, query: str)->str:
    """Sends video to Harmoni to get user context. 
    Call this tool when a face is detected in the video.
    Use the returned context to give a personalized answer.
    """
    return HarmoniTool.harmoni_tool(video_path=video_path, query=query)

#for face detection
@mcp.tool()
def detect_face(video_path:str) -> bool:
    """Returns True if a face is detected in the video
    Call this tool when a video is provided as input"""
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    detected = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            detected = True
            break
    
    cap.release()
    return detected



def main():
    #mcp.run()
    #we can also use 
    mcp.run()
    #if we want to use http and if is only going to run locally


if __name__ == "__main__":
    print("starting server")
    main()