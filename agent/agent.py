import asyncio
import re
import sys
import os
import base64
import requests
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessor.image_processor import ImagePreprocessor
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt.chat_agent_executor import create_react_agent

vision_model = ChatOllama(model="llama3.2-vision", temperature=0)
agent_model = ChatOllama(model="gpt-oss:20b", temperature=0)

async def describe_image(data_uri: str, question:str) -> str:
    """ask the vision model to describe the image."""
    message = HumanMessage(content=[
        {"type": "text", "text": f"Look at this image and describe it in the context of this question {question}"},
        {"type": "image_url", "image_url": data_uri},
    ])
    response = await vision_model.ainvoke([message])
    return response.content
async def run_agent():

    client = MultiServerMCPClient({
        "web-search" :{
            "command" : "python",
            "args" : ["../server/server.py"],
            "transport" : "stdio",
            "cwd": "../server",
        }
    })
        
    tools = await client.get_tools()
    agent = create_react_agent(agent_model, tools)

    while True: 
        print("ask question: ", end="", flush=True)
        user_input = input()
        
        total_timer = time.perf_counter()
        #searching for image url in the message
        image_url_match = re.search(r'https?://\S+\.(?:png|jpg|jpeg)', user_input)

        if image_url_match:
            image_url = image_url_match.group()
            #remove the url from the text to keep only the question
            text = user_input.replace(image_url,"").strip()
            data_uri = ImagePreprocessor.image_to_base64(image_url)

            print("[Vision model analyzing image...]")
            vision_timer = time.perf_counter()
            image_description = await describe_image(data_uri, text)
            vision_elapsed = time.perf_counter() - vision_timer
            print(f"[Image description: {image_description}]")
            print(f"[Vision timer = {vision_elapsed:.2f}s]")

            combined = (
                f"The user asked: {text}\n\n"
                f"A vision model analyzed the image and said: {image_description}"
                f"Using this context, answer user's question. "
            )
            message = HumanMessage(content=combined)
        else: 
            message = HumanMessage(content = user_input)

        agent_timer = time.perf_counter()
        response = await agent.ainvoke({
            "messages" : [message]
        })
        agent_elapsed = time.perf_counter() - agent_timer

        for message in response["messages"]:

            if isinstance(message, HumanMessage):
                print("USER:", message.content)

            elif isinstance(message, AIMessage):
                print("AI:", message.content)

                if message.tool_calls:
                    print("TOOL CALL DETECTED:")
                    for tool in message.tool_calls:
                        print(tool)
                
            elif isinstance(message, ToolMessage):
                print("TOOL RESPONSE:", message.content)
                
            print()

        total_elapsed = time.perf_counter() - total_timer
        print(f"[Agent time: {agent_elapsed:.2f}]s") 
        print(f"[Total: {total_elapsed:.2f}]s")
        print()

    

if __name__ == "__main__":
    asyncio.run(run_agent())
