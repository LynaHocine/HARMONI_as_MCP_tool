import asyncio
import re
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessor import image_processor
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt.chat_agent_executor import create_react_agent


model = ChatOllama(model="gpt-oss:20b", temperature=0)

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
    agent = create_react_agent(model, tools)

    while True: 
        print("ask question: ", end="", flush=True)
        user_input = input()
        
        #searching for image url in the message
        image_url_match = re.search(r'https?://\S+\.(?:png|jpg|jpeg)', user_input)

        if image_url_match:
            image_url = image_url_match.group()
            base64_image = image_processor.ImagePreprocessor.image_to_base64(image_url)

            #remove the url from the text to keep only the question
            text = user_input.replace(image_url,"").strip()

            message = HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": base64_image}},
                {"type": "text", "text": text}
            ])
        else: 
            message = HumanMessage(content = user_input)
        response = await agent.ainvoke({
            "messages" : [message]
        })

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
                

    

if __name__ == "__main__":
    asyncio.run(run_agent())
