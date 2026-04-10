import asyncio
import sys
import os
import base64
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SERVER_DIR = os.path.join(BASE_DIR, "server")
SERVER_FILE = os.path.join(SERVER_DIR, "server.py")

from preprocessor.speech_processor import SpeechToText
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt.chat_agent_executor import create_react_agent

agent_model = ChatOllama(model="gpt-oss:20b", temperature=0)

class MCPAgent : 
    def __init__(self):
        self.model = ChatOllama(model="gpt-oss:20b", temperature=0)
        self.agent = None
        self.chat_history = []
        self.system_prompt = SystemMessage(content=(
            "You are a helpful assistant with access to tools. "
            "When the user provides an image containing a question, "
            "extract the question and answer it — use your tools if needed (e.g. web search for current data). "
            "Do not just describe the image. Actually answer what is being asked."
        ))

    async def initialize(self):
        """
        Initializes the MCP, tools and agent
        """ 
        client = MultiServerMCPClient({
        "web-search" :{
            "command" : "python",
            "args" : [SERVER_FILE],
            "transport" : "stdio",
            "cwd": SERVER_DIR,
        }
        })
            
        tools = await client.get_tools()
        self.agent = create_react_agent(self.model, tools)

    async def run(self, user_input):
        self.chat_history.append(HumanMessage(content=user_input))

        response = await self.agent.ainvoke({
            "messages" : [self.system_prompt] + self.chat_history
        })

        ai_msg = None

        for msg in reversed(response["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                ai_msg = msg
                break

        self.chat_history.append(ai_msg)
        return ai_msg.content


## old version not object oriented
async def run_agent():

    client = MultiServerMCPClient({
        "web-search" :{
            "command" : "python",
            "args" : [SERVER_FILE],
            "transport" : "stdio",
            "cwd": SERVER_DIR,
        }
    })
        
    tools = await client.get_tools()
    agent = create_react_agent(agent_model, tools)

    while True: 
        mode = input("Type 't' for text or 'v' for voice : ")
        if mode == "v" or mode == "V":
            user_input = SpeechToText.speech_to_text(device_index=1)
        else : 
            print("ask question: ", end="", flush=True)
            user_input = input()
        
        total_timer = time.perf_counter()

        message = HumanMessage(content = user_input)

        agent_timer = time.perf_counter()
        response = await agent.ainvoke({
            "messages" : [message]
        })
        agent_elapsed = time.perf_counter() - agent_timer

        for msg in response["messages"]:

            if isinstance(msg, HumanMessage):
                print("USER:", msg.content)

            elif isinstance(msg, AIMessage):
                print("AI:", msg.content)

                if msg.tool_calls:
                    print("TOOL CALL DETECTED:")
                    for tool in msg.tool_calls:
                        print(tool)
                
            elif isinstance(msg, ToolMessage):
                print("TOOL RESPONSE:", msg.content)
                
            print()

        total_elapsed = time.perf_counter() - total_timer
        print(f"[Agent time: {agent_elapsed:.2f}s]") 
        print(f"[Total: {total_elapsed:.2f}s]")
        print()

if __name__ == "__main__":
    asyncio.run(run_agent())
