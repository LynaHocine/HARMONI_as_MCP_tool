import asyncio
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


model = ChatOllama(model="gpt-oss:20b", temperature=0)

async def run_agent():

    client = MultiServerMCPClient({
        "web-search" :{
            "command" : "python",
            "args" : ["../server/server.py"],
            "transport" : "stdio",

        }
    })
        
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)

    while True: 
        print("ask question: ", end="", flush=True)
        user_input = input()

        response = await agent.ainvoke({
            "messages" : [HumanMessage (content = user_input)]
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
