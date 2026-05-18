import asyncio
import sys
import os
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


class MCPAgent:
    def __init__(self):
        self.model = ChatOllama(model="gpt-oss:20b", temperature=0)
        self.agent = None
        self.chat_history = []
        self.system_prompt = SystemMessage(content=(
            "You have tools to process video files. "
            "When the user's message contains a file path ending in .mp4, .mov, or .avi: "
            "1. Call detect_face with that path. "
            "2. If True: call call_harmoni_tool with the path and the user's question. Use the returned context to answer. "
            "3. If False: call transcribe_video with the path. Use the transcript to answer. "
            "If you need current data (weather, news), also call call_web_search_tool. "
            "Do not repeat or refine the same search multiple times unless the previous result is empty."
            "Never call call_image_processing_tool on a video file."
            "If the speaker's name is explicitly known with high confidence, you may greet them naturally by name."
        ))

    async def initialize(self):
        client = MultiServerMCPClient({
            "web-search": {
                "command": "python",
                "args": [SERVER_FILE],
                "transport": "stdio",
                "cwd": SERVER_DIR,
            }
        })
        tools = await client.get_tools()
        self.agent = create_react_agent(self.model, tools)

    async def run(self, user_input):
        self.chat_history.append(HumanMessage(content=user_input))

        response = await self.agent.ainvoke({
            "messages": [self.system_prompt] + self.chat_history
        })

        ai_msg = None
        for msg in response["messages"]:
            if isinstance(msg, HumanMessage):
                print("USER:", msg.content)
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    print("TOOL CALLS:")
                    for tc in msg.tool_calls:
                        print(" →", tc["name"], tc["args"])
                else:
                    ai_msg = msg
                    print("AI:", msg.content)
            elif isinstance(msg, ToolMessage):
                print("TOOL RESPONSE:", msg.content[:300])
            print()

        if ai_msg:
            self.chat_history.append(ai_msg)
            return ai_msg.content
        return "(no response)"


async def main():
    agent = MCPAgent()
    await agent.initialize()

    while True:
        mode = input("Type 't' for text or 'v' for voice : ")
        if mode.lower() == "v":
            user_input = SpeechToText.speech_to_text(device_index=1)
        else:
            print("ask question: ", end="", flush=True)
            user_input = input()

        total_timer = time.perf_counter()
        await agent.run(user_input)
        elapsed = time.perf_counter() - total_timer

        print(f"[Total: {elapsed:.2f}s]\n")


if __name__ == "__main__":
    asyncio.run(main())