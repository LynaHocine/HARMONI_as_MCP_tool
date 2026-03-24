import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import streamlit as st
import asyncio
from agent.agent import MCPAgent

import nest_asyncio
nest_asyncio.apply()

class ChatBotUI:
    """Handles the user interface for the chatbot"""
    def __init__(self):
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.init_session()

    def init_session(self):
        """Initializes the agent"""
        if "agent" not in st.session_state:
            st.session_state.agent = MCPAgent()
            self.loop.run_until_complete(st.session_state.agent.initialize())

    def display_chat(self):
        """Displays the chat history"""
        agent = st.session_state.agent

        for msg in agent.chat_history:
            if msg.type == "human":
                st.chat_message("user").write(msg.content)
            else:
                st.chat_message("assistant").write(msg.content)

    def handle_input(self):
        """Handles the user input"""
        agent = st.session_state.agent

        user_input = st.chat_input("Ask a question ")

        if user_input:
            st.chat_message("user").write(user_input)

            with st.spinner("Agent is thinking..."):
                response = asyncio.run(agent.run(user_input))

            st.chat_message("assistant").write(response)

    def run(self):
        """Runs the UI"""
        st.set_page_config(page_title="MCP Agent", layout="wide")
        st.title("MCP Agent")

        self.display_chat()
        self.handle_input()


if __name__ == "__main__":
    ui = ChatBotUI()
    ui.run()
    