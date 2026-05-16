# Project PAI2D - HARMONI as a Model Context Protocol tool 

Our goal in this project is to implement an integration between HARMONI (codebase available at https://github.com/hamedR96/HARMONI) and the Model Context Protocol (MCP).

This integration aims to make HARMONI accessible as a standardized context provider, enabling foundation models to reason over audio, visual, and long-term user memory in unified manner. 

We seek to position HARMONI as a general-purpose multimodal interface layer that allows language and vision language models to operate robustly in real-world, multi-user interactive environments.

## Project Structure
```
HARMONI_as_MCP_tool/
├── agent/          # ReAct agent using LangGraph + LangChain
├── interface/      # Streamlit UI for interacting with the agent
├── server/         # MCP server exposing web search tool 
├── tool/           # Web search tool using SerpAPI
└── preprocessor/   # Multimodal input processors (image, audio, video)
```

## Prerequisites 

- **Python 3.11 or 3.12**
- **Ollama** installed and running locally https://ollama.com/
- **SerpAPI key** to get from https://serpapi.com/

## Installation 

1. Clone the repository 
```bash
git clone https://github.com/LynaHocine/HARMONI_as_MCP_tool.git
cd HARMONI_as_MCP_tool
```

2. Create and activate a virtual environment (recommended: Python 3.11 or 3.12)
- On Linux/macOS:
```bash
python 3.12 -m venv venv
source venv/bin/activate
```
- On Windows:
```bash
python3.12 -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a `.env` file in the root folder:
```
SERPAPI_KEY=your_key
```
## Execution 

The project requires three separate terminals:

1. One terminal for the HARMONI server
2. One terminal for the MCP tool / agent
3. One terminal for Ollama

### 1. Launch HARMONI

Clone the original HARMONI repository:

```bash
git clone https://github.com/hamedR96/HARMONI.git
cd HARMONI
```

Activate a virtual environment. You may reuse the same virtual environment created for this MCP tool.

- On Linux/macOS:
```bash
source venv/bin/activate
```

- On Windows:
```bash
venv\Scripts\activate
```

Start the HARMONI server:

```bash
uvicorn video_interface.app:app --reload
```

### 2. Launch Ollama

In a third terminal, pull the required model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running locally.

### 3. Run the MCP agent

- To run the code in the terminal:

```bash
cd agent
python agent.py
```

The server starts automatically. 
Example interaction:

```text
Type 't' for text or 'v' for voice : t
ask question: How is the weather in Paris today?
```

- To launch the interface:

```bash
cd interface
streamlit run app.py
```

