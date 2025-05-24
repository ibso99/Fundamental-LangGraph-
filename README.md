🧠 **LangGraph Agents: ReAct, RAG, and Drafter**

This project implements a modular agentic system using LangGraph to orchestrate three key agent types:

-  ReAct Agent: Uses the ReAct pattern to reason and act with tools iteratively. 

-  Drafter Agent: Structures and drafts content based on a given prompt or task. 

-  RAG Agent: Retrieves relevant context from a vector store to ground its responses. 


⚙️ **Tech Stack**

- LangGraph for stateful agent orchestration

- LangChain for agent tools and chains
 
- Ollama (local LLMs) via langchain_ollama

- Python 3.10+ 

```
.
├── Fundamental-LangGraph/
│   ├── Hello_agents_langgraph_2.py     # ReAct-based reasoning agent
│   ├── Hello_agents_langgraph_3.py    # Structured content generation agent
│   └── Hello_agent_langgraph_4.py   # Retrieval-Augmented Generation agent            
└── README.md              # Project documentation

```

```
git clone https://github.com/ibso99/Fundamental-LangGraph-.git
cd Fundamental-LangGraph
```
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
```
pip install -r requirements.txt
```

🧩 Agent Overview
Agent	Purpose
- ReAct	Iterative reasoning with tools and memory
- RAG	Combines search + generation
- Drafter	Structured drafting (e.g., emails, reports)

📌 Notes
Make sure Ollama is running with required models (llama3, etc.).

Configure your .env for any API keys or environment settings.

🛠️ Future Work
- Add UI with Streamlit

- Integrate with external knowledge bases

- Parallel agent orchestration

📄 License
MIT License