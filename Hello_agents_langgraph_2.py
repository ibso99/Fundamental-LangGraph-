'''
ReAct Agent : Reasoning and Acting 

Objective: 
- Learn how to create Tools in LangGraph 
- How to create a ReAct Graph
- Work with different types of Messages such as ToolMessages
- Test out robustness of our graph 

Main Goal: Create a Robust ReAct Agent!
'''
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import (BaseMessage, ToolMessage, SystemMessage) # The foundation class for all message types in langgraph 
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b:int):
    """This is an addition function that adds two numbers together"""
    return a + b

@tool
def subtract(a: int, b:int):
    """Subtraction function """
    return a - b

@tool
def multiply(a: int, b:int):
    """Multiplication function """
    return a * b

tools = [add, subtract, multiply]

model = ChatOllama(model="llama3.2").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are an AI assistant, please answer my query to the best of your ability"
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "add 40 + 12 and then multiply the result by 6 and then subtract 10 from it. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))