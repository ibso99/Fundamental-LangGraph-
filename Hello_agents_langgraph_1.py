'''
Objective 
1. Use different message types - HumanMessage and AIMessage
2. Maintain a full conversation history using both message types
3. Use lamma model uisng langchains ChatOllama
4. Create a sophisticated conversation loop

Main Goal: Create a form of memmory for our Agent
'''

import os 
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="llama3.2")

def process(state: AgentState) -> AgentState:
    """ This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


conversation_history = []

user_input = input("Enter your message:")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})

    conversation_history = result["messages"]

    user_input = input("\nEnter your message:")

with open("logging.txt", "w") as file:
    file.write("Your conversation history:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("\nEnd of conversation history.\n")

print("Conversation history saved to logging.txt")  