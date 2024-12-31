# Imports
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.tools import YouTubeSearchTool
import requests
import json
from CustomLLM import CustomLLM

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# Load environment variables
load_dotenv()

# Tools Initialization
search_tool = TavilySearchResults(max_results=1)
yt_tool = YouTubeSearchTool()
tools = [search_tool, yt_tool]

# Language models
worqhat_llm = CustomLLM(model_name="aicon-v4-nano-160824", api_key="sk-128fd96fdf0d4039bdaef731e091de03")
llm_gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_gpt_with_tools = llm_gpt.bind_tools(tools)

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Graph builder initialization
graph_builder = StateGraph(State)

# Prompt templates
prompt_template = SystemMessage(content=
    "You are only a tool calling agent. Give a tool call only and only if the user's answer will be better by searching the internet. Do not use tools for normal interactions. If not using a tool, simple reply with 'Okay'."
)

# Chatbot logic
def chatbot(state: State):
    user_message = state["messages"][-1].content if state["messages"] else ""

    prompt_message = prompt_template
    context_message = SystemMessage(content="The following is the context that you might need to answer the question. It is the response of some tool calls. Relay the information properly to the user. If there are yt links, just repeat the links in your response.")

    ai_response = llm_gpt_with_tools.invoke([prompt_message, HumanMessage(content=user_message)])
    if "tool_calls" in ai_response.additional_kwargs:
        for toolCall in ai_response.additional_kwargs["tool_calls"]:
            state["messages"].append(context_message)
            state["messages"].append(ai_response)
            return {"messages": state["messages"]}
    return {"messages": state["messages"]}

# Worqhat logic
def worqhat(state: State):
    if(type(state["messages"][-1]) == ToolMessage):
        toolContent = state["messages"][-1].content
        response = AIMessage(content=str(toolContent) + "\n" + worqhat_llm.invoke(state["messages"], st.session_state["conversation_history"]))
        return {"messages": [response]}
    else:
        response = AIMessage(content=worqhat_llm.invoke(state["messages"], st.session_state["conversation_history"]))
        return {"messages": [response]}

# Graph construction
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("worqhat", worqhat)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", END: "worqhat"}
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("worqhat", END)

graph = graph_builder.compile()

# Streamlit UI setup
st.image(graph.get_graph().draw_mermaid_png())
st.title("🤖 TARS BOT")

# Response generation
def generate_response(input):
    response = graph.invoke({"messages": st.session_state.conversation_history})
    return response["messages"][-1]

# User input handling
prompt = st.chat_input("Say something")
if prompt:
    st.session_state.conversation_history.append(HumanMessage(content=prompt))
    response = generate_response(prompt)
    st.session_state.conversation_history.append(response)

# Conversation history display
for message in st.session_state.conversation_history: 
    if type(message) == HumanMessage:
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)
