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



load_dotenv()

search_tool = TavilySearchResults(max_results=1)
llm_gpt = ChatOpenAI(model="gpt-4o-mini")
llm_gpt_with_tools = llm_gpt.bind_tools([search_tool])

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []



class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_gpt_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_node("tools", ToolNode([search_tool]))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    ["tools", END]
)
graph_builder.add_edge("tools", "chatbot")

graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
st.image(graph.get_graph().draw_mermaid_png())

st.title("TARS BOT")




with st.chat_message("user"):
    st.write("Hello 👋")

def generate_response(input):
    response = graph.invoke({"messages": st.session_state.conversation_history})

    return response["messages"][-1]



prompt = st.chat_input("Say something")
if prompt:
    st.session_state.conversation_history.append(HumanMessage(content=prompt))
    st.write("BEFORE: ", st.session_state.conversation_history)
    response = generate_response(prompt)
    st.session_state.conversation_history.append(response)
    st.write("AFTER: ", st.session_state.conversation_history)


for message in st.session_state.conversation_history: 
    if type(message)==HumanMessage:
        
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)