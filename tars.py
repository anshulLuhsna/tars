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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import requests
import json

class CustomLLM:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def _call(self, prompt: list) -> str:
        # Convert messages to a JSON-serializable format
        def serialize_message(message):
            if isinstance(message, HumanMessage):
                return {"type": "user", "content": message.content}
            elif isinstance(message, AIMessage):
                return {"type": "assistant", "content": message.content}
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
        
        serialized_history = [serialize_message(msg) for msg in st.session_state.conversation_history]
        
        payload = json.dumps({
            "question": prompt[-1].content,
            "preserve_history": True,
            "conversation_history": serialized_history,
            "model": self.model_name,
            "stream_data": False,
        })
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        response = requests.post("https://api.worqhat.com/api/ai/content/v4", headers=headers, data=payload)
        response_data = response.json()
        return response_data.get("content", "")

    def invoke(self, prompt_value):
        return self._call(prompt_value)

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

load_dotenv()

search_tool = TavilySearchResults(max_results=1)
worqhat_llm = CustomLLM(model_name="aicon-v4-nano-160824", api_key="sk-128fd96fdf0d4039bdaef731e091de03")
llm_gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_gpt_with_tools = llm_gpt.bind_tools([search_tool])

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


prompt_template = SystemMessage(content=
    "You are only a tool calling agent. Give a tool call only and only if the user's answer will be better by searching the internet. Do not use tools for normal interactions. If not using a tool, simple reply with 'Okay'."
)

def chatbot(state: State):
    # Extract the latest user message
    user_message = state["messages"][-1].content if state["messages"] else ""


    # Create a HumanMessage with the formatted prompt
    prompt_message = prompt_template

    # Invoke the language model with the prompt
    st.write([prompt_message, HumanMessage(content=user_message)])
    ai_response = llm_gpt_with_tools.invoke([prompt_message, HumanMessage(content=user_message)])
    if "tool_calls" in ai_response.additional_kwargs:
        state["messages"].append(ai_response)
        return {"messages": state["messages"]}
    return {"messages": state["messages"]}
    # st.write(ai_response)
    # Replace the user's message with the AI's response in the conversation history
    # if state["messages"]:
    #     state["messages"][-1] = HumanMessage(content=ai_response.content)
    # else:
    #     state["messages"].append(HumanMessage(content=ai_response.content))

    # return {"messages": state["messages"]}

def worqhat(state: State):
    return {"messages": [worqhat_llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode([search_tool]))
graph_builder.add_node("worqhat", worqhat)


graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools":"tools", END: "worqhat"}
)


graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("worqhat", END)

graph = graph_builder.compile()
st.image(graph.get_graph().draw_mermaid_png())

st.title("🤖 TARS BOT")


def generate_response(input):

    response = graph.invoke({"messages": st.session_state.conversation_history})

    return response["messages"][-1]



prompt = st.chat_input("Say something")
if prompt:
    st.session_state.conversation_history.append(HumanMessage(content=prompt))
    response = generate_response(prompt)
    st.session_state.conversation_history.append(response)



for message in st.session_state.conversation_history: 
    if type(message)==HumanMessage:
        
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)