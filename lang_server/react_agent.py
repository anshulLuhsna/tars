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
from langchain_core.messages import HumanMessage, AIMessage
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


load_dotenv()

search_tool = TavilySearchResults(max_results=1)
worqhat_llm = CustomLLM(model_name="aicon-v4-nano-160824", api_key="sk-128fd96fdf0d4039bdaef731e091de03")
llm_gpt = ChatOpenAI(model="gpt-4o-mini")
llm_gpt_with_tools = llm_gpt.bind_tools([search_tool])

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


prompt_template = PromptTemplate.from_template(
    "You are a prompt engineer. Your task is to refine the user's message to enhance clarity and effectiveness. "
    "If necessary, identify and utilize appropriate tools to gather additional information. "
    "Your response will replace the user's original message. "
    "User's message: {user_message}"
)

def chatbot(state: State):
    # Extract the latest user message
    user_message = state["messages"][-1].content if state["messages"] else ""

    # Format the prompt using the template
    formatted_prompt = prompt_template.format(user_message=user_message)

    # Create a HumanMessage with the formatted prompt
    prompt_message = HumanMessage(content=formatted_prompt)

    # Invoke the language model with the prompt
    ai_response = llm_gpt_with_tools.invoke([prompt_message])
    if "tool_calls" in ai_response.additional_kwargs:
        state["messages"].append(ai_response)
        return {"messages": state["messages"]}

    st.write(ai_response)
    # Replace the user's message with the AI's response in the conversation history
    if state["messages"]:
        state["messages"][-1] = HumanMessage(content=ai_response.content)
    else:
        state["messages"].append(HumanMessage(content=ai_response.content))

    return {"messages": state["messages"]}

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