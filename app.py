import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, List, Dict, Literal
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
import json

load_dotenv()

# Initialize the search tool
search_tool = TavilySearchResults(max_results=2)

st.title("😸 TARS BOT 😸")

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Define the custom Worqhat LLM class
class CustomLLM:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def _call(self, prompt: str) -> str:
        url = "https://api.worqhat.com/api/ai/content/v4"
        payload = json.dumps({
            "question": prompt,
            "preserve_history": True,
            "conversation_history": st.session_state.conversation_history,
            "model": self.model_name,
            "stream_data": False,
        })
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        response = requests.post(url, headers=headers, data=payload)
        response_data = response.json()
        return response_data.get("content", "")

    def invoke(self, prompt_value):
        return self._call(prompt_value)

# Initialize the LLMs
worqhat_llm = CustomLLM(model_name="aicon-v4-nano-160824", api_key="sk-128fd96fdf0d4039bdaef731e091de03")
openai_llm = ChatOpenAI(model="gpt-4o-mini")
openai_llm_with_tools = openai_llm.bind_tools([search_tool])

# Define the state structure
class State(TypedDict):
    messages: Annotated[List[Dict[str, str]], add_messages]

# Initialize the state graph
graph_builder = StateGraph(State)

# Define the OpenAI LLM agent function
def openai_agent(state: State) -> Command[Literal["worqhat_agent"]]:
    prompt = [
        SystemMessage(content="You are an assistant that gathers information to assist another LLM. Prompt engineer whatever the user says. Return only the next prompt."),
        HumanMessage(content=state["messages"][-1].content)
    ]
    response = openai_llm_with_tools.invoke(prompt)
    response_content = response.content if response else "No response from OpenAI."
    st.session_state.conversation_history.append({"role": "assistant", "content": response_content})
    return Command(
        goto="worqhat_agent",
        update={"messages": state["messages"] + [AIMessage(content=response_content)]}
    )

# Define the Worqhat LLM agent function
def worqhat_agent(state: State):
    try:
        # Prepare final prompt
        compiled_info = state["messages"][-1].content
        user_query = state["messages"][0].content
        final_prompt = f"User Query: {user_query}\n\nCompiled Information: {compiled_info}"
        

        # Invoke Worqhat LLM
        response_content = worqhat_llm.invoke(final_prompt)
        if not response_content:
            st.error("ERROR: Worqhat LLM returned an empty response.")
            return

        # Display and log response
        with st.chat_message("assistant"):
            st.markdown(response_content)
        st.session_state.conversation_history.append(response_content)
    except Exception as e:
        st.error(f"ERROR: An exception occurred in Worqhat LLM Agent: {e}")

# Build the graph
graph_builder.add_node("openai_agent", openai_agent)
graph_builder.add_node("worqhat_agent", worqhat_agent)
graph_builder.set_entry_point("openai_agent")
graph = graph_builder.compile()

# Function to handle user input
def generate_response(user_input):
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    graph.invoke({"messages": st.session_state.conversation_history})


for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message.content)


# Streamlit chat interface
if prompt := st.chat_input("Enter your message:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    generate_response(prompt)
