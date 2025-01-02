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
from functools import partial
from PyPDF2 import PdfReader

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "question_papers" not in st.session_state:
    st.session_state["question_papers"] = []
if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = []
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = None

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

def get_chatbot_response_questions(user_message, question):
    """Get a response from the chatbot API with context."""
    url = "https://api.worqhat.com/api/ai/content/v4"
    
    question_text = str(question)
    
    payload = json.dumps({
        "question": user_message + question_text,
        "model": "aicon-v4-nano-160824",
        "randomness": 0.1,
        "stream_data": False,
        "training_data": f"You are a question extractor.",
        "response_type": "text"
    })
    
    headers = {
        'Content-Type': 'application/json',
        "Authorization": "Bearer sk-128fd96fdf0d4039bdaef731e091de03"
    }
    
    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code == 200:
        
        return response.json().get("content", "No response received.")
    else:
        return f"Error: {response.status_code}, {response.text}"

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

prompt = """
    This is the text extracted from a file. Your task is to extract and format only the questions from the text. Follow these rules strictly:
RULES:
1) Every question should come on a new line.
2) Reply with only the questions and nothing else.
3) The questions should be clubbed as necessary. Just make sure different questions are on different lines.
4) The sub-questions within a question should come in the same line as the main question.
TEXT: 
"""

# Response generation
def generate_response(input):
    response = graph.invoke({"messages": st.session_state.conversation_history})
    return response["messages"][-1]




# Streamlit UI
# st.image(graph.get_graph().draw_mermaid_png())
st.title("🤖 TARS BOT")

# Initialize session state if not already done
if "all_questions" not in st.session_state:
    st.session_state["all_questions"] = []
if "selected_questions" not in st.session_state:
    st.session_state["selected_questions"] = []

# Upload question papers
uploaded_file = st.file_uploader("Upload Question Paper", type=["txt", "pdf"])
if uploaded_file:
    # Parse uploaded file content
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        file_content = "\n".join(page.extract_text() for page in reader.pages)
    else:
        file_content = uploaded_file.read().decode("utf-8")

    # Extract questions using the chatbot
    questions = get_chatbot_response_questions(prompt, file_content).split("\n")

    # Add new questions to the state without duplicating
    for question in questions:
        if question not in st.session_state["all_questions"]:
            st.session_state["all_questions"].append(question)


# Display questions with checkboxes
st.header("Select Questions")
with st.form("question_selection"):
    selected = st.multiselect(
        "Available Questions",
        options=st.session_state["all_questions"],
        default=st.session_state["selected_questions"],
        help="Select questions to add to the list.",
    )
    submit = st.form_submit_button("Save Selection")

# Update selected questions only if the form is submitted
if submit:
    st.session_state["selected_questions"] = selected

# Display selected questions
st.subheader("Selected Questions")
for idx, question in enumerate(st.session_state["selected_questions"]):
    col1, col2 = st.columns([8, 2])
    with col1:
        st.write(f"{idx + 1}. {question}")
    with col2:
        if st.button("Answer", key=f"answer_{idx}"):
            st.session_state.conversation_history.append(
                HumanMessage(content=f"Please help me answer this question: {question}")
            )
            response = generate_response(question)
            st.session_state.conversation_history.append(response)
            st.success("Success!")


# User input handling
prompt = st.chat_input("Say something")
if prompt:
    st.session_state.conversation_history.append(HumanMessage(content=prompt))
    response = generate_response(prompt)
    st.session_state.conversation_history.append(response)

# Conversation history display
for message in st.session_state.conversation_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)