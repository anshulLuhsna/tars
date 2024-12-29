# react_agent.py

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Initialize the model
model = ChatOpenAI(model="gpt-4", temperature=0)

# Define a sample tool
@tool
def get_current_time() -> str:
    """Returns the current time."""
    from datetime import datetime
    return datetime.now().isoformat()

# Create the ReAct agent
react_agent = create_react_agent(model=model, tools=[get_current_time])
