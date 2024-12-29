# Use Python as the base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install LangGraph CLI
RUN pip install --upgrade "langgraph-cli[inmem]"

# Copy app files to container
COPY . /app

# Install app dependencies in editable mode
RUN pip install -e .
RUN pip install langchain-openai langchain langgraph

# Expose the server port
EXPOSE 8000

# Command to run the LangGraph server in development mode
CMD ["langgraph", "dev"]
