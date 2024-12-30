# Base image with Python
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy application files to the container
COPY . /app

# Install system dependencies (optional, if needed by LangGraph or your app)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --no-cache-dir "langgraph-cli[inmem]" 


# Set environment variables (ensure you have a .env file)
ENV LANGGRAPH_ENV=development
COPY .env /app/.env

# Expose the port the app runs on
EXPOSE 2024
# Command to run the application
CMD ["sh", "-c", "pip install -U 'langgraph-cli[inmem]' && langgraph dev --host 0.0.0.0"]
