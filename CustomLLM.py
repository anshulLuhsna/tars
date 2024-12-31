from langchain_core.messages import HumanMessage, AIMessage
import json
import requests

class CustomLLM:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def _call(self, prompt: list, conversation_history: list) -> str:
        # Convert messages to a JSON-serializable format
        def serialize_message(message):
            if isinstance(message, HumanMessage):
                return {"type": "user", "content": message.content}
            elif isinstance(message, AIMessage):
                return {"type": "assistant", "content": message.content}
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
        
        serialized_history = [serialize_message(msg) for msg in conversation_history]
        
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

    def invoke(self, prompt_value, conversation_history):
        return self._call(prompt_value, conversation_history)