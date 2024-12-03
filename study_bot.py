import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import docx
import tempfile
import requests
import json

def extract_text_with_ocr(file):
    """Use OCR to extract text from a scanned PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    images = convert_from_path(temp_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def extract_text_directly(file):
    """Extract text from a PDF file directly."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        text += page_text if page_text else ""
    return text

def extract_text_from_docx(file):
    """Extract text from a Word document."""
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def filter_common_questions(text_list):
    """Filter out duplicate questions from a list."""
    question_set = set()
    unique_questions = []
    for question in text_list:
        if question.strip() not in question_set:
            question_set.add(question.strip())
            unique_questions.append(question.strip())
    return unique_questions

def get_chatbot_response(question, context):
    """Get a response from the chatbot API with context."""
    url = "https://api.worqhat.com/api/ai/content/v4"
    
    # Add context (previous extracted questions) to the prompt
    context_text = "\n".join(context) if context else "No previous context available."
    
    payload = json.dumps({
        "question": question + context_text,
        "model": "aicon-v4-large-160824",
        "randomness": 0.5,
        "stream_data": False,
        "training_data": f"You are an expert teacher ",
        "response_type": "text"
    })
    
    headers = {
        'Content-Type': 'application/json',
        "Authorization": "Bearer sk-128fd96fdf0d4039bdaef731e091de03"  # Replace with your token
    }
    
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json().get("content", "No response received.")
    else:
        return f"Error: {response.status_code}, {response.text}"

def get_chatbot_response_with_context(question, context, max_chunk_size=5):
    """Send multiple API requests with smaller chunks of context and display each response."""
    responses = []
    
    # Split the context into chunks of size 'max_chunk_size'
    for i in range(0, len(context), max_chunk_size):
        chunk = context[i:i + max_chunk_size]
        response = get_chatbot_response(question, chunk)
        responses.append(response)
        st.write(f"Alex: {response}")  # Display each response immediately
    
    # Combine all the responses from each chunk
    full_response = "\n".join(responses)
    return full_response

def main():
    st.title("Study Bot - Question Processor with Chatbot")
    st.subheader("Upload question papers, extract unique questions, and chat with an AI!")

    # File Upload Section
    uploaded_files = st.file_uploader(
        "Upload question papers (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True
    )

    extracted_questions = []

    if uploaded_files:
        extraction_mode = st.radio(
            "Choose Extraction Method:",
            ("Direct Extraction (for selectable text)", "OCR (for scanned documents)"),
        )
        
        all_questions = []

        for file in uploaded_files:
            file_name = file.name
            st.write(f"Processing: {file_name}")

            if file_name.endswith(".pdf"):
                if extraction_mode == "Direct Extraction (for selectable text)":
                    text = extract_text_directly(file)
                else:
                    text = extract_text_with_ocr(file)
            elif file_name.endswith(".docx"):
                text = extract_text_from_docx(file)
            else:
                st.error("Unsupported file format.")
                continue

            questions = text.split("\n")  # Split text into lines
            all_questions.extend(questions)

        # Filter unique questions
        extracted_questions = filter_common_questions(all_questions)

        st.subheader("Extracted Questions")
        for question in extracted_questions:
            st.write(f"- {question}")

    # Chatbot Section
    st.subheader("Chat with Alex")
    user_question = st.text_input("Ask a question to the chatbot:")

    if st.button("Get Response"):
        if user_question:
            # Get the response from the chatbot with chunked context
            full_response = get_chatbot_response_with_context(user_question, extracted_questions)
            st.write(f"Full Response:\n{full_response}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
