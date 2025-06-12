import os
import io
import base64
import json
import time # Import time for measuring response duration
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import openai

# Local module import
# This imports the DataProcessor class from data_processor.py
from data_processor import DataProcessor

# --- Configuration ---
# Load environment variables from the .env file.
# This ensures that sensitive information like API keys is not hardcoded
# and is loaded securely from the environment.
load_dotenv()

# Get the API key from environment variables.
# This key should be your aipipe.org key, which might start with 'sk-proj-'.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # If the API key is not found, raise an error to stop the application startup
    # and inform the user to set it correctly in their .env file.
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")

# Configure the path to the Tesseract OCR executable.
# This is essential for the Optical Character Recognition (OCR) functionality.
# The path varies by operating system:
# - On Linux/macOS, it's commonly '/usr/bin/tesseract'.
# - On Windows, it's the full path to tesseract.exe (e.g., 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe').
# Ensure Tesseract is installed on your system and the path is correct in .env.
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", '/usr/bin/tesseract')

# --- Initialize FastAPI Application ---
# FastAPI provides a modern, fast (high-performance) web framework for building APIs.
app = FastAPI(
    title="TDS Virtual TA API", # Title for API documentation (e.g., in /docs)
    description="Automatically answers student questions based on course content and Discourse posts.", # Description for API docs
    version="1.0.0" # Version of the API
)

# --- Initialize Data Processor and Load/Create Index ---
# This instance of DataProcessor manages our knowledge base.
# It will attempt to load the pre-built FAISS index and associated documents from disk.
# If these files are not found (e.g., on the very first run, or if they were deleted),
# it will automatically trigger the process to load and chunk data, create embeddings,
# and build a new FAISS index, then save it for future use.
data_processor = DataProcessor()
if not data_processor.load_index():
    print("Building new index (this might take a moment, especially on first run or if data is large)...")
    data_processor.load_and_chunk_data()
    data_processor.create_faiss_index()
    data_processor.save_index()
else:
    print("Loaded existing FAISS index and documents.")

# --- Pydantic Models for API Request and Response ---
# Pydantic models are used by FastAPI for automatic data validation, serialization,
# and generation of OpenAPI (Swagger) documentation, making API development easier
# and less error-prone.

# Defines the expected structure of the incoming POST request to the /api/ endpoint.
class QuestionRequest(BaseModel):
    question: str # The student's question, which is a mandatory string field.
    image: Optional[str] = None # An optional field for a base64 encoded image string.

# Defines the structure for each individual link object that will be returned in the API response.
class Link(BaseModel):
    url: str # The URL of the relevant source (e.g., a Discourse post link).
    text: str # A brief, descriptive text for the link.

# Defines the overall structure of the JSON response that the API will send back.
class AnswerResponse(BaseModel):
    answer: str # The generated answer to the student's question.
    links: List[Link] # A list of relevant links, using the 'Link' model defined above.

# --- Helper Functions ---

def extract_text_from_image(base64_image: str) -> Optional[str]:
    """
    Decodes a base64 encoded image string and performs Optical Character Recognition (OCR)
    to extract any readable text from the image.
    Returns the extracted text as a string, or None if no text is found or an error occurs.
    """
    try:
        # Decode the base64 string back into image bytes.
        image_bytes = base64.b64decode(base64_image)
        # Open the image using Pillow (PIL) from the bytes in memory.
        image = Image.open(io.BytesIO(image_bytes))
        # Use pytesseract to perform OCR on the image and extract text.
        text = pytesseract.image_to_string(image)
        # Return the extracted text, stripped of leading/trailing whitespace.
        # Return None if the extracted text is empty.
        return text.strip() if text else None
    except Exception as e:
        # Catch any errors during the OCR process and print them for debugging.
        print(f"Error during OCR process for image: {e}")
        return None # Indicate failure by returning None.

def get_answer_from_llm(question: str, context_docs: List[Dict]) -> Dict:
    """
    Interacts with the Large Language Model (LLM) to generate an answer.
    This function feeds the student's question and the most relevant context documents
    (retrieved from the FAISS index) to the LLM. It then parses the LLM's response
    to separate the main answer text from any extracted links.
    """
    # Initialize the OpenAI client.
    # CRITICAL CHANGE: The base_url is set to aipipe.org's endpoint.
    # This ensures that all API requests are routed through aipipe.org.
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY, # Your aipipe.org key from .env
        base_url="https://aipipe.org/openai/v1" # The base URL for aipipe.org's OpenAI-compatible API
    )

    # Format the retrieved context documents into a clear string for the LLM.
    # Each document is tagged with its source type (e.g., 'course_content', 'discourse_post')
    # and includes relevant metadata like file names or Discourse URLs/topics.
    context_str_parts = []
    for i, doc in enumerate(context_docs):
        source_info = f"Source {i+1} (Type: {doc['metadata']['source_type']}"
        if doc['metadata']['source_type'] == 'discourse_post':
            # For Discourse posts, include topic title and URL.
            source_info += f", Topic: {doc['metadata'].get('topic_title', 'N/A')}, URL: {doc['metadata'].get('url', 'N/A')}"
        else: # For course_content
            # For course content, include the source filename.
            source_info += f", File: {doc['metadata'].get('source_file', 'N/A')}"
        source_info += ")"
        # Append the formatted source info and the document text to the context parts.
        context_str_parts.append(f"<{source_info}>\n{doc['text']}\n</{source_info}>")
    
    # Join all context parts into a single string, separated by double newlines.
    context_str = "\n\n".join(context_str_parts)

    # Define the prompt messages for the LLM.
    # This is a system message (defining the LLM's role) and a user message (the actual query).
    # The prompt guides the LLM on how to generate the answer, use context, and format links.
    prompt_messages = [
        {"role": "system", "content": "You are a helpful and knowledgeable Teaching Assistant for an IIT Madras Data Science course. Provide accurate and concise answers to student questions based *only* on the provided context. If the context does not contain enough information to answer the question, state that clearly. Always strive to extract and provide relevant URLs from the context with a brief, descriptive text for each link. Format links precisely as '- URL: [link_url], Text: [descriptive_text_for_link]'."},
        {"role": "user", "content": f"""
        Context from course materials and Discourse forum:
        ---
        {context_str}
        ---

        Student Question: "{question}"

        Please provide your answer.
        If you find relevant URLs in the context, list them using the exact format at the end of your answer:
        - URL: [link_url], Text: [descriptive_text_for_link]
        
        Example of link format:
        - URL: https://discourse.onlinedegree.iitm.ac.in/t/example/123, Text: Important clarification on Assignment 3.

        Answer:
        """}
    ]

    try:
        # Make the API call to the OpenAI-compatible endpoint (via aipipe.org).
        response = client.chat.completions.create(
            # UPDATED: Using the generic 'gpt-3.5-turbo' model alias as it's more widely supported by proxies.
            model="gpt-4o-mini", # <--- Use the model suggested by the problem description", # <--- Changed from gpt-3.5-turbo-0125
            messages=prompt_messages, # The conversation history/prompt.
            temperature=0.1, # A low temperature promotes more factual and less creative responses.
            max_tokens=700, # Sets the maximum number of tokens for the LLM's response (answer + links).
            timeout=25 # Sets a timeout for the LLM call to help ensure the overall API response time is met (under 30s).
        )
        # Extract the content of the LLM's response.
        llm_response_content = response.choices[0].message.content.strip()

        answer_lines = []
        extracted_links = []
        
        # Parse the LLM's response line by line to separate the main answer from the formatted links.
        # This parsing logic is robust against variations in how the LLM might format the answer before the links.
        lines = llm_response_content.split('\n')
        
        for line in lines:
            if line.startswith("- URL: "):
                try:
                    # Attempt to extract the URL and text based on the specific format requested from the LLM.
                    parts = line[len("- URL: "):].split(", Text: ", 1)
                    if len(parts) == 2:
                        url_part = parts[0].strip()
                        text_part = parts[1].strip()
                        # Basic validation: ensure the extracted URL starts with http(s).
                        if url_part.startswith("http://") or url_part.startswith("https://"):
                            extracted_links.append({"url": url_part, "text": text_part})
                except Exception as e:
                    # Log an error if a link line cannot be parsed correctly.
                    print(f"Failed to parse link line from LLM response: '{line}'. Error: {e}")
            else:
                # Lines that do not match the link format are considered part of the main answer.
                answer_lines.append(line)

        # Join the answer lines to form the final answer text.
        final_answer = "\n".join(answer_lines).strip()
        # Remove any redundant "Answer:" prefix that the LLM might include at the start of the answer.
        if final_answer.lower().startswith("answer:"):
            final_answer = final_answer[len("answer:"):].strip()

        # Ensure all extracted links are unique to avoid duplicates in the final response.
        unique_links = []
        seen_urls = set()
        for link in extracted_links:
            if link['url'] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link['url'])

        # Return the processed answer and unique links.
        return {"answer": final_answer, "links": unique_links}

    except openai.APIStatusError as e:
        # Handle specific API errors from OpenAI/aipipe.org (e.g., authentication, quota, rate limits).
        # Print the detailed error response for debugging.
        print(f"AI Pipe/OpenAI API Error (Status: {e.status_code}): {e.response.json()}")
        # Raise an HTTPException to propagate the error back to the API caller.
        raise HTTPException(
            status_code=e.status_code,
            detail=f"AI Pipe/OpenAI API Error: {e.response.json().get('message', 'Unknown error')}"
        )
    except openai.APITimeoutError:
        # Handle cases where the LLM call times out (doesn't respond within the allocated time).
        print("AI Pipe/OpenAI API call timed out.")
        raise HTTPException(status_code=504, detail="LLM response timed out. Please try again.")
    except Exception as e:
        # Catch any other unexpected errors during the LLM interaction process.
        print(f"An unexpected error occurred during LLM call: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during LLM processing.")

# --- API Endpoint Definition ---

@app.post("/api/", response_model=AnswerResponse)
async def get_tds_answer(request: QuestionRequest):
    """
    This is the main API endpoint for your Virtual TA.
    It accepts a student's question and an optional base64 encoded image attachment.
    It retrieves relevant information from the indexed course content and Discourse posts,
    uses an LLM to generate an answer based on this context, and then returns
    the answer along with any relevant links.
    """
    # Record the start time to monitor the total response duration, aiming for under 30 seconds.
    start_time = time.monotonic()

    # Initialize the text used for retrieval and LLM prompting with the student's question.
    full_question_text = request.question
    if request.image:
        # If an image is provided, extract text from it using OCR.
        image_text = extract_text_from_image(request.image)
        if image_text:
            print(f"Successfully extracted text from image: '{image_text[:100]}...'")
            # Augment the main question with the extracted image text for better context for RAG and LLM
            full_question_text = f"{request.question}\n\n[Additional context from image]: {image_text}"
        else:
            print("Warning: No readable text could be extracted from the provided image.")

    # Retrieve the most relevant documents from our FAISS index based on the combined question text.
    # Retrieving 7 documents provides ample context to the LLM.
    retrieved_docs = data_processor.search_documents(full_question_text, k=7)

    if not retrieved_docs:
        # If no relevant documents are found, provide a graceful fallback answer to the user.
        answer_text = "I couldn't find specific information related to your question in the available course materials or Discourse posts. Please try rephrasing your question or refer to the course content directly."
        # Return an empty list of links as no relevant context was found.
        return AnswerResponse(answer=answer_text, links=[])

    # Use the LLM to generate the final answer and extract links from the retrieved context.
    llm_output = get_answer_from_llm(full_question_text, retrieved_docs)

    # Calculate the total time taken for the request.
    end_time = time.monotonic()
    response_time = end_time - start_time
    print(f"Request processed in {response_time:.2f} seconds.")
    # Provide a warning if the response time is close to or exceeds the 30-second limit.
    if response_time > 29:
        print("Warning: API response time is approaching or exceeding 30 seconds.")

    # Return the generated answer and links as per the AnswerResponse Pydantic model.
    return AnswerResponse(answer=llm_output["answer"], links=llm_output["links"])

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    A simple GET endpoint to confirm that the API is running and responsive.
    This is useful for monitoring the deployment status of the application.
    """
    return {"status": "ok", "message": "TDS Virtual TA is running and healthy!"}

# --- Main entry point for local development ---
# This block allows you to run the FastAPI application directly using `python app.py`
# for convenience during development. In a production environment, you would typically
# use `uvicorn app:app --host 0.0.0.0 --port 8000` directly or manage it via Docker.
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI application locally...")
    # IMPORTANT REMINDERS FOR LOCAL RUN:
    # 1. Ensure your OPENAI_API_KEY is correctly set in your .env file (this should be your aipipe.org key).
    # 2. Ensure TESSERACT_CMD is configured correctly in your .env file if Tesseract
    #    is not in your system's PATH (e.g., for Windows: TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe").
    uvicorn.run(app, host="0.0.0.0", port=8000)
