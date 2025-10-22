import os
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from mail import send_mail

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import os
from dotenv import load_dotenv
# from transformers import pipeline
import re
# import spacy
import json

# # Load spaCy model for named entity recognition
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("Downloading spaCy English model...")
#     from spacy.cli import download
#     download("en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")

# Sample email mapping (in a real-world scenario, this would come from a database or directory)
EMAIL_DIRECTORY = {
    "martin": "fytron@gmail.com",
    "sarah": "sarah.smith@company.com",
    "mike": "mike.johnson@company.com",
    "emily": "emily.wang@company.com"
}

# Add these global variables at the top of the file
EMAIL_INTENT_FLAG = False
EMAIL_INTENT_CONTEXT = {
    'recipient': None,
    'subject': None,
    'partial_message': None,
    'sender': None
}

# Load environment variables from .env file
load_dotenv()

def extract_recipient_from_query(query: str) -> str:
    """
    Extract the recipient's name from the query using NER and pattern matching.
    
    Args:
        query (str): The input query about sending an email
    
    Returns:
        str: Extracted recipient name or an empty string
    """
    # Preprocess query to lowercase
    query_lower = query.lower()
    
    # Pattern matching for email-related phrases
    email_indicators = [
        "send an email to", 
        "email", 
        "contact", 
        "reach out to", 
        "mail", 
        "write to"
    ]
    
    # Check if query contains email-related phrases
    if not any(indicator in query_lower for indicator in email_indicators):
        return ""
    
    # Use spaCy for named entity recognition
    doc = nlp(query)
    
    # Extract person names
    person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    # If person names found, return the first one
    if person_names:
        return person_names[0].lower()
    
    # Fallback: Try to extract name after email indicators
    for indicator in email_indicators:
        if indicator in query_lower:
            parts = query_lower.split(indicator)
            if len(parts) > 1:
                # Take the part after the indicator and strip whitespace
                potential_name = parts[1].strip().split()[0]
                return potential_name
    
    return ""


def detect_intent_llm(query: str) -> dict:
    """
    Use the language model to determine the intention of the query.
    
    Args:
        query (str): The input query
    
    Returns:
        dict: A dictionary with intent information
    """
    intent_prompt = f"""
    Analyze the following query and extract the intention. What is the query trying to achieve?
    Query: "{query}"
    """

    # Use the same LLM as in the retrieval chain
    print(intent_prompt)
    response = llm.invoke(intent_prompt)
    
    try:
        # Parse the response as JSON 
        # intent_data = json.loads(response.content)
        print(response.content)
        return response.content
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {"is_email_intent": False}

def detect_email_intent_llm(query: str, sender: str = "Chatbot") -> dict:
    """
    Use the language model to determine if the query is about sending an email.
    
    Args:
        query (str): The input query
        sender (str, optional): Name of the sender. Defaults to "Chatbot".
    
    Returns:
        dict: A dictionary with intent information
    """
    intent_prompt = f"""
    Analyze the following query and determine if the user wants to write an email.
    Query: "{query}"

    Return a JSON response with these keys:
    - is_email_intent: boolean (true if the query is about sending an email)
    - recipient: string (name of the recipient)
    - subject: string (concise email subject)
    - message: string (email body, formatted professionally and concisely from the sender {sender})
    - sender: "{sender}"

    If no clear email intent is found, return is_email_intent as false.
    """

    # Use the same LLM as in the retrieval chain
    print(intent_prompt)
    response = llm.invoke(intent_prompt)
    
    try:
        # Parse the response as JSON 
        intent_data = json.loads(response.content)
        print(intent_data)
        return intent_data
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {"is_email_intent": False}


def write_email_intent(query: str, sender: str = "Chatbot") -> dict:
    """
    Use the language model to determine if the query is about sending an email.
    
    Args:
        query (str): The input query
        sender (str, optional): Name of the sender. Defaults to "Chatbot".
    
    Returns:
        dict: A dictionary with intent information
    """
    intent_prompt = f"""
    Analyze the following query and extract details for an email:
    Query: "{query}"
    Sender: "{sender}"

    Provide a concise, professional email message with the following guidance:
    - Create a brief, direct email body that captures the key intent 
    - Include a proper greeting and closing
    - Keep the message straightforward and to the point
    - Use a professional but warm tone
    - Explicitly mention the sender's name {sender} in the closing

    Return a JSON response with these keys:
    - is_email_intent: boolean (true if the query is about sending an email)
    - recipient: string (name of the recipient)
    - subject: string (concise email subject)
    - message: string (email body, formatted professionally and concisely)
    - sender: string (sender's name)

    If no clear email intent is found, return is_email_intent as false.
    """

    # Use the same LLM as in the retrieval chain
    print(intent_prompt)
    response = llm.invoke(intent_prompt)
    
    try:
        # Parse the response as JSON 
        intent_data = json.loads(response.content)
        print(intent_data)
        return intent_data
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {"is_email_intent": False}

# def detect_email_intent_llm(query: str, sender: str = "Chatbot") -> dict:
#     """
#     Use the language model to determine if the query is about sending an email.
    
#     Args:
#         query (str): The input query
#         sender (str, optional): Name of the sender. Defaults to "Chatbot".
    
#     Returns:
#         dict: A dictionary with intent information
#     """
#     intent_prompt = f"""
#     Carefully analyze the following query to determine if it is specifically about sending an email:
#     Query: "{query}"
#     Sender: "{sender}"

#     Strictly follow these criteria to determine email intent:
#     1. The query MUST explicitly request sending an email
#     2. Look for clear email-sending phrases like:
#        - "Send an email to..."
#        - "Write an email about..."
#        - "Email [name] regarding..."
#     3. The query should contain:
#        - A clear recipient
#        - A specific message to be sent
#     4. General conversation or follow-up questions are NOT email intents

#     Return a JSON response with these keys:
#     - is_email_intent: boolean (ONLY true for explicit email-sending requests)
#     - recipient: string (name of the recipient, if clearly specified)
#     - subject: string (concise email subject, if mentioned)
#     - message: string (email body, if clearly defined)
#     - sender: string (sender's name)

#     If the query does not meet these strict criteria, return is_email_intent as false.
#     """

#     # Use the same LLM as in the retrieval chain
#     response = llm.invoke(intent_prompt)
    
#     try:
#         # Parse the response as JSON 
#         intent_data = json.loads(response.content)
        
#         # Additional filtering to prevent false positives
#         if intent_data.get('is_email_intent', False):
#             # Validate that the intent is truly about sending an email
#             email_trigger_phrases = [
#                 "send an email",
#                 "write an email",
#                 "email",
#                 "send email to",
#                 "write to"
#             ]
#             query_lower = query.lower()
            
#             # Check if any trigger phrases exist
#             if not any(phrase in query_lower for phrase in email_trigger_phrases):
#                 intent_data['is_email_intent'] = False
        
#         return intent_data
#     except Exception as e:
#         print(f"Error parsing LLM response: {e}")
#         return {"is_email_intent": False}

def get_email_from_name(name: str) -> str:
    """
    Retrieve email address from a given name.
    
    Args:
        name (str): Name of the recipient
    
    Returns:
        str: Email address or empty string if not found
    """
    # Normalize name (lowercase, remove extra whitespace)
    name_lower = name.lower().strip()
    
    # Direct dictionary lookup
    if name_lower in EMAIL_DIRECTORY:
        return EMAIL_DIRECTORY[name_lower]
    
    # Partial match in case of first name or last name
    for key, email in EMAIL_DIRECTORY.items():
        if name_lower in key:
            return email
    
    return ""

# Optional: Enhanced intent detection with more sophisticated checks
def detect_email_intent_advanced(query: str) -> bool:
    """
    Advanced email intent detection using multiple strategies.
    
    Args:
        query (str): The input query
    
    Returns:
        bool: Whether the query appears to have email-sending intent
    """
    query_lower = query.lower()
    
    # Email intent keywords and phrases
    email_keywords = [
        "send email", 
        "send an email", 
        "email", 
        "contact", 
        "write to", 
        "mail", 
        "reach out"
    ]
    
    # Check for email keywords
    if any(keyword in query_lower for keyword in email_keywords):
        return True
    
    # Check if query contains a name and email-related context
    doc = nlp(query)
    person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    if person_names and any(keyword in query_lower for keyword in email_keywords):
        return True
    
    return False

# Load pre-trained text classification model
# intent_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# def detect_email_intent_nlp(query: str):
#     """
#     Use a pre-trained NLP model to detect email-sending intent in the user's query.
#     """
#     try:
#         # Run classification
#         result = intent_classifier(query)
#         label = result[0]['label']
#         # Assuming the label for email-sending is 'SEND_EMAIL', adjust as needed
#         if label == 'SEND_EMAIL':  
#             return True
#         return False
#     except Exception as e:
#         print(f"Error in intent detection: {str(e)}")
#         return False

def send_email(to_email: str, subject: str, body: str):
    # Email configuration
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    
    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = formataddr(("ChatBot", sender_email))
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, "plain"))
    
    try:
        # Connect to SMTP server and send the email
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return {"status": "success", "message": "Email sent successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}



# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models
class QuestionRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []

class DocumentResponse(BaseModel):
    source: str
    content: str

class QAResponse(BaseModel):
    answer: str
    sources: List[DocumentResponse]

class EmailRequest(BaseModel):
    recipient: str
    subject: str
    message: str

@app.post("/send-email")
def send_email_endpoint(request: EmailRequest):
    email_status = send_email(
        to_email=request.recipient,
        subject=request.subject,
        body=request.message
    )
    if email_status["status"] == "success":
        return {"message": "Email sent successfully"}
    else:
        raise HTTPException(status_code=500, detail=email_status["message"])

# Load configuration
cwd = os.getcwd()
config_path = os.path.join(cwd, 'config.yml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Set API key
os.environ["OPENAI_API_KEY"] = config["api_key"]

# Folder paths
folder_path = "documents"
uploaded_folder_path = "uploaded_documents"

# Document loading

def load_documents(folder_path):
    loaders = []
    supported_extensions = {".pdf", ".txt", ".docx"}

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file_path)

            if ext.lower() == ".pdf":
                loaders.append(PyPDFLoader(file_path))
            elif ext.lower() == ".txt":
                loaders.append(TextLoader(file_path, encoding='utf-8'))
            elif ext.lower() == ".docx":
                loaders.append(UnstructuredFileLoader(file_path))
            else:
                print(f"Skipping unsupported file: {file_path}")

    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {loader}: {e}")

    return documents

def load_all_documents():
    os.makedirs(uploaded_folder_path, exist_ok=True)
    documents = load_documents(folder_path)
    uploaded_documents = load_documents(uploaded_folder_path)

    all_documents = documents + uploaded_documents
    print(f"Loaded {len(all_documents)} documents in total.")
    return all_documents

# Initialize QA system
def initialize_qa_system(new_documents):
    global vector_store, retrieval_chain

    all_documents = split_docs + text_splitter.split_documents(new_documents)
    vector_store = FAISS.from_documents(all_documents, embeddings)

    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 5}),
        document_chain
    )

    return vector_store, retrieval_chain

# Load documents at startup
all_documents = load_all_documents()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(all_documents)

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(split_docs, embeddings)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt_template = ChatPromptTemplate.from_template(
    """
    You are an expert document analyzer. Consider the previous conversation context 
    and the current question. Provide a helpful and contextually relevant answer.

    {previous_context_prompt}

    Current Question:
    {input}

    Context from Documents:
    {context}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt_template)

retrieval_chain = create_retrieval_chain(
    vector_store.as_retriever(search_kwargs={"k": 3}),
    document_chain
)

# API endpoints
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    os.makedirs(uploaded_folder_path, exist_ok=True)

    new_documents = []
    for file in files:
        file_path = os.path.join(uploaded_folder_path, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        _, ext = os.path.splitext(file.filename)
        if ext.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext.lower() == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext.lower() == ".docx":
            loader = UnstructuredFileLoader(file_path)
        else:
            print(f"Skipping unsupported file: {file_path}")
            continue

        try:
            new_documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    vector_store, retrieval_chain = initialize_qa_system(new_documents)

    return {
        "message": f"Uploaded and indexed {len(files)} files and created {len(new_documents)} chunks",
        "file_count": len(files),
        "chunk_count": len(new_documents),
    }

@app.post("/ask", response_model=QAResponse)
async def ask_question(request: QuestionRequest):
    if retrieval_chain is None:
        raise HTTPException(status_code=500, detail="Retrieval chain is not initialized. Upload documents first.")
            
    global EMAIL_INTENT_FLAG, EMAIL_INTENT_CONTEXT

    try:
        query = request.query

        # If email intent flag is already set, handle email composition
        if EMAIL_INTENT_FLAG:
            # Use LLM to help complete the email based on previous context and current input
            complete_email_prompt = f"""
            You are helping to compose an email with the following existing context:
            Recipient: {EMAIL_INTENT_CONTEXT['recipient']}
            Partial Subject: {EMAIL_INTENT_CONTEXT['subject'] or 'No subject'}
            Partial Message: {EMAIL_INTENT_CONTEXT['partial_message'] or ''}
            Sender: {EMAIL_INTENT_CONTEXT['sender'] or 'Chatbot'}

            Current user input: "{query}"

            Help complete the email by:
            1. Incorporating the new input into the existing context
            2. Determining if the email is ready to send
            3. Providing a complete, professional email message

            Return a JSON response with:
            - is_complete: boolean (is the email ready to send)
            - subject: string (final email subject)
            - message: string (complete email body)
            """

            # Use LLM to process the email composition
            response = llm.invoke(complete_email_prompt)
            
            try:
                email_completion = json.loads(response.content)
                
                # Check if email is complete and ready to send
                if email_completion.get('is_complete', False):
                    recipient_email = get_email_from_name(EMAIL_INTENT_CONTEXT['recipient']) or "fytron@gmail.com"
                    
                    # Send the email
                    email_status = send_email(
                        to_email=recipient_email, 
                        subject=email_completion['subject'],
                        body=email_completion['message']
                    )
                    
                    # Reset the email intent flag and context
                    EMAIL_INTENT_FLAG = False
                    EMAIL_INTENT_CONTEXT = {
                        'recipient': None,
                        'subject': None,
                        'partial_message': None,
                        'sender': None
                    }
                    
                    return QAResponse(
                        answer=f"I have sent an email to {recipient_email}.",
                        sources=[]
                    )
                else:
                    # Update the context with new information
                    EMAIL_INTENT_CONTEXT['partial_message'] = email_completion.get('message', '')
                    
                    return QAResponse(
                        answer="I'm still working on composing the email. Please provide more details.",
                        sources=[]
                    )
            
            except Exception as e:
                print(f"Error processing email composition: {e}")
                return QAResponse(
                    answer="I'm having trouble composing the email. Could you clarify?",
                    sources=[]
                )
        intent = detect_intent_llm(query)
        # Detect email intent for new queries
        email_intent = detect_email_intent_llm(intent)
        
        if email_intent.get('is_email_intent', False):
            # Set the email intent flag and store initial context
            EMAIL_INTENT_FLAG = True
            EMAIL_INTENT_CONTEXT = {
                'recipient': email_intent.get('recipient'),
                'subject': email_intent.get('subject'),
                'partial_message': email_intent.get('message'),
                'sender': email_intent.get('sender')
            }
            
            return QAResponse(
                answer=f"I'm ready to help you compose an email to {EMAIL_INTENT_CONTEXT['recipient']}. Please provide more details about what you'd like to say.",
                sources=[]
            )

        # Normal query processing if no email intent
        return await process_normal_query(request)

        

    except Exception as e:
        print(f"Error in /ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
async def process_normal_query(request: QuestionRequest):
    previous_context_prompt = ""
    if request.chat_history:
        previous_context_prompt = "Previous Conversation:\n" + "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in request.chat_history
        ])

    current_document_chain = create_stuff_documents_chain(
        llm,
        prompt_template.partial(
            previous_context_prompt=previous_context_prompt or "No previous context."
        )
    )

    current_retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 5}),
        current_document_chain
    )

    # if "e-mail" in request.query.lower():
    #     # Extract recipient's email dynamically (for now, default to fytron@gmail.com)
    #     recipient_email = "fytron@gmail.com"  # Placeholder: Extract or map dynamically
    #     subject = "Chatbot Request"
    #     message = f"Chatbot message: {request.query}"  # Use query as the message content

    #     # Call the email-sending function
    #     email_status = send_email(to_email=recipient_email, subject=subject, body=message)
    #     if email_status["status"] == "success":
    #         return QAResponse(
    #             answer=f"I have sent an email to {recipient_email}.",
    #             sources=[]
    #         )
    #     else:
    #         raise HTTPException(status_code=500, detail=email_status["message"])


    response = current_retrieval_chain.invoke({
        "input": request.query
    })

    sources = [
        DocumentResponse(
            source=doc.metadata.get('source', 'Unknown'),
            content=doc.page_content[:200] + '...'
        ) for doc in response['context']
    ]

    return QAResponse(
        answer=response['answer'],
        sources=sources
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
