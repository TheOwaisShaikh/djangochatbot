# import os
# import json
# from fuzzywuzzy import fuzz, process
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from msrest.authentication import CognitiveServicesCredentials
# import spacy
# from spacy.lang.en import English
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)

# subscription_key = os.getenv("COMPUTER_VISION_SUBSCRIPTION_KEY")
# endpoint = os.getenv("COMPUTER_VISION_ENDPOINT")

# nlp = spacy.load("en_core_web_sm")

# conversation_history_file = "conversation_history.json"

# # Clear conversation history on page reload
# def clear_conversation_history(conversation_history_file='conversation_history.json'):
#     try:
#         if os.path.exists(conversation_history_file):
#             os.remove(conversation_history_file)
#             print(f"Cleared conversation history: {conversation_history_file}")
#     except Exception as e:
#         print(f"Failed to clear conversation history: {e}")

# clear_conversation_history()

# MAX_CONVERSATIONS = 5  # Define the maximum number of conversations to keep

# def save_conversation(conversation_data, conversation_history_file='conversation_history.json'):
#     try:
#         if os.path.exists(conversation_history_file):
#             with open(conversation_history_file, 'r') as file:
#                 history = json.load(file)
#         else:
#             history = []

#         # Append the new conversation data
#         history.append(conversation_data)

#         # Ensure only the latest MAX_CONVERSATIONS conversations are kept
#         if len(history) > MAX_CONVERSATIONS:
#             history = history[-MAX_CONVERSATIONS:]

#         # Save the updated history back to the file
#         with open(conversation_history_file, 'w') as file:
#             json.dump(history, file)

#         print(f"Conversation saved to {conversation_history_file}")

#     except Exception as e:
#         print(f"Failed to save conversation: {e}")


# def load_conversation(conversation_history_file='conversation_history.json'):
#     try:
#         if os.path.exists(conversation_history_file):
#             with open(conversation_history_file, 'r') as file:
#                 history = json.load(file)
#                 print("Loaded conversation history:", history)
#                 return history
#         return []
#     except Exception as e:
#         print(f"Failed to load conversation: {e}")
#         return []

# def is_relevant_response(response):
#     # Define a list of phrases that indicate an irrelevant response
#     irrelevant_phrases = [
#         "i apologize", "i cannot find the answer", "does not contain any information",
#         "The provided context does not mention anything about"
#     ]
#     # Check if the response contains any of the irrelevant phrases
#     return not any(phrase in response.lower() for phrase in irrelevant_phrases)

# def load_dataset(file_path='dataset.txt'):
#     intents = {}
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         current_category = None
#         for line in lines:
#             line = line.strip()
#             if not line or line.startswith('#'):
#                 continue
#             if ':' in line:
#                 key, value = line.split(':', 1)
#                 intents[key.strip().lower()] = value.strip()
#     return intents

# dataset = load_dataset()

# def get_best_intent(user_input, intents, threshold=80):
#     user_input = user_input.lower().strip()
#     best_match, best_score = None, 0
#     for intent in intents.keys():
#         score = fuzz.ratio(user_input, intent)
#         if score > best_score:
#             best_match, best_score = intent, score
#     if best_score >= threshold:
#         return best_match
#     return None

# def handle_common_intents(user_question, intents):
#     best_intent = get_best_intent(user_question, intents)
#     if best_intent:
#         return intents[best_intent]
#     return None

# def get_computervision_client():
#     credentials = CognitiveServicesCredentials(subscription_key)
#     client = ComputerVisionClient(endpoint, credentials)
#     return client

# def extract_text_from_image(image_stream):
#     client = get_computervision_client()
#     ocr_result = client.recognize_printed_text_in_stream(image_stream)
#     lines = []
#     for region in ocr_result.regions:
#         for line in region.lines:
#             lines.append(" ".join([word.text for word in line.words]))
#     return " ".join(lines)

# def load_pdf_from_directory(pdf_path):
#     try:
#         text = ""
#         pdf_reader = PdfReader(pdf_path)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#         return text
#     except FileNotFoundError as e:
#         print(f"PDF file '{pdf_path}' not found.")
#         return None

# def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return text_splitter.split_text(text)

# def get_vector_store(text_chunks, model_path="models/embedding-001"):
#     print("Creating vector store...")
#     embeddings = GoogleGenerativeAIEmbeddings(model=model_path)
#     vector_store = FAISS.from_texts(text_chunks, embeddings)
#     vector_store.save_local("faiss_index")
#     print("Vector store created.")
#     return vector_store

# def load_or_get_vector_store(session_key, text_chunks=None, model_path="models/embedding-001"):
#     if session_key not in globals() or text_chunks is not None:
#         print("Loading or creating vector store...")
#         globals()[session_key] = get_vector_store(
#             text_chunks, model_path=model_path)
#     print("Vector store loaded.")
#     return globals()[session_key]

# def get_conversation_chain(model="gemini-pro", temperature=0):
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, and don't provide the wrong answer. If the answer cannot be found within this context, use the Gemini API to generate the response.\n\n
#     Context:\n{context}?\n
#     Question:\n{question}\n
#     """
#     chat_model = ChatGoogleGenerativeAI(model=model, temperature=temperature)
#     prompt = PromptTemplate(template=prompt_template,
#                             input_variables=["context", "question"])
#     return load_qa_chain(chat_model, prompt=prompt)

# def process_user_input(user_question, image_text, conversation_data):
#     print("Processing user input...")
#     print(f"User asked: {user_question}")

#     # Check for common intents
#     intent_response = handle_common_intents(user_question, dataset)
#     if intent_response:
#         print("Common intent detected, responding directly.")
#         return intent_response

#     faiss_index_path = '/root/altacademy/polls/faiss_index'

#     # Load previous conversations
#     conversation_history = load_conversation()

#     # Initialize last_bot_response
#     last_bot_response = ""

#     # Combine user question with extracted image text and previous context
#     if conversation_history:
#         last_bot_response = conversation_history[-1].get('bot_response', "")
#         if is_relevant_response(last_bot_response):
#             combined_input = f"{last_bot_response} {user_question}. Extracted Information: {image_text}"
#             print("Combined input with previous bot response:", combined_input)
#         else:
#             combined_input = f"{user_question}. Extracted Information: {image_text}"
#             print("Previous bot response not relevant, using only current user question.")
#     else:
#         combined_input = f"{user_question}. Extracted Information: {image_text}"
#         print("No previous conversation history found, using only current user question.")

#     # Update conversation context
#     conversation_data['context'] = combined_input

#     # Maintain conversation history
#     if 'history' not in conversation_data:
#         conversation_data['history'] = []

#     # Append current interaction to history
#     conversation_data['history'].append({
#         'user_question': user_question,
#         'image_text': image_text
#     })

#     # Load the vector store and perform similarity search
#     model_path = "models/embedding-001"
#     embeddings = GoogleGenerativeAIEmbeddings(model=model_path)
#     faiss_index = FAISS.load_local(
#         faiss_index_path, embeddings, allow_dangerous_deserialization=True)
#     docs = faiss_index.similarity_search(combined_input)

#     # Get conversation chain and invoke it
#     chain = get_conversation_chain()
#     response = chain.invoke({"input_documents": docs,
#                              "question": combined_input}, return_only_outputs=True)
#     generated_response = response["output_text"]

#     # Check if response is empty or irrelevant
#     if not generated_response.strip() or any(error_phrase in generated_response.lower() for error_phrase in ["i apologize", "The provided context does not mention anything about", "i cannot find the answer", "does not contain any information"]):
#         # Fallback to Gemini model using the latest user question as prompt
#         print("Fallback to Gemini model using latest question.")
#         latest_question = conversation_data['history'][-1]['user_question']
#         generated_response = generate_fallback(latest_question, last_bot_response)  # Pass last_bot_response here
    
#     # Save the current conversation
#     conversation_data['bot_response'] = generated_response
#     save_conversation(conversation_data)

#     print("Generated response:", generated_response)
#     print("User input processed.")

#     return generated_response

# def generate_fallback(question, previous_response):
#     print("Using fallback with Gemini model.")
    
#     # Only include the previous response if it is relevant and exists
#     if previous_response and is_relevant_response(previous_response):
#         prompt = f"""
#         Previous Bot Response: {previous_response}
#         User Question: {question}
#         Ensure the response aligns with the previous conversation. If it doesn't, prioritize the user's current question.
#         Only answer psychology-related questions. If the question is not related to psychology, respond with: "Kindly ask a psychology-related question. Thank you."
#         """
#     else:
#         prompt = f"""
#         User Question: {question}
#         Ensure the response is detailed and related to psychology. If the question is not related to psychology, respond with: "Kindly ask a psychology-related question. Thank you."
#         """
#         print("Previous bot response not relevant or missing, using only current user question for fallback.")
    
#     print("Fallback prompt:", prompt)
    
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     response = model.generate_content(prompt)
#     return response.text



# # Function to ask questions and get responses
# def ask_questions():
#     # Define the questions
#     question1 = "Evaluate the causes of impulse control disorders and non-substance addictive disorder, including a discussion of reductionism."
#     question2 = "Explain in detail."

#     # Process the first question
#     response1 = process_user_input(question1, "", conversation_data={})
#     print(f"Response to first question: {response1}")

#     # Process the second question
#     response2 = process_user_input(question2, "", conversation_data={})
#     print(f"Response to second question: {response2}")
import os
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import spacy
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import logging
import json
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the TOKENIZERS_PARALLELISM environment variable to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
subscription_key = os.getenv("COMPUTER_VISION_SUBSCRIPTION_KEY")
endpoint = os.getenv("COMPUTER_VISION_ENDPOINT")

# Configure the generative AI API
logging.debug("Configuring generative AI API...")
# genai.configure(api_key=api_key)  # Uncomment if using Google API

# Load the Sentence Transformer model
logging.debug("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract Q&A from PDF
def extract_qa_from_pdf(pdf_path):
    logging.debug(f"Extracting Q&A from PDF: {pdf_path}")
    qa_data = {}
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            # Split text by lines
            lines = text.split('\n')
            current_question = None
            for line in lines:
                if line.startswith("Question"):
                    current_question = line
                elif line.startswith("Answer") and current_question:
                    answer_start_index = lines.index(line) + 1
                    answer_lines = lines[answer_start_index:]
                    answer = " ".join(answer_lines).strip()
                    qa_data[current_question] = answer
                    current_question = None
    logging.debug(f"Extracted Q&A data: {qa_data}")
    return qa_data

# Function to load PDF from directory
def load_pdf_from_directory(pdf_path):
    logging.debug(f"Loading PDF from directory: {pdf_path}")
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except FileNotFoundError as e:
        logging.error(f"PDF file '{pdf_path}' not found.")
        return None

# Function to split text into chunks
def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    logging.debug(f"Splitting text into chunks of size {chunk_size} with overlap {chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Function to create a vector store
def get_vector_store(text_chunks, model_path="models/embedding-001"):
    logging.debug("Creating vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model=model_path)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    logging.debug("Vector store created.")
    return vector_store

# Initialize Computer Vision Client
logging.debug("Initializing Computer Vision Client...")
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Function to extract text from an image using Azure Computer Vision
def extract_text_from_image(image_stream):
    logging.debug("Extracting text from image...")
    ocr_result = computervision_client.recognize_printed_text_in_stream(image_stream)
    lines = []
    for region in ocr_result.regions:
        for line in region.lines:
            lines.append(" ".join([word.text for word in line.words]))
    extracted_text = " ".join(lines)
    logging.debug(f"Extracted text: {extracted_text}")
    return extracted_text

def process_query(text=None, image_text=None):
    logging.debug("Processing query...")
    combined_prompt = ""
    if image_text:
        combined_prompt += f"Extracted text from image: {image_text}\n"
    if text:
        combined_prompt += f"User Prompt: {text}"
    if not combined_prompt:
        logging.warning("No text or image text provided.")
        return "Please provide either a text query or an image."
    logging.debug(f"Combined prompt: {combined_prompt}")
    return chat(combined_prompt)
def chat(user_input, email, session_id, image_text=None):
    logging.debug("Chat function invoked...")
    last_bot_response = get_last_bot_response(email, session_id)
    logging.debug(f"Last bot response: {last_bot_response}")

    # Combine user question and image text (if available) into a single prompt
    combined_input = user_input
    if image_text:
        combined_input = f"{image_text}\n{user_input}" if user_input else image_text

    if last_bot_response:
        combined_prompt = f"Previous Bot Response: {last_bot_response}\nUser Question: {combined_input}"
    else:
        combined_prompt = f"User Question: {combined_input}"

    # Search for the answer in the PDF data first
    pdf_answer = search_pdf_data(combined_input)
    if (pdf_answer):
        # Align the PDF extracted answer using the Gemini API
        prompt = (f"You are a polite academic helper from AltAcademy. Your role is to assist students with chemistry-related "
                  f"questions specifically for A levels and O levels, CAIE, and AQA. Align the following answer for better understanding: {pdf_answer}. "
                  f"If the question is not related to chemistry, respond with: 'Kindly ask a chemistry-related question. Thank you.'\n\n{combined_prompt}")
        aligned_response = generate_response(prompt)
        response = aligned_response
    else:
        # If no suitable answer is found in the PDF data, use the API
        prompt = (f"You are a polite academic helper from AltAcademy. Your role is to assist students with chemistry-related "
                  f"questions specifically for A levels and O levels, CAIE, and AQA. Always answer as if you are helping students at these academic levels, "
                  f"but do not mention the levels in your response. If the question is not related to chemistry, respond with: 'Kindly ask a chemistry-related question. Thank you.'\n\n{combined_prompt}")
        response = generate_response(prompt)
    
    logging.debug(f"Generated response: {response}")
    log_conversation(combined_input, response, email, session_id)
    return response




# Function to search PDF data
def search_pdf_data(user_input):
    logging.debug("Searching PDF data...")
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    if question_embeddings is None or len(question_embeddings) == 0:
        logging.debug("No question embeddings available.")
        return None
    cos_scores = util.pytorch_cos_sim(user_input_embedding, question_embeddings)[0]
    best_match_idx = cos_scores.argmax().item()
    best_match_score = cos_scores[best_match_idx].item()
    matched_question = questions[best_match_idx]
    if best_match_score > 0.7:  # Adjust the threshold as needed
        logging.debug(f"Best match found with score {best_match_score}")
        return qa_data[matched_question]
    logging.debug("No suitable match found in PDF data.")
    return None

# Function to generate a response
def generate_response(prompt):
    logging.debug("Generating response from generative model...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    logging.debug(f"Response generated: {response.text}")
    return response.text

# Function to log conversation specific to a user
def log_conversation(user_input, bot_response, email, session_id):
    logging.debug("Logging conversation...")
    # Create a unique file path for each user based on their email and session ID
    log_dir = os.path.join('conversations', email)
    os.makedirs(log_dir, exist_ok=True)
    file_name = os.path.join(log_dir, f'{session_id}.txt')
    
    with open(file_name, "a") as file:
        file.write(f"You: {user_input}\n")
        file.write(f"Bot: {bot_response}\n\n")

# Function to get the last bot response specific to a user
def get_last_bot_response(email, session_id):
    log_dir = os.path.join('conversations', email)
    file_name = os.path.join(log_dir, f'{session_id}.txt')

    if not os.path.exists(file_name):
        return ""

    with open(file_name, "r") as file:
        lines = file.readlines()

    for line in reversed(lines):
        if line.startswith("Bot:"):
            return line[5:].strip()
    return ""
# Load the Q&A data and compute the embeddings
logging.debug("Loading Q&A data and computing embeddings...")
qa_data = extract_qa_from_pdf("testing.pdf")
questions = list(qa_data.keys())
question_embeddings = model.encode(questions, convert_to_tensor=True) if questions else []
logging.debug("Initialization complete.")






# # Test function to ask questions
# def ask_questions():
#     # Define the questions
#     question1 = "tell me about benzene"
#     question2 = "Explain in detail."

#     # Process the first question
#     response1 = process_query(text=question1)
#     print(f"Response to first question: {response1}")

#     # Process the second question
#     response2 = process_query(text=question2)
#     print(f"Response to second question: {response2}")

# ask_questions()

