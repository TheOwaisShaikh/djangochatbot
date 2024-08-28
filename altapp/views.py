import os
from .models import User, ChatSession, Message, Conversation
from django.shortcuts import get_object_or_404
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import render
import uuid

from django.views.decorators.csrf import csrf_exempt
from .utils import process_query, extract_text_from_image, chat,load_pdf_from_directory, get_text_chunks, get_vector_store
import logging
import base64
import json
from django.core.files.storage import default_storage
import multiprocessing
import logging

from io import BytesIO
from django.views.decorators.csrf import ensure_csrf_cookie

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
def debug_semaphores():
    semaphores = multiprocessing.active_children()
    logging.debug(f"Active semaphores: {len(semaphores)}")
@csrf_exempt
def save_message(request):
    if request.method == 'POST':
        logger.debug("Processing save_message request...")
        data = json.loads(request.body)
        email = data.get('email')
        session_id = data.get('session_id')
        sender = data.get('speaker')
        message = data.get('message')
        image_data = data.get('image', None)

        user = get_object_or_404(User, email=email)
        session, created = ChatSession.objects.get_or_create(session_id=session_id, user=user)

        message_instance = Message.objects.create(session=session, sender=sender, message=message)

        if image_data:
            format, imgstr = image_data.split(';base64,')
            ext = format.split('/')[-1]
            message_instance.image.save(f'image.{ext}', ContentFile(base64.b64decode(imgstr)), save=True)

        logger.debug("Message saved successfully.")
        return JsonResponse({'success': True})

    logger.error("Invalid request method for save_message.")
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def load_conversation(request):
    if request.method == 'POST':
        logger.debug("Processing load_conversation request...")
        data = json.loads(request.body)
        email = data.get('email')
        session_id = data.get('session_id')

        user = get_object_or_404(User, email=email)
        session = get_object_or_404(ChatSession, session_id=session_id, user=user)
        messages = Message.objects.filter(session=session).values('sender', 'message', 'image', 'timestamp')

        logger.debug("Conversation loaded successfully.")
        return JsonResponse(list(messages), safe=False)
@csrf_exempt
def handle_uploaded_file(f):
    file_path = default_storage.save('uploads/' + f.name, ContentFile(f.read()))
    return file_path
@csrf_exempt
def create_new_session(request):
    if request.method == 'POST':
        logger.debug("Processing create_new_session request...")
        data = json.loads(request.body)
        email = data.get('email')

        user = get_object_or_404(User, email=email)
        session_id = 'session_' + str(ChatSession.objects.filter(user=user).count() + 1)
        ChatSession.objects.create(user=user, session_id=session_id)

        # Clear the conversation log file when a new session is started
        with open("conversation_log.txt", "w") as file:
            file.write("")

        logger.debug("New session created successfully.")
        return JsonResponse({'session_id': session_id, 'success': True})
@csrf_exempt
def get_sessions(request):
    if request.method == 'POST':
        logger.debug("Processing get_sessions request...")
        data = json.loads(request.body)
        email = data.get('email')

        user = get_object_or_404(User, email=email)
        sessions = ChatSession.objects.filter(user=user).values('session_id', 'created_at')

        logger.debug("Sessions retrieved successfully.")
        return JsonResponse(list(sessions), safe=False)


def load_resources(request):
    if 'processed' not in request.session:
        try:
            logger.debug("Loading resources from PDF...")
            raw_text = load_pdf_from_directory(PDF_PATH) # type: ignore
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                request.session['processed'] = True
            logger.debug("Resources loaded or already present.")
            return JsonResponse({'status': 'Resources loaded or already present'})
        except FileNotFoundError:
            logger.error("PDF file not found.")
            return JsonResponse({'error': 'PDF file not found'}, status=404)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
# @csrf_exempt
# def chatbot_view(request):
#     if request.method == 'POST':
#         logger.debug("Processing chatbot_view POST request...")
#         try:
#             data = json.loads(request.body)
#             logger.debug(f"Received data: {data}")
            
#             user_question = data.get('question', '').strip()
#             # email = data.get('email')
#             session_id = data.get('session_id')
            
#             if not session_id:
#                 logger.error("Session ID is required.")
#                 return JsonResponse({'error': 'Session ID is required'}, status=400)
            
#             image_text = ""
#             image_data = None
#             if 'image' in data:
#                 logger.debug("Extracting text from image...")
#                 format, imgstr = data['image'].split(';base64,')
#                 image_stream = BytesIO(base64.b64decode(imgstr))
#                 image_text = extract_text_from_image(image_stream)
#                 image_stream.seek(0)
#                 image_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')

#             # user, created = User.objects.get_or_create(email=email)
#             # conversation, _ = Conversation.objects.get_or_create(user=user, session_id=session_id)

#             # if conversation.conversation_data is None:
#             #     conversation.conversation_data = {'context': "", 'history': []}

#             # if not isinstance(conversation.conversation_data, dict):
#             #     conversation.conversation_data = {'context': "", 'history': []}

#             new_interaction = {'question': user_question, 'reply': None, 'image': image_data}
#             conversation_data = {
#                 'context': "", 'history': [new_interaction]
#             }

#             reply = process_query(text=user_question, image_text=image_text)
#             conversation_data['history'][-1]['reply'] = reply

#             # conversation.conversation_data = conversation_data
#             # conversation.save()

#             logger.debug("Chatbot response generated and saved.")
#             return JsonResponse({'response': reply, 'history': conversation_data['history']})

#         except json.JSONDecodeError:
#             logger.error("Invalid JSON")
#             return JsonResponse({'error': 'Invalid JSON'}, status=400)
#         except Exception as e:
#             logger.error(f"Unexpected error: {str(e)}")
#             return JsonResponse({'error': str(e)}, status=500)

#     elif request.method == 'GET':
#         logger.debug("Processing chatbot_view GET request...")
#         email = request.GET.get('email')
#         session_id = request.GET.get('session_id')
#         if not email or not session_id:
#             logger.error("Email and session ID are required.")
#             return JsonResponse({'error': 'Email and session ID are required'}, status=400)

#         user = get_object_or_404(User, email=email)
#         conversation = get_object_or_404(Conversation, user=user, session_id=session_id)
#         history = conversation.conversation_data.get('history', [])

#         logger.debug("Conversation history retrieved successfully.")
#         return JsonResponse({'history': history})

#     logger.error("Invalid request method for chatbot_view.")
#     return JsonResponse({'error': 'Invalid request method'}, status=405)
@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        try:
            if request.content_type.startswith('multipart/form-data'):
                data = request.POST
                files = request.FILES  # This will capture any files sent in the request
            else:
                return JsonResponse({'error': 'Unsupported content type'}, status=400)

            user_question = data.get('question', '').strip()  # Ensure stripping whitespace and check for empty string
            session_id = data.get('session_id')
            email = data.get('email')

            if not session_id or not email:
                return JsonResponse({'error': 'Session ID and email are required.'}, status=400)

            image_text = ""
            image_data = None
            if 'image' in files:
                # Process the image if it's provided in the request
                image = files['image']
                image_stream = BytesIO(image.read())  # Convert the image to a BytesIO stream
                image_text = extract_text_from_image(image_stream)  # Extract text from the image
                image_stream.seek(0)
                image_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')

            # Call the chat function with the necessary arguments, including image_text
            reply = chat(user_question, email, session_id, image_text=image_text)  # Pass email, session_id, and image_text

            # Log and save the conversation as before
            session_folder = os.path.join('sessions', email)
            os.makedirs(session_folder, exist_ok=True)
            session_file = os.path.join(session_folder, f'{session_id}.json')

            # Load existing conversation data or start a new one
            if os.path.exists(session_file):
                with open(session_file, 'r') as file:
                    conversation_data = json.load(file)
            else:
                conversation_data = {'context': '', 'history': []}

            # Include image data in the new interaction
            new_interaction = {'question': user_question, 'reply': reply, 'image': image_data}
            conversation_data['history'].append(new_interaction)

            # Save the updated conversation history back to the file
            with open(session_file, 'w') as file:
                json.dump(conversation_data, file)

            return JsonResponse({'response': reply, 'history': conversation_data['history']})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)




@ensure_csrf_cookie
def set_csrf_token(request):
    return JsonResponse({'detail': 'CSRF cookie set'})


@ensure_csrf_cookie
def set_csrf_token(request):
    return JsonResponse({'detail': 'CSRF cookie set'})

@csrf_exempt
def handle_email(request):
    if request.method == 'POST':
        logger.debug("Processing handle_email request...")
        try:
            data = json.loads(request.body.decode('utf-8'))
            email = data.get('email')

            if not email:
                logger.error("Email not provided.")
                return JsonResponse({'success': False, 'error': 'Email not provided'}, status=400)

            user, created = User.objects.get_or_create(email=email)
            if created:
                logger.info(f"New user created with email: {email}")
            else:
                logger.info(f"Existing user found with email: {email}")

            return JsonResponse({'success': True, 'message': 'Email processed successfully'})
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {str(e)}")
            return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    logger.error("Invalid request method for handle_email.")
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)

@ensure_csrf_cookie
def home(request):
    print("Chat view called")
    return render(request, 'index.html')