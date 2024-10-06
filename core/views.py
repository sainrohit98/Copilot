from django.shortcuts import render
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.schema import AIMessage, HumanMessage
import base64

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings():
    return HuggingFaceInstructEmbeddings(model_name="allenai/longformer-base-4096")

def get_vectorstore(text_chunks):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 2048},
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_pdf_upload(request):
    if 'pdf_files' in request.FILES:
        pdf_files = request.FILES.getlist('pdf_files')
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        
        # Serialize and encode the vectorstore
        vectorstore_bytes = vectorstore.serialize_to_bytes()
        vectorstore_base64 = base64.b64encode(vectorstore_bytes).decode('utf-8')
        
        # Store the base64 encoded vectorstore in the session
        request.session['vectorstore'] = vectorstore_base64
        
        return render(request, 'core/upload.html', {'message': 'File uploaded successfully'})
    
    return render(request, 'core/upload.html')


def handle_user_question(request):
    user_question = request.POST.get('user_question')
    chat_history = request.POST.getlist('chat_history')
    vectorstore_base64 = request.session.get('vectorstore')
    
    # Convert chat_history to a list of dictionaries
    chat_history = [{'type': msg.split(':', 1)[0], 'content': msg.split(':', 1)[1]} for msg in chat_history if ':' in msg]
    
    if not user_question and not request.POST:
        # Initial page load or no user input
        return render(request, 'core/ask_question.html', {'chat_history': []})
    
    if user_question and vectorstore_base64:
        # Decode and deserialize the vectorstore
        vectorstore_bytes = base64.b64decode(vectorstore_base64)
        embeddings = get_embeddings()
        vectorstore = FAISS.deserialize_from_bytes(vectorstore_bytes, embeddings, allow_dangerous_deserialization=True)
        
        # Create a new conversation chain for each question
        conversation_chain = get_conversation_chain(vectorstore)
        
        response = conversation_chain({"question": user_question})
        ai_message = next((message.content for message in response['chat_history'] if isinstance(message, AIMessage)), None)

        if ai_message:
            chat_history.append({'type': 'bot', 'content': ai_message})
        return render(request, 'core/ask_question.html', {'chat_history': chat_history})
    
    return render(request, 'core/ask_question.html', {'error': 'Please upload PDFs before asking questions'})

