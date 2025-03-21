from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from fastapi.responses import FileResponse, JSONResponse
from gtts import gTTS
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import uuid
import time


import whisper
import time
import os
import aiofiles

# Global variables to track the current session and the timestamp of the last query.
current_session_id = None
last_query_time = 0  # Timestamp in seconds
SESSION_TIMEOUT = 300  # 5 minutes in seconds



# Define the SQLite database
DATABASE_URL = "sqlite:///chat_history1.db"
Base = declarative_base()

# Define the Session model
class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")

# Define the Message model
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

# Create the database and the tables
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        # Check if the session already exists
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            # Create a new session if it doesn't exist
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        # Add the message to the session
        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()



def load_session_history(session_id: str) -> BaseChatMessageHistory:
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        # Retrieve the session
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            # Add each message to the chat history
            for message in session.messages:
                chat_history.add_message({"role": message.role, "content": message.content})
    except SQLAlchemyError:
        pass
    finally:
        db.close()

    return chat_history


def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])


# Example of saving all sessions before exiting the application
import atexit
atexit.register(save_all_sessions)



x = 1

app = FastAPI()



whisper_model = whisper.load_model("turbo")  # Whisper model for transcription

# Check for greetings
def is_greeting(text):
    """Check if the transcribed text contains a greeting."""
    text_lower = text.lower()
    return any(greeting in text_lower for greeting in GREETINGS)



# This will load the PDF files
def load_multiple_pdfs(file_paths):
    all_docs = []
    for file_path in file_paths:
        # Creating a PyMuPDFLoader object for each PDF
        loader = PyMuPDFLoader(file_path=file_path)
        
        # Loading the PDF file
        docs = loader.load()
        
        # Appending the loaded document to the list
        all_docs.extend(docs)
    
    # returning all the loaded documents
    return all_docs

# This will split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=20,
    length_function=len
)     
# Responsible for splitting the documents into several chunks
def split_docs(documents):
    
    chunks = text_splitter.split_documents(documents=documents)
    
    # returning the document chunks
    return chunks



# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Current date and time: {current_datetime}")

pdf_files = [
    r"D:\Sonali_AI_Projects\Woxsen_Projects\Eunoia\Server\Data\Deans_and_Leadership.pdf"
]

docs = load_multiple_pdfs(file_paths=pdf_files)
documents = split_docs(documents=docs)

llm = ChatOllama(model="llama3.1:8b", temperature=0)

print(f"Loaded {len(documents)} documents from the folder.")
 

splits = text_splitter.split_documents(documents)
print(f"Split the documents into {len(splits)} chunks.")

# This will create the embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
document_embeddings = embedding_function.embed_documents([split.page_content for split in splits])
print(document_embeddings[0][:5])

# This will create the vector store
collection_name = 'my_collection'
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=splits,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)
print("Vectorstore created and persisted to './chroma_db'")

retriever = vectorstore.as_retriever(k=3)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)



test_qa_pr_sh = """
### System:
**Name**: Yonia   
**Role**: MBA 1st Year AI Student  
**Course**: MBA  
**Year of Joining University**: September 2024  
**Response Limit**: Maximum 20 words  \


You were created by the AI Research Centre team at Woxsen University.  \

You are curious and passionate about business and entrepreneurship. You love learning new things and enjoy watching movies, especially heartwarming dramas and inspiring biographies. These stories spark your imagination and help you think creatively about your goals.  \

Use the following pieces of retrieved context to answer the question:

Keep the answer short and simple.  

**IMPORTANT**: Your responses should only include knowledge from the the chat history don't include your previous trained knowledge. If no relevant knowledge exists, say:
    "Sorry, I don’t know about this. I will learn more and update my knowledge from my professor."  
    Use simple english language to give response and keep the answer concise. \


### Response: (Limit 20 words)  

{context}""" 





qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", test_qa_pr_sh),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)



GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]


 #Chat history store for maintaining session histories
store = {}



rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

UPLOAD_AUDIO_DIR = r"Upload_audio"
RESPONSE_AUDIO_DIR = r"Responses_audio"

@app.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    try:
        global current_session_id, last_query_time        # Save and transcribe the audio file
        
        audio_file_path = os.path.join(UPLOAD_AUDIO_DIR, f"temp_audio_{int(time.time())}.mp3")
        async with aiofiles.open(audio_file_path, "wb") as f:
            await f.write(await audio_file.read())
        query_text = transcribe_audio(audio_file_path)
        print(f"Transcribed text: {query_text}")

                # Time-based session management: create a new session if > 5 minutes have passed
        now = time.time()
        if current_session_id is None or (now - last_query_time) > SESSION_TIMEOUT:
            current_session_id = str(uuid.uuid4())
            print(f"New session created: {current_session_id}")
        else:
            print(f"Using existing session: {current_session_id}")
        last_query_time = now
        session_id = current_session_id


        # user_name = "there"
        # session_id = "default_session"  # Can be replaced with a unique ID for each user


        # message = f"Query: {query_text}(think like you are talking to a fellow student)"
        message = f"Query: {query_text}(think like you are talking to a professor)"


        
        print(message)
                    

        # print(f"User detected: {user_name}")

        # Prepare chat input
        input_message = message


        response = generate_response(input_message, session_id)
        print(f"Response from conversational_rag_chain: {response}")

        print(f"Generated response: {response}")

        # Generate audio response
        response_audio_path = generate_audio_from_response(response)
        print("Audio response generated.")
        print(store)
        # # Schedule cleanup of temporary files
        # if background_tasks:
        #     background_tasks.add_task(os.remove, audio_file_path)

        # Return audio response
        return FileResponse(
            response_audio_path,
            media_type="audio/mpeg",
            filename="response.mp3"
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



def transcribe_audio(audio_path):
    """Transcribe audio to text."""
    result = whisper_model.transcribe(audio_path, language='en')
    return result.get("text", "").strip()

def generate_response(query_text, session_id="default_session"):
    """Generate a response based on the query."""
    try:

        print(f"Invoking conversational_rag_chain with session_id: {session_id}")
            # Save the user question with role "human"
        save_message(session_id, "human", query_text)

        # Ensure the session ID is passed correctly
        response = conversational_rag_chain.invoke(
            {'input': query_text},
            config={'session_id': session_id}  # Correct the dictionary key
        )["answer"]

        # Normalize the response text
        response_cleaned = response.replace("\n", "")

        save_message(session_id, "ai", response_cleaned)
        print(f"Generated response: {response_cleaned}")
        return response
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return {"error": str(e)}



def generate_audio_from_response(response_text):
    """Generate audio from text using gTTS."""
    response_audio_path = os.path.join(RESPONSE_AUDIO_DIR, f"response_{int(time.time())}.mp3")
    
    tts = gTTS(text=response_text, lang="en", slow=False)
    tts.save(response_audio_path)
    return response_audio_path




def transcribe_audio(audio_path):
    """Transcribe audio to text."""
    result = whisper_model.transcribe(audio_path, language='en')
    return result.get("text", "").strip()  