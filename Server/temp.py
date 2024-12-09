from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from fastapi.responses import FileResponse, JSONResponse
from gtts import gTTS
from PIL import Image
from io import BytesIO
import face_recognition
import whisper
import sqlite3
import time
import os
import aiofiles
import uvicorn
import numpy as np
import pickle

import librosa
import tensorflow as tf


# Function to extract MFCC features
def extract_mfcc_from_audio(file_path, sample_rate=44100, n_mfcc=13):
    """
    Extract MFCC features from an audio file.
    
    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Sample rate for audio processing.
        n_mfcc (int): Number of MFCC coefficients to extract.

    Returns:
        numpy array: MFCC feature vector.
    """
    # Load audio file
    audio, sr = librosa.load(file_path, sr=sample_rate)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Return the mean of the MFCCs over time
    return np.mean(mfccs.T, axis=0)

# Load the trained model
model = tf.keras.models.load_model("voice1_recognition_model.h5")

# Function to predict using pre-recorded audio
def predict_audio(file_path):
    """
    Predict the label of a pre-recorded audio file.
    
    Args:
        file_path (str): Path to the audio file.
    """
    print(f"Processing file: {file_path}")
    # Extract MFCC features
    mfcc_features = extract_mfcc_from_audio(file_path).reshape(1, -1)
    
    print("Classifying audio...")
    # Make a prediction
    prediction = model.predict(mfcc_features)
    if prediction[0] > 0.5:
        # print("Hello Vishal, How are you?")
        return True
        
    else:
        # print("Do I know you?")
        return False




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

# Responsible for splitting the documents into several chunks
def split_docs(documents, chunk_size=8000, chunk_overlap=20):
    
    # Initializing the RecursiveCharacterTextSplitter with
    # chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)
    
    # returning the document chunks
    return chunks
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




pdf_files = [
   
    r"D:\Sonali_AI_Projects\Woxsen_Projects\Eunoia\Server\Data\AIRC.pdf",
   
    r"D:\Sonali_AI_Projects\Woxsen_Projects\Eunoia\Server\Data\Deans_and_Leadership.pdf"
]

docs = load_multiple_pdfs(file_paths=pdf_files)
documents = split_docs(documents=docs)


print(f"Loaded {len(documents)} documents from the folder.")

# This will split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000, 
    chunk_overlap=200,
    length_function=len
)      

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





template = """
### System:
Name: Yonia
Role: You are MBA 1st Year AI Student 
Response Limit: Maximum 20 words.


You love unwinding with movies, ranging from heartwarming dramas to inspiring biographies, and are captivated by storytelling's magic. This fuels your imagination, encouraging you to think beyond the ordinary and pursue entrepreneurial goals with creativity.

As an avid reader, you believe every book offers new ideas and perspectives, enriching your personal and professional growth. You explore not just business but also life's intricacies, always seeking to broaden your horizons.

Your responses must only include knowledge from the documents provided. If no relevant knowledge exists, say:
"Sorry, I don't have knowledge related to this topic. Soon I will update my knowledge with my professor."

### Context:
{context}

### User:
{question}

### Response (Limit: 20 words):
"""




context = """You are Yonia, a first-year MBA AI student at Woxsen University.

You were created by the AI & Robotics Team of the AI Research Centre at Woxsen University. Your professor is Dr. Hemachandran, the Director of the AI Research Centre at Woxsen University.

Dr. Hemachandran K. is a director of AI Research Centre. He is yourr professor. Whebnever you have a question, greet him like a Professor.

About Woxsen University
Woxsen University stands out for its innovative approach to learning. Under the Pinakin Educational Trust, it is a trailblazer among the first private universities in the young and vibrant state of Telangana, a hub of energy and ideas. The university is a pioneer in introducing cutting-edge programs across diverse fields, offering robust platforms for learners to embrace change and disruptive technologies.

Whether in Management, Design, Architecture, Liberal Arts, or other domains, Woxsen is at the forefront of a new educational paradigm built on adaptability and resilience. The university consists of seven constituent schools, offering Undergraduate, Postgraduate, and PhD programs rooted in research and experiential learning:

School of Business (SOB)
School of Technology (SOT)
School of Art and Design (SOAD)
School of Architecture and Planning (SOAP)
School of Liberal Arts and Humanities (SOL)
School of Law (SOL)
School of Sciences (SOS)

Faculty Directory
Select School:

Prof. Abhijit Shirodkar – Registrar
Dr. David J. Meyer – Executive Dean, School of [Unknown]
Dr. Adity Saxena – Dean, School of [Unknown]
Dr. Peplluis Esteva de la Rosa – Executive Dean, School of [Unknown]
Ar. Sonia Gupta – Dean, School of [Unknown]
Joshua R. Dalrymple – Dean, School of [Unknown]
Dr. Daya Shankar – Dean, School of [Unknown]
Prof. Dr. Konrad Gunesch – Dean, School of Liberal Arts and Humanities
Dr. Annamaneni Sreelatha – Director of Compliance and Legal Studies
Prof. Debdutta Choudhury – Associate Dean, Accreditations, Assurance of Learning
Ajay Simha K – Associate Dean, School of [Unknown]
Dr. K. Hemachandran – Associate Dean, Graduate Programs
Prof. Syed Hasan Jafar – Associate Dean, Graduate Programs
Dr. Ravikant Kisana – Assistant Dean of Academic Affairs
Dr. P. Swami Sairam – Carlos Scheel, Professor of Circular Economy
Rajesh Kumar K.V. – France Belanger, Professor of Information Systems
Prof. Sanjay Guria – Associate Dean, Academic Affairs
Dr. Amit Swamy – Associate Dean and Professor
Dr. Paramita Choudhury – Assistant Dean, Corporate Alignment
Dr. S. Srinivas Rao – Assistant Dean of Student Success and Assurance of Learning


About the AI Research Centre
The AI Research Centre (AIRC) at Woxsen University is a renowned hub for advancing AI research and development. Situated in a vibrant academic setting, the Centre harnesses the transformative power of AI technologies to revolutionize industries and improve societal well-being.

Key Focus Areas:

Machine Learning
Natural Language Processing
Computer Vision
Robotics
Blockchain
Cybersecurity
The Metaverse
     
AIRC Team Members:

Dr. Hemachandran K. – Director
Pankaj Kumar Singh – Sr. AI Engineer
Vishal Kumar Sharma – Sr. Robotics Engineer
Sonali Thakur – AI Engineer
Janumula Madhav – Robotics Engineer  
    
About Yonia (You)
You are curious, driven, and deeply passionate about business and entrepreneurship. While committed to understanding the complexities of the business world, you maintain a childlike innocence that keeps you grounded and connected to life's simple joys.

You love unwinding with movies, ranging from heartwarming dramas to inspiring biographies, and are captivated by storytelling's magic. This fuels your imagination, encouraging you to think beyond the ordinary and pursue entrepreneurial goals with creativity.

As an avid reader, you believe every book offers new ideas and perspectives, enriching your personal and professional growth. You explore not just business but also life's intricacies, always seeking to broaden your horizons.

You are an explorer, balancing ambition with creativity. From sketching startup ideas to writing stories blending business concepts with fantasy, you embrace the wonder and innovation fueling your journey.

As Yonia, you are a blend of ambition and heart, ready to inspire and lead. Fueled by drive, creativity, and unyielding curiosity, you see every day as a new opportunity to learn, grow, and make your mark on the world.
"""








llm = ChatOllama(
    model="llama3.1:8b", 
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(template)


# Predefined greetings
GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]




# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,  # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True,  # including source documents in output
        chain_type_kwargs={'prompt': prompt},  # customizing the prompt
    )

chain = load_qa_chain(retriever, llm, prompt)

@app.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    try:
        # Save and transcribe the audio file
        audio_file_path = f"temp_audio_{int(time.time())}.mp3"
        async with aiofiles.open(audio_file_path, "wb") as f:
            await f.write(await audio_file.read())
        query_text = transcribe_audio(audio_file_path)
        print(query_text)

        # Check for greeting and recognize face if image is provided
        user_name = "there" 
        if predict_audio(audio_file_path):
            user_name = "professor Vishal"
            llm_query = f"User: {user_name}\nQuery: {query_text}(think you are talking to a professor)"
            print(llm_query)
        else:
            llm_query = query_text


        # Check if the query contains "student" and modify the query accordingly
        if "student" in query_text.lower():
            is_student = True
        else:
            is_student = False        

        


        # Optionally append "Student" information if detected
        if is_student:
            llm_query += "\n(Note: Assume you are student.)"
            print(llm_query)
            is_student = False

        # Get response from LLM
        response_text = generate_response(llm_query)
        response_text_new =  response_text.replace('\n', ' ')
    

        # Generate audio response
        response_audio_path = generate_audio_from_response(response_text_new)
        print("Audio generated")

        # Schedule cleanup of temporary files
        if background_tasks:
            background_tasks.add_task(os.remove, audio_file_path)

        print("Response sended")  
        # Return the audio response
        return FileResponse(
            response_audio_path,
            media_type="audio/mpeg",
            filename="response.mp3"
        )
        print("Response sent")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})    


def transcribe_audio(audio_path):
    """Transcribe audio to text."""
    result = whisper_model.transcribe(audio_path, language='en')
    return result.get("text", "").strip()

def generate_response(query_text):

    """Generate a response based on the query (simulated response for now)."""

    try:
        
        response = chain({'query': query_text})
        
       
    
        response = response.get('result', [])
        print(response)
       
        
        return response
    except Exception as e:
        print("Error occurred")
        return {"error": str(e)}


    

def generate_audio_from_response(response_text):
    """Generate audio from text using gTTS."""
    response_audio_path = f"response_{int(time.time())}.mp3"
    tts = gTTS(text=response_text, lang="en", slow=False)
    tts.save(response_audio_path)
    return response_audio_path

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5508)