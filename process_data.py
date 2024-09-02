from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store and save it
def create_vector_store(pdf_docs, output_dir="faiss_index"):
    # Extract and split text
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the vector store locally
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vector_store.save_local(output_dir)
    print(f"Vector store saved to {output_dir}")

if __name__ == "__main__":
    # Read PDFs from the data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pdf_docs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    # Create and save the vector store
    create_vector_store(pdf_docs)
