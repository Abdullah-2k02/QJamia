import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import matplotlib.pyplot as plt
import faiss
from sklearn.decomposition import PCA

# Create a .env file
with open('.env', 'w') as f:
    f.write('GOOGLE_API_KEY=AIzaSyDz4pErwQTpaLisRyGT8sg_J6Nw2q5bNKQ')

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
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

    # Visualize embeddings
    visualize_embeddings(output_dir)

# Function to visualize embeddings
def visualize_embeddings(vector_store_dir):
    # Load the FAISS index
    index_file = os.path.join(vector_store_dir, 'index.faiss')
    index = faiss.read_index(index_file)

    # Get the embeddings
    vectors = index.reconstruct_n(0, index.ntotal)  # Retrieve all vectors

    # Reduce dimensionality if needed (e.g., using PCA)
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Plot the embeddings
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], s=10)
    plt.title("Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

if __name__ == "__main__":
    # Specify the uploaded PDF file
    pdf_docs = ["QJamia.pdf"]  # Use the correct file path

    # Create and save the vector store
    create_vector_store(pdf_docs)
