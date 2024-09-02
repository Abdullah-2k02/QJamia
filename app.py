import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state for storing previous searches
if 'previous_searches' not in st.session_state:
    st.session_state.previous_searches = []

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    You are an AI assistant that provides detailed answers based on the context provided. If the answer is not available in the context, reply with "Can you please make your question more clear & check if your question is related to admission or not. Thanks :)".

    Context: {context}
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Print context and question for debugging help to check if system is retrieving the targeted content or not and if it used the right query to search or not
    for doc in docs:
        print("Context:\n", doc.page_content)
    print("Question:\n", user_question)


    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print("Response:", response["output_text"]) #print response on console
    st.write("Reply: ", response["output_text"]) #print response on app

    # Save the search to session state
    st.session_state.previous_searches.append(user_question)

# Main function to run the Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title="QJamia", layout="wide")

    # Create a three-column layout
    col1, col2, col3 = st.columns([1.7, 6, 1.7])

    with col1:
        # Left column for Important Announcements and Previous Searches
        st.markdown("<h1 style='font-size: 20px;'>📢Important Announcements</h1>", unsafe_allow_html=True)
        st.write("""Fee challan submission date for 1st merit list for admission Fall 2024 has been extended till 16th August 2024 on the request of candidates except LLB program.
        Please deposit your fee challan by 16th August 2024.""")

        st.markdown("<h2 style='font-size: 20px;'>🔍Previous Searches</h2>", unsafe_allow_html=True)
        if st.session_state.previous_searches:
            for search in st.session_state.previous_searches:
                st.write(f"- {search}")

    with col2:
        # Main content
        st.header("QJamia")

        # Frequently asked questions
        st.subheader("Frequently Asked Questions:")
        faqs = [
            "what is the last date to apply?",
            "How do I sign up for an account if I don't have one?",
            "How do I download the fee challan?",
            "My transcripts are not in English. Will you accept them?"
        ]

        for question in faqs:
            if st.button(question):
                user_input(question)

        # Input field for user question
        user_question = st.text_input('Search QJamia', placeholder="Feel Free to ask 'Admission Related Queries'")

        # Handle user input
        if user_question:
            user_input(user_question)

    with col3:
        # Right column for Deadlines and Important Links
        st.markdown("<h2 style='font-size: 20px;'>⏳Deadlines</h2>", unsafe_allow_html=True)
        st.write("	Last date for submission of Application Form : July 29")
        st.write("	Last Date to deposit Fee Challan(except LLB) : August 16")

        st.markdown("<h2 style='font-size: 20px;'>🔗Important Links</h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='font-size: 20px;'>Important Links</h2>", unsafe_allow_html=True)
        st.write("[University Website](https://www.iiu.edu.pk/)")
        st.write("[Apply Now](https://cms.iiu.edu.pk/web/signup)")
        st.write("[Contact Us](https://www.iiu.edu.pk/contacts.php)")

if __name__ == "__main__":
    main()
