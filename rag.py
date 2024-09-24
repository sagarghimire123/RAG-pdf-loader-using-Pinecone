import streamlit as st
import os
import PyPDF2
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key and Pinecone API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# Function to extract text from PDF
def pdf_to_text(pdf):
    pdf_reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Streamlit app
def main():
    st.title("PDF Question Answering with LangChain")

    # Upload PDF
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_pdf is not None:
        st.write("Processing your PDF...")
       
        # Convert PDF to text
        pdf_text = pdf_to_text(uploaded_pdf)
       
        # Save extracted text to a temporary file
        temp_file_path = "extracted_text.txt"
        # Use UTF-8 encoding to avoid UnicodeEncodeError
        with open(temp_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(pdf_text)
       
        # Load the extracted text
        loader = TextLoader(temp_file_path)
        text_documents = loader.load()
 
        # Split the text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, length_function=len)
        documents = text_splitter.split_documents(text_documents)
 
        # Set up the OpenAI model
        model = ChatOpenAI()
 
        # Set up embeddings and vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)
 
        # Set up the prompt template
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
 
        # Set up the chain
        chain = (
            {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | model
        )
 
        # Input question
        user_question = st.text_input("Ask a question about the PDF:")
        if st.button("Get Answer"):
            if user_question:
                response = chain.invoke(user_question)
                st.write("Answer:", response.content)
            else:
                st.write("Please ask a question.")
 
if __name__ == "__main__":
    main()
