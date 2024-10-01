
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.schema import Document
import tempfile
from langchain.llms import Ollama 
import os

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize LLM
llm = Ollama(model="llama2")

# Initialize vector store
vector_store = None

def process_file(file):
    # Read file contents
    content = file.read().decode('utf-8')
    
    # Create a Document object
    return Document(page_content=content, metadata={"source": file.name})

def main():
    st.set_page_config(page_title="Document Chatbot")
    st.header("Document Chatbot with Custom Knowledge Base")

    # File uploader for multiple files
    uploaded_files = st.file_uploader("Choose files", type=["txt"], accept_multiple_files=True)

    if uploaded_files:
        all_documents = []
        for file in uploaded_files:
            document = process_file(file)
            all_documents.append(document)
        
        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(all_documents)
        
        # Create the vector store
        global vector_store
        vector_store = FAISS.from_documents(texts, embeddings)
        
        st.success(f"{len(uploaded_files)} file(s) processed successfully!")

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize the conversation chain
    if vector_store is not None:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )

        # Chat interface
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            response = conversation_chain({"question": user_question})
            st.write("Answer:", response['answer'])

if __name__ == "__main__":
    main()