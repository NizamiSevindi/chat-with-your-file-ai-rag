import os
import torch
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template
from langchain.chains import ConversationalRetrievalChain

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.classes.__path__ = []


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    vectorstore = FAISS.from_texts(embedding=embeddings, texts=text_chunks)
    return vectorstore


def get_conversation_chain(vectorstore):
    endpoint = os.environ["ENDPOINT_URL"]   
    subscription_key = os.environ["AZURE_OPENAI_API_KEY"] 

    llm = AzureChatOpenAI(
        model="gpt-4o",
        api_key=subscription_key,  
        api_version="2024-05-01-preview",
        azure_endpoint=endpoint
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):

    response = st.session_state.conversation.invoke({'question': user_question})

    st.session_state.chat_history = response['chat_history']

    history_pairs = list(zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]))
    history_pairs.reverse()  
    
    for user_msg, bot_msg in history_pairs:
        st.write(user_template.replace("{{MSG}}", user_msg.content), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", bot_msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
 

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == "__main__":
    main()