import streamlit as slit
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

OpenAI_API_KEY = "sk-proj-qmOntj7Ub28-XoVsTJzoWkdQEICcVF1d80GJN1EnnYygeBGafWmSHyhT3BlbkFJqaI7OpvOwkQbvQgk_fbHnyIieS3KOx1qYFuPy69uxg0r6emBHLZ4NHef4bScsSqjrtAr7sm9IA"

#Giving header/name to bot
slit.header("NoteBot")

#Including a simple sidebar to our interface
with slit.sidebar:
    slit.title("My notes")
    file=slit.file_uploader("Upload pdf here", type="pdf")

# Extracting the text from pdf file
if file is not None:
    My_pdf = PdfReader(file)
    text=""
    for page in My_pdf.pages:
        text += page.extract_text()
        # slit.write(text) Just to show text on the interface

    # Break text into small chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks=splitter.split_text(text)
    # slit.write(chunks) This is to print the chunks to show if splitting is done or not

    #creating object of OpenAIEmbeddings class that will allow us to connect with openai embedding models
    embeddings=OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    #creating vectorDB and storing embeddings into it
    vector_store=FAISS.from_texts(chunks,embeddings)

    #get user query
    user_query=slit.text_input("Ask your question")

    # semantic search from vector store/obtaining relevant chunks from vector store
    if user_query:
        relevant_chunks=vector_store.similarity_search(user_query)

        # define the LLM
        llm=ChatOpenAI(
            api_key=OpenAI_API_KEY,
            max_tokens=200,
            temperature=0,
            model="gpt-3.5-turbo"
        )

        #generate response 1st method(deprecated soon)
        # chain=load_qa_chain(llm,chain_type="stuff")
        # output=chain.run(question=user_query, input_document=relevant_chunks)
        # slit.write(output)

        #generate response 2nd method(compulsory to give prompt)
        customized_prompt=ChatPromptTemplate.from_template(
            """ You are my assistant tutor. Answer the question based on the following context and
            if you did not get the context simply say "I don't know" :{context}
            Question :{input}
            """
        )
        chain=create_stuff_documents_chain(llm,customized_prompt)
        output=chain.invoke({"input":user_query, "context":relevant_chunks})
        slit.write(output)