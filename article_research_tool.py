# allows you to find information from articles provided as URLs. Runs on streamlit

# pip install selenium unstructured faiss-cpu sentence-transformers google-generativeai langchain-google-genai streamlit

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

import os
import pickle

file_path = "vector_index.pkl"


def main():
    st.set_page_config(page_title="Article research tool", page_icon="🔎")
    st.title("Article research tool")

    google_api_key = st.sidebar.text_input("Input your Google generative AI API key")

    st.sidebar.title("Paste article URLs")

    urls = []
    for i in range(4):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    # empty UI element which shows the progress between stages and later the question input bar
    placeholder_element = st.empty()

    process_button_clicked = st.sidebar.button("Process URLs")
    if process_button_clicked:
        # load data
        loader = SeleniumURLLoader(urls=urls)
        placeholder_element.text("Loading data...")
        data = loader.load()

        # split data to chunks
        splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                ".",
                " ",
            ],  # splitting along a list of separators by priority; if the chunks are too small they are merged
            chunk_size=1000,
            chunk_overlap=100,
        )
        placeholder_element.text("Splitting data to chunks...")
        chunks = splitter.split_documents(data)

        # create embeddings; save to FAISS index
        embeddings = HuggingFaceEmbeddings()
        placeholder_element.text("Creating vector embeddings...")
        vector_index = FAISS.from_documents(chunks, embeddings)

        # save FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vector_index, f)

    query = placeholder_element.text_input("Question: ")
    if google_api_key and query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vector_index = pickle.load(f)

        llm = load_llm(google_api_key)
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, retriever=vector_index.as_retriever()
        )
        output = chain({"question": query}, return_only_outputs=True)
        st.header("Answer:")
        st.write(output["answer"])

        sources = output.get("sources", "")
        if sources:
            st.subheader("Sources: ")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)


def load_llm(google_api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=google_api_key,
        temperature=0.7,
    )
    return llm


if __name__ == "__main__":
    main()
