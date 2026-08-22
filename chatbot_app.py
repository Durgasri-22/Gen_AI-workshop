import os
import streamlit as st
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader


# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    temperature=0,
    model="openai/gpt-oss-120b",
    api_key=api_key
)


# ChromaDB
client = chromadb.Client()

collection = client.get_or_create_collection(
    "career_knowledge_base"
)


# PDF ingestion
def ingest_pdf(file):
    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text

    return text


# Streamlit UI
st.title("Career Guidance Chatbot")

st.markdown(
    "Get personalized, grounded career guidance "
    "based on your resume and job description."
)


uploaded_files = st.file_uploader(
    "Upload Career Resources (PDFs)",
    type=["pdf"],
    accept_multiple_files=True
)


# Process PDFs
if uploaded_files:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    for file in uploaded_files:

        text = ingest_pdf(file)

        chunks = splitter.split_text(text)

        collection.add(
            documents=chunks,
            ids=[
                f"{file.name}_{i}"
                for i in range(len(chunks))
            ]
        )

    st.success(
        f"{len(uploaded_files)} PDFs ingested successfully!"
    )


# User question
user_query = st.text_input(
    "Ask a career-related question:"
)


if st.button("Get Advice") and user_query:

    # Vector search
    vector_results = collection.query(
        query_texts=[user_query],
        n_results=5,
    )

    vector_docs = vector_results["documents"][0]

    # Keyword search
    keywords = user_query.lower().split()

    keyword_docs = [
        doc
        for doc in vector_docs
        if any(k in doc.lower() for k in keywords)
    ]

    # Hybrid results
    hybrid_docs = list(
        set(vector_docs + keyword_docs)
    )


    # Reranking prompt
    rerank_prompt = PromptTemplate.from_template(
        """
        User Query:
        {query}

        Documents:
        {docs}

        Rank the documents from most relevant
        to least relevant for providing career advice,
        including skills and companies.

        Return the ranked list.
        """
    )


    rerank_chain = rerank_prompt | llm

    reranked_output = rerank_chain.invoke({
        "query": user_query,
        "docs": hybrid_docs
    })


    # Select top results
    top_context = (
        reranked_output.content
        .split("\n")[:3]
    )


    # Final career advice prompt
    final_prompt = PromptTemplate.from_template(
        """
        You are a career guidance AI assistant.

        Based on the following resources:

        {context}

        Provide a personalized roadmap for the user:

        - Skills to learn
        - Recommended companies
        - Steps to improve career readiness

        User Query:
        {query}
        """
    )


    rag_chain = final_prompt | llm


    career_advice = rag_chain.invoke({
        "context": "\n".join(top_context),
        "query": user_query
    })


    # Display results
    st.subheader("Top Retrieved Context")

    for doc in top_context:
        st.write("- ", doc)


    st.subheader("Personalized Career Advice")

    st.write(career_advice.content)