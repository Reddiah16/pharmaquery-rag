import os
import streamlit as st
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def get_embedding_model(api_key):
    """Initialize embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=api_key
    )


@st.cache_resource
def get_db(api_key):
    """Initialize Chroma database."""
    embedding_model = get_embedding_model(api_key)

    return Chroma(
        collection_name="pharma_database",
        embedding_function=embedding_model,
        persist_directory="./pharma_db"
    )


@st.cache_resource
def get_chat_model(api_key):
    """Initialize Gemini model."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )


def add_to_db(uploaded_files):
    """Process uploaded PDFs and add chunks to database."""

    if not uploaded_files:
        st.error("No files uploaded!")
        return

    db = get_db(st.session_state.get("gemini_api_key"))

    for uploaded_file in uploaded_files:

        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        doc_metadata = [doc.metadata for doc in data]
        doc_content = [doc.page_content for doc in data]

        text_splitter = RecursiveCharacterTextSplitter(
             chunk_size=3000,
             chunk_overlap=300
        )

        chunks = text_splitter.create_documents(
            doc_content,
            doc_metadata
        )
        st.write("Number of chunks created:", len(chunks))

        ids = [
            f"{uploaded_file.name}_{i}"
            for i in range(len(chunks))
        ]

        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            current_chunks = chunks[i:i + batch_size] 
            current_ids = ids[i:i + batch_size] 
            success = False
            while not success:
                try:
                    db.add_documents(
                        documents=current_chunks,
                        ids=current_ids
                    ) 
                    success = True 
                except Exception as e:
                    st.warning(
                         f"Temporary error. Retrying...\n{e}" 
                    ) 
            time.sleep(10) 
        time.sleep(3)
        

        os.remove(temp_file_path)


def run_rag_chain(query):
    """Run RAG chain and return answer."""

    db = get_db(st.session_state.get("gemini_api_key"))

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

    PROMPT_TEMPLATE = """
You are a highly knowledgeable assistant specializing in pharmaceutical sciences.

Answer ONLY using the provided context.

If the answer is unavailable in the context, say:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    chat_model = get_chat_model(
        st.session_state.get("gemini_api_key")
    )

    output_parser = StrOutputParser()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | chat_model
        | output_parser
    )

    docs = retriever.invoke(query)

    response = rag_chain.invoke(query)

    return response, docs


def main():
    """Main Streamlit Application."""

    st.set_page_config(
        page_title="PharmaQuery",
        page_icon="🧪"
    )

    st.header("Pharmaceutical Insight Retrieval System")

    query = st.text_area(
        "💡 Enter your query about the Pharmaceutical Industry:",
        placeholder="e.g., What are the AI applications in drug discovery?"
    )

    if st.button("Submit"):

        if "gemini_api_key" not in st.session_state:
            st.warning("Please enter your Gemini API key.")

        elif not query:
            st.warning("Please ask a question.")

        else:

            with st.spinner("Thinking..."):

                result, docs = run_rag_chain(query)

                st.subheader("Answer")
                st.write(result)

                with st.expander("Source Documents"):

                    for doc in docs:

                        st.write(
                            f"Source: {doc.metadata.get('source', 'Unknown')}"
                        )

                        st.write(
                            f"Page: {doc.metadata.get('page', 'N/A')}"
                        )

                        st.markdown("---")

    with st.sidebar:

        st.title("API Keys")

        gemini_api_key = st.text_input(
            "Enter your Gemini API key:",
            type="password"
        )

        if st.button("Enter"):

            if gemini_api_key:

                st.session_state.gemini_api_key = gemini_api_key

                st.success("API key saved!")

            else:

                st.warning(
                    "Please enter your Gemini API key."
                )

    with st.sidebar:

        st.markdown("---")

        pdf_docs = st.file_uploader(
            "Upload research PDFs (Optional)",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):

            if not pdf_docs:

                st.warning("Please upload files.")

            elif "gemini_api_key" not in st.session_state:

                st.warning("Please enter Gemini API key first.")

            else:

                with st.spinner(
                    "Processing documents..."
                ):

                    add_to_db(pdf_docs)

                    st.success(
                        "Documents successfully added!"
                    )

    st.sidebar.markdown("---")
    st.sidebar.write(
        "Built with ❤️ using Streamlit + LangChain + Gemini + Chroma"
    )


if __name__ == "__main__":
    main()
