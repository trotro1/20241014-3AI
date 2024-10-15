import fitz  # PyMuPDF
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

###### dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


def init_page() -> None:
    st.set_page_config(
        page_title="Upload PDF(s)",
        page_icon="\ud83d\udcc4"
    )
    st.sidebar.title("Options")


def init_messages() -> None:
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    clear_button = st.sidebar.button("Clear DB", key="clear")
    if clear_button and st.session_state.vectorstore is not None:
        del st.session_state.vectorstore
        st.success("Vector store cleared successfully.")


def get_pdf_text() -> list[str] | None:
    pdf_file = st.file_uploader(
        label='Upload your PDF \ud83d\ude07',
        type='pdf'  # PDFファイルのみアップロード可
    )
    if pdf_file:
        pdf_text = []
        try:
            with st.spinner("Loading PDF ..."):
                pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                for page_num, page in enumerate(pdf_doc, 1):
                    pdf_text.append(page.get_text())
                    st.progress(page_num / pdf_doc.page_count)
        except Exception as e:
            st.error(f"An error occurred while reading the PDF: {e}")
            return None

        # RecursiveCharacterTextSplitter でチャンクに分割する
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0
        )
        return text_splitter.split_text(" ".join(pdf_text))
    else:
        return None


def build_vector_store(pdf_text: list[str]) -> None:
    with st.spinner("Saving to vector store ..."):
        try:
            if st.session_state.vectorstore is not None:
                st.session_state.vectorstore.add_texts(pdf_text)
            else:
                st.session_state.vectorstore = FAISS.from_texts(
                    pdf_text,
                    OpenAIEmbeddings(model="text-embedding-ada-002")
                )
                st.success("Vector store initialized and documents added successfully.")
        except Exception as e:
            st.error(f"An error occurred while building the vector store: {e}")


def page_pdf_upload_and_build_vector_db() -> None:
    st.title("PDF Upload \ud83d\udcc4")
    pdf_text = get_pdf_text()
    if pdf_text:
        build_vector_store(pdf_text)


def main() -> None:
    init_page()
    init_messages()
    page_pdf_upload_and_build_vector_db()


if __name__ == '__main__':
    main()