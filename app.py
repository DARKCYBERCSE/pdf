"""
Streamlit ChatPDF app (fixed & more robust)

Save as app.py and run:
    streamlit run app.py

Requirements (example):
    pip install streamlit PyPDF2 langchain langchain-community langchain-groq langchain-google-genai faiss-cpu google-generative-ai fpdf pillow python-dotenv

Note: Some langchain / vectorstore APIs have changed across versions.
This code contains helpful try/except fallback logic for common API differences.
"""

import os
import io
import traceback
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fpdf import FPDF
from PIL import Image
import time

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ---------- Helpers ----------

def extract_pages_from_uploaded_files(uploaded_files):
    """
    Returns list of dicts: {"text": ..., "page": n, "source": filename}
    """
    pages = []
    for uploaded_file in uploaded_files:
        try:
            # st.file_uploader returns a stream-like object; ensure we use BytesIO
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            st.warning(f"Couldn't open {uploaded_file.name} with PdfReader: {e}")
            continue

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                # fallback if extract_text fails on a page
                text = ""
            text = text.strip()
            pages.append({"text": text, "page": i + 1, "source": uploaded_file.name})
    return pages

def chunk_pages(pages, chunk_size=1000, chunk_overlap=200):
    """
    Split every page into chunks and keep metadata (page & source).
    Returns texts[], metadatas[]
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = []
    metadatas = []
    for p in pages:
        if not p["text"]:
            continue
        chunks = splitter.split_text(p["text"])
        for c in chunks:
            texts.append(c)
            metadatas.append({"page": p["page"], "source": p["source"]})
    return texts, metadatas

def save_vector_store(vector_store, path="faiss_index"):
    """
    Try several ways to persist FAISS index depending on versions.
    """
    try:
        vector_store.save_local(path)
        return True
    except Exception:
        try:
            # older versions use persist or persist_directory
            vector_store.persist(path)
            return True
        except Exception:
            try:
                # fallback: use python's save (not recommended but try)
                faiss_index = getattr(vector_store, "index", None)
                if faiss_index is not None:
                    import pickle
                    with open(os.path.join(path, "index.pkl"), "wb") as f:
                        pickle.dump(vector_store, f)
                    return True
            except Exception:
                return False

def load_vector_store(path="faiss_index", embeddings=None):
    """
    Try to load FAISS vectorstore with several common signatures.
    Returns vector_store or raises Exception.
    """
    # common signature
    if embeddings is None:
        raise ValueError("Embeddings must be provided to load FAISS index.")
    try:
        return FAISS.load_local(path, embeddings)
    except TypeError:
        # some versions expect different args
        try:
            return FAISS.load_local(path)
        except Exception as e:
            raise e
    except Exception as e:
        raise e

def call_llm_with_fallback(llm, prompt, max_retries=1):
    """
    Try common LLM call styles and return text answer.
    """

    last_exc = None
    for attempt in range(max_retries):
        try:
            # try __call__
            out = llm(prompt)
            if isinstance(out, str):
                return out
            # langchain LLM may return a dict-like or object
            if hasattr(out, "generations"):
                # LangChain LLMResult interface
                gens = out.generations
                if gens and gens[0] and len(gens[0]) > 0:
                    return gens[0][0].text
            if isinstance(out, dict) and "text" in out:
                return out["text"]
        except Exception as e:
            last_exc = e

        # try .predict
        try:
            if hasattr(llm, "predict"):
                pred = llm.predict(prompt)
                if isinstance(pred, str):
                    return pred
        except Exception as e:
            last_exc = e

        # try .generate (LangChain newer API)
        try:
            if hasattr(llm, "generate"):
                # generate expects list of prompts in many wrappers
                res = llm.generate([prompt])
                # try to extract
                if hasattr(res, "generations"):
                    gens = res.generations
                    if gens and gens[0] and len(gens[0]) > 0:
                        return gens[0][0].text
        except Exception as e:
            last_exc = e

    # If we reach here, all attempts failed
    raise RuntimeError(f"LLM call failed. Last exception:\n{traceback.format_exc()}\nCause: {last_exc}")

def generate_pdf_bytes_from_history(history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)
    for msg in history:
        role = msg.get("type", "human").capitalize()
        avatar = msg.get("avatar", "")
        content = msg.get("content", "")
        # ensure encoding compatibility
        try:
            content = content.encode("latin1", "replace").decode("latin1")
        except Exception:
            pass
        # write with multi_cell
        pdf.multi_cell(0, 8, f"{role} ({avatar}): {content}")
        pdf.ln(1)
    return pdf.output(dest="S").encode("latin1")

# ---------- Core functions ----------

def build_vector_store_from_uploaded_pdfs(pdf_files, embeddings_model="models/embedding-001", index_path="faiss_index"):
    pages = extract_pages_from_uploaded_files(pdf_files)
    if not pages:
        raise ValueError("No pages extracted from uploaded PDFs.")
    texts, metadatas = chunk_pages(pages)
    if not texts:
        raise ValueError("No text chunks created from PDF pages (maybe blank pages).")

    embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
    try:
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    except Exception as e:
        # fallback: try without metadatas if that signature fails
        vector_store = FAISS.from_texts(texts, embeddings)
        # attempt to reattach metadatas to docs if possible (best-effort)
        try:
            for i, md in enumerate(metadatas):
                if i < len(vector_store.docstore._dict):
                    pass
        except Exception:
            pass

    saved = save_vector_store(vector_store, path=index_path)
    if not saved:
        st.warning("Saving FAISS index to disk didn't work using known methods. The index will still be in memory for this session.")
    return vector_store

def retrieve_context_from_store(vector_store, query, k=4):
    """
    Return (joined_context_string, docs_list)
    """
    # Try standard retriever
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
    except Exception:
        # fallback to similarity_search
        try:
            docs = vector_store.similarity_search(query, k=k)
        except Exception:
            try:
                docs_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=k)
                docs = [d for d, _score in docs_with_scores]
            except Exception as e:
                raise RuntimeError(f"Failed to run retrieval on vector_store: {e}")

    # Build context with page references (if available)
    context_pieces = []
    for d in docs:
        # handle either Document objects or plain strings
        text = getattr(d, "page_content", None) or (d if isinstance(d, str) else "")
        meta = getattr(d, "metadata", {}) or {}
        page = meta.get("page") or meta.get("source_page") or meta.get("pageno")
        source = meta.get("source") or meta.get("filename")
        header = []
        if page:
            header.append(f"Page {page}")
        if source:
            header.append(f"Source: {source}")
        header_str = " — ".join(header) if header else ""
        if header_str:
            context_pieces.append(f"{header_str}\n{text}")
        else:
            context_pieces.append(text)
    joined = "\n\n---\n\n".join(context_pieces)
    return joined, docs

def get_answer_from_docs(query, context, model_name, groq_api_key_local):
    """
    Prompt the LLM with the extracted context and the user's question and return the LLM answer.
    """
    prompt_template = """
Extract and present comprehensive insights from the provided context in the PDF, ensuring accuracy and clarity. Your response must address the following:

1. Insight Accuracy: Provide detailed and accurate insights based only on the provided context, refraining from any assumptions.
2. Contextual Relevance: Ensure the response is closely tied to the context, offering any additional or supplementary information that could enhance the user's understanding of the topic.
3. Numerical Data Representation: If the information includes numerical data, always mention the relevant units or currency (e.g., dollars, rupees, percentages, etc.). Specify quantities in their exact form, ensuring no data is overlooked.
4. Specificity in Data: Where applicable, break down data into categories or subcomponents (e.g., financial breakdowns by department, year, or region, etc.), and provide comparisons or trends if present in the context. 
5. Extra Insights: If possible, identify any noteworthy patterns, anomalies, or areas for deeper exploration based on the provided data. Highlight key takeaways or summarizations to help the user grasp the significance of the data.
6. Page References: Always mention the page number(s) from which you extracted the information to allow for easy reference.
7. Unavailable Information: If the requested insight is not available in the provided context, explicitly state, "Answer is not available in the context." Do not speculate or provide vague information.
8. Clarity & Detail: Use clear, concise language to avoid confusion. If there are technical terms, provide brief definitions or explanations if needed to ensure clarity.

<context>
{context}
</context>

Question: {question}
"""

    formatted = prompt_template.format(context=context or "No context available.", question=query)

    # create ChatGroq LLM and call it robustly
    llm = ChatGroq(groq_api_key=groq_api_key_local, model_name=model_name)
    answer = call_llm_with_fallback(llm, formatted, max_retries=1)
    return answer

# ---------- Streamlit App ----------

def main():
    st.set_page_config(page_title="ChatPDF", 
                       page_icon="📄", 
                       layout="wide", 
                       initial_sidebar_state="expanded",
                       menu_items={'About': "This is a compact ChatPDF app — improved error handling & page refs."})

    col1, col2, col3 = st.columns((1, 2, 1))
    with col2:
        # If you have assets/Header.png, Streamlit will show it. If not, ignore.
        try:
            st.image(Image.open("assets/Header.png"))
        except Exception:
            st.markdown("## ChatPDF")

    st.markdown("""##### Suggestions:
▶️ Summarize the document.  ▶️ List keywords and identify key terms.  ▶️ What is the primary goal or objective of this document?""")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "show_confirmation" not in st.session_state:
        st.session_state.show_confirmation = False
    if "reset_confirmed" not in st.session_state:
        st.session_state.reset_confirmed = False

    model_options = {
        "Gemma2-9B": "gemma2-9b-it",
        "Llama3-8b": "llama3-8b-8192",
        "Llama3-70B": "llama3-70b-8192",
        "Llama 3.1 70B": "llama-3.1-70b-versatile",
        "Mixtral-8x7B": "mixtral-8x7b-32768",
    }

    with st.sidebar:
        st.title("Menu")
        selected_model = st.selectbox("Select LLM Model", options=list(model_options.keys()))
        selected_model_name = model_options[selected_model]
        st.write(f"Selected model: **{selected_model_name}**")

        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Submit & Process", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file first.")
            else:
                try:
                    with st.spinner("Processing PDFs and building vector index..."):
                        # Build in-memory vector store (and try to save to disk)
                        vector_store = build_vector_store_from_uploaded_pdfs(pdf_docs, embeddings_model="models/embedding-001", index_path="faiss_index")
                        st.success("Vector index created (and saved if possible). You can now ask questions.")
                except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")
                    st.exception(traceback.format_exc())

        if st.session_state.history:
            try:
                pdf_bytes = generate_pdf_bytes_from_history(st.session_state.history)
                st.download_button("Download Chat History", data=pdf_bytes, file_name="chat_history.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.warning(f"Failed to create chat history PDF: {e}")

        if st.button("Reset Chat", use_container_width=True):
            st.session_state.show_confirmation = True

        if st.session_state.show_confirmation:
            st.error("Are you sure you want to delete the chat history?")
            colA, colB = st.columns(2)
            with colA:
                if st.button("Confirm"):
                    st.session_state.history = []
                    st.session_state.show_confirmation = False
                    st.session_state.reset_confirmed = True
                    st.experimental_rerun()
            with colB:
                if st.button("Close"):
                    st.session_state.show_confirmation = False
                    st.experimental_rerun()

        if st.session_state.reset_confirmed:
            st.success("Chat history has been reset")
            st.session_state.reset_confirmed = False

        st.markdown("<p style=' margin-top: 120px;'>Powered by Groq</p>", unsafe_allow_html=True)

    # Display existing chat history
    for msg in st.session_state.history:
        st.chat_message(msg["type"], avatar=msg["avatar"]).write(msg["content"])

    # Input and answering logic
    user_question = st.chat_input("Ask a question from the processed PDF files")
    if user_question:
        # display human message
        st.chat_message("human", avatar='assets/human.png' if os.path.exists("assets/human.png") else None).write(user_question)
        st.session_state.history.append({"type": "human", "content": user_question, "avatar": 'assets/human.png'})

        # Load vector store (try local saved index first, else error)
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            try:
                vector_store = load_vector_store("faiss_index", embeddings=embeddings)
            except Exception:
                st.info("Local FAISS index not found or failed to load; trying to rebuild from uploaded files (if any).")
                if pdf_docs:
                    vector_store = build_vector_store_from_uploaded_pdfs(pdf_docs, embeddings_model="models/embedding-001", index_path="faiss_index")
                else:
                    raise RuntimeError("No FAISS index and no uploaded PDFs available. Please upload PDFs & click Submit & Process.")

            with st.spinner("Retrieving relevant context from vector store..."):
                context, docs = retrieve_context_from_store(vector_store, user_question, k=4)

            with st.spinner("Querying the language model..."):
                try:
                    answer = get_answer_from_docs(user_question, context, selected_model_name, groq_api_key)
                except Exception as e:
                    st.error("Failed to generate an answer from the LLM. See details below.")
                    st.exception(traceback.format_exc())
                    answer = f"LLM error: {e}"

            # show AI message
            st.chat_message("ai", avatar='assets/ai.png' if os.path.exists("assets/ai.png") else None).write(answer)
            st.session_state.history.append({"type": "ai", "content": answer, "avatar": 'assets/ai.png'})

            # save last question
            st.session_state.last_question = user_question

        except Exception as e:
            st.error(f"Error while answering question: {e}")
            st.exception(traceback.format_exc())

    # Regenerate button
    if st.session_state.get("last_question"):
        if st.button("Regenerate"):
            last_q = st.session_state.last_question
            st.chat_message("human", avatar='assets/human.png' if os.path.exists("assets/human.png") else None).write(last_q)
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = load_vector_store("faiss_index", embeddings=embeddings)
                context, docs = retrieve_context_from_store(vector_store, last_q, k=4)
                answer = get_answer_from_docs(last_q, context, selected_model_name, groq_api_key)
                st.chat_message("ai", avatar='assets/ai.png' if os.path.exists("assets/ai.png") else None).write(answer)
                st.session_state.history.append({"type": "ai", "content": answer, "avatar": 'assets/ai.png'})
            except Exception as e:
                st.error(f"Failed to regenerate: {e}")
                st.exception(traceback.format_exc())


if __name__ == "__main__":
    main()
