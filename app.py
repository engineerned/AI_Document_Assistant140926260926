import io
import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Document Assistant")
st.caption("PDF • DOCX • TXT • MD • Local Upload • Google Drive")


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append({
                "filename": filename,
                "page": page_number,
                "text": text.strip(),
            })

    return documents


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text.strip())

    text = "\n".join(paragraphs)

    if not text.strip():
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text,
    }]


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text.strip(),
    }]


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text.strip(),
    }]


def extract_document(file_bytes, filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# -----------------------------
# Chunking
# -----------------------------
def chunk_documents(documents, chunk_size=800, overlap=120):
    chunks = []

    for document in documents:
        words = document["text"].split()

        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "filename": document["filename"],
                "page": document["page"],
                "text": chunk_text,
            })

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def chunks_hash(chunks):
    raw = "\n".join(
        f"{c['filename']}|{c['page']}|{c['text']}"
        for c in chunks
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_vector_store(chunks):
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "what", "which", "who", "how",
    "why", "when", "where", "does", "do", "did", "can", "could",
    "would", "should", "from", "with", "about", "this", "that",
    "these", "those", "it", "its", "be", "as", "by", "at",
}


def important_words(text):
    words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]{2,}\b", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def keyword_score(question, chunk_text):
    question_words = set(important_words(question))
    chunk_words = set(important_words(chunk_text))

    if not question_words:
        return 0.0

    matches = question_words.intersection(chunk_words)
    return len(matches) / len(question_words)


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, chunks, index, top_k=5):
    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # Retrieve a larger candidate pool before hybrid ranking
    candidate_k = min(max(top_k * 4, 10), len(chunks))
    semantic_scores, semantic_ids = index.search(
        question_embedding,
        candidate_k,
    )

    candidates = []

    for score, idx in zip(semantic_scores[0], semantic_ids[0]):
        if idx < 0:
            continue

        chunk = chunks[int(idx)]
        kw_score = keyword_score(question, chunk["text"])

        # Semantic similarity gets more weight, keyword matching adds precision.
        combined_score = (0.75 * float(score)) + (0.25 * kw_score)

        candidates.append({
            "filename": chunk["filename"],
            "page": chunk["page"],
            "text": chunk["text"],
            "semantic_score": float(score),
            "keyword_score": float(kw_score),
            "score": combined_score,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# -----------------------------
# Groq answer
# -----------------------------
def answer_question(question, retrieved_chunks, api_key, model_name):
    if not retrieved_chunks:
        return "I could not find relevant information in the loaded documents."

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        page_text = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"[Source {i}]\n"
            f"Filename: {chunk['filename']}\n"
            f"{page_text}\n"
            f"Text:\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an AI document assistant.

Answer the user's question using ONLY the document context provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not contained in the context, clearly say:
   "The information is not available in the provided documents."
4. Keep the answer clear and directly related to the question.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    client = get_groq_client(api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Answer only from the supplied document context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


# -----------------------------
# Google Drive loading
# -----------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def download_drive_source(url):
    temp_dir = tempfile.mkdtemp(prefix="drive_docs_")

    # gdown can download public/shared Google Drive files and folders.
    if "drive.google.com" not in url and "docs.google.com" not in url:
        raise ValueError("Please provide a Google Drive or Google Docs link.")

    downloaded = gdown.download_folder(
        url,
        output=temp_dir,
        quiet=True,
        use_cookies=False,
    )

    if downloaded is None:
        # Try the URL as a single public file.
        file_path = os.path.join(temp_dir, "drive_file")
        result = gdown.download(
            url,
            output=file_path,
            quiet=True,
            fuzzy=True,
        )

        if result:
            downloaded = [result]

    files = []

    if isinstance(downloaded, list):
        files = [Path(item) for item in downloaded if item]
    elif downloaded:
        files = [Path(downloaded)]

    # Folder downloads can return files recursively.
    if not files:
        files = [
            path
            for path in Path(temp_dir).rglob("*")
            if path.is_file()
        ]

    supported_files = [
        path for path in files
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return supported_files


def process_file_paths(file_paths):
    all_documents = []
    loaded_names = []

    for path in file_paths:
        try:
            file_bytes = path.read_bytes()
            extracted = extract_document(file_bytes, path.name)

            if extracted:
                all_documents.extend(extracted)
                loaded_names.append(path.name)

        except Exception as exc:
            st.warning(f"Could not read {path.name}: {exc}")

    return all_documents, loaded_names


# -----------------------------
# Session state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "loaded_names" not in st.session_state:
    st.session_state.loaded_names = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    chunk_size = st.slider(
        "Chunk size (words)",
        min_value=300,
        max_value=1500,
        value=800,
        step=100,
    )

    overlap = st.slider(
        "Chunk overlap (words)",
        min_value=0,
        max_value=300,
        value=120,
        step=20,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=5,
    )

    response_size = st.selectbox(
        "Response size",
        ["Short", "Medium", "Detailed"],
        index=1,
    )

    groq_model = st.selectbox(
        "Groq model",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ],
        index=0,
    )

    st.divider()
    st.info(
        "Google Drive links must be accessible without private account "
        "authentication. Supported types: PDF, DOCX, TXT and MD."
    )


# -----------------------------
# Document sources
# -----------------------------
st.subheader("1. Add documents")

local_files = st.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

drive_url = st.text_input(
    "Google Drive file or folder link",
    placeholder="Paste a public/shared Google Drive link",
)

col1, col2 = st.columns(2)

with col1:
    load_documents = st.button(
        "📥 Load & Process Documents",
        type="primary",
        use_container_width=True,
    )

with col2:
    clear_documents = st.button(
        "🗑️ Clear Documents",
        use_container_width=True,
    )


if clear_documents:
    st.session_state.chunks = []
    st.session_state.faiss_index = None
    st.session_state.embeddings = None
    st.session_state.processed_hash = None
    st.session_state.documents = []
    st.session_state.loaded_names = []
    st.rerun()


# -----------------------------
# Process documents once
# -----------------------------
if load_documents:
    all_documents = []
    all_names = []

    if local_files:
        for uploaded_file in local_files:
            try:
                file_bytes = uploaded_file.getvalue()
                extracted = extract_document(
                    file_bytes,
                    uploaded_file.name,
                )

                if extracted:
                    all_documents.extend(extracted)
                    all_names.append(uploaded_file.name)

            except Exception as exc:
                st.error(
                    f"Could not process {uploaded_file.name}: {exc}"
                )

    if drive_url.strip():
        with st.spinner("Loading files from Google Drive..."):
            try:
                drive_paths = download_drive_source(drive_url.strip())

                if not drive_paths:
                    st.warning(
                        "No supported PDF, DOCX, TXT or MD files were found "
                        "at that Google Drive link."
                    )
                else:
                    drive_documents, drive_names = process_file_paths(
                        drive_paths
                    )
                    all_documents.extend(drive_documents)
                    all_names.extend(drive_names)

            except Exception as exc:
                st.error(f"Google Drive loading failed: {exc}")

    if not all_documents:
        st.warning("Please add at least one readable document.")
    else:
        new_chunks = chunk_documents(
            all_documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        new_hash = chunks_hash(new_chunks)

        if new_hash == st.session_state.processed_hash:
            st.success("These documents are already processed. Reusing embeddings.")
        else:
            with st.spinner(
                f"Creating embeddings for {len(new_chunks)} chunks..."
            ):
                index, embeddings = build_vector_store(new_chunks)

                st.session_state.chunks = new_chunks
                st.session_state.faiss_index = index
                st.session_state.embeddings = embeddings
                st.session_state.processed_hash = new_hash

        st.session_state.documents = all_documents
        st.session_state.loaded_names = list(dict.fromkeys(all_names))

        st.success(
            f"Loaded {len(st.session_state.loaded_names)} document(s) "
            f"and created {len(new_chunks)} chunks."
        )


# -----------------------------
# Document information
# -----------------------------
if st.session_state.documents:
    st.subheader("2. Extracted document information")

    total_characters = sum(
        len(document["text"])
        for document in st.session_state.documents
    )

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Documents",
        len(st.session_state.loaded_names),
    )

    info2.metric(
        "Extracted sections/pages",
        len(st.session_state.documents),
    )

    info3.metric(
        "Text characters",
        f"{total_characters:,}",
    )

    with st.expander("Show extracted document details"):
        for document in st.session_state.documents:
            page_label = (
                f"Page {document['page']}"
                if document["page"] is not None
                else "Page not available"
            )

            st.markdown(
                f"**{document['filename']}** — {page_label}"
            )
            st.write(document["text"][:1500])

    st.info(
        f"📦 {len(st.session_state.chunks)} chunks are stored and ready "
        "for semantic + keyword search."
    )


# -----------------------------
# Question answering
# -----------------------------
st.subheader("3. Ask a question")

question = st.text_input(
    "Ask something about your documents",
    placeholder="Example: What are the main requirements?",
)

ask = st.button(
    "🔎 Search Documents & Ask AI",
    use_container_width=True,
)

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.faiss_index is None:
        st.warning("Please load documents first.")
    else:
        api_key = st.secrets.get("GROQ_API_KEY", "")

        if not api_key:
            st.error(
                "GROQ_API_KEY is missing. Add it in Streamlit Secrets."
            )
        else:
            with st.spinner("Searching the documents..."):
                retrieved = hybrid_search(
                    question,
                    st.session_state.chunks,
                    st.session_state.faiss_index,
                    top_k=top_k,
                )

            # Add a simple response-length instruction.
            size_instruction = {
                "Short": "Keep the answer concise.",
                "Medium": "Give a clear answer with useful detail.",
                "Detailed": "Give a detailed answer, but stay within the supplied context.",
            }[response_size]

            modified_question = (
                f"{question}\n\nResponse preference: {size_instruction}"
            )

            with st.spinner("Generating answer with Groq..."):
                answer = answer_question(
                    modified_question,
                    retrieved,
                    api_key,
                    groq_model,
                )

            st.subheader("🤖 Answer")
            st.write(answer)

            st.subheader("📚 Retrieved Sources")

            for number, source in enumerate(retrieved, start=1):
                page_label = (
                    f"Page {source['page']}"
                    if source["page"] is not None
                    else "Page not available"
                )

                with st.expander(
                    f"{number}. {source['filename']} — {page_label}"
                ):
                    st.caption(
                        f"Hybrid score: {source['score']:.3f} | "
                        f"Semantic: {source['semantic_score']:.3f} | "
                        f"Keyword: {source['keyword_score']:.3f}"
                    )
                    st.write(source["text"])


st.divider()
st.caption(
    "Embeddings are created only when the processed document set changes. "
    "Questions reuse the existing FAISS index."
)
