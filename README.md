# 📄 AI Document Assistant

A simple Streamlit RAG-style document assistant using:

- Streamlit
- PyPDF
- python-docx
- Sentence Transformers
- FAISS
- Groq
- gdown for public/shared Google Drive files and folders

The app supports:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Local file uploads
- Public/shared Google Drive file links
- Public/shared Google Drive folder links
- Text extraction
- Text chunking with overlap
- Sentence Transformer embeddings
- FAISS semantic search
- Keyword search
- Hybrid search
- Groq question answering
- Retrieved source display
- Streamlit session-state reuse

---

## 1. Project files

Keep exactly these three files in your GitHub repository:

```text
app.py
requirements.txt
README.md
```

---

## 2. How the app works

The application uses this pipeline:

```text
Local Upload / Google Drive
          ↓
     Text Extraction
          ↓
       Chunking
          ↓
Sentence Transformer Embeddings
          ↓
       FAISS Index
          ↓
   User Question
          ↓
 ┌───────────────────────┐
 │ Semantic Search       │
 │ +                     │
 │ Keyword Search        │
 └───────────────────────┘
          ↓
     Hybrid Ranking
          ↓
  Top Relevant Chunks
          ↓
         Groq
          ↓
       AI Answer
          ↓
    Retrieved Sources
```

---

## 3. Important optimization

The application does **not** create embeddings every time you ask a question.

When documents are loaded:

1. Text is extracted.
2. Text is split into chunks.
3. All chunks are embedded once.
4. The embeddings are placed into a FAISS index.
5. The FAISS index is stored in Streamlit session state.

When the user asks another question:

1. Only the question is embedded.
2. FAISS finds semantically similar chunks.
3. Keyword matching scores the same candidate chunks.
4. Both scores are combined.
5. The best chunks are sent to Groq.

This makes repeated questions much faster.

---

## 4. Hybrid search

The search uses two signals.

### Semantic score

Sentence Transformers converts:

- the document chunks
- the user's question

into numerical vectors.

FAISS compares the question vector with the document vectors.

### Keyword score

Important words from the question are extracted.

Common stop words such as:

```text
the
a
and
is
are
what
how
```

are ignored.

The app checks how many important question words occur in each candidate chunk.

### Combined score

The current weighting is:

```text
Hybrid Score =
    75% semantic similarity
    +
    25% keyword matching
```

You can change these weights in `app.py` if required.

---

## 5. Supported document metadata

Every extracted section/chunk keeps:

```text
filename
page
text
```

PDF files preserve page numbers.

DOCX, TXT and MD files do not have reliable page numbers during normal text extraction, so their page value is shown as:

```text
Page not available
```

---

# 6. Streamlit Secrets

Do **not** put your Groq API key inside `app.py`.

In Streamlit Cloud:

1. Open your deployed application.
2. Open **Settings**.
3. Open **Secrets**.
4. Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Make sure the key uses normal straight quotes.

Correct:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxx"
```

Incorrect:

```toml
GROQ_API_KEY = “gsk_xxxxxxxxx”
```

The second example uses curly quotation marks and is not valid TOML.

---

# 7. GitHub setup

Create a new GitHub repository.

Upload:

```text
app.py
requirements.txt
README.md
```

Do not upload:

```text
.env
.env.local
secrets.toml
```

Do not put your Groq API key in GitHub.

---

# 8. Deploy on Streamlit Community Cloud

Go to Streamlit Community Cloud and create a new app.

Select:

```text
Repository: your GitHub repository
Branch: main
Main file path: app.py
```

Deploy the application.

Then add the Groq key through Streamlit's **Secrets** settings.

---

# 9. Local testing

Install the requirements:

```bash
pip install -r requirements.txt
```

Create:

```text
.streamlit/secrets.toml
```

Put:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Run:

```bash
streamlit run app.py
```

---

# 10. Google Drive

The app accepts a Google Drive file or folder link.

Example:

```text
https://drive.google.com/...
```

For a folder, the folder should be accessible without requiring the app to log into a private Google account.

Supported Drive files:

```text
.pdf
.docx
.txt
.md
```

A Drive folder can contain multiple supported files. The app downloads the supported files and sends them through exactly the same pipeline as local uploads:

```text
Google Drive
    ↓
Extraction
    ↓
Chunking
    ↓
Embedding
    ↓
FAISS
    ↓
Hybrid Search
    ↓
Groq
```

### Important Google Drive limitation

This simple version is designed for **public/shared links that can be accessed without private Google authentication**.

It does not implement OAuth login to a user's private Google Drive.

That keeps the project simple and avoids adding Google Cloud OAuth credentials.

---

# 11. Running in Google Colab

You can also test the application in Colab.

Install the dependencies:

```python
!pip install -r requirements.txt
```

Then run:

```python
!streamlit run app.py &>/content/streamlit.log &
```

For a public temporary URL, you can use a Cloudflare tunnel separately.

The application itself does not require ngrok.

---

# 12. Groq model

The default model in the UI is:

```text
openai/gpt-oss-20b
```

The UI also provides:

```text
openai/gpt-oss-120b
```

The model is instructed to answer only from the retrieved document context.

If the retrieved context does not contain the answer, the application asks the model to say:

```text
The information is not available in the provided documents.
```

---

# 13. Security note

Never expose your Groq API key in:

- `app.py`
- GitHub
- README.md
- screenshots
- public notebooks

Use Streamlit Secrets instead.

---

# 14. Simple explanation of each major function

### `extract_pdf()`

Reads each PDF page and keeps the page number.

### `extract_docx()`

Reads paragraphs from a DOCX document.

### `extract_txt()`

Reads a normal text file.

### `extract_md()`

Reads a Markdown file as text.

### `chunk_documents()`

Splits large text into smaller overlapping pieces.

### `build_vector_store()`

Creates Sentence Transformer embeddings and puts them into FAISS.

### `keyword_score()`

Checks important words from the question against a chunk.

### `hybrid_search()`

Combines semantic similarity and keyword matching.

### `answer_question()`

Sends the question and retrieved context to Groq.

### `download_drive_source()`

Downloads supported public/shared Google Drive files or folders.

---

# 15. Recommended first test

Start with one small TXT or PDF file.

Ask a question whose answer is clearly inside the document.

Then check:

1. Extracted document information
2. Number of chunks
3. Retrieved sources
4. Filename
5. Page number for PDFs
6. Retrieved text
7. Final Groq answer

Then try a question whose answer is **not** in the document.

The assistant should state that the information is not available in the provided documents.
