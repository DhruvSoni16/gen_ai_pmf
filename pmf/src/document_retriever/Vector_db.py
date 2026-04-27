import os
import fitz  # PyMuPDF
import docx  # python-docx
import faiss
import numpy as np
import pickle

# Try sentence_transformers (requires torch). If torch DLLs fail on Windows,
# fall back to TF-IDF + TruncatedSVD which needs only scikit-learn.
_BACKEND = "none"
try:
    from sentence_transformers import SentenceTransformer
    _BACKEND = "st"
    print("[DocumentRetriever] Using sentence_transformers backend.")
except Exception as _st_err:
    print(f"[DocumentRetriever] sentence_transformers unavailable ({_st_err}), trying TF-IDF fallback.")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        _BACKEND = "tfidf"
        print("[DocumentRetriever] Using TF-IDF + SVD backend.")
    except Exception as _sk_err:
        print(f"[DocumentRetriever] sklearn also unavailable ({_sk_err}). No embedding backend found.")


class _TFIDFModel:
    """TF-IDF + TruncatedSVD drop-in for SentenceTransformer. No torch required."""
    DIM = 512

    def __init__(self):
        self._tfidf = TfidfVectorizer(max_features=30000, sublinear_tf=True)
        self._svd = TruncatedSVD(n_components=self.DIM, random_state=0)
        self.fitted = False

    def fit(self, texts: list):
        mat = self._tfidf.fit_transform(texts)
        nc = min(self.DIM, mat.shape[1] - 1, mat.shape[0] - 1)
        self._svd.n_components = max(1, nc)
        self._svd.fit(mat)
        self.fitted = True

    def encode(self, text, convert_to_tensor=False):
        if not self.fitted:
            raise RuntimeError("TF-IDF model has not been fitted yet. Call process_documents() first.")
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        emb = self._svd.transform(self._tfidf.transform(texts)).astype(np.float32)
        return emb[0] if single else emb


def _to_numpy(embedding):
    """Convert a tensor or array to a plain float32 numpy array."""
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()
    return np.array(embedding, dtype=np.float32)


class DocumentRetriever:
    def __init__(self, documents_dir, vector_db_path="vector_db", model_name="all-mpnet-base-v2"):
        if _BACKEND == "none":
            raise RuntimeError(
                "No embedding backend is available. Install sentence_transformers or scikit-learn."
            )
        print(f"Initializing DocumentRetriever (backend={_BACKEND})...")
        self.documents_dir = documents_dir
        self.vector_db_path = vector_db_path
        self._model_name = model_name
        self.index = None
        self.document_info = {}
        self.embeddings = None

        if _BACKEND == "st":
            self.model = SentenceTransformer(model_name)
        else:
            self.model = _TFIDFModel()

        os.makedirs(vector_db_path, exist_ok=True)
        print("Initialization complete.")

    def clear_database(self):
        print("Clearing existing vector database...")
        for fname in ("vector_db.pkl", "faiss_index.bin"):
            path = os.path.join(self.vector_db_path, fname)
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted: {path}")

    def extract_text_from_pdf(self, pdf_path):
        print(f"Extracting text from PDF: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            text = "".join(page.get_text() for page in doc)
            print(f"Extracted {len(text)} characters from PDF.")
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    def extract_text_from_docx(self, docx_path):
        print(f"Extracting text from DOCX: {docx_path}")
        try:
            doc = docx.Document(docx_path)
            text = "\n".join(p.text for p in doc.paragraphs)
            print(f"Extracted {len(text)} characters from DOCX.")
            return text
        except Exception as e:
            print(f"Error extracting text from DOCX {docx_path}: {e}")
            return ""

    def extract_text_from_file(self, file_path):
        if file_path.endswith(".pdf"):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            return self.extract_text_from_docx(file_path)
        print(f"Unsupported file type: {file_path}")
        return ""

    def generate_embedding(self, text):
        if not text.strip():
            return None
        try:
            emb = _to_numpy(self.model.encode(text, convert_to_tensor=(_BACKEND == "st")))
            norm = np.linalg.norm(emb)
            if norm == 0:
                return None
            return emb / norm
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def process_documents(self):
        print("Starting document processing...")
        self.clear_database()

        db_file = os.path.join(self.vector_db_path, "vector_db.pkl")
        index_file = os.path.join(self.vector_db_path, "faiss_index.bin")

        # Collect valid files and their texts
        file_texts = []
        file_paths = []
        for file in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, file)
            if os.path.isfile(file_path) and file.lower().endswith((".pdf", ".docx")):
                text = self.extract_text_from_file(file_path)
                if text:
                    file_texts.append(text)
                    file_paths.append((file_path, file))

        if not file_texts:
            print("No documents were processed.")
            return

        # TF-IDF needs to be fitted on the full corpus before encoding
        if _BACKEND == "tfidf":
            print("Fitting TF-IDF model on corpus...")
            self.model.fit(file_texts)

        # Generate embeddings
        all_embeddings = []
        doc_id = 0
        for text, (file_path, file) in zip(file_texts, file_paths):
            print(f"Generating embedding for: {file}")
            embedding = self.generate_embedding(text)
            if embedding is None:
                print(f"Skipping {file}: embedding failed.")
                continue
            all_embeddings.append(embedding)
            self.document_info[doc_id] = {"path": file_path, "filename": file}
            doc_id += 1

        # Build FAISS index
        print("Building FAISS index...")
        self.embeddings = np.vstack(all_embeddings).astype(np.float32)
        print(f"Embeddings shape: {self.embeddings.shape}")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.add(self.embeddings)
        print("Embeddings added to FAISS index.")

        # Persist
        with open(db_file, "wb") as f:
            pickle.dump({
                "document_info": self.document_info,
                "embeddings": self.embeddings,
                "tfidf_model": self.model if _BACKEND == "tfidf" else None,
            }, f)
        faiss.write_index(self.index, index_file)
        print(f"Processed {len(self.document_info)} documents.")

    def search(self, query, top_k=5):
        print(f"Searching for: {query}")
        if self.index is None:
            print("FAISS index not initialized. Run process_documents() first.")
            return []

        q_emb = _to_numpy(self.model.encode(query, convert_to_tensor=(_BACKEND == "st")))
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm
        q_emb = q_emb.reshape(1, -1).astype(np.float32)

        if q_emb.shape[1] != self.index.d:
            print(f"Dimension mismatch: index={self.index.d}, query={q_emb.shape[1]}")
            return []

        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.document_info):
                info = self.document_info[idx]
                results.append((info["path"], info["filename"], distance))
        print(f"Found {len(results)} results.")
        return results
