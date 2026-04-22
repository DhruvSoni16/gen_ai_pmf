import os
import fitz  # PyMuPDF
import docx  # python-docx
import faiss
import numpy as np
import pickle

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception as _st_err:
    _ST_AVAILABLE = False
    _ST_LOAD_ERR = _st_err

class DocumentRetriever:
    # def __init__(self, documents_dir, vector_db_path="vector_db", model_name="all-MiniLM-L6-v2"):
    def __init__(self, documents_dir, vector_db_path="vector_db", model_name="all-mpnet-base-v2"):
        if not _ST_AVAILABLE:
            raise RuntimeError(
                f"sentence_transformers could not be loaded ({_ST_LOAD_ERR}). "
                "Fix torch by running:\n"
                "  pip uninstall torch torchvision torchaudio -y\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        print("Initializing DocumentRetriever...")
        self.documents_dir = documents_dir
        self.vector_db_path = vector_db_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.document_info = {}
        self.embeddings = None
 
        # Ensure vector database directory exists
        os.makedirs(vector_db_path, exist_ok=True)
        print("Initialization complete.")
 
    def clear_database(self):
        """Clear the vector database and FAISS index."""
        print("Clearing existing vector database...")
        db_file = os.path.join(self.vector_db_path, "vector_db.pkl")
        index_file = os.path.join(self.vector_db_path, "faiss_index.bin")
 
        if os.path.exists(db_file):
            os.remove(db_file)
            print(f"Deleted database file: {db_file}")
        if os.path.exists(index_file):
            os.remove(index_file)
            print(f"Deleted FAISS index file: {index_file}")
 
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        print(f"Extracting text from PDF: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            print(f"Extracted {len(text)} characters from PDF.")
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
 
    def extract_text_from_docx(self, docx_path):
        """Extract text from a DOCX file."""
        print(f"Extracting text from DOCX: {docx_path}")
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            print(f"Extracted {len(text)} characters from DOCX.")
            return text
        except Exception as e:
            print(f"Error extracting text from DOCX {docx_path}: {e}")
            return ""
 
    def extract_text_from_file(self, file_path):
        """Extract text based on file type."""
        print(f"Determining file type for: {file_path}")
        if file_path.endswith(".pdf"):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            return self.extract_text_from_docx(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return ""
 
    def generate_embedding(self, text):
        """Generate and normalize embedding for the given text."""
        print(f"Generating embedding for text of length {len(text)}...")
        if not text.strip():
            print("Text is empty. Skipping embedding generation.")
            return None
 
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=True)  # Returns a PyTorch tensor
            embedding = embedding.detach().cpu().numpy()  # Convert tensor to NumPy array
 
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print("Embedding norm is zero. Skipping normalization.")
                return None
            embedding = embedding / norm
            print("Embedding generated and normalized successfully.")
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
 
    
 
    def process_documents(self):
        """Process documents and build the vector database using IndexHNSWFlat."""
        print("Starting document processing...")
        self.clear_database()  # Clear the database before processing
 
        db_file = os.path.join(self.vector_db_path, "vector_db.pkl")
        index_file = os.path.join(self.vector_db_path, "faiss_index.bin")
 
        print("Processing documents...")
        all_embeddings = []
        doc_id = 0
 
        for file in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, file)
            if os.path.isfile(file_path) and file.lower().endswith((".pdf", ".docx")):
                print(f"Processing file: {file}")
                text = self.extract_text_from_file(file_path)
                if not text:
                    print(f"Skipping empty document: {file}")
                    continue
 
                # Generate embedding for the document
                print(f"Generating embedding for document: {file}")
                embedding = self.generate_embedding(text)
                if embedding is None:
                    print(f"Skipping document due to embedding generation failure: {file}")
                    continue
                all_embeddings.append(embedding)
 
                # Save document info
                self.document_info[doc_id] = {"path": file_path, "filename": file}
                doc_id += 1
 
        # Build FAISS index
        if all_embeddings:
            print("Building FAISS index using IndexHNSWFlat...")
            self.embeddings = np.vstack(all_embeddings).astype(np.float32)
            print(f"Embeddings shape: {self.embeddings.shape}, dtype: {self.embeddings.dtype}")
 
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the number of neighbors in the graph
            print(f"FAISS index dimensionality: {self.index.d}, number of neighbors: 32")
 
            # Add embeddings to the index
            try:
                self.index.add(self.embeddings)
                print("Embeddings added to FAISS index successfully.")
            except Exception as e:
                print(f"Error adding embeddings to FAISS index: {e}")
                return
 
            # Save vector database and index
            print("Saving vector database...")
            with open(db_file, "wb") as f:
                pickle.dump({"document_info": self.document_info, "embeddings": self.embeddings}, f)
 
            print("Saving FAISS index...")
            faiss.write_index(self.index, index_file)
            print(f"Processed {len(self.document_info)} documents.")
        else:
            print("No documents were processed.")
 
    def search(self, query, top_k=5):
        """Search for relevant documents based on a query."""
        print(f"Starting search for query: {query}")
        if self.index is None:
            print("FAISS index is not initialized. Run process_documents() first.")
            return []
 
        # Generate query embedding
        print("Generating embedding for query...")
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        query_embedding = query_embedding.detach().cpu().numpy()  # Convert tensor to NumPy array
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        query_embedding = np.array([query_embedding]).astype(np.float32)
 
        # Check query embedding shape
        print(f"Query embedding shape: {query_embedding.shape}")
        if query_embedding.shape[1] != self.index.d:
            print(f"Query embedding dimension mismatch. Expected {self.index.d}, got {query_embedding.shape[1]}.")
            return []
 
        # Search the index
        print("Searching FAISS index...")
        distances, indices = self.index.search(query_embedding, top_k)
        print("Search completed.")
 
        # Retrieve document info
        print("Processing search results...")
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.document_info):
                continue
            doc_info = self.document_info[idx]
            results.append((doc_info["path"], doc_info["filename"], distance))
 
        print(f"Search complete. Found {len(results)} results.")
        return results
 