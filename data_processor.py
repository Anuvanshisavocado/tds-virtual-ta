import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Optional # Import List and Dict from typing

class DataProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the DataProcessor with a SentenceTransformer model.
        'all-MiniLM-L6-v2' is a good balance of speed and performance.
        """
        print(f"Initializing SentenceTransformer model: {model_name}...")
        # This line might download the model if it's not cached locally
        self.model = SentenceTransformer(model_name)
        self.course_content_dir = "data/course_content"
        self.discourse_posts_dir = "data/discourse_posts"
        # This is the metadata file we created in the previous step
        self.discourse_metadata_file = "data/discourse_posts/all_discourse_posts_metadata.json"
        
        self.documents: List[Dict] = [] # Stores list of dictionaries: {"text": "...", "metadata": {}}
        self.faiss_index: Optional[faiss.Index] = None

    def load_and_chunk_data(self):
        """
        Loads text content from course materials and discourse posts,
        and prepares them as documents for indexing. Simple chunking applied.
        """
        print("Loading and chunking data from disk...")
        
        # Load Course Content
        for filename in os.listdir(self.course_content_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.course_content_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # Simple chunking: split by double newline (paragraph)
                    # This helps when answers might come from specific paragraphs
                    chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
                    if not chunks: # If no paragraph splits, treat whole content as one chunk
                        chunks = [content.strip()]
                    
                    for i, chunk in enumerate(chunks):
                        if chunk: # Ensure the chunk is not empty
                            self.documents.append({
                                "text": chunk,
                                "metadata": {
                                    "source_type": "course_content",
                                    "source_file": filename,
                                    "chunk_id": i
                                }
                            })
        print(f"Loaded {len([d for d in self.documents if d['metadata']['source_type'] == 'course_content'])} course content chunks.")

        # Load Discourse Posts (using the metadata file for richer info)
        if os.path.exists(self.discourse_metadata_file):
            with open(self.discourse_metadata_file, "r", encoding="utf-8") as f:
                discourse_data_metadata = json.load(f)

            for post_meta in discourse_data_metadata:
                post_id = post_meta.get('post_id')
                if post_id:
                    # Assuming the content was saved to individual .txt files
                    filepath = os.path.join(self.discourse_posts_dir, f"discourse_post_{post_id}.txt")
                    if os.path.exists(filepath):
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                            if content: # Ensure the content is not empty
                                self.documents.append({
                                    "text": content,
                                    "metadata": {
                                        "source_type": "discourse_post",
                                        "topic_id": post_meta.get('topic_id'),
                                        "post_id": post_meta.get('post_id'),
                                        "topic_title": post_meta.get('topic_title'),
                                        "url": post_meta.get('url'),
                                        "username": post_meta.get('username'),
                                        "created_at": post_meta.get('created_at')
                                    }
                                })
            print(f"Loaded {len([d for d in self.documents if d['metadata']['source_type'] == 'discourse_post'])} discourse post chunks.")
        else:
            print(f"Warning: {self.discourse_metadata_file} not found. Discourse posts might not be indexed properly or with full metadata.")

    def create_faiss_index(self):
        """
        Generates embeddings for all loaded documents and builds a FAISS index.
        """
        if not self.documents:
            print("No documents loaded to index. Please run load_and_chunk_data() first.")
            return

        print(f"Creating embeddings for {len(self.documents)} documents...")
        texts_to_embed = [doc["text"] for doc in self.documents]
        # Convert to float32 as FAISS often expects this data type
        embeddings = self.model.encode(texts_to_embed, show_progress_bar=True).astype('float32')
        
        dimension = embeddings.shape[1] # The dimensionality of the embeddings
        
        print(f"Building FAISS index with dimension {dimension}...")
        # IndexFlatL2 is a simple index that stores vectors directly and uses L2 (Euclidean) distance
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
        print(f"FAISS index created with {self.faiss_index.ntotal} documents indexed.")

    def search_documents(self, query_text: str, k: int = 5) -> List[Dict]:
        """
        Performs a similarity search in the FAISS index for the given query.
        Returns the top k most relevant documents.
        """
        if not self.faiss_index:
            print("FAISS index not available. Please create or load it first.")
            return []

        query_embedding = self.model.encode([query_text]).astype('float32')
        
        # D: distances, I: indices of the nearest neighbors
        distances, indices = self.faiss_index.search(query_embedding, k)

        results = []
        # indices[0] because search returns results for multiple queries; we have one.
        for i, distance in zip(indices[0], distances[0]):
            if i != -1: # FAISS returns -1 for empty slots if k is larger than ntotal
                doc_data = self.documents[i]
                # It's good practice to copy to avoid modifying the original documents list
                # when adding ephemeral search-specific data like 'distance'
                result_doc = doc_data.copy() 
                # result_doc["distance"] = float(distance) # Uncomment if you want to see distance
                results.append(result_doc)
        return results

    def save_index(self, index_path="faiss_index.bin", doc_path="documents_with_metadata.json"):
        """
        Saves the FAISS index and the associated document metadata to disk.
        """
        if self.faiss_index:
            faiss.write_index(self.faiss_index, index_path)
            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=4)
            print(f"FAISS index and documents saved to {index_path} and {doc_path}")
        else:
            print("No FAISS index to save.")

    def load_index(self, index_path="faiss_index.bin", doc_path="documents_with_metadata.json"):
        """
        Loads the FAISS index and document metadata from disk.
        Returns True if successful, False otherwise.
        """
        if os.path.exists(index_path) and os.path.exists(doc_path):
            print(f"Loading FAISS index from {index_path} and documents from {doc_path}...")
            try:
                self.faiss_index = faiss.read_index(index_path)
                with open(doc_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                print("FAISS index and documents loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading FAISS index or documents: {e}. Will rebuild.")
                return False
        print("Existing FAISS index or documents not found. A new one will be created.")
        return False

# --- Example Usage for creating/loading the index (for standalone test) ---
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Try to load existing index; if not found or corrupted, create it
    if not processor.load_index():
        processor.load_and_chunk_data()
        processor.create_faiss_index()
        processor.save_index()

    # Test search functionality with a sample query
    test_query = "How do I filter rows in Pandas using multiple conditions?"
    print(f"\nPerforming a sample search for: '{test_query}'")
    results = processor.search_documents(test_query, k=3)
    
    if results:
        print(f"Found {len(results)} relevant documents:")
        for i, res in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(f"Source Type: {res['metadata']['source_type']}")
            if res['metadata']['source_type'] == 'discourse_post':
                print(f"  URL: {res['metadata'].get('url', 'N/A')}")
                print(f"  Topic: {res['metadata'].get('topic_title', 'N/A')}")
            else:
                print(f"  File: {res['metadata'].get('source_file', 'N/A')}")
            print(f"  Content (first 150 chars):\n{res['text'][:150]}...\n")
    else:
        print("No relevant documents found for the test query.")