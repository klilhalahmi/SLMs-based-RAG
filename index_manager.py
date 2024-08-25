import faiss
import numpy as np
import pickle
import time

from settings import INDEX_NAME, INDEX_MAPPER_NAME
from logger_config import logger


class SingleIndexManager:
    def __init__(self, embedding_dim):
        """
        Initialize the SingleIndexManager with the desired embedding dimension.

        Args:
            embedding_dim (int): The dimension of the embeddings.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index_to_doc_id = {}

    def add_documents(self, embeddings, doc_ids):
        """
        Add a batch of document embeddings to the FAISS index.

        Args:
            embeddings (np.ndarray): A 2D numpy array where each row is an embedding vector.
            doc_ids (list): A list of document IDs corresponding to the embeddings.

        Raises:
            ValueError: If the number of embeddings and document IDs do not match.
        """
        if len(embeddings) != len(doc_ids):
            raise ValueError(
                "Number of embeddings and document IDs must be the same.")

        # Add embeddings to FAISS index
        self.index.add(embeddings)

        # Map the index positions to document IDs
        start_pos = len(self.index_to_doc_id)
        for i, doc_id in enumerate(doc_ids):
            self.index_to_doc_id[start_pos + i] = doc_id

    def search(self, query_embedding, top_k=10):
        """
        Search the FAISS index for the top_k most similar embeddings.

        Args:
            query_embedding (np.ndarray): A 1D numpy array representing the query embedding.
            top_k (int): The number of top similar results to retrieve (default is 10).

        Returns:
            list: A list of tuples where each tuple contains a distance and a document ID.
        """
        # Perform the search
        distances, indexes = self.index.search(
            np.array([query_embedding]), top_k)

        # Filter out invalid indices
        valid_indices = [i for i in indexes[0] if i >= 0]

        # Retrieve document IDs from the mapping
        retrieved_doc_ids = [self.index_to_doc_id[i] for i in valid_indices]

        # zip the distances and retrieved_doc_ids
        return list(zip(distances[0, :len(valid_indices)], retrieved_doc_ids))

    def save_index(self, index_filepath, mapping_filepath):
        """
        Save the FAISS index and the index to document ID mapping to disk.

        Args:
            index_filepath (str): The file path to save the FAISS index.
            mapping_filepath (str): The file path to save the index to document ID mapping.
        """
        faiss.write_index(self.index, index_filepath)
        with open(mapping_filepath, "wb") as f:
            pickle.dump(self.index_to_doc_id, f)

    def load_index(self, index_filepath, mapping_filepath):
        """
        Load the FAISS index and the index to document ID mapping from disk.

        Args:
            index_filepath (str): The file path to load the FAISS index from.
            mapping_filepath (str): The file path to load the index to document ID mapping from.
        """
        self.index = faiss.read_index(index_filepath)
        with open(mapping_filepath, "rb") as f:
            self.index_to_doc_id = pickle.load(f)


def index_corpus(corpus, model, model_manager, index_output=None):
    """
    Indexes the given corpus using the provided model and model manager.

    Args:
        corpus (dict): A dictionary containing the corpus data.
        model: The model used for encoding the documents.
        model_manager: The manager responsible for adding documents to the model.
        index_output (str, optional): The output directory for saving the index. Defaults to None.
    """
    start_time = time.time()
    docs_embeddings = model.encode(
        [doc['title'] + ' ' + doc['text'] for doc in corpus.values()], batch_size=32, 
        show_progress_bar=True)
    docs_ids = list(corpus.keys())
    model_manager.add_documents(docs_embeddings, docs_ids)
    if index_output:
        index_output.mkdir(parents=True, exist_ok=True)
        model_manager.save_index(
            index_output / INDEX_NAME, index_output / INDEX_MAPPER_NAME)
    end_time = time.time()
    logger.info("Indexing completed in %.2f seconds.", end_time - start_time)
