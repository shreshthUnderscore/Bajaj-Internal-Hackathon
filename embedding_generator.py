import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingGenerator:
    """
    A class to generate sentence embeddings using a pre-trained Sentence-Transformer model.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the EmbeddingGenerator with a specified Sentence-Transformer model.
        Args:
            model_name (str): The name of the pre-trained model to load.
                              Common choices: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'.
        """
        self.model_name = model_name
        self.model = None
        try:
            logging.info(f"Loading Sentence-Transformer model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logging.info(f"Model {self.model_name} loaded successfully.")
            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logging.info(f"Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logging.error(f"Error loading Sentence-Transformer model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings.
        Args:
            texts (List[str]): A list of text strings (e.g., document chunks, queries).
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        if not self.model:
            logging.error("Embedding model not loaded. Cannot generate embeddings.")
            return []
        if not texts:
            return []

        try:
            logging.info(f"Generating embeddings for {len(texts)} texts...")
            # Changed: Removed convert_to_numpy=False. Default is True, which returns a numpy array.
            # numpy arrays also have a .tolist() method.
            embeddings = self.model.encode(texts).tolist()
            logging.info(f"Successfully generated embeddings for {len(texts)} texts.")
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return []

if __name__ == "__main__":
    # Example usage:
    print("--- Testing Embedding Generation Module ---")

    # Initialize the generator
    try:
        embedding_gen = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
        print(f"Model loaded. Embedding dimension: {embedding_gen.embedding_dimension}")
    except Exception as e:
        print(f"Failed to initialize EmbeddingGenerator: {e}")
        exit()

    # Sample texts (e.g., chunks from document_processor)
    sample_chunks = [
        "This is a sample sentence about insurance policies and their terms.",
        "The quick brown fox jumps over the lazy dog.",
        "GDPR regulations are important for data privacy and compliance in Europe.",
        "An employment contract outlines the terms and conditions of work."
    ]

    # Generate embeddings
    embeddings = embedding_gen.generate_embeddings(sample_chunks)

    print(f"\nGenerated {len(embeddings)} embeddings.")
    if embeddings:
        print(f"Dimension of first embedding: {len(embeddings[0])}")
        print("First embedding (first 10 values):")
        print(embeddings[0][:10])

    # Test with empty list
    empty_embeddings = embedding_gen.generate_embeddings([])
    print(f"\nGenerated {len(empty_embeddings)} embeddings for empty list.")

    # Test with a very short text
    short_text_embeddings = embedding_gen.generate_embeddings(["hello"])
    print(f"\nGenerated {len(short_text_embeddings)} embeddings for short text.")
    if short_text_embeddings:
        print(f"Dimension of short text embedding: {len(short_text_embeddings[0])}")
