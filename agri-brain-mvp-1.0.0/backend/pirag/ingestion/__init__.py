"""piRAG data ingestion pipeline: parsing, embedding, and vector storage."""
from .parser import parse_document
from .embedder import TFIDFEmbedder
from .vector_store import VectorStore
