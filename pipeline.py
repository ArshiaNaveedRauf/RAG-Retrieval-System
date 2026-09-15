from src.data.scraper import WebsiteScraper
from src.data.embedder import DataEmbedder
from src.vector_db.chroma_db import VectorDatabase
from config import Url

class Pipeline:
    def __init__(self):
        self.chunker = WebsiteScraper()
        self.embedder = DataEmbedder()
        self.db = VectorDatabase()

    def run_ingestion_pipeline(self):
        collection = self.db.create_collection()
        if collection.count()>0:
            return collection
        chunks = self.chunker.get_website_data(Url)
        embeddings = self.embedder.embedding_visualizer(chunks)
        database = self.db.collect_vectors(embeddings,chunks)
        return database 


