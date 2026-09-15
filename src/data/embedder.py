from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()


class DataEmbedder:
    def __init__(self):
        self.client = InferenceClient(
            provider="auto",
            api_key=os.getenv("HF_TOKEN")
        )

        self.model_name = "Qwen/Qwen3-Embedding-0.6B"

    def embedding_generator(self, texts):
        embeddings = self.client.feature_extraction(
            texts,
            model=self.model_name
        )

        return embeddings.tolist()

    def embedding_visualizer(self, chunks):
        texts = [chunk.page_content for chunk in chunks]

        embeddings = self.embedding_generator(texts)

        print("number of embeddings:", len(embeddings))
        print("first embedding:", embeddings[0])
        print("embedding size:", len(embeddings[0]))

        return embeddings