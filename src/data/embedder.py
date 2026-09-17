from huggingface_hub import InferenceClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)

class DataEmbedder:
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
        
        if self.provider == "hf":
            self.hf_client = InferenceClient(
                provider="auto",
                api_key=os.getenv("HF_TOKEN")
            )
            self.hf_model_name = "Qwen/Qwen3-Embedding-0.6B"
            print("[Embedder Setup] Hugging Face initialized.")
        else:
            self.gemini_embedder = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2"
            )
            print("[Embedder Setup] Gemini initialized.")

    def embedding_generator(self, texts):
        # makes embedding
        if self.provider == "hf":
            embeddings = self.hf_client.feature_extraction(texts, model=self.hf_model_name)
            return embeddings.tolist()
        else:
            return self.gemini_embedder.embed_documents(texts)

    def embedding_visualizer(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_generator(texts)

        print("number of embeddings:", len(embeddings))
        print("first embedding size:", len(embeddings[0]))

        return embeddings



