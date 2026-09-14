from sentence_transformers import SentenceTransformer

class DataEmbedder:
    def __init__(self,model_name='all-MiniLM-L6-v2'):
        self.model= SentenceTransformer(model_name)

    def embedding_generator(self,chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings= self.model.encode(texts)
        return embeddings

    def embedding_visualizer(self,chunks):
        embeddings= self.embedding_generator(chunks)
        print("number of embeddings: ",len(embeddings))
        print("first embedding: ", embeddings[0])
        print("embedding size: ", embeddings[0].shape)
        return embeddings


