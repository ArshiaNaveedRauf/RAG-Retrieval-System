from src.data.embedder import DataEmbedder
from langchain_core.tools import tool
from pipeline import Pipeline
from config import Top_k
class QueryRetrival:

    def __init__(self):
        self.embedder= DataEmbedder()

    def query_embedder(self,query):
        embedded_query= self.embedder.embedding_generator([query])
        return embedded_query[0]

    def search_vector_db(self,db,query,top_k):
        embedded_query = self.query_embedder(query)
        retrieved_docs= db.query(
            query_embeddings=[embedded_query],
            n_results=top_k
        )
        return retrieved_docs


@tool("website_knowledge_base")
def website_knowledge_base(query):
    """Search the website knowledge base for relevant information."""
    retrieval= QueryRetrival()
    pipeline = Pipeline()
    collection = pipeline.run_ingestion_pipeline()
    results= retrieval.search_vector_db(db= collection,query=query,top_k=Top_k)
    documents= results["documents"][0]
    print ( "\n\n".join(documents))
    return "\n\n".join(documents)

# for testing
website_knowledge_base.invoke({"query": "what is ned university"})

    
