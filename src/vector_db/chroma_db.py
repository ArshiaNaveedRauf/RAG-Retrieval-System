import chromadb

class VectorDatabase:
    def __init__ (self):
        self.client= chromadb.PersistentClient(path="./chroma_db")

    def create_collection(self):
        collection= self.client.get_or_create_collection(
            name="main_collection"
        )
        print ("collection loaded")
        return collection

    def collect_vectors(self,embeddings,chunks):
        collection = self.create_collection()    
        metadatas = [chunk.metadata for chunk in chunks]
        documents =  [chunk.page_content for chunk in chunks]
        ids= [str(x) for x in range(len(chunks))]
        collection.add(
            embeddings= embeddings,
            metadatas=metadatas,
            documents= documents,
            ids=ids
        )
        result = collection.get(limit=5)
        print("ids:", result["ids"])
        print("documents: ", result["documents"])
        print("metadatas", result["metadatas"])
        return collection 