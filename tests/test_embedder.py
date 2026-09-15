from src.data.embedder import DataEmbedder
import numpy as np
from langchain_core.documents import Document

def test_embedder():
    chunks =[ Document( page_content="MMSE Lab is a research laboratory at NED University.", metadata={"source": "fake-website"} ),
              Document( page_content="The laboratory focuses on research and technological innovation.", metadata={"source": "fake-website"} ),
              Document( page_content="Students and researchers work together on different projects.", metadata={"source": "fake-website"} ) ]

    embedder= DataEmbedder()
    embeddings= embedder.embedding_visualizer(chunks)

    assert isinstance(embeddings, list)



       