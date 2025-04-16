from src.vector_store.loader import get_vectorstore
from typing import List
from langchain.schema import Document


def retriever(query: str) -> List[Document]:

    #query = "Cuando fue creada la carrera licenciatura en informatica?"
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise ValueError("Vectorstore no está cargado")
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs