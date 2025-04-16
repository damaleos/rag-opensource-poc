from langchain_ollama.llms import OllamaLLM
from src.retriever.similarity_retriever import retriever

class OllamaClient:

    llm = None

    @classmethod
    def load_model(cls):
        if cls.llm is None:
            cls.llm = OllamaLLM(model="llama3.2:1b")
            print("✅ Modelo LLM cargado")

    @classmethod
    def invoke(cls, query: str) -> str:
        if cls.llm is None:
            raise ValueError("El modelo no ha sido cargado")
        
        print("Retriever invoke")
        retrieved_docs = retriever(query)

        print(f"retrieved_docs:")            
        for doc in retrieved_docs:
            print(doc.page_content)

        context = ' '.join([doc.page_content for doc in retrieved_docs])
                
        #return cls.llm.invoke(query)
        response = cls.llm.invoke(f"""Answer the question according to the context given very briefly:
           Question: {query}.
           Context: {context}
        """)
        

        return response
        
    