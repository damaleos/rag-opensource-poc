import os
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM

os.environ['USER_AGENT']='dd'



#bs4_strainer = bs4.SoupStrainer(class_=("content-area"))
#loader = WebBaseLoader(
#    web_paths=("https://pythonology.eu/using-pandas_ta-to-generate-technical-indicators-and-signals/",),
#    bs_kwargs={"parse_only": bs4_strainer},
#)

bs4_strainer = bs4.SoupStrainer(class_=("content"))
loader = WebBaseLoader(
    web_paths=("https://www.info.unlp.edu.ar/carreras-de-grado-lic-en-informatica/",),
    bs_kwargs={"parse_only": bs4_strainer},
)


docs = loader.load()

for doc in docs:
    print(doc.page_content)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Total chunks: {len(all_splits)}")
#for i, doc in enumerate(all_splits, start=1):
#    print(f"\n--- Chunk {i} ---")
#    print("Content:")
#    print(doc.page_content)
#    print("Metadata:")
#    print(doc.metadata)


local_embeddings = OllamaEmbeddings(model="all-minilm")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

question = "Cuando fue creada la carrera licenciatura en informatica?"
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
retrieved_docs = retriever.invoke(question)

print(f"retrieved_docs:")
      
for doc in retrieved_docs:
    print(doc.page_content)

context = ' '.join([doc.page_content for doc in retrieved_docs])



llm = OllamaLLM(model="llama3.2:1b")
response = llm.invoke(f"""Answer the question according to the context given very briefly:
           Question: {question}.
           Context: {context}
""")

print(f"Respuesta: {response}")