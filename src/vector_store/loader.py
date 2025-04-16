import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

os.environ['USER_AGENT']='dd'
vectorstore = None

def load_vectorstore():
    global vectorstore

    bs4_strainer = bs4.SoupStrainer(class_=("content"))
    loader = WebBaseLoader(
    web_paths=("https://www.info.unlp.edu.ar/carreras-de-grado-lic-en-informatica/",),
    bs_kwargs={"parse_only": bs4_strainer},
    )

    print(f"------ Loading data - start ------")
    docs = loader.load()
    print(f"------ Loading data - end --------")

    '''
        for doc in docs:
            print(doc.page_content)
    '''

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

    print("✅ Vectorstore cargado exitosamente")

def get_vectorstore():
    return vectorstore