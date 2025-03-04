from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def ingest():

    paths = [
        "data/Mobily AR_2022_English.pdf",
        "data/Operation-and-Maintenance-Manual_SEBU8407-06.pdf",
    ]

    docs = [PyPDFLoader(path).load() for path in paths]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    pc = Pinecone()
    index_name = "tactical-edge-rag-index"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # dimension for text-embedding-3-small
            metric='cosine',
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    vectorstore = PineconeVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        index_name="tactical-edge-rag-index"
    )
    print("Documents successfully ingested into Pinecone!")

if __name__ == "__main__":
    ingest()