import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client.http.models import VectorParams, Distance

load_dotenv()
collection_name = "articles"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest():
    urls = [
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2022-06-09-vlm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_flattened = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=25
    )

    doc_splits = text_splitter.split_documents(docs_flattened)

    vectorstore = QdrantVectorStore.from_documents(
        doc_splits,
        embedding_model,
        url=os.getenv("QDRANT_URL"),
        prefer_grpc=True,
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name,
    )


def get_retriever():
    vectorstore = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embedding_model,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    return vectorstore.as_retriever()
