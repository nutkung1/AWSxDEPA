from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv(override=True)


class MongoDBEmbedding:
    def __init__(self, path) -> None:
        self.mongo_uri = os.environ["MONGO_URI"]
        self.db_name = "AWSxDEPA"
        self.collection_name = "AWSxDEPA"
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.path = path

    def load_data_to_mongodb(self) -> None:
        # Load and split documents
        loader = CSVLoader(file_path=self.path)
        documents = loader.load()

        # Split documents into chunks
        self.collection.delete_many({})
        print("Collection cleared.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)

        # # Initialize HuggingFace Embeddings
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            model_kwargs={
                "dimensions": 512,
                "normalize": True,
            },
        )

        # # Perform vector search insertion into MongoDB
        MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embeddings,
            collection=self.collection,
            index_name="embedding",
        )
        print("Data loaded to MongoDB successfully.")


embedding = MongoDBEmbedding("Data/burger-king-menu.csv")
embedding.load_data_to_mongodb()
