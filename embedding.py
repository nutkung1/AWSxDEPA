from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
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
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
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

        # Initialize HuggingFace Embeddings
        hf = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
            multi_process=False,  # Disable multiprocessing for simplicity
        )

        # Perform vector search insertion into MongoDB
        MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=hf,
            collection=self.collection,
            index_name="embedding",
        )
        print("Data loaded to MongoDB successfully.")


embedding = MongoDBEmbedding("Data/burger-king-menu.csv")
embedding.load_data_to_mongodb()
