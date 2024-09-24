from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os


class mongoDBEmbedding:
    def __init__(self) -> None:
        self.MONGO_URI = os.environ("MONGO_URI")
        self.DB_NAME = "DepaXAws"
        self.COLLECTION_NAME = "AWS"
        self.client = MongoClient(self.MONGO_URI)
        self.db = self.client[self.DB_NAME]
        self.collection = self.db[self.COLLECTION_NAME]
