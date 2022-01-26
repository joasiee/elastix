from pymongo import MongoClient
import os

class DBClient:
    def __init__(self) -> None:
        self.client = MongoClient(os.environ.get("DB_URI"))

if __name__ == "__main__":
    db = DBClient()
    