from typing import Any, Dict
from pymongo import MongoClient
from pymongo.errors import InvalidOperation
from tinydb import TinyDB
import os
import logging

logger = logging.getLogger("DBClient")

class DBClient:
    def __init__(self, path: str) -> None:
        self.db = TinyDB(path)

    def save_results(self, results_dict: Dict[str, Any]) -> bool:
        self.db.insert(results_dict)


class MongoDBClient(DBClient):
    def __init__(self) -> None:
        self.client = MongoClient(os.environ.get("DB_URI"))
        self.results = self.client.experiments.results

    def save_results(self, results_dict: Dict[str, Any]) -> bool:
        super().save_results(results_dict)
        try:
            self.results.insert_one(results_dict)
        except InvalidOperation as err:
            logger.error(f"results with id: {results_dict['id']} to MongoDB. Error message: {str(err)}.")
            return False

        logger.info(f"Saved results with id: {results_dict['id']} to MongoDB.")
        return True
