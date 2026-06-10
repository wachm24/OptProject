from dotenv import load_dotenv
load_dotenv()
import os
from pymongo import MongoClient

client = MongoClient(os.environ["MONGO_URI"])
db = client.notes_db
notes_collection = db["notes"]

folders_collection = db["folders"]