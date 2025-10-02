
from pymongo import MongoClient

# Local MongoDB
local_client = MongoClient("mongodb://localhost:27017")
local_db = local_client["online_retail_db"]

# Atlas
atlas_client = MongoClient("mongodb+srv://retail_user:retail_pass@cluster0.ay6r6kl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
atlas_db = atlas_client["online_retail_db"]

# Copy collections
for col_name in ["invoices_txn", "customers_cust"]:
    docs = list(local_db[col_name].find())
    if docs:
        atlas_db[col_name].insert_many(docs)
