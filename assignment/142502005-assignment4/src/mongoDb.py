import pandas as pd
import pymongo
from pymongo.errors import BulkWriteError, ConnectionFailure
from pathlib import Path

# Load dataset
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Online Retail.xlsx"
df = pd.read_excel(DATA_PATH)

# Drop rows with missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Convert data types
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# MongoDB connection
try:
    client = pymongo.MongoClient(
        "mongodb://localhost:27017",
        maxPoolSize=50,
        serverSelectionTimeoutMS=5000
    )
    client.admin.command('ping')
    print("✅ Connected to MongoDB")
except ConnectionFailure as e:
    print("❌ MongoDB connection failed:", e)
    exit(1)

db = client["online_retail_db"]

# Transaction-centric collection
invoices_col = db["invoices_txn"]
transactions = []
for invoice_no, group in df.groupby("InvoiceNo"):
    try:
        transactions.append({
            "_id": str(invoice_no),
            "invoice_date": str(group["InvoiceDate"].iloc[0]),
            "customer_id": int(group["CustomerID"].iloc[0]),
            "items": [
                {
                    "StockCode": str(row["StockCode"]),
                    "Description": str(row["Description"]),
                    "Quantity": int(row["Quantity"]),
                    "UnitPrice": float(row["UnitPrice"])
                }
                for idx, row in group.iterrows()
            ]
        })
    except Exception as e:
        print(f"Skipping invoice {invoice_no}: {e}")

try:
    result = invoices_col.insert_many(transactions, ordered=False)
    print(f"✅ Inserted {len(result.inserted_ids)} transaction-centric documents")
except BulkWriteError as e:
    skipped = len(e.details.get("writeErrors", []))
    inserted = len(transactions) - skipped
    print(f"⚠ Inserted {inserted}, skipped {skipped} duplicate transaction documents")

#Customer-centric collection
customers_col = db["customers_cust"]
customers = []
for cust_id, group in df.groupby("CustomerID"):
    try:
        customers.append({
            "_id": cust_id,
            "country": group["Country"].iloc[0],
            "invoices": [
                {
                    "invoice_no": str(row["InvoiceNo"]),
                    "invoice_date": str(row["InvoiceDate"]),
                    "items": [
                        {
                            "StockCode": str(row["StockCode"]),
                            "Description": str(row["Description"]),
                            "Quantity": int(row["Quantity"]),
                            "UnitPrice": float(row["UnitPrice"])
                        }
                    ]
                }
                for idx, row in group.iterrows()
            ]
        })
    except Exception as e:
        print(f"Skipping customer {cust_id}: {e}")

try:
    result = customers_col.insert_many(customers, ordered=False)
    print(f"✅ Inserted {len(result.inserted_ids)} customer-centric documents")
except BulkWriteError as e:
    skipped = len(e.details.get("writeErrors", []))
    inserted = len(customers) - skipped
    print(f"⚠ Inserted {inserted}, skipped {skipped} duplicate customer documents")
