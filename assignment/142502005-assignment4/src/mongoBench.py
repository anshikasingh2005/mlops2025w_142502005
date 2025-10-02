import time
import csv
import pymongo
from pymongo.errors import ConnectionFailure
from pathlib import Path


# MongoDB connection setup
try:
    client = pymongo.MongoClient(
        "mongodb://localhost:27017",
        maxPoolSize=50,
        serverSelectionTimeoutMS=5000
    )
    client.admin.command("ping")
    print("✅ Connected to MongoDB")
except ConnectionFailure as e:
    print("❌ MongoDB connection failed:", e)
    exit(1)

db = client["online_retail_db"]

# Transaction-centric collection
invoices_col = db["invoices_txn"]


# Utility: measure execution time
def timed(fn, n=3):
    times = []
    for _ in range(n):
        start = time.time()
        fn()
        times.append(time.time() - start)
    mean = sum(times) / n
    return mean, times


# Dummy invoice document
dummy_invoice = {
    "_id": "dummy_0001",
    "invoice_date": "2025-10-02 10:00:00",
    "customer_id": 99999,
    "items": [
        {"StockCode": "DUMMY001", "Description": "Dummy Product A", "Quantity": 1, "UnitPrice": 9.99},
        {"StockCode": "DUMMY002", "Description": "Dummy Product B", "Quantity": 2, "UnitPrice": 19.99}
    ]
}

# Benchmark operations
benchmark_results = []

# Insert (skip if already exists)
def insert_op():
    if not invoices_col.find_one({"_id": dummy_invoice["_id"]}):
        invoices_col.insert_one(dummy_invoice)

mean_insert, times_insert = timed(insert_op)
benchmark_results.append(["Insert", mean_insert, times_insert])
print(f"Insert avg time: {mean_insert:.6f} sec, runs={times_insert}")

# Read
mean_read, times_read = timed(lambda: list(invoices_col.find({"_id": dummy_invoice["_id"]})))
benchmark_results.append(["Read", mean_read, times_read])
print(f"Read avg time: {mean_read:.6f} sec, runs={times_read}")

# Update
mean_update, times_update = timed(
    lambda: invoices_col.update_one({"_id": dummy_invoice["_id"]}, {"$set": {"bench_flag": True}})
)
benchmark_results.append(["Update", mean_update, times_update])
print(f"Update avg time: {mean_update:.6f} sec, runs={times_update}")

# Delete
mean_delete, times_delete = timed(lambda: invoices_col.delete_one({"_id": dummy_invoice["_id"]}))
benchmark_results.append(["Delete", mean_delete, times_delete])
print(f"Delete avg time: {mean_delete:.6f} sec, runs={times_delete}")


# Save benchmark results to CSV
csv_path = Path(__file__).resolve().parents[1] / "benchmark_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Operation", "Average Time (sec)", "Run Times (sec)"])
    writer.writerows(benchmark_results)

print(f"✅ Benchmark results saved to {csv_path}")
