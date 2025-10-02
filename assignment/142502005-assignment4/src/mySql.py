import mysql.connector
import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data"

# Connect to MySQL (without selecting DB first)
conn = mysql.connector.connect(
    host="localhost",
    user="retail_user",
    password="retail_pass"
)
cur = conn.cursor()

# Create DB if not exists
cur.execute("CREATE DATABASE IF NOT EXISTS online_retail")
conn.database = "online_retail"

# Create tables (2NF schema)
DDL = """
CREATE TABLE IF NOT EXISTS customers (
  customer_id INT PRIMARY KEY,
  country VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS products (
  stock_code VARCHAR(20) PRIMARY KEY,
  description TEXT,
  default_unit_price DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS invoices (
  invoice_no VARCHAR(20) PRIMARY KEY,
  invoice_date DATETIME,
  customer_id INT,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS invoice_items (
  id INT AUTO_INCREMENT PRIMARY KEY,
  invoice_no VARCHAR(20),
  stock_code VARCHAR(20),
  quantity INT,
  unit_price DECIMAL(10,2),
  FOREIGN KEY (invoice_no) REFERENCES invoices(invoice_no),
  FOREIGN KEY (stock_code) REFERENCES products(stock_code)
);
"""
for stmt in DDL.strip().split(";"):
    if stmt.strip():
        cur.execute(stmt)
conn.commit()


def load_csv(table, file, cols, colmap=None):
    """
    Load CSV into table.
    cols   = DB column names
    colmap = mapping from DB col → CSV col (if headers differ)
    """
    with open(file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            # Map CSV headers → DB cols if needed
            if colmap:
                row = tuple(r[colmap[c]] for c in cols)
            else:
                row = tuple(r[c] for c in cols)
            rows.append(row)

    placeholders = ",".join(["%s"] * len(cols))
    cur.executemany(
        f"INSERT IGNORE INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    print(f"Inserted into {table}: {len(rows)} rows")


# === Load data into each table ===
load_csv(
    "customers",
    DATA_PATH / "customers.csv",
    ["customer_id", "country"],
    {"customer_id": "CustomerID", "country": "Country"}
)

load_csv(
    "products",
    DATA_PATH / "products.csv",
    ["stock_code", "description", "default_unit_price"],
    {"stock_code": "StockCode", "description": "Description", "default_unit_price": "DefaultUnitPrice"}
)

load_csv(
    "invoices",
    DATA_PATH / "invoices.csv",
    ["invoice_no", "invoice_date", "customer_id"],
    {"invoice_no": "InvoiceNo", "invoice_date": "InvoiceDate", "customer_id": "CustomerID"}
)

load_csv(
    "invoice_items",
    DATA_PATH / "invoice_items.csv",
    ["invoice_no", "stock_code", "quantity", "unit_price"],
    {"invoice_no": "InvoiceNo", "stock_code": "StockCode", "quantity": "Quantity", "unit_price": "UnitPrice"}
)

cur.close()
conn.close()
