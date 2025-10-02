import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)
INFILE = DATA_PATH / "Online Retail.xlsx"

def clean_and_split():
    df = pd.read_excel(INFILE, engine="openpyxl")
    df = df.dropna(subset=["InvoiceNo", "StockCode"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Quantity"] = df["Quantity"].astype(int)
    df["UnitPrice"] = df["UnitPrice"].astype(float)

    # Products
    products = df[["StockCode", "Description", "UnitPrice"]].drop_duplicates("StockCode")
    products = products.rename(columns={"UnitPrice": "DefaultUnitPrice"})
    products.to_csv(DATA_PATH / "products.csv", index=False)

    # Customers
    customers = df[["CustomerID", "Country"]].dropna().drop_duplicates("CustomerID")
    customers.to_csv(DATA_PATH / "customers.csv", index=False)

    # Invoices
    invoices = df[["InvoiceNo", "InvoiceDate", "CustomerID"]].drop_duplicates("InvoiceNo")
    invoices.to_csv(DATA_PATH / "invoices.csv", index=False)

    # Invoice Items
    items = df[["InvoiceNo", "StockCode", "Quantity", "UnitPrice"]]
    items.to_csv(DATA_PATH / "invoice_items.csv", index=False)

    print("CSV files written in /data")

if __name__ == "__main__":
    clean_and_split()

