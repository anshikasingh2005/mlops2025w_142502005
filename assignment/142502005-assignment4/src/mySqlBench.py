import mysql.connector, time, statistics, csv

# MySQL connection
conn = mysql.connector.connect(
    host="localhost",
    user="retail_user",
    password="retail_pass",
    database="online_retail"
)
cur = conn.cursor()

# Utility: measure execution time
def timed(fn, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    return mean, times

# Pick a sample invoice_no
cur.execute("SELECT invoice_no FROM invoices LIMIT 1")
sample_invoice = cur.fetchone()[0]

# Benchmark operations

# Insert dummy product and invoice_item
def insert_item():
    # Insert dummy product if not exists
    cur.execute("""
        INSERT IGNORE INTO products (stock_code, description, default_unit_price)
        VALUES (%s, %s, %s)
    """, ("DUMMY001", "Dummy Product", 9.99))
    
    # Insert dummy invoice item
    cur.execute("""
        INSERT INTO invoice_items (invoice_no, stock_code, quantity, unit_price)
        VALUES (%s, %s, %s, %s)
    """, (sample_invoice, "DUMMY001", 1, 9.99))
    conn.commit()

# Read invoice items
def read_invoice():
    cur.execute("SELECT * FROM invoice_items WHERE invoice_no=%s", (sample_invoice,))
    cur.fetchall()

# Update invoice item
def update_invoice_item():
    cur.execute("""
        UPDATE invoice_items SET quantity = quantity + 1
        WHERE invoice_no=%s AND stock_code='DUMMY001' LIMIT 1
    """, (sample_invoice,))
    conn.commit()

# Delete invoice item
def delete_invoice_item():
    cur.execute("""
        DELETE FROM invoice_items
        WHERE invoice_no=%s AND stock_code='DUMMY001' LIMIT 1
    """, (sample_invoice,))
    conn.commit()

# Run benchmarks
benchmark_results = []

mean_insert, runs_insert = timed(insert_item)
benchmark_results.append(["Insert", mean_insert, runs_insert])

mean_read, runs_read = timed(read_invoice)
benchmark_results.append(["Read", mean_read, runs_read])

mean_update, runs_update = timed(update_invoice_item)
benchmark_results.append(["Update", mean_update, runs_update])

mean_delete, runs_delete = timed(delete_invoice_item)
benchmark_results.append(["Delete", mean_delete, runs_delete])

# Save results to CSV (same format as MongoDB)
with open("mysql_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Operation", "Average Time (sec)", "Run Times (sec)"])
    for op, mean, runs in benchmark_results:
        writer.writerow([op, mean, str(runs)])

# Cleanup
cur.close()
conn.close()

print("✅ MySQL benchmark completed. Results saved to mysql_results.csv")
