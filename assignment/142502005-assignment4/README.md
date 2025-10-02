- uv sync 
- uv run python src/dataPrep.py
- uv run python src/mongoDb.py
- uv run python src/mongoBench.py
- uv run python src/mySql.py
- uv run python src/mySqlBench.py

Results are saved in files as follows:
- data has all the intermediate csv for 2NF 
- benchmark_results.csv contains timings for MongoDB dummy CRUD operations
- mysql_results.csv contains timings for MySQL dummy CRUD operations
- You can find comparision in ResultComparisions.pdf file.