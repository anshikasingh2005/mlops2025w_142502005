import wandb
import json
import os
import sys

# --- CONFIGURATION ---
# We just need the project path, not the specific sweep ID
PROJECT_PATH = "142502029-iit-palakkad/NCERT-RAG-Sweep"
METRIC_TO_OPTIMIZE = "average_score" # The metric we want to sort by
OUTPUT_FILE = "best_config.json"

print(f"Connecting to W&B project: {PROJECT_PATH}")

try:
    # 1. Login to W&B API
    api = wandb.Api()

    # 2. Get all runs from the project, sorted by our metric
    #    - We add a filter to only include runs that
    #      successfully logged our metric (ignoring crashes).
    #    - We use 'order' to put the best run at the top.
    print(f"Fetching best run, ordered by '{METRIC_TO_OPTIMIZE}' (descending)...")
    runs = api.runs(
        path=PROJECT_PATH,
        order=f"-summary_metrics.{METRIC_TO_OPTIMIZE}",
        filters={"summary_metrics." + METRIC_TO_OPTIMIZE: {"$exists": True}}
    )

    # 3. Check if we found any runs
    if not runs:
        print(f"Error: No successful runs found in project '{PROJECT_PATH}'.")
        print("Please run the 'run_sweep.py' script first.")
        sys.exit(1) # Exit with an error

    # 4. Get the best run (it's the first one in the sorted list)
    best_run = runs[0]

    print(f"---")
    print(f"✅ Found best run: {best_run.name}")
    print(f"Score ({METRIC_TO_OPTIMIZE}): {best_run.summary[METRIC_TO_OPTIMIZE]:.2f}")
    print(f"Config: {best_run.config}")
    print(f"---")

    # 5. Save this run's config to a file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(best_run.config, f, indent=2)

    print(f"✅ Successfully saved best config to {OUTPUT_FILE}")

except Exception as e:
    print(f"Error connecting to W&B or finding runs.")
    print(f"Error: {e}")
    sys.exit(1)