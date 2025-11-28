import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Data Warehouse Assignment")

parser.add_argument("-t", "--type", choices=["clustering", "etl", "dss"], help="Type of analysis to perform on the data.")

args = parser.parse_args()

if __name__ == "__main__":
    if args.type == "clustering":
        from src.pipelines.clustering import ClusteringPipeline

        clustering_pipeline = ClusteringPipeline(warehouse_schema_name="dwh")
        clustering_pipeline.run()
    elif args.type == "etl":
        from src.pipelines.etl import ETLPipeline

        etl_pipeline = ETLPipeline(warehouse_schema_name="dwh")
        etl_pipeline.run()
    elif args.type == "dss":
        print("Running Decision Support System...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "Clustering_Analysis.py"])
    else:
        print("Please specify a type: clustering, etl, or dss")
