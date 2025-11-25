import argparse

parser = argparse.ArgumentParser(description="Data Warehouse Assignment")

parser.add_argument("-t", "--type", choices=["clustering", "etl"], help="Type of analysis to perform on the data.")

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
    else:
        print("Running Decision Support System...")