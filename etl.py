from src.pipelines.etl import ETLPipeline

etl_pipeline = ETLPipeline(warehouse_schema_name="dwh")
etl_pipeline.run()