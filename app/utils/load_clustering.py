import pandas as pd

from src.config.warehouse_connection import postgre_engine

clusters_data_df = pd.read_sql(f'SELECT * FROM dwh."ClustersData"', postgre_engine)