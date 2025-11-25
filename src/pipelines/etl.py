from src.config.database_connection import mssql_engine
from src.config.warehouse_connection import postgre_engine
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateSchema
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)

class ETLPipeline:
    def __init__(self, warehouse_schema_name: str, db_engine: Engine = mssql_engine, wh_engine: Engine = postgre_engine):
        self.db_engine = db_engine
        self.wh_engine = wh_engine
        self.warehouse_schema_name = warehouse_schema_name
        self.logger = logging.getLogger(__name__)

    def _extract_data(self, query: str) -> pd.DataFrame:
        query = query.strip()
        return pd.read_sql(query, self.db_engine)

    def _load_data(self, data: pd.DataFrame, target_table: str) -> None:
        data.to_sql(target_table, self.wh_engine, schema=self.warehouse_schema_name, if_exists='append', index=False)


    def _assign_interval(self, fact_df: pd.DataFrame, dim_df: pd.DataFrame, date_col="OrderDate") -> pd.DataFrame:
        dim = dim_df.sort_values("StartDate")
        fact = fact_df.sort_values(date_col)

        merged = pd.merge_asof(
            fact,
            dim,
            left_on=date_col,
            right_on="StartDate",
            direction="backward"
        )

        return merged[
            (merged[date_col] >= merged["StartDate"]) &
            ((merged["EndDate"].isna()) | (merged[date_col] <= merged["EndDate"]))
        ]

    def run(self, override_exists: bool = True) -> None:
        self.logger.info("ETL Pipeline started ! ")

        ##################
        ### Extract phase
        # Load table Product
        self.logger.info("Stage 1/3: Extracting data from source database...")
        dimProduct_df = self._extract_data("""SELECT ProductID, Name, ProductSubcategoryID, FinishedGoodsFlag FROM CompanyX.Production.Product""")
        dimProduct_df = dimProduct_df[dimProduct_df['FinishedGoodsFlag'] == 1].reset_index(drop=True).drop(columns=['FinishedGoodsFlag'])
        salable_products = list(dimProduct_df['ProductID'].unique())

        # Load table ProductSubCategory
        dimProductSubcategory_df = self._extract_data("""SELECT ProductSubcategoryID, Name, ProductCategoryID FROM CompanyX.Production.ProductSubcategory""")

        # Load table ProductCategory
        dimProductCategory_df = self._extract_data("""SELECT ProductCategoryID, Name FROM CompanyX.Production.ProductCategory""")

        # Load table ProductCostHistory, ProductListPriceHistory
        input_pch_df = self._extract_data("""SELECT ProductID, StartDate, EndDate, StandardCost FROM CompanyX.Production.ProductCostHistory""")
        input_plph_df = self._extract_data("""SELECT ProductID, StartDate, EndDate, ListPrice FROM CompanyX.Production.ProductListPriceHistory""")
        merge_df = pd.merge(input_pch_df, input_plph_df, on=['ProductID', 'StartDate', 'EndDate'], how='inner')
        merge_df = merge_df[merge_df['ProductID'].isin(salable_products)].reset_index(drop=True)

        # Load table SaleOrderDetail and SalesOrderHeader
        input_soh_df = self._extract_data("""SELECT SalesOrderID, OrderDate, CustomerID FROM CompanyX.Sales.SalesOrderHeader""")
        input_sod_df = self._extract_data("""SELECT ProductID, OrderQty, LineTotal, SalesOrderID FROM CompanyX.Sales.SalesOrderDetail""")
        input_sod_df = pd.merge(input_sod_df, input_soh_df[['SalesOrderID', 'OrderDate', 'CustomerID']], on='SalesOrderID', how='left')

        # Only keep salable products
        input_sod_df = input_sod_df[input_sod_df['ProductID'].isin(salable_products)].reset_index(drop=True)

        ##################
        ### Transform phase
        self.logger.info("Stage 2/3: Transforming data...")
        dimDate_df = merge_df[['StartDate', 'EndDate']].copy()
        dimDate_df.drop_duplicates(inplace=True)
        dimDate_df['Id'] = dimDate_df.index + 1

        # ProductPriceCostHistory
        dimProductPriceCostHistory_df = pd.merge(merge_df, right=dimDate_df, on=['StartDate', 'EndDate'], how='inner').drop(columns=['StartDate', 'EndDate'])
        dimProductPriceCostHistory_df.rename(columns={'Id': 'Interval'}, inplace=True)

        factProductSales_df = self._assign_interval(input_sod_df, dimDate_df, date_col="OrderDate").drop(columns=['StartDate', 'EndDate'])
        factProductSales_df.rename(columns={'Id': 'Interval'}, inplace=True)
        factProductSales_df.dropna(inplace=True)                                # Deal with missing value: simply drop

        ##################
        ### Load phase
        self.logger.info("Stage 3/3: Loading data into data warehouse...")
        with postgre_engine.connect() as connection:
            try:
                connection.execute(CreateSchema(self.warehouse_schema_name, if_not_exists=override_exists))
                connection.commit()
                print(f"Schema '{self.warehouse_schema_name}' created successfully (or already exists).")
            except Exception as e:
                connection.rollback()
                print(f"Error creating schema '{self.warehouse_schema_name}': {e}")

        dimDate_df.to_sql('DimDate', postgre_engine, schema=self.warehouse_schema_name, if_exists='replace', index=False)
        dimProduct_df.to_sql('DimProduct', postgre_engine, schema=self.warehouse_schema_name, if_exists='replace', index=False)
        dimProductSubcategory_df.to_sql('DimProductSubcategory', postgre_engine, schema=self.warehouse_schema_name, if_exists='replace', index=False)
        dimProductCategory_df.to_sql('DimProductCategory', postgre_engine, schema=self.warehouse_schema_name, if_exists='replace', index=False)
        dimProductPriceCostHistory_df.to_sql('DimProductPriceCostHistory', postgre_engine, schema=self.warehouse_schema_name, if_exists='replace', index=False)
        factProductSales_df.to_sql('FactProductSales', postgre_engine, schema=self.warehouse_schema_name, if_exists='replace', index=False)

        self.logger.info("ETL Pipeline finished ! ")