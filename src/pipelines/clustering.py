from src.config.warehouse_connection import postgre_engine
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateSchema

import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

MATRIX_FEATURES = ['profit', 'profit_margin', 'average_unit_price', 'avg_quantity_sold', 
            'nunique_customer', 'customer_loyalty', 'revenue_growth', 'selling_duration']

N_CLUSTERS = 4

class PreprocessingUtils:
    def __init__(self, dimProductPriceCostHistory_df: pd.DataFrame, dimDate_df: pd.DataFrame):
        self.dimProductPriceCostHistory_df = dimProductPriceCostHistory_df
        self.dimDate_df = dimDate_df

        self._intervals_mapping()
        self._standard_cost_mapping()

    def _intervals_mapping(self) -> None:
        self.intervals = {}
        for _, row in self.dimDate_df.iterrows():
            id, interval_start, interval_end = row['Id'], row['StartDate'], row['EndDate']
            if interval_end is pd.NaT:
                interval_end = datetime.max
            self.intervals[id] = (interval_start, interval_end)

    def _standard_cost_mapping(self) -> None:
        self.standard_cost_mapping = {}
        for _, row in self.dimProductPriceCostHistory_df.iterrows():
            product_id = row['ProductID']
            interval_id = row['Interval']
            standard_cost = row['StandardCost']
            self.standard_cost_mapping[(product_id, interval_id)] = standard_cost

    def get_interval_id(self, order_date: datetime) -> int:
        """
        Get the interval ID for a given order date.
        """
        for id, (start_date, end_date) in self.intervals.items():
            if start_date <= order_date <= end_date:
                return id
        return None

    def get_standard_cost(self, product_id: int, order_date: datetime) -> float:
        """
        Get the standard cost for a given product ID and order date.
        """
        interval_id = self.get_interval_id(order_date)
        return self.standard_cost_mapping.get((product_id, interval_id), None)

class ClusteringPipeline:
    def __init__(self, warehouse_schema_name: str, wh_engine: Engine = postgre_engine):
        self.wh_engine = wh_engine
        self.warehouse_schema_name = warehouse_schema_name
        self.logger = logging.getLogger(__name__)

    def _extract_data(self, dim: str) -> pd.DataFrame:
        query = f'SELECT * FROM {self.warehouse_schema_name}."{dim}"'
        return pd.read_sql(query, self.wh_engine)

    def run(self):
        self.logger.info("Running clustering pipeline...")

        ###################
        self.logger.info("Stage 1/3: Extracting data from data warehouse...")

        dimProductPriceCostHistory_df = self._extract_data('DimProductPriceCostHistory')
        dimDate_df = self._extract_data('DimDate')
        factProductSales_df = self._extract_data('FactProductSales')
        self.processing_util = PreprocessingUtils(dimProductPriceCostHistory_df, dimDate_df)

        factProductSales_df['IntervalID'] = factProductSales_df['OrderDate'].apply(self.processing_util.get_interval_id)
        factProductSales_df['StandardCost'] = factProductSales_df.apply(lambda row: self.processing_util.get_standard_cost(row['ProductID'], row['OrderDate']), axis=1)
        factProductSales_df['profit'] = factProductSales_df['LineTotal'] - factProductSales_df['StandardCost'] * factProductSales_df['OrderQty']
        factProductSales_df = factProductSales_df.dropna(subset=['StandardCost', 'IntervalID'])
        factProductSales_df.drop(columns=['IntervalID', 'Interval'], inplace=True)

        ###################
        self.logger.info("Stage 2/3: Do feature extraction...")
        FINAL_DATE = factProductSales_df['OrderDate'].max() + pd.Timedelta(days=1)      # datetime(2014, 7, 1)
        ACTIVE_TIME_MOCK = FINAL_DATE - pd.Timedelta(days=365)

        print("Get the up-to-date active information from {}...".format(ACTIVE_TIME_MOCK))
        filtered_df = factProductSales_df[factProductSales_df['OrderDate'] >= ACTIVE_TIME_MOCK]
        print(f"Number of active products: {filtered_df['ProductID'].nunique()}")

        product_df = filtered_df.groupby('ProductID').aggregate({
            'OrderQty': 'sum',
            'LineTotal': 'sum',
            'profit': 'sum',
            'CustomerID': 'nunique',
            'SalesOrderID': 'nunique'
        }).rename(columns={
            'CustomerID': 'nunique_customer',
            'SalesOrderID': 'order_frequency',
        })

        product_df['average_unit_price'] = product_df['LineTotal'] / product_df['OrderQty']
        product_df['profit_margin'] = product_df['profit'] / product_df['LineTotal']
        product_df['customer_loyalty'] = product_df['order_frequency'] / product_df['nunique_customer']

        # Retrieve the performance growth features
        first_date = factProductSales_df.groupby('ProductID').aggregate({
            'OrderDate': 'min'
        })['OrderDate']

        first_date = (FINAL_DATE - first_date).dt.days

        product_df['selling_duration'] = product_df.index.map(first_date)
        product_df['overall_revenue'] = factProductSales_df.groupby('ProductID').aggregate({
            'LineTotal': 'sum'
        })['LineTotal']

        product_df['overall_revenue'] = (product_df['overall_revenue'] / product_df['selling_duration'] * 365)
        product_df['revenue_growth'] = product_df['LineTotal'] / product_df['overall_revenue']
        product_df['avg_quantity_sold'] = product_df['OrderQty'] / product_df['order_frequency']

        ###################
        self.logger.info("Stage 3/3: Perform clustering...")
        feature_matrix = product_df[MATRIX_FEATURES].copy()
        data = feature_matrix.to_numpy()
        from sklearn.preprocessing import StandardScaler
        data = StandardScaler().fit_transform(data)

        from sklearn.cluster import KMeans
        predicts = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit_predict(data)

        feature_matrix['cluster'] = predicts

        self.logger.info("Clustering pipeline finished. Save results to data warehouse...")
        feature_matrix.to_sql('product_clustering', self.wh_engine, schema=self.warehouse_schema_name, if_exists='replace')