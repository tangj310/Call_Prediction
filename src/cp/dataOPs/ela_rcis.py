from src.cp.SparkOPs import DataBricks
import logging

from src.cp.dataOPs.table_extract import TableExtract

logger = logging.getLogger("ela_rcis_DataOPs")


class ElaRcis(TableExtract):
    """
        This class wraps data operations for customer account details
    """

    ### Table names & queries on Databricks ###
    ec_account_table = "ela_rcis.ec_account"
    ec_cust_account_table  = "ela_rcis.ec_cust_acct"
    all_tables = [ec_account_table, ec_cust_account_table]

    @staticmethod
    def load_data(table, condition=None):
        """
        Loading ela_rcis tables data to spark dataframes

        :return: spark dataframe
        """
        query = f"SELECT * FROM {table}"

        if condition is not None:
            query = f"{query} where {condition}"

        logger.info(f"reading {table} from Databricks.")
        logger.debug(f"running queries: {query}\n")

        return DataBricks.spark.sql(query)


    @classmethod
    def load_all_data(cls):
        """
        Loading all tables from ela_rcis into pyspark dataframes
        :return: list of pyspark dataframes
        """
        queries = []
        for table in cls.all_tables:
            query = f"SELECT * FROM {table}"
            queries.append(query)
        logger.info(f"reading all ela_rcis tables from Databricks.")
        logger.debug(f"running queries: {[query for query in queries]}\n")

        return [DataBricks.spark.sql(query) for query in queries]