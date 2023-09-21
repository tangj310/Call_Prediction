from src.cp.SparkOPs import DataBricks
import logging

from src.cp.dataOPs.table_extract import TableExtract

logger = logging.getLogger(__name__)


class AppDcc(TableExtract):
    """
        This class wraps data operations for AppDcc database
    """

    # Table names & queries on Databricks
    cbc_icm_cases_table = "app_dcc.cbc_icm_case"
    cbc_icm_table = "app_dcc.cbc_icm"

    columns = {cbc_icm_table: f"*",
               cbc_icm_cases_table: "ban as account_number, creation_time, CASE_TYPE_LVL1, X_CASE_TYPE_LVL5, "
                                    "CASE_TYPE_LVL2, CASE_TYPE_LVL3 "}

    @staticmethod
    def load_data(table, condition=None):
        """
        Load specific table with conditions
        :param table:
        :param condition:
        :return:
        """
        query = f"select {AppDcc.columns[table]} from {table}"
        if condition is not None:
            query = f"{query} where {condition}"

        logger.info(f"reading {table} from Databricks.")
        logger.debug(f"running query: {query}\n")

        return DataBricks.spark.sql(query)

    @staticmethod
    def load_all_data():
        """
        Loading all tables from hem into pyspark dataframes
        :return: list of pyspark dataframes
        """
        queries = {table: f"select {AppDcc.columns[table]} from {table}" for table in AppDcc.columns}
        logger.info(f"reading all hem tables from Databricks.")
        logger.debug(f"running queries: {[queries[table] for table in queries]}\n")

        return {table: DataBricks.spark.sql(queries[table]) for table in queries}