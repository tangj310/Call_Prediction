from src.cp.SparkOPs import DataBricks
import logging

from src.cp.dataOPs.table_extract import TableExtract

logger = logging.getLogger("ium_DataOPs")


class Ium(TableExtract):
    """
        This class wraps data operations for hem database
    """

    ### Table names & queries on Databricks ###
    resi_usage_ranked_daily_table = "delta_ium.resi_usage_daily"
    mac_usage_fct_table = "ium.mac_usage_fct"
    all_tables = [resi_usage_ranked_daily_table, mac_usage_fct_table]
    columns = {resi_usage_ranked_daily_table: "cm_mac_addr as mac, event_date, hsi_down_1d_rank, hsi_up_1d_rank,"
                                              "IPTV_Down_1D_rank, IPTV_Up_1D_rank, "
                                              "RHP_Down_1D_rank, RHP_Up_1D_rank"}

    @staticmethod
    def load_data(table, condition=None):
        """
        Loading ium tables data to spark dataframes

        :return: spark dataframe
        """

        query = f"select {Ium.columns[table]} from {table}"
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
        queries = {table: f"select {Ium.columns[table]} from {table}" for table in Ium.all_tables}
        logger.info(f"reading all hem tables from Databricks.")
        logger.debug(f"running queries: {[queries[table] for table in queries]}\n")

        return {table: DataBricks.spark.sql(queries[table]) for table in queries}
