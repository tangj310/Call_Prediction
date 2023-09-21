from src.cp.SparkOPs import DataBricks
import logging

from src.cp.dataOPs.table_extract import TableExtract

logger = logging.getLogger("iptv_DataOPs")


class Iptv(TableExtract):
    """
        This class wraps data operations for iptv database
    """

    ### Table names & queries on Databricks ###
    ch_whix_daily_table = "iptv.ch_whix_daily"
    all_tables = [ch_whix_daily_table]
    columns = {ch_whix_daily_table: "GW_MAC_ADDRESS as gw_mac, event_date, AVG_DAY_WIFI_HAPPINESS_INDEX/100 as "
                                    "whix_daily, "
                                    "AVG_DAILY_WIFI_DISCONNECTS as daily_wifi_disconnects, COUNT_DAILY_GW_WIFI_TOOHOT"
                                    }

    @staticmethod
    def load_data(table, condition=None):
        """
        Loading Iptv tables data to spark dataframes

        :return: spark dataframe
        """

        query = f"select {Iptv.columns[table]} from {table}"
        if condition is not None:
            query = f"{query} where {condition}"

        logger.info(f"reading {table} from Databricks.")
        logger.debug(f"running query: {query}\n")

        return DataBricks.spark.sql(query)

    @staticmethod
    def load_all_data():
        """
        Loading all tables from iptv into pyspark dataframes
        :return: list of pyspark dataframes
        """
        queries = {table: f"select {Iptv.columns[table]} from {table}" for table in Iptv.all_tables}
        logger.info(f"reading all iptv tables from Databricks.")
        logger.debug(f"running queries: {[queries[table] for table in queries]}\n")

        return {table: DataBricks.spark.sql(queries[table]) for table in queries}
