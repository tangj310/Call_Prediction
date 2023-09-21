from src.cp.SparkOPs import DataBricks
import logging

from src.cp.dataOPs.table_extract import TableExtract

logger = logging.getLogger("hem_DataOPs")


class Hem(TableExtract):
    """
        This class wraps data operations for hem database
    """

    ### Table names & queries on Databricks ###
    modem_accessibility_table = "delta_hem.modem_accessibility_daily"
    modem_attainability_table = "delta_hem.modem_attainability_daily"
    modem_service_quality_table = "delta_hem.modem_service_quality_daily"
    intermittency_daily_table = "delta_hem.intermittency_daily"
    resi_usage_ranked_daily_table = "delta_hem.resi_usage_daily"
    mac_usage_fct_table = "delta_hem.mac_usage_fct"

    all_tables = [modem_accessibility_table, modem_attainability_table, modem_service_quality_table,
                  intermittency_daily_table, resi_usage_ranked_daily_table, mac_usage_fct_table]
    columns = {modem_service_quality_table: "mac, channel_number , codeword_error_rate,"
                                            "cmts_cm_us_signal_noise_avg/364 as cmts_cm_us_signal_noise_avg,"
                                            "cmts_cm_us_rx_power_avg/10 as cmts_cm_us_rx_power_avg,"
                                            "cmts_cm_us_signal_noise_max/380 as cmts_cm_us_signal_noise_max,"
                                            "cmts_cm_us_rx_power_max/12 as cmts_cm_us_rx_power_max, "
                                            "cmts_cm_us_signal_noise_sum/1000 as cmts_cm_us_signal_noise_sum,"
                                            "cmts_cm_us_rx_power_sum/200 as cmts_cm_us_rx_power_sum,event_date",
               modem_attainability_table: "CM_MAC_ADDR as mac, US_SPEED_ATTAINABLE_FULL , "
                                          "DS_SPEED_ATTAINABLE_FULL, Attainability_Pct , event_date,"
                                          "UP_MBYTES/1000 as up_gb, DOWN_MBYTES/1000 as down_gb",
               modem_accessibility_table: "mac, FULL_ACCOUNT_NUMBER as account_number, outage_count, "
                                          "ACCESSIBILITY_PERC_FULL_DAY/100 as ACCESSIBILITY_PERC_FULL_DAY, "
                                          "ACCESSIBILITY_PERC_PRIME/100 as ACCESSIBILITY_PERC_PRIME, event_date",
               intermittency_daily_table: "cm_mac_addr as mac, Intermittent_hrs, transitions, Offline_counts,"
                                          "event_date",
               resi_usage_ranked_daily_table: "cm_mac_addr as mac, event_date, hsi_down_1d_rank, hsi_up_1d_rank,"
                                              "IPTV_Down_1D_rank, IPTV_Up_1D_rank, "
                                              "RHP_Down_1D_rank, RHP_Up_1D_rank"
               }

    @staticmethod
    def load_data(table, condition=None):
        """
        Loading ela_rcis tables data to spark dataframes

        :return: spark dataframe
        """

        query = f"select {Hem.columns[table]} from {table}"
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
        queries = {table: f"select {Hem.columns[table]} from {table}" for table in Hem.all_tables}
        logger.info(f"reading all hem tables from Databricks.")
        logger.debug(f"running queries: {[queries[table] for table in queries]}\n")

        return {table: DataBricks.spark.sql(queries[table]) for table in queries}