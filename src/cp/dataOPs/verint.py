from src.cp.SparkOPs import DataBricks
import logging

from src.cp.dataOPs.table_extract import TableExtract

logger = logging.getLogger("verint_DataOPs")


class Verint(TableExtract):
    """
        This class wraps data operations for verint database
    """

    ### Table names & queries on Databricks ###
    sessions_categories_table = "verint.sessions_categories"
    categories_table = "verint.categories"
    sessions_booked_table = "verint.sessions_booked"
    cbu_rog_conv_sumfct_table = "verint.cbu_rog_conversation_sumfct"
    all_tables = [sessions_booked_table, categories_table,sessions_categories_table,cbu_rog_conv_sumfct_table]
    columns = {sessions_categories_table: "distinct sid, category_key",
               categories_table: "distinct category_key, category_name",
               sessions_booked_table: "distinct CONCAT(unit_num, '0', channel_num) as speech_id_verint, sid, unit_num",
               cbu_rog_conv_sumfct_table: "distinct SPEECH_ID_VERINT, \
                                           LEFT(speech_id_verint,6) as unit_num_verint, \
                                           RIGHT(speech_id_verint,9) as channel_num_verint, \
                                           customer_id as account_number, \
                                           conversation_date as event_date"}

    @staticmethod
    def load_data(table, condition=None):
        """
        Loading verint tables data to spark dataframes

        :return: spark dataframe
        """

        query = f"select {Verint.columns[table]} from {table}"
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
        queries = {table: f"select {Verint.columns[table]} from {table}" for table in Verint.all_tables}
        logger.info(f"reading all hem tables from Databricks.")
        logger.debug(f"running queries: {[queries[table] for table in queries]}\n")

        return {table: DataBricks.spark.sql(queries[table]) for table in queries}
