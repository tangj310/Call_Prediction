from src.cp.SparkOPs import DataBricks
import logging

from src.cp.dataOPs.table_extract import TableExtract

logger = logging.getLogger("bold_chat_DataOPs")


class BoldChat(TableExtract):
    """
        This class wraps data operations for customer chat  details
    """

    ### Table names & queries on Databricks ###
    boldchat_table = "salman_workspace.bold_id_session"
    accounts_table = "data_pods.hash_lookup"
    live_sas_table = "BOLDCHAT.LIVECHAT_SAS_SUMMARY"

    all_tables = [boldchat_table, accounts_table, live_sas_table]
    columns = {boldchat_table: "ecid as hash_account, date_part as chat_date, bold_id",
               accounts_table: "hash_account, account as account_number",
               live_sas_table: "session_id as bold_id, message_text, message_technician, message_customer"}

    @staticmethod
    def load_data(table, condition=None):
        """
        Loading bold_chat tables data to spark dataframes

        :return: spark dataframe
        """
        query = f"SELECT {BoldChat.columns[table]} FROM {table}"

        if condition is not None:
            query = f"{query} where {condition}"

        logger.info(f"reading {table} from Databricks.")
        logger.debug(f"running queries: {query}\n")

        return DataBricks.spark.sql(query)


    @classmethod
    def load_all_data(cls):
        """
        Loading all tables from bolchat sessions into pyspark dataframes
        :return: list of pyspark dataframes
        """
        queries = []
        for table in cls.all_tables:
            query = f"SELECT * FROM {table}"
            queries.append(query)
        logger.info(f"reading all boldchat sessions tables from Databricks.")
        logger.debug(f"running queries: {[query for query in queries]}\n")

        return [DataBricks.spark.sql(query) for query in queries]