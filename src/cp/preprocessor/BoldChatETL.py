import logging

from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from pyspark.sql import Window

from src.cp.dataOPs.boldchat import BoldChat
from src.cp.dataOPs.app_dcc import AppDcc

class BoldChatETL:

    def __init__(self):
        """
        Constructor for Boldchat ETL
        """

        self.logger = logging.getLogger(__name__)
        self.bold_chat_sessions_df: DataFrame = BoldChat.load_data(BoldChat.boldchat_table)
        self.accounts_df: DataFrame = BoldChat.load_data(BoldChat.accounts_table)
        self.live_sas_df: DataFrame = BoldChat.load_data(BoldChat.live_sas_table)
        self.icm_cases_df = AppDcc.load_data(AppDcc.cbc_icm_cases_table)
        self.__df = self.bold_chat_sessions_df.join(self.accounts_df, on=["hash_account"]) \
                                              .join(self.live_sas_df, on=["bold_id"])

    def __chats_today(self, df):
        """
        Function to calculate the number of times account has chatted today
        :param df:
        :return:
        """

        chat_today_w = Window().partitionBy("account_number") \
            .orderBy(f.col("creation_time")
                     .cast("timestamp")
                     .cast("long")) \
            .rangeBetween(0, 0)
        has_chatted = self.icm_cases_df.select("account_number", "creation_time"). \
            withColumn(f"chat_count_0", f.count("creation_time").over(chat_today_w))
        has_chatted = has_chatted.withColumnRenamed("account_number", "account_number_icm")   
        df = df.join(has_chatted, (df.chat_date == has_chatted.creation_time) & \
                                  (df.account_number == has_chatted.account_number_icm), how='left'). \
                                  fillna({f'chat_count_0': 0})
        
        return df


    def __chat_history(self, window: int = 60):
        """
        Function to add chat history feature to dataframe.
        To be used with the transform feature of pyspark
        :param window: int
        :return:
        """

        def transform(df: DataFrame):
            days = lambda i: i * 86400
            chat_count_w = Window().partitionBy("account_number") \
                .orderBy(f.col("chat_date")
                         .cast("timestamp")
                         .cast("long")) \
                .rangeBetween(-days(window), -days(1))

            df = df.withColumn(f"chat_count_{window}", f.sum("chat_count_0").over(chat_count_w))
            return df

        return transform


    def etl(self):

        """
        ETL function to return a final pyspark dataframe for training dataset
        :return:Dataframe
        """
        final_cols = ["account_number", "chat_date", "chat_count_0", "chat_count_10", "chat_count_30", "chat_count_60",
                      "chat_count_120", "message_text", "message_technician", "message_customer"]

        self.__df = self.__df.transform(self.__chats_today)
        self.__df = self.__df.transform(self.__chat_history())
        self.__df = self.__df.transform(self.__chat_history(10))
        self.__df = self.__df.transform(self.__chat_history(30))
        self.__df = self.__df.transform(self.__chat_history(120))
        self.__df = self.__df.withColumnRenamed('chat_date', 'event_date')
        
        return self.__df.select(*final_cols)
