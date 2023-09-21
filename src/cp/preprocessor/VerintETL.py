import logging

from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from pyspark.sql import Window

from src.cp.dataOPs.verint import Verint


class VerintETL:

    def __init__(self):
        """
        Constructor for Verint ETL
        """

        self.logger = logging.getLogger(__name__)
        self.sessions_categories_df: DataFrame = Verint.load_data(Verint.sessions_categories_table)
        self.sessions_booked_df: DataFrame = Verint.load_data(Verint.sessions_booked_table)
        self.categories_df: DataFrame = Verint.load_data(Verint.categories_table)
        self.cbu_rog_conv_sumfct_df: DataFrame = Verint.load_data(Verint.cbu_rog_conv_sumfct_table)

    def etl(self, categories_filter):
        """
        ETL function to return a final pyspark dataframe for training dataset
        :return:Dataframe
        """
        
        cat_name_df = self.sessions_categories_df.join(self.categories_df, on=["category_key"], how="left"). \
            filter(f.col("category_name").isin(categories_filter))
        # cat_name_df = cat_name_df.select("sid", "category_name").distinct()

        # Verint join before and equal to 2022-12-09
        inter_df_1 = self.cbu_rog_conv_sumfct_df.join(self.sessions_booked_df, on=["speech_id_verint"])
        final_df_1: DataFrame = inter_df_1.join(cat_name_df, on=["sid"])
        final_df_1 = final_df_1.withColumn("called", f.lit(1))
        final_df_1 = final_df_1.select("account_number", "event_date","category_name","called")

        # Verint join after 2022-12-09
        inter_df_2 = self.cbu_rog_conv_sumfct_df.join(self.sessions_booked_df, \
            (self.cbu_rog_conv_sumfct_df.unit_num_verint == self.sessions_booked_df.unit_num) & \
            (self.cbu_rog_conv_sumfct_df.channel_num_verint == self.sessions_booked_df.sid)
            )
        final_df_2: DataFrame = inter_df_2.join(cat_name_df, on=["sid"])
        final_df_2 = final_df_2.withColumn("called", f.lit(1))
        final_df_2 = final_df_2.select("account_number", "event_date","category_name","called")

        final_df = final_df_1.union(final_df_2)

        return final_df.distinct()