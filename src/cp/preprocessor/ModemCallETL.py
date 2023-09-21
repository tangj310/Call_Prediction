import logging

from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from pyspark.sql import Window


class ModemCallETL:
    agg_dict = {"outage_count": ["avg", "var", "kurtosis"], "codeword_error_rate": ["avg", "var", "kurtosis"],
                "cmts_cm_us_rx_power_avg": ["avg", "var", "kurtosis"],
                "cmts_cm_us_signal_noise_avg": ["avg", "var", "kurtosis"],
                "ACCESSIBILITY_PERC_FULL_DAY": ["avg", "var", "kurtosis"],
                "US_SPEED_ATTAINABLE_FULL": ["avg", "var", "kurtosis"],
                "DS_SPEED_ATTAINABLE_FULL": ["avg", "var", "kurtosis"], "attainability_pct": ["avg", "var", "kurtosis"],
                "down_gb": ["avg", "var", "kurtosis"], "up_gb": ["avg", "var", "kurtosis"],
                "Intermittent_hrs": ["avg", "var", "kurtosis"], "transitions": ["avg", "var", "kurtosis"],
                "Offline_counts": ["avg", "var", "kurtosis"], "hsi_down_1d_rank": ["avg", "var", "kurtosis"],
                "hsi_up_1d_rank": ["avg", "var", "kurtosis"], "IPTV_Down_1D_rank": ["avg", "var", "kurtosis"],
                "IPTV_Up_1D_rank": ["avg", "var", "kurtosis"], "RHP_Down_1D_rank": ["avg", "var", "kurtosis"],
                "RHP_Up_1D_rank": ["avg", "var", "kurtosis"],
                "whix_daily": ["avg", "var", "kurtosis"],
                "daily_wifi_disconnects": ["avg", "var", "kurtosis"],
                "COUNT_DAILY_GW_WIFI_TOOHOT": ["avg", "var", "kurtosis"]}

    def __init__(self, modem_accessibility_df: DataFrame, modem_service_quality_df: DataFrame,
                 modem_attainability_df: DataFrame, intermittency_df: DataFrame,
                 icm_cases_df: DataFrame, resi_df: DataFrame, whix_df, agg_window: int):
        """


        :param modem_accessibility_df:
        :param modem_service_quality_df:
        :param modem_attainability_df:
        """

        self.logger = logging.getLogger(__name__)
        self.icm_cases_df: DataFrame = icm_cases_df
        self.__df: DataFrame = modem_accessibility_df.join(modem_service_quality_df, ['mac', 'event_date']) \
            .join(modem_attainability_df, ["mac", "event_date"]).join(intermittency_df, ["mac", "event_date"]).join(
            resi_df, on=["mac", "event_date"])
        self.__df = self.__df.withColumn("gw_mac", f.upper(f.regexp_replace("mac", "-", "")))
        self.__df = self.__df.join(whix_df, on=["gw_mac", "event_date"], how='left').fillna(
            {"whix_daily": 1, "daily_wifi_disconnects": 0,
             "COUNT_DAILY_GW_WIFI_TOOHOT": 0})

        self.__agg_window = Window.partitionBy("account_number").orderBy("event_date").rowsBetween(agg_window, 0)

    def __apply_agg(self, column_name, agg):
        new_column = f"{column_name}_{agg}"

        def transform(df):
            if agg == "avg":
                return df.withColumn(new_column, f.avg(column_name).over(self.__agg_window))
            elif agg == "var":
                return df.withColumn(new_column, f.avg(f.pow(f.col(column_name), 2)).over(self.__agg_window) -
                                     f.pow(f.col(f"{column_name}_avg"), 2))
            elif agg == "sum":
                return df.withColumn(new_column, f.sum(column_name).over(self.__agg_window))
            elif agg == "max":
                return df.withColumn(new_column, f.max(column_name).over(self.__agg_window))
            elif agg == "kurtosis":
                return df.withColumn(new_column, f.avg(f.pow(f.col(column_name), 4)).over(self.__agg_window))

            else:
                return df

        return transform

    def __calls_today(self, df):

        """
        Function to calculate the number of times account has called today
        :param df:
        :return:
        """

        call_today_w = Window().partitionBy("account_number") \
            .orderBy(f.col("creation_time")
                     .cast("timestamp")
                     .cast("long")) \
            .rangeBetween(0, 0)
        has_called = self.icm_cases_df.select("account_number", "creation_time"). \
            withColumn(f"call_count_0", f.count("creation_time").over(call_today_w))
        has_called = has_called.withColumnRenamed("creation_time", "event_date")
        df = df.join(has_called, on=["account_number", "event_date"], how='left').fillna({f'call_count_0': 0})
        return df

    def __call_history(self, window: int = 60):
        """
        Function to add call history feature to dataframe.
        To be used with the transform feature of pyspark
        :param window: int
        :return:
        """

        def transform(df: DataFrame):
            days = lambda i: i * 86400
            call_count_w = Window().partitionBy("account_number") \
                .orderBy(f.col("event_date")
                         .cast("timestamp")
                         .cast("long")) \
                .rangeBetween(-days(window), -days(1))

            df = df.withColumn(f"call_count_{window}", f.sum("call_count_0").over(call_count_w))
            return df

        return transform

    # def __target_feature(self,df:DataFrame):
    #     """
    #     Function to add the target feature (call or no call) to
    #     dataframe. To be used with the transform feature of pyspark
    #     :param df: Dataframe
    #     :return:
    #     """
    #     case_lvl1_filter = ['Rogers GCL Only - Cable', 'Tech - Cable', 'Tech - Rogers - IPTV',
    #                         'Tech - Rogers - Residential', 'Tech - Rogers – Résidentiel',
    #                         'Tech. Câble']
    #
    #     case_lvl5_filter = ['Ticket', 'Billet', 'N/A', 'S/O', 'Truck', 'Camion', 'Service Request',
    #                         'Demande de service']
    #     icm_filtered_df = self.icm_cases_df.filter(f.col("CASE_TYPE_LVL1").isin(case_lvl1_filter)).filter(
    #         f.col("X_CASE_TYPE_LVL5").isin(case_lvl5_filter))
    #     icm_filtered_df = icm_filtered_df.select("account_number", "X_CASE_TYPE_LVL5 ")
    #
    #     df = df.join(icm_filtered_df, on=[icm_filtered_df["creation_time"] == f.date_add(df["event_date"],1)], how='left')
    #     df = df.withColumn('target', f.when(f.col("X_CASE_TYPE_LVL5").isNull(), 0).otherwise(1))
    #
    #     return df

    def etl(self):

        """
        ETL function to return a final pyspark dataframe for training dataset
        :return:Dataframe
        """
        final_cols = ["account_number", "event_date", "call_count_0", "call_count_30", "call_count_60", "call_count_10",
                      "call_count_120"]
        for col in self.agg_dict:

            for agg_func in self.agg_dict[col]:
                new_column_name = f"{col}_{agg_func}"
                final_cols.append(new_column_name)

                self.__df = self.__df.transform(self.__apply_agg(col, agg_func))
        self.__df = self.__df.transform(self.__calls_today)
        self.__df = self.__df.transform(self.__call_history())
        self.__df = self.__df.transform(self.__call_history(30))
        self.__df = self.__df.transform(self.__call_history(10))
        self.__df = self.__df.transform(self.__call_history(120))
        return self.__df.select(*final_cols)