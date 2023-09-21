import argparse
import sys
from datetime import date, timedelta
from azureml.core import Run

from pyspark.shell import spark
from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as f
from src.cp.dataOPs import AppDcc, Hem, Iptv
from src.cp.preprocessor.ModemCallETL import ModemCallETL
from src.cp.preprocessor.VerintETL import VerintETL
from src.cp.SparkOPs import DataBricks


def main(train_start_date, train_end_date, test_start_date, test_end_date):
    categories_filter = [
        "Cable Customer Frustration",
        "L1R - Tech Issues: Internet",
        "BP1 Technical Support",
        "BP3 Technical Support",
        "HOT TOPIC: National Outage",
        "Ignite TV",
        "L1R - Tech Issues: TV",
        "L2R - INT: Modem Offline",
        "HOT TOPIC: Outage Compensation",
        "L1F - Tech Issues: Internet",
        "L2R - General Inquiries: Internet"
    ]
    service_quality_df = Hem.load_data(Hem.modem_service_quality_table,
                                       condition="(channel_number =0)")
    accessibility_df = Hem.load_data(Hem.modem_accessibility_table)
    attainability_df = Hem.load_data(Hem.modem_attainability_table)
    intermittency_df = Hem.load_data(Hem.intermittency_daily_table)
    ium_df = Hem.load_data(Hem.resi_usage_ranked_daily_table)
    icm_cases_df = AppDcc.load_data(AppDcc.cbc_icm_cases_table)
    whix_df = Iptv.load_data(Iptv.ch_whix_daily_table)
    verint_df = VerintETL().etl(categories_filter)

    windows = [10]
    for window in windows:
        print(f"Window: {window}")
        cp_etl = ModemCallETL(modem_accessibility_df=accessibility_df,
                              modem_attainability_df=attainability_df,
                              modem_service_quality_df=service_quality_df,
                              intermittency_df=intermittency_df,
                              icm_cases_df=icm_cases_df,
                              resi_df=ium_df,
                              whix_df=whix_df,
                              agg_window=-14)
        df: DataFrame = cp_etl.etl()
        # bold_chat_df = bold_chat_etl()
        # df = df.join(bold_chat_df, on=["account_number", "event_date"], how="left").fillna({"bold_chat": 0})
        # df = bold_chat_history(df, "bold_chat")
        # df = bold_chat_history(df, "bold_chat", 20)
        # df = bold_chat_history(df, "bold_chat", 30)
        # df = df.where(f"event_date BETWEEN '{train_start_date}' AND '2022-11-01'")
        verint_df = verint_df.where(f"event_date BETWEEN '{train_start_date}' AND '{test_end_date}'")

        df = has_called(df, verint_df=verint_df)
        df = window_target(df, "called", window)
        # df: DataFrame = add_target_feature(df, icm_cases_df)
        train_df = df.where(f"event_date BETWEEN '{train_start_date}' AND '{train_end_date}'")
        test_df = df.where(f"event_date BETWEEN '{test_start_date}' AND '{test_end_date}'")
        test_df = test_df.limit(500000)
        train0 = train_df.filter(f'target=0')
        train1 = train_df.filter(f'target=1')

        train0: DataFrame = train0.limit(train1.count())
        # stack datasets back together
        train_df = train0.union(train1)

        print(f"Creating dataset from {train_start_date} to {train_end_date}")

        print('*** PRE-PROCESSED CP Dataset COMPLETE ***')
        # print("Number of cp dataset rows:", test_df.count())

        print('*** Writing FINAL CALL PREDICTION OUTPUT To Azure Storage Blob ***')
        train_df.write.mode("overwrite").option("header", True).parquet(train_output)
        test_df.write.mode("overwrite").option("header", True).parquet(test_output)
        print('*** Completed writing FINAL CALL PREDICTION OUTPUT To Azure Storage Blob ***')

        # time.sleep(10)
        # DataBricks.spark.stop()


def has_called(df: DataFrame, verint_df: DataFrame):
    df = df.join(verint_df, on=["account_number", "event_date"], how='left').fillna(
        {'called': 0, "category_name": "no_call"})

    return df


# def bold_chat_etl():
#     boldchat_df = DataBricks.spark.sql("select ecid as hash_account, chat_date as event_date from "
#                                        "salman_workspace.bold_id_session where event_date between '2022-09-01' and "
#                                        "'2022-11-01'")
#     accounts_df = DataBricks.spark.sql("select hash_account, account as account_number from data_pods.hash_lookup")
#     boldchat_df = boldchat_df.join(accounts_df, on=["hash_account"], how='inner')
#     boldchat_df = boldchat_df.withColumn("bold_chat", f.lit(1))
#     return boldchat_df.select("account_number, event_date, bold_chat")


# def bold_chat_history(df: DataFrame, column, window: int = 10):
#     days = lambda i: i * 86400
#     bold_chat_window = Window().partitionBy("account_number") \
#         .orderBy(f.col("event_date")
#                  .cast("timestamp")
#                  .cast("long")) \
#         .rangeBetween(days(-1), -days(window))

#     df = df.withColumn(f"bold_chat_{window}", f.sum(column).over(bold_chat_window))

#     return df


def window_target(df: DataFrame, column, window: int = 15):
    """

    :param column:
    :param df:
    :param window:
    :return:
    """
    days = lambda i: i * 86400
    call_count_w = Window().partitionBy("account_number") \
        .orderBy(f.col("event_date")
                 .cast("timestamp")
                 .cast("long")) \
        .rangeBetween(days(1), days(window))
    df = df.withColumn("window", f.lit(window))
    df = df.withColumn(f"target", f.sum(column).over(call_count_w))
    df = df.withColumn(f"target", f.when(f.col(f"target") > 0, 1).otherwise(0))
    return df


if __name__ == "__main__":
    print(sys.argv)
    parser = argparse.ArgumentParser("cp_dataset_build")
    parser.add_argument("--train_output")
    parser.add_argument("--test_output")
    parser.add_argument("--TRAIN_START_DATE")
    parser.add_argument("--TRAIN_END_DATE")
    parser.add_argument("--TEST_START_DATE")
    parser.add_argument("--TEST_END_DATE")
    parser.add_argument("--AZUREML_SCRIPT_DIRECTORY_NAME")
    parser.add_argument("--AZUREML_RUN_TOKEN")
    parser.add_argument("--AZUREML_RUN_TOKEN_EXPIRY")
    parser.add_argument("--AZUREML_RUN_ID")
    parser.add_argument("--AZUREML_ARM_SUBSCRIPTION")
    parser.add_argument("--AZUREML_ARM_RESOURCEGROUP")
    parser.add_argument("--AZUREML_ARM_WORKSPACE_NAME")
    parser.add_argument("--AZUREML_ARM_PROJECT_NAME")
    parser.add_argument("--AZUREML_SERVICE_ENDPOINT")
    parser.add_argument("--AZUREML_EXPERIMENT_ID")
    parser.add_argument("--AZUREML_WORKSPACE_ID")
    parser.add_argument("--MLFLOW_EXPERIMENT_ID")
    parser.add_argument("--MLFLOW_EXPERIMENT_NAME")
    parser.add_argument("--MLFLOW_RUN_ID")
    parser.add_argument("--MLFLOW_TRACKING_URI")
    parser.add_argument("--MLFLOW_TRACKING_TOKEN")
    args = parser.parse_args()
    print(args)
    train_output = args.train_output
    test_output = args.test_output
    test_start_date = args.TEST_START_DATE
    train_start_date = args.TRAIN_START_DATE
    test_end_date = args.TEST_END_DATE
    train_end_date = args.TRAIN_END_DATE

    main(train_start_date, train_end_date, test_start_date, test_end_date)
