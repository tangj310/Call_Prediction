import argparse
import json
import sys
from datetime import date, timedelta
from azureml.core import Run

from src.cp.dataOPs.iptv import Iptv
from src.cp.preprocessor import VerintETL
from pyspark.sql import DataFrame
from src.cp.dataOPs import AppDcc, Hem
from src.cp.preprocessor.ModemCallETL import ModemCallETL


def data_quality_check(dataframes, inference_date):
    missing_data = False
    error_message = ""
    for name in dataframes:
        df = dataframes[name]
        if df.filter(df["event_date"] == inference_date).count() == 0:
            missing_data = True
            error_message += f"{name} df is missing data for {inference_date}\n"
            print(f"{name} df is missing data for {inference_date}")

    return missing_data, error_message


def main():
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
    dataframes = {"svq": service_quality_df, "accessibility": accessibility_df,
                  "attainability": attainability_df, "intermittency": intermittency_df,
                  "resi": ium_df, "verint": verint_df}

    if inference_date_stamp == "Today":
        inference_date = date.today() - timedelta(2)
    else:
        inference_date = date.fromisoformat(inference_date_stamp)

    missing_data, error_message = data_quality_check(dataframes, inference_date)
    if missing_data:
        print("Empty Dataframe, missing data")
        run.fail(error_code="missing_data",error_details=error_message)
        sys.exit()
    else:
        cp_etl = ModemCallETL(modem_accessibility_df=accessibility_df,
                              modem_attainability_df=attainability_df,
                              modem_service_quality_df=service_quality_df,
                              intermittency_df=intermittency_df,
                              icm_cases_df=icm_cases_df,
                              resi_df=ium_df,
                              whix_df=whix_df,
                              agg_window=-14)

        df: DataFrame = cp_etl.etl()
        df = df.filter(f"event_date='{inference_date}'")
        verint_df = verint_df.filter(f"event_date='{inference_date}'")
        df = has_called(df, verint_df)

        df = df.drop_duplicates(['account_number'])
        df = df.fillna(0)
        print('*** ETL COMPLETE ***')
        # df = spark.createDataFrame(df.rdd, schema=new_schema)
        print('*** Writing FINAL CALL PREDICTION OUTPUT To Azure Storage Blob ***')
        df.write.mode("overwrite").option("header", True).parquet(output_blob_folder)
        print("Finished writing output")


def has_called(df: DataFrame, verint_df: DataFrame):
    df = df.join(verint_df, on=["account_number", "event_date"], how='left').fillna({'called': 0,
                                                                                     'category_name': 'no_call'})

    return df


if __name__ == "__main__":
    print(sys.argv)
    parser = argparse.ArgumentParser("cp_etl_test")
    parser.add_argument("--INFERENCE_DATE")
    parser.add_argument("--etl_output")
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
    run = Run.get_context()
    args = parser.parse_args()
    output_blob_folder = args.etl_output
    inference_date_stamp = args.INFERENCE_DATE
    main()
