import sys

sys.path.append("/Users/Masih.Sultani/Desktop/CallPrediction/src")
from rogers.cp.preprocessor import VerintETL
from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as f
from rogers.cp.dataOPs import AppDcc, Hem
from rogers.cp.SparkOPs import DataBricks
from rogers.cp.preprocessor.ModemCallETL import ModemCallETL


def main():
    service_quality_df = Hem.load_data(Hem.modem_service_quality_table,
                                       condition="(channel_number =0)")
    accessibility_df = Hem.load_data(Hem.modem_accessibility_table)
    attainability_df = Hem.load_data(Hem.modem_attainability_table)
    intermittency_df = Hem.load_data(Hem.intermittency_daily_table)
    ium_df = Hem.load_data(Hem.resi_usage_ranked_daily_table)
    icm_cases_df = AppDcc.load_data(AppDcc.cbc_icm_cases_table)
    windows = [15]
    for window in windows:
        print(f"Window: {window}")
        cp_etl = ModemCallETL(modem_accessibility_df=accessibility_df,
                              modem_attainability_df=attainability_df,
                              modem_service_quality_df=service_quality_df,
                              intermittency_df=intermittency_df,
                              icm_cases_df=icm_cases_df,
                              resi_df=ium_df,
                              agg_window=-20)
        df: DataFrame = cp_etl.etl()
        df = df.where("event_date BETWEEN '2021-08-05' AND '2021-10-15'")

        df = window_target(df, "target_icm", window)
        # df: DataFrame = add_target_feature(df, icm_cases_df)
        train_df = df.where("event_date BETWEEN '2021-09-05' AND '2021-09-15'")
        train_df = train_df.sampleBy(f'target', {1: 0.2, 0: 0.2})
        test_df = df.where("event_date BETWEEN '2021-09-16' AND '2021-09-20'")
        test_df = test_df.sampleBy(f'target', {1: 0.05, 0: 0.05})
        print("Creating dataset from 2021-09-05 to 2021-09-20")

        print('*** PRE-PROCESSED CP Dataset COMPLETE ***')
        # print("Number of cp dataset rows:", test_df.count())

        train0 = train_df.filter(f'target=0')
        train1 = train_df.filter(f'target=1')

        test0 = test_df.filter(f'target=0')
        test1 = test_df.filter(f'target=1')
        # split datasets into training and testing
        # val0, test0 = test0.randomSplit([0.3, 0.7], seed=1234)
        # val1, test1 = test1.randomSplit([0.3, 0.7], seed=1234)
        # train1, val1, test1 = ones.randomSplit([0.7, 0.1, 0.2], seed=1234)
        train0: DataFrame = train0.sample(withReplacement=False, fraction=0.5) \
            .limit(train1.select("account_number").count())
        # stack datasets back together
        train = train0.union(train1)
        # val = val0.union(val1)
        test = test0.union(test1)
        storage_name = "mazcacnpedigitaldls01"
        container_name = "ml-etl-output-data"
        db = DataBricks(container_name, storage_name)
        print('*** Writing FINAL CALL PREDICTION OUTPUT To Azure Storage Blob ***')
        db.write_to_storage(train, f'call_prediction/cp_train_ium_15')
        db.write_to_storage(test, f'call_prediction/cp_test_ium_15')
        # db.write_to_storage(test, 'call_prediction/call_prediction_test_wdate')
        print('*** Completed writing FINAL CALL PREDICTION OUTPUT To Azure Storage Blob ***')

        # time.sleep(10)
        # DataBricks.spark.stop()


def icm_window_target_feature(df: DataFrame, icm_cases_df: DataFrame):
    """
    Function to add the target feature (call or no call) to
    dataframe. To be used with the transform feature of pyspark
    :param icm_cases_df:
    :param df: Dataframe
    :return:
    """

    case_lvl2_filter = ['Internet']

    case_lvl3_filter = ['RF Signal', 'Network Performance', 'Signal RF', 'Modem']
    icm_filtered_df = icm_cases_df.filter(f.col("CASE_TYPE_LVL2").isin(case_lvl2_filter)).filter(
        f.col("CASE_TYPE_LVL3").isin(case_lvl3_filter))
    icm_filtered_df = icm_filtered_df.select("account_number", "CASE_TYPE_LVL3", "creation_time").distinct()
    icm_filtered_df = icm_filtered_df.withColumnRenamed("creation_time", "event_date")

    df = df.join(icm_filtered_df, on=["account_number", "event_date"], how='left')
    df = df.withColumn('target_icm', f.when(f.col("CASE_TYPE_LVL3").isNull(), 0).otherwise(1))
    df = df.drop("CASE_TYPE_LVL3")
    return df


def has_called(df: DataFrame, verint_df: DataFrame):
    verint_df = verint_df.withColumnRenamed("target", "called")

    df = df.join(verint_df, on=["account_number", "event_date"], how='left').fillna({'called': 0})

    return df


def window_target(df: DataFrame, column, window: int = 10, ):
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
    main()
