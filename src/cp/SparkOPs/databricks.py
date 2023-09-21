import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession


class DataBricks:
    """
        This Static class provides support for connecting the project to the Azure Databricks
        environment.
    """

    ###  Spark Session  ###
    spark = SparkSession.builder \
        .config("spark.driver.memory", "16G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.1.0") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")\
        .config('spark.driver.extraJavaOptions', '-Xss32M')\
        .getOrCreate()

    def __init__(self, container_name, storage_name):
        """

        :param container_name:
        :param storage_name:
        """
        self.__storage_name = storage_name
        self.__container_name = container_name

    def write_to_storage(self, df, folder, mode="overwrite", header=True):
        """
        Writing spark dataframe to Azure Blob Storage in Parquet Format
        :param df: Spark Dataframe to be written to storage
        :param folder: custom folder in blob storage containing the dataOPs
        :param mode: overwrite, append, etc.
        :param header: True/False
        """

        # sas_key = "BGU/Sg/TS40/bywqYA88MxU+cRV45uGZhj4X7v0iLPtQscClTQ6FgczU+0crsMj8gnDskj0NYTaitifNZiNDkA=="
        load_dotenv()
        sas_key = os.environ.get("storage_account_key")
        DataBricks.spark.conf.set(
            f"fs.azure.account.key.{self.__storage_name}.blob.core.windows.net",
            sas_key)

        output_container_path = f"wasbs://{self.__container_name}@{self.__storage_name}.blob.core.windows.net"
        output_blob_folder = f"{output_container_path}/{folder}"

        df \
            .coalesce(1) \
            .write \
            .mode(mode) \
            .option("header", header) \
            .parquet(output_blob_folder)
