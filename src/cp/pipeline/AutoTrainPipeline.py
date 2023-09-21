from azure.core.pipeline import Pipeline
from azureml.core import RunConfiguration, Datastore
from azureml.core.databricks import EggLibrary, PyPiLibrary
from azureml.pipeline.core import Pipeline, PipelineEndpoint, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep, DatabricksStep
from azureml.data import OutputFileDatasetConfig

from src.cp.pipeline.CPPipeline import CPPipeline


class AutoTrainPipeline(CPPipeline):

    def schedule(self):
        pass

    def __init__(self, conf_file, azure_conf, deployment=False):
        """
        Training Pipeline class for building, publishing and scheduling training pipeline

        :param conf_file: str
        The path to the pipeline config file
        :param azure_conf: str
        The path to azure config
        """
        super().__init__(conf_file, azure_conf, deployment)
        self.repo_link = self.config["repo_link"].get(str)
        self.train_start_date = self.config["train_start_date"].get(str)
        self.train_end_date = self.config["train_end_date"].get(str)
        self.test_start_date = self.config["test_start_date"].get(str)
        self.test_end_date = self.config["test_end_date"].get(str)
        self.cluster_logs_path = self.config["cluster_logs_path"].get(str)
        self.pypi_packages = self._get_pypi_packages(self.config["pypi_packages"].get(list))
        self.egg_lib = EggLibrary(library=self.config['egg_lib_path'].get(str))
        self.compute_target = self._get_aml_compute(self.config['aml_compute_target'].get(str))
        self.db_compute = self._get_db_compute(self.config['db_attached_compute'].get(str))
        self.dataset_etl_step_script = self.config["dataset_etl_step_script"].get(str)
        self.etl_datastore = Datastore.get(self.aml_workspace, self.config['etl_datastore'].get(str))
        self.aml_env = self._get_or_create_env(env_name=self.config['env_name'].get(str))
        self.source_directory = self.config['script_dir'].get(str)
        self.training_script_name = self.config['training_script_name'].get(str)
        self.autotrain_score_datastore = Datastore.get(self.aml_workspace, self.config['derived_scores_datastore'].get(str))
        self.output_dir_autotrain_score = self.config['output_dir_autotrain_score'].get(str)

    def build(self):
        """
        Build pipeline and steps, set self.pipeline to built pipeline
        :return:
        """
        pipeline_run_config = RunConfiguration()
        pipeline_run_config.target = self.compute_target
        pipeline_run_config.environment = self.aml_env
        train_start = PipelineParameter(name="train_start_date", default_value=self.train_start_date)
        train_end = PipelineParameter(name="train_end_date", default_value=self.train_end_date)
        test_start = PipelineParameter(name="test_start_date", default_value=self.test_start_date)
        test_end = PipelineParameter(name="test_end_date", default_value=self.test_end_date)
        train_output = PipelineData("train_output", self.etl_datastore).as_dataset()
        test_output = PipelineData("test_output", self.etl_datastore).as_dataset()

        etl_step = DatabricksStep(name="DB_dataset_creation_step",
                                  source_directory=self.source_directory,
                                  python_script_name=self.dataset_etl_step_script,
                                  run_name='cp_etl_run',
                                  compute_target=self.db_compute,
                                  allow_reuse=False,
                                  egg_libraries=[self.egg_lib],
                                  outputs=[train_output, test_output],
                                  node_type="Standard_F32s_v2",
                                  min_workers=1,
                                  max_workers=5,
                                  spark_version="7.3.x-scala2.12",
                                  spark_env_variables={
                                      "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                                      "spark.kryoserializer.buffer.max": "2000M"},
                                  pypi_libraries=self.pypi_packages,
                                  cluster_log_dbfs_path=self.cluster_logs_path,
                                  python_script_params=[train_start,
                                                        train_end,
                                                        test_start,
                                                        test_end])


        output_dir_autotrain_score = OutputFileDatasetConfig(name="output_dir_autotrain_score_output",
                                                       destination=(self.autotrain_score_datastore, self.output_dir_autotrain_score))
        
        train_step = PythonScriptStep(
            source_directory=self.source_directory,
            name="CP training",
            script_name=self.training_script_name,
            arguments=['--output_dir_autotrain_score', output_dir_autotrain_score.as_upload(overwrite=True),
                       '--TRAIN_START_DATE', train_start,
                       '--TRAIN_END_DATE', train_end,
                       '--TEST_START_DATE', test_start,
                       '--TEST_END_DATE', test_end],
            inputs=[train_output.parse_parquet_files(), test_output.parse_parquet_files()],
            compute_target=self.compute_target,
            runconfig=pipeline_run_config,
            allow_reuse=False)
        
        # Run the pipeline
        training_pipeline = Pipeline(workspace=self.aml_workspace, steps=[etl_step, train_step])
        self.pipeline = training_pipeline

    def publish(self):
        """
        Publish pipeline to endpoint and return published endpoint
        :return:
        """
        self.published_pipeline = self.pipeline.publish("CP_Auto_Training_Pipeline",
                                                        "Call Prediction Training Pipeline")

        if self.endpoint is None:
            self.endpoint = PipelineEndpoint.publish(workspace=self.aml_workspace, name=self.config["endpoint_name"],
                                                     description="CP Training Pipeline",
                                                     pipeline=self.published_pipeline)
        else:
            self.endpoint.add_default(self.published_pipeline)

    def _get_pypi_packages(self, pypi_packages):
        """
        Return list of PyPiLibraries for dbstep
        :param pypi_packages:
        :return:
        """

        return [PyPiLibrary(package=x, repo=self.repo_link) for x in pypi_packages]
