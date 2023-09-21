from azure.core.pipeline import Pipeline
from azureml.core import Datastore, RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, Schedule, PipelineEndpoint

from src.cp.pipeline.CPPipeline import CPPipeline


class TrainingPipeline(CPPipeline):

    def __init__(self, conf_file, azure_conf):
        """
        Training Pipeline class for building, publishing and scheduling training pipeline

        :param conf_file: str
        The path to the pipeline config file
        :param azure_conf: str
        The path to azure config
        """
        super().__init__(conf_file, azure_conf)
        self.aml_env = self._get_or_create_env(env_name=self.config['env_name'].get(str))
        self.compute_target = self._get_aml_compute(self.config['compute_target'].get(str))
        self.source_directory = self.config['script_dir'].get(str)
        self.training_script_name = self.config['training_script_name'].get(str)

    def build(self):
        """
        Build pipeline and steps, set self.pipeline to built pipeline
        :return:
        """
        pipeline_run_config = RunConfiguration()
        pipeline_run_config.target = self.compute_target
        pipeline_run_config.environment = self.aml_env

        train_step = PythonScriptStep(
            source_directory=self.source_directory,
            name="CP training",
            script_name=self.training_script_name,
            compute_target=self.compute_target,
            runconfig=pipeline_run_config,
            allow_reuse=False
        )
        training_pipeline = Pipeline(workspace=self.aml_workspace, steps=[train_step])
        self.pipeline = training_pipeline

    def schedule(self, name):
        """
        Create a schedule to track path on datastore and run pipeline when new data is uploaded
        :param name:
        :return:
        """
        datastore = Datastore(workspace=self.aml_workspace, name=self.config["datastore"])

        Schedule.create(self.aml_workspace, name=name,
                        description="Based on input file change.",
                        pipeline_id=self.published_pipeline.id,
                        experiment_name=self.experiment_name,
                        datastore=datastore, path_on_datastore="call_prediction/cp_test_latest")

    def publish(self):
        """
        Publish pipeline to endpoint and return published endpoint
        :return:
        """
        self.published_pipeline = self.pipeline.publish("CP_Training_Pipeline", "Call Prediction Training Pipeline")

        if self.endpoint is None:
            self.endpoint = PipelineEndpoint.publish(workspace=self.aml_workspace, name=self.config["endpoint_name"],
                                                     description="CP Training Pipeline",
                                                     pipeline=self.published_pipeline)
        else:
            self.endpoint.add_default(self.published_pipeline)
