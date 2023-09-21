from abc import ABC, abstractmethod

import confuse
from azureml.core import Environment, ComputeTarget, Workspace, Experiment
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.compute import DatabricksCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.exceptions import ComputeTargetException
from azureml.pipeline.core import PipelineEndpoint


class CPPipeline(ABC):

    def __init__(self, conf_file, azure_conf, deployment=False):
        """
        Abstact class for CallPrediction Pipelines
        :param conf_file:
        :param azure_conf:
        """

        self.config = confuse.Configuration("batch_inference_pipeline_conf", __name__)
        self.config.set_file(conf_file)

        if deployment:
            cli_auth = AzureCliAuthentication()
            self.aml_workspace = Workspace.from_config(path=azure_conf, auth=cli_auth)
        else:
            self.aml_workspace = Workspace.from_config(azure_conf)

        self.endpoint = self.get_endpoint(self.config["endpoint_name"].get(str))

        self.conda_packages = self.config['conda_packages'].get(list)
        self.pip_packages = self.config['pip_packages'].get(list)
        self.experiment_name = self.config['experiment_name'].get(str)
        self.python_version = self.config["python_version"].get(str)
        self.published_pipeline = None
        self.pipeline = None

    def _get_aml_compute(self, aml_compute_name):
        """
        Get the compute target for azure ml
        :param aml_compute_name:
        :return: ComputeTarget
        """
        try:
            aml_compute = ComputeTarget(workspace=self.aml_workspace, name=aml_compute_name)
            print('AML Compute target {} already exists'.format(aml_compute_name))
            return aml_compute
        except ComputeTargetException:
            print('AML Compute not found!')
            return ComputeTargetException

    def _get_db_compute(self, db_compute_name):
        """
        Get the databricks compute
        :param db_compute_name:
        :return:
        """
        try:
            databricks_compute = DatabricksCompute(workspace=self.aml_workspace, name=db_compute_name)
            print('Compute target {} already exists'.format(db_compute_name))
            return databricks_compute
        except ComputeTargetException:
            print('Databricks Compute not found!')
        return ComputeTargetException

    def _get_or_create_env(self, env_name):
        """
        Create or get environment if it exists
        :param env_name:
        :return:
        """
        env = Environment(env_name)
        packages = CondaDependencies.create(conda_packages=self.conda_packages,
                                            pip_packages=self.pip_packages,
                                            python_version=self.python_version)
        env.python.conda_dependencies = packages

        env.docker.enabled = True
        env.docker.base_image = DEFAULT_CPU_IMAGE
        env.register(workspace=self.aml_workspace)  # Register the environment
        env = Environment.get(self.aml_workspace, env_name)
        return env

    def set_published_pipeline(self):

        self.published_pipeline = self.endpoint.get_pipeline()

    def get_endpoint(self, endpoint_name):

        names = set([x.name for x in PipelineEndpoint.list(workspace=self.aml_workspace)])
        if endpoint_name in names:
            endpoint = PipelineEndpoint.get(workspace=self.aml_workspace, name=endpoint_name)

            return endpoint
        else:
            return None

    def run(self):
        """
        Run the pipeline experiment and wait for results
        :return:
        """
        if self.pipeline is not None:
            pipeline_run = Experiment(self.aml_workspace, self.experiment_name).submit(self.pipeline)
            # pipeline_run.wait_for_completion(show_output=True) # comment out so that VM agent will not time out

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def publish(self):
        pass

    @abstractmethod
    def schedule(self):
        pass
