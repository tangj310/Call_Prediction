from azureml.core import Datastore
from azureml.core.databricks import PyPiLibrary
from azureml.core.runconfig import EggLibrary
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig

from azureml.pipeline.core import Pipeline, PipelineData, Schedule, ScheduleRecurrence, TimeZone, PipelineParameter
from azureml.pipeline.core.pipeline_output_dataset import PipelineOutputFileDataset
from azureml.pipeline.steps import DatabricksStep, PythonScriptStep
from azureml.pipeline.core import PipelineEndpoint
from src.cp.pipeline.CPPipeline import CPPipeline
from datetime import timezone


class BatchInferencePipeline(CPPipeline):

    def __init__(self, conf_file, azure_conf, deployment=False):
        """
        Batch Inference pipeline class

        :param conf_file: str
        The path to the pipeline config file
        :param azure_conf: str
        The path to azure config
        """

        super().__init__(conf_file, azure_conf, deployment)
        self.spark_conf = self.config['spark_conf'].get(dict)
        self.spark_version = self.config['spark_version'].get(str)
        self.node_type = self.config['node_type'].get(str)
        self.num_workers = self.config['num_workers'].get(int)
        self.instance_pool_id = self.config["instance_pool_id"].get(str)
        self.repo_link = self.config["repo_link"].get(str)
        self.pypi_packages = self._get_pypi_packages(self.config["pypi_packages"].get(list))
        self.schedule_frequency = self.config["schedule"]["frequency"].get(str)
        self.start_time = self.config["schedule"]["start_time"].get(str)
        self.schedule_name = self.config["schedule"]["schedule_name"].get(str)
        self.schedule_interval = self.config['schedule']['schedule_interval'].get(int)
        self.aml_schedule_timezone = self.config["schedule"]['schedule_timezone'].get(str)
        self.model_name = self.config["model_name"].get(str)
        self.output_dir = self.config["output_dir"].get(str)
        self.aml_env = self._get_or_create_env(env_name=self.config['env_name'].get(str))
        self.compute_target = self._get_aml_compute(self.config['aml_compute_target'].get(str))

        self.db_compute = self._get_db_compute(self.config['db_attached_compute'].get(str))
        self.source_directory = self.config['script_dir'].get(str)
        self.etl_step_script_name = self.config['etl_step_script_name'].get(str)
        self.infer_save_step_script_name = self.config['inference_save_step_script_name'].get(str)
        self.egg_lib = EggLibrary(library=self.config['egg_lib_path'].get(str))
        self.etl_datastore = Datastore.get(self.aml_workspace, self.config['etl_datastore'].get(str))
        self.inference_datastore = Datastore.get(self.aml_workspace, self.config['derived_scores_datastore'].get(str))
        self.cluster_log_path = self.config["cluster_logs_path"].get(str)
        self.inference_date = self.config["inference_date"].get(str)

# Local Exp Inference Pythonstep yaml
        self.local_exp_save_step_script_name = self.config['local_exp_save_step_script_name'].get(str)
        self.output_dir_local_exp = self.config['output_dir_local_exp'].get(str)

# Etl output store in datastore yaml
        self.etl_output_save_step_script_name = self.config['etl_output_save_step_script_name'].get(str)
        self.output_dir_elt_store = self.config['output_dir_elt_store'].get(str)

    def build(self):
        """
        Build pipeline and steps, set self.pipeline to built pipeline
        :return:
        """
        print("execute function")
        run_config = RunConfiguration()
        run_config.target = self.compute_target
        run_config.environment = self.aml_env

        inference_date = PipelineParameter(name="inference_date",
                                           default_value=self.inference_date)

        # DB Step
        etl_out_dir = PipelineData("etl_output", 
                                   datastore=self.etl_datastore,
                                   output_mode="upload",
                                   output_path_on_compute=self.output_dir_elt_store,
                                   output_overwrite=True).as_dataset()


        etl_step = DatabricksStep(name="DB_etl_step",
                                  source_directory=self.source_directory,
                                  python_script_name=self.etl_step_script_name,
                                  run_name='etl_run',
                                  compute_target=self.db_compute,
                                  allow_reuse=False,
                                  outputs=[etl_out_dir],
                                  egg_libraries=[self.egg_lib],
                                  node_type=self.node_type,
                                  # instance_pool_id=self.instance_pool_id,
                                  min_workers=2,
                                  max_workers=10,
                                  spark_version=self.spark_version,
                                  spark_env_variables=self.spark_conf,
                                  pypi_libraries=self.pypi_packages,
                                  cluster_log_dbfs_path=self.cluster_log_path,
                                  python_script_params=[inference_date])


        # Inference and Save step
        output_dir = OutputFileDatasetConfig(name="scores_output",
                                             destination=(self.inference_datastore, self.output_dir))
        
        output_dir_local_exp = OutputFileDatasetConfig(name="local_exp_output",
                                                       destination=(self.inference_datastore, self.output_dir_local_exp))
        
        output_dir_elt_store = OutputFileDatasetConfig(name="etl_store_output",
                                                       destination=(self.inference_datastore, self.output_dir_elt_store))


        inference_save_step = PythonScriptStep(name="Inference and Save Data",
                                               source_directory=self.source_directory,
                                               script_name=self.infer_save_step_script_name,
                                               arguments=['--model_name', self.model_name,
                                                          '--output_dir', output_dir.as_upload(overwrite=True)],
                                               inputs=[etl_out_dir.parse_parquet_files()],
                                               compute_target=self.compute_target,
                                               runconfig=run_config,
                                               allow_reuse=False)
        
        local_exp_save_step = PythonScriptStep(name="Local Exp Inference and Save Data",
                                        source_directory=self.source_directory,
                                        script_name=self.local_exp_save_step_script_name,
                                        arguments=['--model_name', self.model_name,
                                                   '--output_dir_local_exp', output_dir_local_exp.as_upload(overwrite=True)],
                                        inputs=[etl_out_dir.parse_parquet_files()],
                                        compute_target=self.compute_target,
                                        runconfig=run_config,
                                        allow_reuse=False)

        etl_output_save_step = PythonScriptStep(name="Etl Output Save Data",
                                        source_directory=self.source_directory,
                                        script_name=self.etl_output_save_step_script_name,
                                        arguments=['--output_dir_elt_store', output_dir_elt_store.as_upload(overwrite=True)],
                                        inputs=[etl_out_dir.parse_parquet_files()],
                                        compute_target=self.compute_target,
                                        runconfig=run_config,
                                        allow_reuse=False)
        
        # Run the pipeline
        pipeline = Pipeline(workspace=self.aml_workspace, steps=[etl_step, 
                                                                 inference_save_step,
                                                                 local_exp_save_step,
                                                                 etl_output_save_step
                                                                 ])
        self.pipeline = pipeline

    def publish(self, set_default=True):
        """
        Publish pipeline to endpoint and return published endpoint
        :return:
        """
        self.published_pipeline = self.pipeline.publish("CP_Inferencing_Pipeline", "Call Inferencing Pipeline")

        if self.endpoint is None:
            self.endpoint = PipelineEndpoint.publish(workspace=self.aml_workspace, name=self.config["endpoint_name"],
                                                     description="CP Inferencing Pipeline",
                                                     pipeline=self.published_pipeline)
            print("New Pipeline Endpoint created")
        elif set_default:
            self.endpoint.add_default(self.published_pipeline)
            print("pipeline endpoint exists, default pipeline updated")
        else:
            self.endpoint.add(self.published_pipeline)

    def schedule(self):
        """
        Create daily schedule for batch inferencing
        :return:
        """
        recurrence = ScheduleRecurrence(frequency=self.schedule_frequency,
                                        interval=self.schedule_interval, start_time=self.start_time,
                                        time_zone=getattr(TimeZone, self.aml_schedule_timezone))
        Schedule.create(self.aml_workspace, name=self.schedule_name,
                        pipeline_id=self.published_pipeline.id,
                        experiment_name=self.experiment_name, recurrence=recurrence)

    def _get_pypi_packages(self, pypi_packages):
        """
        Return list of PyPiLibraries for dbstep
        :param pypi_packages:
        :return:
        """

        return [PyPiLibrary(package=x, repo=self.repo_link) for x in pypi_packages]
