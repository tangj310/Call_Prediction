
from azureml.core import ComputeTarget, Datastore, Dataset, Environment, Workspace
from azureml.pipeline.core import PipelineEndpoint
import logging

ws = Workspace.from_config("conf/pipeline/inferencing/dev/dev_azure_config.json")

def get_environments():
    envs = Environment.list(ws)
    for env in envs:
        if env.startswith("AzureML"):
            print("Name", env)
            # print("packages", envs[env].python.conda_dependencies.serialize_to_string())


def get_compute_targets():
    print("Compute Targets:")
    for compute_name in list(ws.compute_targets)[:2]:
        compute = ws.compute_targets[compute_name]
        print("\t", compute_name, " : ", compute.type)


def get_datastores():

    print("data stores:")
    for datastore_name in ws.datastores:
        datastore = Datastore.get(ws, datastore_name)
        print("\t", datastore_name, " : ", datastore.datastore_type)


def get_datasets():
    print("data sets:")
    for dataset_name in list(ws.datasets.keys())[:2]:
        dataset = Dataset.get_by_name(ws, dataset_name)
        print("\t", dataset.name)

def get_pipelines():
    print("pipelines")
    for endpoint in PipelineEndpoint.list(ws)[:2]:
        print(endpoint.name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    get_pipelines()
    get_environments()
    get_datasets()
    get_datastores()
    get_compute_targets()

