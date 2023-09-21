import argparse
from src.cp.pipeline import AutoTrainPipeline


def main():
    pipeline = AutoTrainPipeline(args.pipeline_config, args.azure_config, deployment=False)
    pipeline.build()
    pipeline.run()
    # pipeline.publish()
    # pipeline.schedule()
    print("pipeline_built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("cp_auto_train_pipeline_run")
    parser.add_argument("--pipeline_config")
    parser.add_argument("--azure_config")
    args = parser.parse_args()
    main()
