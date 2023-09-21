import argparse
from src.cp.pipeline import TrainingPipeline


def main():
    pipeline = TrainingPipeline(args.pipeline_config, args.azure_config)
    pipeline.build()
    pipeline.run()
    # pipeline.publish()
    # pipeline.schedule(name="CP_ReactiveTraining")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CP_training_pipeline_run")
    parser.add_argument("--pipeline_config")
    parser.add_argument("--azure_config")
    args = parser.parse_args()
    main()