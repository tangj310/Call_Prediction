import argparse
import os
from azureml.core import Run
import sys

def save_elt_output(input_data):

    print("Saving Data...")
    batch_date = input_data["event_date"].max()
    to_date_str = batch_date.strftime('%Y-%m-%d')

        # *** Write to LATEST folder ***
    output_path = os.path.join(output_dir_elt_store, 'LATEST')
    os.makedirs(output_path, exist_ok=True)
    input_data.to_parquet(os.path.join(output_path, 'cp_etl_output.parquet'), index=False)

    # *** Write to ARCHIVE folder ***
    output_path = os.path.join(output_dir_elt_store, 'ARCHIVE', f"DATE_PART={to_date_str}")
    os.makedirs(output_path, exist_ok=True)
    input_data.to_parquet(os.path.join(output_path, 'cp_etl_output.parquet'), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir_elt_store')
    args = parser.parse_args()
    output_dir_elt_store = args.output_dir_elt_store
    print(output_dir_elt_store)

    # Get the experiment run context
    run = Run.get_context()
    ws = run.experiment.workspace

    etl_df = run.input_datasets["etl_output"].to_pandas_dataframe()
    if etl_df.empty:
        print("Empty Dataframe, missing data")
        run.complete()
        sys.exit()
    else:
        save_elt_output(etl_df)
