import argparse
import os
import pickle
import sys
from datetime import date

import pandas
from azureml.core import Model, Run



def run_inference(input_data):
    # make inference

    df = input_data[input_data.columns.difference(["account_number", "event_date", "join_date","category_name",
                                                  "target", "window"])]
    X = df.fillna(0).values
    pred = model.predict(X)
    score = model.predict_proba(X)[:, 1]

    # cleanup
    input_data['pred'] = pred
    input_data["score"] = score
    return input_data


def run_saving():
    # Save the split data
    print("Saving Data...")
    batch_date = inference_pd["event_date"].max()
    to_date_str = batch_date.strftime('%Y-%m-%d')



    # *** Write to LATEST folder ***
    output_path = os.path.join(output_dir, 'LATEST')
    os.makedirs(output_path, exist_ok=True)
    inference_pd.to_parquet(os.path.join(output_path, 'cp_prediction.parquet'), index=False)

    # *** Write to ARCHIVE folder ***
    output_path = os.path.join(output_dir, 'ARCHIVE', f"DATE_PART={to_date_str}")
    os.makedirs(output_path, exist_ok=True)
    inference_pd.to_parquet(os.path.join(output_path, 'cp_prediction.parquet'), index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    # parser.add_argument(("--output_dir2"))
    parser.add_argument('--model_name', dest="model_name", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir
    # output_dir2 = args.output_dir2
    model_name = args.model_name
    print(model_name)
    print(output_dir)
    # print(output_dir2)

    # Scoring data

    # Get the experiment run context
    run = Run.get_context()
    ws = run.experiment.workspace

    # load the model
    model_path = Model.get_model_path(args.model_name)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    etl_df = run.input_datasets["etl_output"].to_pandas_dataframe()
    if etl_df.empty:
        print("Empty Dataframe, missing data")
        run.complete()
        sys.exit()
    else:
        etl_df["account_number"] = etl_df["account_number"].astype(str)
        inference_pd = run_inference(etl_df)
        run_saving()