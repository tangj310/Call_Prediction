import os
import argparse
import pickle
import sys

from azureml.core import Model, Run
from interpret_community.mimic import MimicExplainer
from interpret_community.mimic.models import LGBMExplainableModel

import pandas as pd
from datetime import date

today = date.today()

def run_save_local_exp(input_data):

    batch_date = input_data["event_date"].max()
    to_date_str = batch_date.strftime('%Y-%m-%d')

    local_exp_input = input_data[input_data.columns.difference(["account_number", "event_date", "join_date","category_name",
                                                  "target", "window"])]
    explainer = MimicExplainer(model=local_exp_model,
                               initialization_examples=local_exp_input,
                               explainable_model=LGBMExplainableModel,
                               augment_data=False,
                               features=local_exp_input.columns,
                            #    classes=["nocall","call"]
                               )

    local_exp_output = explainer.explain_local(local_exp_input)

    local_exp_output_pd = pd.DataFrame(local_exp_output.local_importance_values[1], columns=local_exp_output.features)
    
    # test out to putback event_date, and account_number
    local_exp_output_pd['account_number'] = input_data['account_number']
    local_exp_output_pd['event_date'] = input_data['event_date']


    # *** Write to LATEST folder ***
    output_path = os.path.join(output_dir_local_exp, 'LATEST')
    os.makedirs(output_path, exist_ok=True)
    local_exp_output_pd.to_parquet(os.path.join(output_path, 'cp_explanation.parquet'), index=False)

    # *** Write to ARCHIVE folder ***
    output_path = os.path.join(output_dir_local_exp, 'ARCHIVE', f"DATE_PART={to_date_str}")
    os.makedirs(output_path, exist_ok=True)
    local_exp_output_pd.to_parquet(os.path.join(output_path, 'cp_explanation.parquet'), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir_local_exp')
    parser.add_argument('--model_name', dest="model_name", required=True)
    args = parser.parse_args()
    output_dir_local_exp = args.output_dir_local_exp
    model_name = args.model_name
    print(model_name)
    print(output_dir_local_exp)
    
    # Get the experiment run context
    run = Run.get_context()
    ws = run.experiment.workspace

    # load the model
    model_path = Model.get_model_path(args.model_name)
    with open(model_path, 'rb') as f:
        local_exp_model = pickle.load(f)

    etl_df = run.input_datasets["etl_output"].to_pandas_dataframe()
    if etl_df.empty:
        print("Empty Dataframe, missing data")
        run.complete()
        sys.exit()
    else:
        etl_df["account_number"] = etl_df["account_number"].astype(str)
        run_save_local_exp(etl_df)