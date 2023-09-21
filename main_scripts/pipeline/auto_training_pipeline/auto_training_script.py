import os
import argparse
from azureml.core import Run
from azureml.interpret import ExplanationClient
from interpret_community.mimic import MimicExplainer
from interpret_community.mimic.models import LGBMExplainableModel

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd

from datetime import date

today = date.today()


def explain_sklearn(model, features):
    features_score = {}
    try:
        scores = model.feature_importances_
    except AttributeError:
        return None
    for idx, feature in enumerate(features):
        features_score[feature] = scores[idx]

    return pd.DataFrame(features_score.items(), columns=["feature", "importance"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir_autotrain_score')
    parser.add_argument("--TRAIN_START_DATE")
    parser.add_argument("--TRAIN_END_DATE")
    parser.add_argument("--TEST_START_DATE")
    parser.add_argument("--TEST_END_DATE")
    args = parser.parse_args()
    output_dir_autotrain_score = args.output_dir_autotrain_score
    test_start_date = args.TEST_START_DATE
    train_start_date = args.TRAIN_START_DATE
    test_end_date = args.TEST_END_DATE
    train_end_date = args.TRAIN_END_DATE
    print(output_dir_autotrain_score)
    print(test_start_date)
    print(train_start_date)
    print(test_end_date)
    print(train_end_date)

    with mlflow.start_run():
        run = Run.get_context()
        # train_df = run.input_datasets['train_set'].to_pandas_dataframe()
        # test_df = run.input_datasets["test_set"].to_pandas_dataframe()
        client = ExplanationClient.from_run(run)
        ws = run.experiment.workspace
        train_df = run.input_datasets["train_output"].to_pandas_dataframe()
        test_df = run.input_datasets["test_output"].to_pandas_dataframe()

        train_df = train_df[train_df.columns.difference(["account_number", "event_date", "join_date",
                                                        "category_name", "window"])]

        test_df = test_df[test_df.columns.difference(["account_number", "event_date", "join_date",
                                                    "category_name", "window"])]
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)
        X_train = train_df[train_df.columns.difference(["target"])]

        y_train = train_df.target.values

        X_test = test_df[train_df.columns.difference(["target"])]

        y_test = test_df.target.values

        mlflow.log_text(f"Training shape: {train_df.shape}\nTest shape:{test_df.shape}\n"
                        f"Columns: {X_train.columns}\n"
                        f"Training Target Distribution: {train_df.target.value_counts()}\n"
                        f"Test Target Distribution: {test_df.target.value_counts()}\n", "Data metrics.txt")


        # Define models
        model1 = RandomForestClassifier()
        model2 = GradientBoostingClassifier()
        model3 = StackingClassifier([('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier())])
        models = {"RandomForest": model1, "GradientBoosting": model2, "Stacking": model3}


        # Train models
        model1.fit(X_train, y_train)
        y_pred_prob = model1.predict_proba(X_test)[:, 1]
        y_pred = model1.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # log metrics
        mlflow.log_metric(f"{next(iter(models))} ROC AUC", roc_auc)
        mlflow.log_metric(f"{next(iter(models))} Recall", recall)
        mlflow.log_metric(f"{next(iter(models))} Precision", precision)

        ### COMMENTED OUT FOR NOW, NEED TO FIGURE OUT HOW TO SAVE FIGURES TO MLFLOW ###
        # # plot and save roc auc curve
        # fig, ax = plt.subplots()
        # plot_roc_curve(model, X_test, y_test, drop_intermediate=False, ax=ax)
        # ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        # ax.legend()
        # fig.savefig(f"{model_name} ROC AUC curve.png")
        # mlflow.log_artifact(f"{model_name} ROC AUC curve.png")

        # # plot and save precision and recall curve
        # fig, ax = plt.subplots()
        # plot_precision_recall_curve(model, X_test, y_test, ax=ax)
        # ax.plot([0, 1], [0, 0], linestyle='--', label='No Skill')
        # ax.legend()
        # fig.savefig(f"{model_name} pr curve.png")
        # mlflow.log_artifact(f"{model_name} pr curve.png")

        # log the model
        mlflow.sklearn.log_model(model1, f"cp_{next(iter(models))}_v1_{today}")

        # model explanation
        explainer = MimicExplainer(model1,
                                X_train,
                                LGBMExplainableModel,
                                augment_data=True,
                                max_num_of_augmentations=10,
                                features=X_train.columns,
                                classes=["nocall", "call"])
        explanation = explainer.explain_global(X_test)

        # explain model
        original_model = run.register_model(model_name=f"cp_{next(iter(models))}_v1_{today}",
                                            model_path=f"cp_{next(iter(models))}_v1_{today}/model.pkl")
        comment = f'Global explanation on {next(iter(models))} model trained on call prediction dataset {today}'
        client.upload_model_explanation(explanation, comment=comment, model_id=original_model.id)

        # save AUC, precision, recall
        score_data = {'AUC': [roc_auc], 'precision': [precision], 'recall': [recall]}
        autotrain_score_pd = pd.DataFrame(data = score_data)

        to_date_str = pd.Timestamp.today().strftime('%Y-%m-%d')

        autotrain_score_pd['event_date'] = to_date_str
        autotrain_score_pd['train_start_date'] = train_start_date
        autotrain_score_pd['train_end_date'] = train_end_date
        autotrain_score_pd['test_start_date'] = test_start_date
        autotrain_score_pd['test_end_date'] = test_end_date


        # *** Write to LATEST folder ***
        output_path = os.path.join(output_dir_autotrain_score, 'LATEST')
        os.makedirs(output_path, exist_ok=True)
        autotrain_score_pd.to_parquet(os.path.join(output_path, 'autotrain_score.parquet'), index=False)

        # *** Write to ARCHIVE folder ***
        output_path = os.path.join(output_dir_autotrain_score, 'ARCHIVE', f"DATE_PART={to_date_str}")
        os.makedirs(output_path, exist_ok=True)
        autotrain_score_pd.to_parquet(os.path.join(output_path, 'autotrain_score.parquet'), index=False)