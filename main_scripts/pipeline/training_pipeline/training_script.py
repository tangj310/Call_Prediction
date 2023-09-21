from azureml.core import Run, Dataset
from azureml.interpret import ExplanationClient
from interpret_community.mimic import MimicExplainer
from interpret_community.mimic.models import LGBMExplainableModel

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, \
    VotingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, plot_roc_curve, plot_precision_recall_curve
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from datetime import date

today = date.today()

with mlflow.start_run():
    run = Run.get_context()
    # train_df = run.input_datasets['train_set'].to_pandas_dataframe()
    # test_df = run.input_datasets["test_set"].to_pandas_dataframe()
    client = ExplanationClient.from_run(run)
    ws = run.experiment.workspace
    train_df = Dataset.get_by_name(workspace=ws, name="call_prediction_train1").to_pandas_dataframe()
    test_df = Dataset.get_by_name(workspace=ws, name="call_prediction_test1").to_pandas_dataframe()

    X_train = train_df[train_df.columns.difference(["account_number", "event_date", "join_date",
                                                    "target"])]

    y_train = train_df.target.values

    X_test = test_df[test_df.columns.difference(["account_number", "event_date", "join_date",
                                                 "target"])]
    y_test = test_df.target.values

    mlflow.log_text(f"Training shape: {train_df.shape}\nTest shape:{test_df.shape}\n"
                    f"Columns: {X_train.columns}", "Data metrics.txt")

    # Define models
    model1 = RandomForestClassifier()
    model2 = GradientBoostingClassifier()
    model3 = StackingClassifier([('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier())])
    models = {"RandomForest": model1, "GradientBoosting": model2, "Stacking": model3}

    for model_name in models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # log metrics
        mlflow.log_metric(f"{model_name} ROC AUC", roc_auc)
        mlflow.log_metric(f"{model_name} Recall", recall)
        mlflow.log_metric(f"{model_name} Precision", precision)

        # plot and save roc auc curve
        fig, ax = plt.subplots()
        plot_roc_curve(model, X_test, y_test, drop_intermediate=False, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        ax.legend()
        fig.savefig(f"{model_name} ROC AUC curve.png")
        mlflow.log_artifact(f"{model_name} ROC AUC curve.png")

        # plot and save precision and recall curve
        fig, ax = plt.subplots()
        plot_precision_recall_curve(model, X_test, y_test, ax=ax)
        ax.plot([0, 1], [0, 0], linestyle='--', label='No Skill')
        ax.legend()
        fig.savefig(f"{model_name} pr curve.png")
        mlflow.log_artifact(f"{model_name} pr curve.png")

        # log the model
        mlflow.sklearn.log_model(model, f"cp_{model_name}_v1_{today}")

        # model explanation
        explainer = MimicExplainer(model,
                                   X_train,
                                   LGBMExplainableModel,
                                   augment_data=True,
                                   max_num_of_augmentations=10,
                                   features=X_train.columns,
                                   classes=["nocall","call"])
        explanation = explainer.explain_global(X_test)
        # explain model
        original_model = run.register_model(model_name=f"cp_{model_name}_v1_{today}",
                                            model_path=f"cp_{model_name}_v1_{today}/model.pkl")
        comment = f'Global explanation on {model_name} model trained on call prediction dataset {today}'
        client.upload_model_explanation(explanation, comment=comment, model_id=original_model.id)
