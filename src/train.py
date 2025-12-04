import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from joblib import dump

import mlflow
import mlflow.sklearn

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_PATH = DATA_DIR / "processed" / "train.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

def main():
    df = pd.read_csv(TRAIN_PATH)

    X = df[["Length1", "Length2", "Length3", "Height", "Width"]]
    y = df["Weight"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # define/usa experimento
    mlflow.set_experiment("fish_weight_regression")

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        print("MAE:", mae)

        # loga métrica no MLflow
        mlflow.log_metric("mae", mae)

        # (opcional) logar alguns parâmetros
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # salva modelo no disco (como antes)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "linear_regression_fish.joblib"
        dump(model, model_path)
        print("Modelo salvo em:", model_path)

        # loga modelo também no MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    main()
