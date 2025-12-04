import pandas as pd
from pathlib import Path
from joblib import load

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "linear_regression_fish.joblib"

def predict_weight(length1, length2, length3, height, width):
    model = load(MODEL_PATH)

    data = pd.DataFrame([{
        "Length1": length1,
        "Length2": length2,
        "Length3": length3,
        "Height": height,
        "Width": width,
    }])

    pred = model.predict(data)[0]
    return pred

def main():
    predicted = predict_weight(
        length1=23.2,
        length2=25.4,
        length3=30.0,
        height=11.52,
        width=4.02,
    )
    print("Peso previsto:", predicted)

if __name__ == "__main__":
    main()
