from pathlib import Path
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def main():
    ref = pd.read_csv(TRAIN_PATH)
    cur = pd.read_csv(TEST_PATH)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    html_path = REPORTS_DIR / "data_drift_report.html"
    report.save_html(html_path)
    print("Relatório de drift salvo em:", html_path)

if __name__ == "__main__":
    main()
