import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "raw" / "fish.csv"
PROCESSED_DIR = DATA_DIR / "processed"

def main():
    print("Lendo:", RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    print("Colunas:", df.columns.tolist())
    print("Primeiras linhas:")
    print(df.head())

    # separa treino/teste (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print("Arquivos salvos em data/processed/")

if __name__ == "__main__":
    main()
