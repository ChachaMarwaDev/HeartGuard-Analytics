import pandas as pd
from pathlib import Path

def load_data(df: pd.DataFrame, output_filename: str) -> None:
    output_path = Path("Data\processed_data") / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Data loaded to: {output_path.resolve()}')