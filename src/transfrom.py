import pandas as pd
import matplotlib.pyplot as plt

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Section 1: Data inspection
    print(df.head(5))
    return df