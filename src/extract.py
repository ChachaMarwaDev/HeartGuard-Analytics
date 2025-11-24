from src.config import DATA_DIR

import pandas as pd

def extract_csv(filename: str) -> pd.DataFrame:
    """
    Extract data from a CSV file located in the configured data directory.
    This function reads a CSV file from the DATA_DIR and loads it into a pandas DataFrame.
    The full file path is printed to the console for verification purposes.
    Args:
        filename (str): The name of the CSV file to be loaded (without the full path).
    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
    Example:
        >>> df = extract_csv('data.csv')
        Loading:  /c:/Users/user/Desktop/HeartGuard Analytics/data/data.csv
    """
    file_path = DATA_DIR / filename
    print('Loading: ', file_path.resolve())

    df = pd.read_csv(file_path)

    return df