from src.extract import extract_csv
from src.transfrom import clean_data
# from src.load import load_data

def main():
    try:
        raw_data = extract_csv("Heart_disease_statlog.csv") 
    except FileNotFoundError:
        print("ERROR: CSV file not found. Please verify the filename and location")
        return

    cleaned = clean_data(raw_data)

    print(f"Rows before cleaning: {len(raw_data)}")
    print(f"Rows after cleaning: {len(cleaned)}")

    # cleaned.to_csv("cleaned_diabetes02.csv", index=False) #Save to a csv file
    # print("clean data exported")
    # Heart_disease_statlog.csv

    # load_data(cleaned, "diabetes_cleaned.csv")
    return cleaned

if __name__ == "__main__":
    cleaned_df = main()