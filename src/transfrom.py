import pandas as pd
import matplotlib.pyplot as plt

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Section 1: Data inspection
    df = df.copy() # makes an independent copy of our dataframe

    # print(df.describe())
    # print(df.dtypes)
    print('BEFORE CLEANING')
    print(df.head(5))
    # print(df.tail(5))
    # print(df.columns) #['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    # print(df.isnull().sum()) # we have zero null values
    # print(df.nunique())
    # print(df.duplicated().value_counts()) # we have zero duplicate values
    
    # mapping; sex -> male,female , fbs -> true,false , exang -> Yes,No , slope -> Upsloping,flat,downsloping , target -> disease_present,normal
    print('AFTER CLEANING')
    df['sex'] = df['sex'].map({1 :'Male', 0:'Female'})
    df['fbs'] = df['fbs'].map({1 :'True', 0:'False'})
    df['slope'] = df['slope'].map({2: 'Downsloping', 1 :'Flat', 0:'Upsloping'})
    df['target'] = df['target'].map({1 :'Disease_present', 0:'Normal'})
    df['exang'] = df['exang'].map({1 :'Yes', 0:'No'})
    print(df.head(6))
    
    return df