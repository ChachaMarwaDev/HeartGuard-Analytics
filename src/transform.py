import pandas as pd
import matplotlib.pyplot as plt

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Section 1: Data inspection
    df = df.copy()  # makes an independent copy of our dataframe

    # print('BEFORE CLEANING')
    # print(df.describe())
    # print(df.dtypes)
    # print(df.head(5))
    # print(df.tail(5))
    # print(df.columns)  # ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    # print(df.isnull().sum())  # we have zero null values
    # print(df.nunique())
    # print(df.duplicated().value_counts())  # we have zero duplicate values
    
    # mapping; sex -> male,female , fbs -> true,false , exang -> Yes,No , slope -> Upsloping,flat,downsloping , target -> disease_present,normal
    # print('AFTER CLEANING')
    df['sex'] = df['sex'].map({1: 'Male', 0: 'Female'})
    df['fbs'] = df['fbs'].map({1: 'True', 0: 'False'})
    df['slope'] = df['slope'].map({2: 'Downsloping', 1: 'Flat', 0: 'Upsloping'})
    df['exang'] = df['exang'].map({1: 'Yes', 0: 'No'})
    # print(df.head(6))
    # print(df.dtypes)

    # Section 2: analysis
    # 1. Descriptive Statistics
    # Age distribution
    # age_distribution = df['age'].describe()

    # Create ordered age groups using pd.cut to handle unexpected ages robustly
    # bins = [28, 39, 49, 59, 69, 200]  # 29-39, 40-49, 50-59, 60-69, 70+
    # labels = ['29-39', '40-49', '50-59', '60-69', '70+']
    # df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True, ordered=True)

    # age_group_counts = df['AgeGroup'].value_counts().sort_index()
    # age_group_counts.plot(kind='bar')
    # plt.title('Age group numbers')
    # plt.show()

    # Male-to-female ratio
    # sex_counts = df['sex'].value_counts()
    # Use explicit label lookup to avoid FutureWarning when Series index is non-integer (e.g., 'Male'/'Female')
    # male_count = sex_counts.get('Male', 0)
    # female_count = sex_counts.get('Female', 0)
    # sex_ratio = male_count / female_count if female_count != 0 else None
    # print('Sex ratio: ', sex_ratio)

    # Heart disease prevalence by sex
    # disease_by_sex = df.groupby('sex')['target'].mean()  # working with object type so on was to be type int64
    # print('Disease by sex: ', disease_by_sex)

    # Continuous variables statistics
    # continuous_vars = ['trestbps', 'chol', 'thalach', 'oldpeak']
    # continuous_stats = df[continuous_vars].describe()
    # print(continuous_stats)

    # 2. Target Variable Analysis
    # Percentage with heart disease
    # target_distribution = df['target'].value_counts(normalize=True) * 100
    # print(target_distribution)
    # target_counts = df['target'].value_counts()
    # print(target_counts)

    # Check if dataset is balanced
    # is_balanced = abs(target_distribution[0] - target_distribution[1]) < 10  # Within 10% difference
    # print(is_balanced)

    # 3. Demographic Relationships
    # Heart disease prevalence by age group
    # bins = [28, 39, 49, 59, 69, 200]  # 29-39, 40-49, 50-59, 60-69, 70+
    # labels = ['29-39', '40-49', '50-59', '60-69', '70+']
    # df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True, ordered=True)
    # disease_by_age = df.groupby('AgeGroup')['target'].mean() * 100
    # print(disease_by_age)

    # Heart disease rates by sex
    # disease_by_sex = df.groupby('sex')['target'].mean()
    # print(disease_by_sex)

    # Age distribution by sex and disease status
    # age_sex_disease = df.groupby(['sex', 'target'])['age'].describe()
    # print(age_sex_disease)

    # 4. Clinical Symptoms Analysis
    # Chest pain type association with heart disease
    # cp_disease = df.groupby('cp')['target'].mean().sort_values(ascending=False)
    # print(cp_disease)

    # Asymptomatic patients with heart disease
    # asymptomatic_disease = df[df['cp'] == 3]['target'].mean()
    # print(asymptomatic_disease)

    # Resting blood pressure and heart disease
    # specify observed=False to retain current behavior and silence FutureWarning about Categorical groupby
    # bp_disease = df.groupby(pd.cut(df['trestbps'], bins=5), observed=False)['target'].mean()
    # print(bp_disease)

    # specify observed=False to retain current behavior and silence FutureWarning about Categorical groupby
    # chol_disease = df.groupby(pd.cut(df['chol'], bins=5), observed=False)['target'].mean()
    # print(chol_disease)

    # Fasting blood sugar and heart disease
    # fbs_disease = df.groupby('fbs')['target'].mean()
    # print(fbs_disease)

    # 5. Cardiac Test Results
    # Resting ECG patterns
    # restecg_disease = df.groupby('restecg')['target'].mean().sort_values(ascending=False)
    # print(restecg_disease)

    # Max heart rate correlation
    # thalach_correlation = df[['thalach', 'target']].corr().iloc[0, 1]
    # print(thalach_correlation)
    # thalach_by_disease = df.groupby('target')['thalach'].describe()
    # print(thalach_by_disease)

    # Exercise-induced angina
    # exang_disease = df.groupby('exang')['target'].mean()
    # print(exang_disease)

    # ST-depression and disease
    # oldpeak_disease = df.groupby(pd.cut(df['oldpeak'], bins=5), observed=False)['target'].mean()
    # print(oldpeak_disease)

    # ST segment slope
    # slope_disease = df.groupby('slope')['target'].mean().sort_values(ascending=False)
    # print(slope_disease)
    
    # 6. Blood Disorder Analysis
    # Major vessels and heart disease
    # ca_disease = df.groupby('ca')['target'].mean().sort_values(ascending=False)
    # print(ca_disease)

    # Thalassemia classification
    # thal_disease = df.groupby('thal')['target'].mean().sort_values(ascending=False)
    # print(thal_disease)

    # Check for NULL thalassemia values
    # null_thal = df[df['thal'] == 0]
    # null_thal_count = len(null_thal)
    # print(null_thal_count)

    # 7. Multivariate Analysis
    # Combination of risk factors
    # df['multiple_risk'] = ((df['chol'] > 240) & (df['trestbps'] > 140) & (df['fbs'] == 1)).astype(int)
    # multiple_risk_disease = df.groupby('multiple_risk')['target'].mean()
    # print(multiple_risk_disease)

    # Interaction between age, sex and chest pain
    # bins = [28, 39, 49, 59, 69, 200]  # 29-39, 40-49, 50-59, 60-69, 70+
    # labels = ['29-39', '40-49', '50-59', '60-69', '70+']
    # df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True, ordered=True)
    # age_sex_cp_disease = df.groupby(['AgeGroup', 'sex', 'cp'])['target'].mean().reset_index()
    # print(age_sex_cp_disease)

    # Correlation matrix for key variables
    # correlation_matrix = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']].corr()
    # print(correlation_matrix)

    # 8. Feature Importance
    # Correlation with target variable
    # target_correlations = df.corr()['target'].sort_values(ascending=False).drop('target')
    # print(target_correlations)

    # Check for redundant features
    # feature_correlation = df.corr().abs()
    
    # Identify highly correlated features (correlation > 0.8)
    # high_corr = []
    # for i in range(len(feature_correlation.columns)):
    #     for j in range(i + 1, len(feature_correlation.columns)):
    #         if feature_correlation.iloc[i, j] > 0.8:
    #             high_corr.append(
    #                 (feature_correlation.columns[i], 
    #                  feature_correlation.columns[j], 
    #                  feature_correlation.iloc[i, j])
    #             )

    # print(high_corr)

    # 9. Outlier & Data Quality
    # Check for outliers using IQR method
    # def find_outliers(series):
    #     Q1 = series.quantile(0.25)
    #     Q3 = series.quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
    #     return series[(series < lower_bound) | (series > upper_bound)]
    #
    # outliers = {}
    # for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
    #     outliers[col] = find_outliers(df[col])
    #
    # print(outliers[col])

    # Check for missing values
    # missing_values = df.isnull().sum()
    # print(missing_values)

    # Check medically plausible ranges
    # medically_implausible = {
    #     'trestbps': df[(df['trestbps'] < 80) | (df['trestbps'] > 250)],
    #     'chol': df[(df['chol'] < 100) | (df['chol'] > 600)],
    #     'thalach': df[(df['thalach'] < 60) | (df['thalach'] > 220)]
    # }
    # print(medically_implausible)

    # 10. Comparative Analysis
    # Patients with vs without exercise-induced angina
    # exang_comparison = df.groupby('exang').agg({
    #     'age': 'mean',
    #     'trestbps': 'mean', 
    #     'chol': 'mean',
    #     'thalach': 'mean',
    #     'target': 'mean'
    # })
    # print(exang_comparison)

    # Characteristic profiles by chest pain type
    # cp_profiles = df.groupby('cp').agg({
    #     'age': 'mean',
    #     'sex': 'mean',  # proportion male
    #     'trestbps': 'mean',
    #     'chol': 'mean',
    #     'target': 'mean'
    # })
    # print(cp_profiles)

    # Cardiac test results by disease status
    # disease_comparison = df.groupby('target').agg({
    #     'thalach': ['mean', 'std'],
    #     'oldpeak': ['mean', 'std'],
    #     'trestbps': ['mean', 'std'],
    #     'chol': ['mean', 'std']
    # })
    # print(disease_comparison)

    return df