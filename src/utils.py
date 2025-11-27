import matplotlib.pyplot as plt

# Function to create pretty summary tables
def create_summary_table(df, group_col, target_col='target'):
    summary = df.groupby(group_col).agg({
        target_col: ['count', 'mean', 'std'],
        'age': 'mean',
        'sex': 'mean'
    }).round(3)
    return summary

# Quick visualization helper
def plot_disease_rate_by_category(df, category_col):
    rates = df.groupby(category_col)['target'].mean()
    rates.plot(kind='bar', title=f'Heart Disease Rate by {category_col}')
    plt.ylabel('Disease Rate')
    plt.show()