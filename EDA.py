import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to load data with retry mechanism
def load_data(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Trying fallback URL...")
                try:
                    fallback_url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
                    return pd.read_csv(fallback_url)
                except:
                    raise ConnectionError("All data sources failed. Check your internet connection.")
            time.sleep(2)

# Main EDA function
def perform_eda(data):
    # Basic Data Inspection
    print("="*50)
    print("BASIC DATA INSPECTION")
    print("="*50)
    print("\nFirst 5 rows:")
    display(data.head())
    
    print("\nDataset Info:")
    print(data.info())
    
    print("\nDescriptive Statistics:")
    display(data.describe(include='all'))
    
    # Missing Values Analysis
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    missing = data.isnull().sum()
    print("\nMissing Values:")
    display(missing[missing > 0])
    
    # Data Cleaning
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data.drop('Cabin', axis=1, inplace=True)
    
    # Visualization Settings
    plt.style.use('seaborn')
    color_palette = ['#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    
    # Create figure
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[3, :])
    
    # Age Distribution
    sns.histplot(data['Age'], bins=30, kde=True, ax=ax1, color=color_palette[0])
    ax1.set_title('Age Distribution', fontsize=14)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Count')
    
    # Survival by Gender
    sns.barplot(x='Sex', y='Survived', data=data, ax=ax2, palette=color_palette[1:3])
    ax2.set_title('Survival Rate by Gender', fontsize=14)
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Survival Rate')
    
    # Survival by Class
    sns.barplot(x='Pclass', y='Survived', data=data, ax=ax3, palette=color_palette[2:5])
    ax3.set_title('Survival Rate by Passenger Class', fontsize=14)
    ax3.set_xlabel('Passenger Class')
    ax3.set_ylabel('Survival Rate')
    
    # Family Size Analysis
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    sns.barplot(x='FamilySize', y='Survived', data=data, ax=ax4, palette=color_palette)
    ax4.set_title('Survival Rate by Family Size', fontsize=14)
    ax4.set_xlabel('Family Size')
    ax4.set_ylabel('Survival Rate')
    
    # Fare Distribution
    sns.boxplot(x='Survived', y='Fare', data=data, ax=ax5, palette=color_palette[1:3])
    ax5.set_title('Fare Distribution by Survival', fontsize=14)
    ax5.set_xlabel('Survived')
    ax5.set_ylabel('Fare')
    
    # Correlation Heatmap
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = data[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax6)
    ax6.set_title('Feature Correlation Heatmap', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical Testing
    print("\n" + "="*50)
    print("STATISTICAL TESTING")
    print("="*50)
    
    # T-test for Age difference between survivors/non-survivors
    survivors = data[data['Survived'] == 1]['Age']
    non_survivors = data[data['Survived'] == 0]['Age']
    t_stat, p_value = stats.ttest_ind(survivors, non_survivors)
    
    print(f"\nAge Difference (T-test):\nT-statistic = {t_stat:.3f}, p-value = {p_value:.4f}")
    alpha = 0.05
    if p_value < alpha:
        print(f"P-value < {alpha}: Significant difference in age between survivors and non-survivors.")
    else:
        print(f"P-value ≥ {alpha}: No significant age difference between groups.")
    
    # Chi-square test for class and survival
    contingency_table = pd.crosstab(data['Pclass'], data['Survived'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nClass vs Survival (Chi-square test):\nChi2 = {chi2:.3f}, p-value = {p:.4f}")
    
    # Key Insights
    print("\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    print("\n1. Survival Rate: {:.1%} of passengers survived".format(data['Survived'].mean()))
    print("2. Female survival rate: {:.1%}".format(data[data['Sex']=='female']['Survived'].mean()))
    print("3. 1st class survival rate: {:.1%}".format(data[data['Pclass']==1]['Survived'].mean()))
    print("4. Average age of passengers: {:.1f} years".format(data['Age'].mean()))
    print("5. Highest correlation with survival: Fare ({:.2f})".format(corr_matrix['Survived']['Fare']))

# Main execution
if __name__ == "__main__":
    try:
        print("Starting Titanic Dataset EDA...")
        
        # Try primary data source first
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        titanic_data = load_data(url)
        
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        perform_eda(titanic_data)
        print("\nEDA completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your internet connection or try again later.")
