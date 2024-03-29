import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif  # Assuming chi-square for now
from sklearn.preprocessing import MinMaxScaler  # Consider if necessary
from sklearn.model_selection import train_test_split
import numpy as np
from dataset import Dataset

# Import dataset
newDataset = Dataset('src/housing.csv')
dataset = newDataset.dataset
df = newDataset.df

# Identify missing values
print(df.info())
mv = df.isna().sum()
print('\nJumlah missing value tiap kolom:\n', mv)

# Identify correlation between numerical variables
numerical_columns = df.select_dtypes(include=[np.number])
correlation = numerical_columns.corr(method='pearson')
print(f'\n correlation \n {correlation}')  # Uncomment to view correlation matrix
df['block'] = pd.cut(df['households'], bins=20, labels=False)

# Handle missing values (using mean for this example)
def fill_nan_with_block_mean(group):
    return group.fillna(group.mean())

df['total_bedrooms'] = df.groupby('block')['total_bedrooms'].transform(fill_nan_with_block_mean)
df.drop(columns=['block'], inplace=True)  # Remove block column after filling

# Encode categorical data
label_encoder_x = LabelEncoder()
df['ocean_proximity'] = label_encoder_x.fit_transform(df['ocean_proximity'])

# Scatter plots (optional)
sns.scatterplot(x='median_income', y='median_house_value', data=df)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Income vs House Value')
plt.show()
plt.figure(figsize=(15,8))
sns.scatterplot(x='latitude', y='longitude', data=df, hue='median_house_value')
scaler = MinMaxScaler()
df[['latitude', 'longitude']] = scaler.fit_transform(df[['latitude', 'longitude']])


X = df.drop(columns=['median_house_value'])
#ekstraksi variabel dependen 
y = df['median_house_value']

def selection(type):
    # Feature selection (using chi-square here)
    selector = SelectKBest(score_func=type, k=7)  # Choose number of features (k)
    X_selected = selector.fit_transform(X, y)
    # Analyze eliminated features
    eliminated_features_index = ~selector.get_support()
    eliminated_features = X.columns[eliminated_features_index]
    print('\n')
    print(f"Fitur-fitur yang tereliminasi {type}:")
    print(eliminated_features)
    return X_selected

# Data splitting
def dataSplitting(type):
    X_train, X_test, y_train, y_test = train_test_split(selection(type), df['median_house_value'], test_size=0.15, random_state=0)
    print(f'Shape of X_train :{X_train.shape}')
    print(f'Shape of X_test :{X_test.shape}')
    print(f'Shape of y_train :{y_train.shape}')
    print(f'Shape of y_test :{y_test.shape}')
    
dataSplitting(chi2)
dataSplitting(f_classif)
dataSplitting(mutual_info_classif)
