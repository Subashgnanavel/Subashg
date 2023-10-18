import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Upload dataset  'Data_Gov_Tamil_Nadu.csv'
data = pd.read_csv('Data_Gov_Tamil_Nadu.csv')

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# Example preprocessing steps
# Handle missing values
data.fillna(0, inplace=True)

# Encode categorical variables (if any)
data = pd.get_dummies(data, columns=['categorical_column'])

# Standardize numerical features
scaler = StandardScaler()
data['numerical_column'] = scaler.fit_transform(data[['numerical_column']])

# Example data visualization
sns.countplot(x='categorical_column', data=data)
plt.title('Distribution of Categorical Column')
plt.show()

X = data.drop('target_column', axis=1)
y = data['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
