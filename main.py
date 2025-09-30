import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#Load dataset
data = pd.read_csv('insurance.csv')
view_top5 = data.head()
view_lat5 = data.tail()
view_unique = data.nunique()
view_info = data.info() 
view_desc = data.describe()
view_missing = data.isnull().sum()
view_duplicates = data.duplicated().sum()

print("Top 5 rows:\n", view_top5)
print("Last 5 rows:\n", view_lat5)
print("Unique values per column:\n", view_unique)
print("Dataset info:\n", view_info)
print("Statistical description:\n", view_desc)
print("Missing values per column:\n", view_missing)
print("Number of duplicate rows:\n", view_duplicates)

#Visualize and get insights
def age_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['age'], bins=10, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

def gender_distribution(data):
    plt.figure(figsize=(6, 4))
    sns.countplot(data['sex'])
    plt.title('Gender Distribution')
    plt.xlabel('Count')
    plt.ylabel('Gender')
    plt.show()

def smoker_distribution(data):
    plt.figure(figsize=(6,4))
    sns.countplot(x= 'smoker', data=data)
    plt.title('Smoker Distribution')
    plt.xlabel('Smoker')
    plt.ylabel('Count')
    plt.show()
    
def region_distribution(data):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='region', data=data)
    plt.title('Region Distribution')
    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.show()

def charges_distribution(data):
    plt.figure(figsize=(8, 5))
    plt.scatter(data['charges'], data['age'], alpha=0.5) #Change this depending on what you want to compare
    plt.title('Charges Distribution')
    plt.xlabel('Charges')
    plt.ylabel('Count')
    plt.show()

#Preprocess data
def preprocess_data(data):
    data = data.dropna()
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
    scaler = StandardScaler()
    data[['age', 'bmi']] = scaler.fit_transform(data[['age', 'bmi']])
    return data

print("Verify the shape:\n", data.shape)

#train-test split
data = preprocess_data(data)
X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#Model training and evaluation
def linear_regression_model(X_train, y_train, X_test, y_test): #Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R^2 Score: {r2}')
    print(f'Mean Absolute Error: {mae}')
    return y_pred

#Plot actual vs predicted
def plot_actual_vs_predicted(y_test, y_pred):
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, c=errors, cmap='coolwarm', alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Charges')
    plt.show()

