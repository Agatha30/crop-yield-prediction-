import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from numpy import cov
from scipy.stats import pearsonr

train_df = pd.read_csv(r'C:\Users\Ropa\Desktop\project\yield3.csv')
train_df.dropna(inplace=True)
print(train_df.head())

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# check if the are any missing values in dataset entries, true if yes and false if no
print(train_df.isnull())

# check sum of missing values for each dataset column
print(train_df.isnull().sum())

# drop year of production since its not important
train_df.drop(train_df.columns[[2]], axis=1, inplace=True)
print(train_df.columns)
# drop all missing or null values in the dataset
train_df.dropna(inplace=True)
print(train_df.head())

# Converting Catergoric column data to numeric using one hot encoder
# Crop Name and Place dataset columns contained catergoric variables
crop_data = pd.get_dummies(train_df['Crop_Name'], drop_first=True)
print(crop_data)
place_data = pd.get_dummies(train_df['Place'], drop_first=True)
print(place_data)

# Concatinating catergorically converted numeric data with the original dataset
train_df = pd.concat([train_df, crop_data, place_data], axis=1)
print(train_df)

# Dropping older Crop Name and Place dataset columns since they are now unecessary
train_df.drop(['Crop_Name', 'Place'], axis=1, inplace=True)

# Printing new columns and data for the finally processed dataset
# print finally processed dataset
print('Finally Processed Dataset....')
print(train_df.columns)
print(train_df)

X = train_df[['Rainfall', 'Humidity', 'Temperature', 'Pesticides', 'Soil_ph',
              'N', 'P', 'K', 'Area_Planted']]

y = train_df['Total_production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# X_train = np.array(X_train).reshape(-1, 1)
# X_test = np.array(X_test).reshape(-1, 1)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

from sklearn.ensemble import RandomForestRegressor

# printing the accuracy of the model
# print(accuracy_score(y_test, rf_predict))

# MODEL EVALUATION
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
random_forest = RandomForestRegressor(random_state=0)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print("MAE", mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
print("RMSE", sqrt(mean_squared_error(y_test, y_pred)))
print("R2_Score", r2_score(y_test, y_pred))
train_score = random_forest.score(X_train, y_train)
test_score = random_forest.score(X_test, y_test)
print('\n')
print("The score of the model on test data is:", test_score)
print('\n')
print("The predicted output array is:", y_pred)

# Checking if y-test actual and y_predict are in correlation
plt.scatter(y_test, y_pred)
plt.title('Random Forest Regression Actual Vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

covariance = cov(y_test, y_pred)
print("Covariance of the actual and predicted values is", covariance)

# calculating Pearson's correlation
correlation, _ = pearsonr(y_test, y_pred)
print('Pearsons correlation: %.3f' % correlation)

# iterating through random forest model
models = [RandomForestRegressor()]

for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(type(model).__name__)
    print("MAE", mean_absolute_error(y_test, pred))
    print('MSE: ', mean_squared_error(y_test, pred))
    print("RMSE", sqrt(mean_squared_error(y_test, pred)))
    print("Optimised Model Accuracy is ", train_score)

import pickle

# save the model to disk
filename = 'rr.pkl'
pickle.dump(filename, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))






