import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


data = pd.read_csv("Data\creditcard.csv")
print(len(data))

#Data cleaning
data.fillna(data.mean(), inplace=True)
#data.dropna(inplace=True)
print(len(data)) #no empty cells

data.drop_duplicates(inplace=True)
print(len(data))

data.drop(columns=['Time'], inplace=True) #information not needed

#Normalization
scaler = RobustScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])


#train test split
X = data.drop(columns=['Class'])
y = data['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set size:", len(X_train))
print("Test set size:", len(X_test))




