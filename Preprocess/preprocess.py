import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle



def preprocess_data(file = 'creditcard.csv', test_size = 0.2,  random_state=42):

    data = pd.read_csv(file)

    data.drop_duplicates(inplace=True)
    print("Removed duplicate rows.")

    data.fillna(data.mean(numeric_only= True) ,inplace=True)
    print("Filled missing values with column means.")

    data.drop(columns=['Time'], inplace=True)
    print("Dropped 'Time' column as it is not needed.")

    scaler = RobustScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'])

    X = data.drop(columns=['Class']).values
    y = data['Class'].values

    X = data.drop(columns=['Class']).values
    y = data['Class'].values

    X, y = shuffle(X, y, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    return (X_train, y_train), (X_test, y_test)


(X_train, y_train), (X_test, y_test) = preprocess_data(file='creditcard.csv', test_size=0.2, random_state=42)








































    