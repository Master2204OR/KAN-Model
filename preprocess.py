import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle


def preprocess_data(file = 'data/creditcard.csv', test_data = 0.2):
    data = pd.read_csv(file)



































def load_credit_card_data(filepath='data/creditcard.csv', test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)

    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows.")

    
    df.fillna(df.mean(numeric_only=True), inplace=True)

    if 'Time' in df.columns:
        df.drop(columns=['Time'], inplace=True)

    scaler = RobustScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    X = df.drop(columns=['Class']).values
    y = df['Class'].values

    X, y = shuffle(X, y, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return (X_train, y_train), (X_test, y_test)
