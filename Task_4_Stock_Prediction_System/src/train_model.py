import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def load_data():
    path = r"C:\Users\PUSHKAR\Documents\Algonive_ML_Internship\Task_4_Stock_Prediction_System\data\WIPRO.NS.csv"
    df = pd.read_csv(path)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    return df

def train_model():
    df = load_data()

    # Feature = previous day's Close price
    df['Prev_Close'] = df['Close'].shift(1)
    df = df.dropna()

    X = df[['Prev_Close']]        # input
    y = df['Close']               # target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    # model save
    joblib.dump(model, r"C:\Users\PUSHKAR\Documents\Algonive_ML_Internship\Task_4_Stock_Prediction_System\models\linear_model.pkl")

    print("Model trained successfully.")
    print("Test Accuracy (R² Score):", score)

if __name__ == "__main__":
    train_model()
