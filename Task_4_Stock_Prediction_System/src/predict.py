import pandas as pd
import joblib

def load_latest_data():
    path = r"C:\Users\PUSHKAR\Documents\Algonive_ML_Internship\Task_4_Stock_Prediction_System\data\WIPRO.NS.csv"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def predict_next_day():
    # Load data + model
    df = load_latest_data()
    model = joblib.load(
        r"C:\Users\PUSHKAR\Documents\Algonive_ML_Internship\Task_4_Stock_Prediction_System\models\linear_model.pkl"
    )

    # Last day's Close price
    last_close = df['Close'].iloc[-1]
    next_close = model.predict([[last_close]])

    print("Last Close Price:", last_close)
    print("Predicted Next Day Price:", float(next_close[0]))

if __name__ == "__main__":
    predict_next_day()
