import pandas as pd

def load_stock_data():
    # CSV file ka exact path
    path = r"C:\Users\PUSHKAR\Documents\Algonive_ML_Internship\Task_4_Stock_Prediction_System\data\WIPRO.NS.csv"

    df = pd.read_csv(path)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df


if __name__ == "__main__":
    data = load_stock_data()
    print(data.head())
    print("\nTotal Rows:", len(data))
