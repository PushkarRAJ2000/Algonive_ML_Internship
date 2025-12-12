import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    path = r"C:\Users\PUSHKAR\Documents\Algonive_ML_Internship\Task_4_Stock_Prediction_System\data\WIPRO.NS.csv"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def plot_close_price(df):
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Close'])
    plt.title("WIPRO Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    plot_close_price(df)
