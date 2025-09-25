import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

RESULT_CSV = 'results-NIFTY-240-1-LSTM/predictions.csv'
MODEL_PATH = 'models-NIFTY-240-1-LSTM/final_lstm_nifty.h5'

model = load_model(MODEL_PATH)

df_prev = pd.read_csv(RESULT_CSV)
df_prev['date'] = pd.to_datetime(df_prev['date']) 

last_date = df_prev['date'].max()
tomorrow = last_date + pd.Timedelta(days=1)

df_numeric = df_prev.copy()
df_numeric['Numeric'] = df_numeric['Predicted'].map({'Down': 0, 'Up': 1})

stocks = df_numeric['Name'].unique()
stock_matrix = pd.DataFrame(index=df_numeric['date'].unique(), columns=stocks)

for st in stocks:
    stock_matrix[st] = df_numeric[df_numeric['Name'] == st].set_index('date')['Numeric']

stock_matrix = stock_matrix.fillna(method='ffill')

window = 240
all_predictions = []

for st in stocks:
    if len(stock_matrix[st]) < window:
        continue

    x_input = stock_matrix[st].values[-window:].reshape(1, window, 1).astype(np.float32)

    scaler = RobustScaler()
    x_input = scaler.fit_transform(x_input.reshape(-1,1)).reshape(1, window, 1)

    pred = model.predict(x_input)
    pred_class = np.argmax(pred, axis=1)[0]

    all_predictions.append({
        'Date': tomorrow.strftime('%Y-%m-%d'),
        'Name': st,
        'Predicted': 'Up' if pred_class == 1 else 'Down'
    })

result_df = pd.DataFrame(all_predictions)
print(result_df)

result_df.to_csv('results-NIFTY-240-1-LSTM/predictions_nifty_tomorrow.csv', index=False)
print(f"Predictions for {tomorrow.strftime('%Y-%m-%d')} saved to predictions_nifty_tomorrow.csv")
