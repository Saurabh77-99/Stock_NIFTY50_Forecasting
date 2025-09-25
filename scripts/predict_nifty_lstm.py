import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model

OPEN_PATH = 'data/NIFTY_open.csv'
CLOSE_PATH = 'data/NIFTY_close.csv'
MODEL_PATH = 'models-NIFTY-240-1-LSTM/final_lstm_nifty.h5'
RESULT_PATH = 'results-NIFTY-240-1-LSTM/predictions.csv'

model = load_model(MODEL_PATH)

df_open = pd.read_csv(OPEN_PATH, index_col=0)
df_close = pd.read_csv(CLOSE_PATH, index_col=0)

df_open = df_open.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')
df_close = df_close.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')

df_open.columns = [f"{col.split('.')[0]}_OPEN" for col in df_open.columns]
df_close.columns = [f"{col.split('.')[0]}_CLOSE" for col in df_close.columns]

df_open['date'] = df_open.index
df_close['date'] = df_close.index

def create_stock_data(df_open, df_close, st, m=240):
    open_col = f'{st}_OPEN'
    close_col = f'{st}_CLOSE'
    if open_col not in df_open.columns or close_col not in df_close.columns:
        return np.array([]), pd.DataFrame()

    daily_change = df_close[close_col] / df_open[open_col] - 1

    # Build all lagged columns at once
    shifted_cols = {f'IntraR{k}': daily_change.shift(k) for k in range(m)[::-1]}
    shifted_cols['IntraR-future'] = daily_change.shift(-1)
    st_data = pd.concat([df_close[['date']].copy(), pd.DataFrame(shifted_cols)], axis=1)
    st_data['Name'] = st
    st_data.dropna(inplace=True)

    if len(st_data) < m:
        # Not enough data for LSTM sequence
        return np.array([]), pd.DataFrame()

    return np.array(st_data), st_data

def scalar_normalize(train_data, test_data, start_col=1, end_col=None):
    if end_col is None:
        end_col = train_data.shape[1] - 1
    scaler = RobustScaler()
    scaler.fit(train_data[:, start_col:end_col].astype(np.float32))
    train_data[:, start_col:end_col] = scaler.transform(train_data[:, start_col:end_col].astype(np.float32))
    test_data[:, start_col:end_col] = scaler.transform(test_data[:, start_col:end_col].astype(np.float32))

m = 240
stock_names = [col.split("_")[0] for col in df_open.columns if "_OPEN" in col]
all_predictions = []

for st in stock_names:
    st_array, st_df = create_stock_data(df_open, df_close, st, m=m)
    if st_array.size == 0:
        continue

    scalar_normalize(st_array, st_array, start_col=1, end_col=1+m)

    test_x = st_array[:, 1:1+m]
    test_x = test_x[-(len(test_x)//m*m):]
    test_x = np.reshape(test_x, (-1, m, 1)).astype(np.float32)

    preds = model.predict(test_x)
    pred_classes = np.argmax(preds, axis=1)

    st_df = st_df.iloc[-len(pred_classes):].copy()
    st_df['Predicted'] = pred_classes
    all_predictions.append(st_df[['date', 'Name', 'Predicted']])

if all_predictions:
    result_df = pd.concat(all_predictions)
    result_df['Predicted'] = result_df['Predicted'].map({0:'Down', 1:'Up'})
    result_df.to_csv(RESULT_PATH, index=False)
    print(f"Predictions saved to {RESULT_PATH}")
else:
    print("No valid stock data found for prediction!")