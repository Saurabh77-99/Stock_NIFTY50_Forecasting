import pandas as pd
import numpy as np
import random, os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# Seed setup
# =====================================================
SEED = 9
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OPEN_PATH = 'data/NIFTY_open.csv'
CLOSE_PATH = 'data/NIFTY_close.csv'
MODEL_FOLDER = 'models-NIFTY-240-1-LSTM'
RESULT_FOLDER = 'results-NIFTY-240-1-LSTM'
FEATURE_DATA_FOLDER = 'feature_engineered_data'

for folder in [MODEL_FOLDER, RESULT_FOLDER, FEATURE_DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

df_open = pd.read_csv(OPEN_PATH, index_col=0)
df_close = pd.read_csv(CLOSE_PATH, index_col=0)

df_open.columns = df_open.columns.str.strip().str.split('.').str[0].str.upper()
df_close.columns = df_close.columns.str.strip().str.split('.').str[0].str.upper()

df_open = df_open.apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_close = df_close.apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_open.fillna(method='ffill', inplace=True)
df_close.fillna(method='ffill', inplace=True)

df_open['DATE'] = df_open.index
df_close['DATE'] = df_close.index

def create_label(df_open, df_close, perc=[0.5, 0.5]):
    perc = [0.] + list(np.cumsum(perc))
    numeric_cols = df_open.select_dtypes(include=np.number).columns
    returns = df_close[numeric_cols] / df_open[numeric_cols] - 1
    label = returns.apply(
        lambda x: pd.qcut(x.rank(method='first'), perc, labels=False, duplicates='drop'),
        axis=1
    )
    return label

label = create_label(df_open, df_close)

def create_stock_data(df_open, df_close, st, m=240, save_path=None):
    st_data = pd.DataFrame()
    st_data['DATE'] = df_close['DATE']
    st_data['NAME'] = st

    open_col = f'{st}'
    close_col = f'{st}'

    if open_col not in df_open.columns or close_col not in df_close.columns or st not in label.columns:
        return np.array([]), np.array([])

    daily_change = df_close[close_col] / df_open[open_col] - 1

    for k in range(m)[::-1]:
        st_data[f'IntraR{k}'] = daily_change.shift(k)
    st_data['IntraR-future'] = daily_change.shift(-1)
    st_data['label'] = label[st].values

    st_data = st_data.dropna()

    if save_path and not st_data.empty:
        st_data.to_csv(save_path, index=False)
        print(f"Saved feature data for {st} to {save_path}")

    if len(st_data) == 0:
        return np.array([]), np.array([])

    return np.array(st_data), np.array(st_data)

def scalar_normalize(train_data, test_data):
    scaler = RobustScaler()
    scaler.fit(train_data[:, 2:-2].astype(np.float32))
    train_data[:, 2:-2] = scaler.transform(train_data[:, 2:-2].astype(np.float32))
    test_data[:, 2:-2] = scaler.transform(test_data[:, 2:-2].astype(np.float32))

def makeLSTM():
    inputs = Input(shape=(240, 1))
    x = LSTM(25, return_sequences=False)(inputs)
    x = Dropout(0.1)(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(), metrics=['accuracy'])
    model.summary()
    return model

def trainer(train_data):
    np.random.shuffle(train_data)
    train_x, train_y = train_data[:, 2:-2], train_data[:, -1]
    train_x = np.reshape(train_x, (len(train_x), 240, 1)).astype(np.float32)
    train_y = np.reshape(train_y, (-1, 1)).astype(np.int32)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_y)
    enc_y = enc.transform(train_y).toarray().astype(np.float32)

    model = makeLSTM()
    model.fit(train_x, enc_y, epochs=10, validation_split=0.2, batch_size=512)
    return model

stock_names = [col for col in df_open.columns if col != 'DATE']
valid_stocks = [st for st in stock_names if st in df_close.columns and st in label.columns]

print("Valid stocks for training:", valid_stocks)
if len(valid_stocks) == 0:
    raise ValueError("No valid stock data found. Check your CSV files and label columns!")

train_data, test_data = [], []
for st in valid_stocks:
    csv_save_path = os.path.join(FEATURE_DATA_FOLDER, f"{st}_features.csv")

    st_train_data, st_test_data = create_stock_data(df_open, df_close, st, save_path=csv_save_path)

    if len(st_train_data) > 0:
        train_data.append(st_train_data)
        test_data.append(st_test_data)

train_data = np.concatenate([x for x in train_data])
test_data = np.concatenate([x for x in test_data])
scalar_normalize(train_data, test_data)

model = trainer(train_data)
model.save(os.path.join(MODEL_FOLDER, "final_lstm_nifty.h5"))
print("Training complete and model saved.")