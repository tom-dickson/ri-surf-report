import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

block = pd.read_fwf('bi.txt')
block=block.iloc[::2, :].reset_index()
nj = pd.read_fwf('nj.txt')
montauk = pd.read_fwf('mk.txt')
nantucket = pd.read_fwf('nk.txt')

block['location'] = 'block_island'
nj['location'] = 'new_jersey'
montauk['location'] = 'montauk'
nantucket['location'] = 'nantucket'

print(nj.head())

frames = [block, nj, montauk, nantucket]
df = pd.concat(frames, ignore_index=True)

df = df[['MM', 'DD', 'hh', 'SwH', 'SwP', 'MWD', 'location']]
df.to_csv('all_data.csv')


def scale(frame):
    """
    Selects and scales data from data frame
    """
    scaler = MinMaxScaler()
    arr = frame[['SwH', 'SwP', 'MWD']].astype(float).to_numpy()
    print(arr)
    arr_scaled = scaler.fit_transform(arr)
    df = pd.DataFrame(arr_scaled, columns=['SwH', 'SwP', 'MWD'])
    return df


def prepare_windows(location, frame, window):
    """
    Prepares/splits time series data for input into model
    """
    subset = scale(frame[frame.location==location])
    print(subset.head())
    windows = []
    data = subset.to_numpy()
    for i in range(data.shape[0] - window):
        windows.append(data[i : i+window, :])
    arr = np.array(windows)
    return arr

bi_prep = prepare_windows('block_island', df, 5)
nj_prep = prepare_windows('new_jersey', df, 5)
mn_prep = prepare_windows('montauk', df, 5)
na_prep = prepare_windows('nantucket', df, 5)

full_arr = np.vstack((bi_prep, nj_prep, mn_prep, na_prep))

def train_test_split(arr, amt=.8):
    index = int(len(arr)*amt)
    train = arr[:index, :, :]
    test = arr[index:, :, :]
    return train[:, :-1, :], train[:, -1:, :], test[:, :-1, :], test[:, -1:, :]

def train_save_model(output, filename: str):
    """
    Output param: 0 for wave height, 1 for ave period, 2 for direction
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='linear'))
    model.compile(optimizer='adam',
                    loss='mse',
                    metrics=['accuracy'])

    X_train, y_train, X_test, y_test = train_test_split(full_arr)
    model.fit(X_train, y_train[:, :, output], epochs=5)

    predicted = model.predict(bi_prep[:,:-1,:])
    target = np.squeeze(bi_prep[:,-1:, output])
    plt.plot(predicted)
    plt.plot(target, color='r', alpha=.8)
    plt.show()
    model.save(filename)

train_save_model(0, 'wvhtmodel')
train_save_model(1, 'apdmodel')
train_save_model(2, 'mwdmodel')
