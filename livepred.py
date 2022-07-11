import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

wvht = tf.keras.models.load_model('wvhtmodel')
apd = tf.keras.models.load_model('apdmodel')
mwd = tf.keras.models.load_model('mwdmodel')

df = pd.read_csv('daily_collection.csv')
bi = df[df.location == 'block_island']
nj = df[df.location == 'new_jersey']
mn = df[df.location == 'montauk']
nk = df[df.location == 'nantucket']

frames = [bi, nj, mn, nk]
locations = ['block_island',
            'new_jersey',
            'montauk',
            'nantucket']


def get_data(frame, window):
    scaler = MinMaxScaler()
    frame = frame[['SwH', 'SwP', 'MWD']]
    arr = scaler.fit_transform(frame.to_numpy())
    arr = arr[:window, :]
    return arr, scaler

def predict(arr, scaler):
    """
    Returns list of predictions for 4, 8 and 12 hours out
    """
    arr = np.array([arr])
    predicted_vals = []
    for i in range(1, 13):
        wave_pred = wvht.predict(arr)
        pd_pred = apd.predict(arr)
        dir_pred = mwd.predict(arr)
        results = np.squeeze(np.array([wave_pred, pd_pred, dir_pred])).transpose()
        if (i % 4 == 0):
            predicted_vals.append(np.squeeze(scaler.inverse_transform(np.array([results]))).tolist())
        prev = np.squeeze(arr)
        arr = np.vstack((prev, results))[1:, :]
        arr = np.array([arr])
    return predicted_vals

def m_to_f(x):
    return x*3.28084

def dir_to_card(x):
    if 337.5 < x <= 22.5:
        return 'N'
    elif 22.5 < x <= 67.5:
        return 'NE'
    elif 67.5 < x <= 112.5:
        return 'E'
    elif 112.5 < x <= 157.5:
        return 'SE'
    elif 157.5 < x <= 202.5:
        return 'S'
    elif 202.5 < x <= 247.5:
        return 'SW'
    elif 249.5 < x <= 292.5:
        return 'W'
    elif 292.5 < x <= 337.5:
        return 'NW'

def to_frame(predictions, location):
    row = [val for vals in predictions for val in vals]
    print(row)
    cols = ['wvht_4', 'apd_4', 'mwd_4', 'wvht_8', 'apd_8', 'mwd_8'
            , 'wvht_12', 'apd_12', 'mwd_12']
    df = pd.DataFrame([row], columns=cols)
    df[['wvht_4', 'wvht_8', 'wvht_12']] = df[['wvht_4', 'wvht_8', 'wvht_12']].apply(m_to_f)
    df[['mwd_4', 'mwd_8', 'mwd_12']] = df[['mwd_4', 'mwd_8', 'mwd_12']].round(2)
    df['swdir_4'] = df['mwd_4'].astype(str) + " " + df['mwd_4'].apply(dir_to_card)
    df['swdir_8'] = df['mwd_8'].astype(str) + ' ' +df['mwd_8'].apply(dir_to_card)
    df['swdir_12'] = df['mwd_12'].astype(str) + ' ' + df['mwd_12'].apply(dir_to_card)
    df['location'] = location
    return df

pred_frames = []
for i in range(len(frames)):
    d, scaler = get_data(frames[i], 4)
    predictions = predict(d, scaler)
    pred_frames.append(to_frame(predictions, locations[i]))

full_df = pd.concat(pred_frames, ignore_index=True)
full_df.to_csv('daily_predictions.csv')
print(full_df.swdir_12)
