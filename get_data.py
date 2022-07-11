import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

biurl = 'https://www.ndbc.noaa.gov/data/realtime2/44097.spec'
mnurl = 'https://www.ndbc.noaa.gov/data/realtime2/44017.spec'
nkurl = 'https://www.ndbc.noaa.gov/data/realtime2/44008.spec'
njurl = 'https://www.ndbc.noaa.gov/data/realtime2/44066.spec'

urls = [biurl, mnurl, nkurl, njurl]
names = ['bi.txt', 'mk.txt', 'nk.txt', 'nj.txt']
locations = ['block_island', 'montauk', 'nantucket', 'new_jersey']

def get_frames():
    frames = []
    for i in range(len(names)):
        html = requests.get(urls[i])
        with open(names[i], 'w') as f:
            count = 0
            for line in html.iter_lines():
                line = line.decode("utf-8")
                if line.startswith('#yr'):
                    continue
                if 'MM' in line and count > 0:
                    continue
                f.write(line)
                f.write('\n')
                count += 1
        df = pd.read_fwf(names[i])
        if urls[i] == biurl:
            df = df.iloc[::2, :].reset_index()
        df = df[['MM', 'DD', 'hh',	'SwH',	'SwP', 'MWD']]
        df['location'] = locations[i]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

all_data = get_frames()
all_data.to_csv('daily_collection.csv')
print(all_data.shape)
