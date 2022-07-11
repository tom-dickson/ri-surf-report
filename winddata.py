import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

locations = ['Narragansett', 'East Matunuck']

gansetturl = 'https://f1.weather.gov/MapClick.php?w3=sfcwind&w3u=1&w10u=0&w13u=1&w14u=1&AheadHour=0&Submit=Submit&FcstType=digital&textField1=41.4271&textField2=-71.4669&site=all&unit=0&dd=&bw='
matunuckurl = 'https://f1.weather.gov/MapClick.php?w3=sfcwind&w3u=1&w10u=0&w13u=1&w14u=1&AheadHour=0&Submit=Submit&FcstType=digital&textField1=41.3783&textField2=-71.5255&site=all&unit=0&dd=&bw='

gansett = requests.get(gansetturl)
matunuck = requests.get(matunuckurl)

wind_data = []
vals = [gansett, matunuck]
for i in range(len(vals)):
    html1 = bs(vals[i].text, 'html.parser')
    html2 = bs(vals[i].text, 'html.parser')

    nums = html1.find_all('font', color="#990099")
    nums = [line.get_text() for line in nums][1:13:4]
    dirs = html1.find_all('font', color="#666666")
    dirs = [line.get_text() for line in dirs][1:13:4]
    full = [locations[i]] + [str(nums[i]) + ' ' + dirs[i] for i in range(len(dirs))]
    wind_data.append(full)

df = pd.DataFrame(wind_data, columns=['location', 'wind_4hr', 'wind_8hr', 'wind_12hr'])
df.to_csv('wind.csv')
print(df)
