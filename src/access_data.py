# access climate data thorugh GEOMET service

import requests 
import os
import io
import pandas as pd 
from datetime import date, timedelta


current_date = str(date.today() - timedelta(days=2))
previous_date = str(date.today() - timedelta(days=3))

base_url = """https://geo.weather.gc.ca/geomet/features/collections/climate-daily/items?time={}%2000:00:00/{}%2000:00:00&STN_ID=30165&sortby=PROVINCE_CODE,STN_ID,LOCAL_DATE&f=csv&limit=150000&offset=0""".format(previous_date, current_date)

print(base_url)

def get_climate_data(request_url):
	r = requests.get(request_url)
	data = r.content.decode('utf8')
	df_head = pd.read_csv(io.StringIO(data), header=0)

	

	return df_head




df_data = get_climate_data(base_url)

