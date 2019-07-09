# preprocessing weather data

import pandas as pd 
import numpy as np

df = pd.read_csv('D:/MeteoHack/crime-prediction/weather_data/climate-daily.csv', header=0, index_col='LOCAL_DATE',
				usecols=['LOCAL_DATE', 'MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_PRECIPITATION', 
				'SNOW_ON_GROUND', 'DIRECTION_MAX_GUST', 'SPEED_MAX_GUST', 'HEATING_DEGREE_DAYS', 'MIN_REL_HUMIDITY', 'MAX_REL_HUMIDITY'])

for column in df.columns:
	norm_col = column + '_VARIANCE'
	df[norm_col] = df[column].pct_change()

df = df.fillna(0)

X_train_climate = df.to_numpy()
np.save('X_train_weather.npy', X_train_climate)