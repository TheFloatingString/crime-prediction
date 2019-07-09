import pandas as pd 
import numpy as np 

weather_data_mtl = pd.read_csv("../datasets/climate-daily.csv")

crime_data = pd.read_csv("../datasets/pdq_3.csv", index_col='Unnamed: 0').drop(columns=['jour','nuit','soir'])
print("crime data shape")
print(crime_data.shape)

preprocessed_weather_data_mtl = np.load("../preprocessed_data/X_train_weather.npy")
preprocessed_weather_data_mtl[preprocessed_weather_data_mtl == np.inf] = 0

print("weather data shape")
print(preprocessed_weather_data_mtl.shape)

for column in crime_data.columns.values:
	for feature in range(len(preprocessed_weather_data_mtl[0])):
		print(str(np.corrcoef(preprocessed_weather_data_mtl[:, feature], crime_data[column].values)[0][1])[0:4], end=' ')
	print()

