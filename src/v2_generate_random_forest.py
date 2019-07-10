import pandas as pd
import csv
import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from math import *


pdq_list = [1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 20.0, 21.0, 22.0, 23.0, 24.0, 26.0, 27.0, 30.0, 31.0, 33.0, 35.0, 38.0, 39.0, 42.0, 44.0, 45.0, 46.0, 48.0, 49.0, 50.0, 55.0]

for pdq_number in pdq_list:

	print("pdq_number: "+str(pdq_number))
	df_unique_crime_pdq = pd.read_csv("datasets/pdq_"+str(int(pdq_number))+".csv", index_col="Unnamed: 0").drop(columns=["jour", "nuit", "soir"])
	print(df_unique_crime_pdq.head())
	# df_nasdaq = pd.read_csv("datasets/NASDAQ_DATA.csv", index_col="Date")
	# print(df_nasdaq.head())
	nasdaq_data = np.load("preprocessed_data/X_train_nasdaq.npy")
	# print(nasdaq_data[0])
	weather_array = np.load("preprocessed_data/X_train_weather.npy")

	# x[~np.isnan(x).any(axis=1)]
	# x[x == -inf] = 0

	X_data = np.concatenate((nasdaq_data, weather_array), axis=1)
	y_data = df_unique_crime_pdq.values

	X_data[X_data == np.inf] = 0
	X_data[X_data == -np.inf] = 0

	# scaling data
	X_min_max_scaler = MinMaxScaler()
	X_data = X_min_max_scaler.fit_transform(X_data)
	joblib.dump(X_min_max_scaler, "static/saved_models/X_min_max_scaler.pkl")

	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

	# print(weather_array[0])
	clf = RandomForestRegressor()
	clf.fit(X_data, y_data)
	joblib.dump(clf, "static/saved_models/v2_rn_forest_"+str(pdq_number)+".pkl")
	print(clf.score(X_test, y_test))

	# mse
	predictions = clf.predict(X_test)
	mse = sqrt(mean_squared_error(predictions, y_test))
	print("mse: "+str(mse))

	random_test_index = random.randrange(0, len(X_test))
	# print(random_test_index)

	# print(clf.predict([X_test[random_test_index]]))
	# print(y_test[random_test_index])