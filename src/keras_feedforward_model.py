import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# from keras.models import Sequential
# from keras.layers import Dense

weather_data_mtl = pd.read_csv("datasets/climate-daily.csv")

pdq_list = [1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 20.0, 21.0, 22.0, 23.0, 24.0, 26.0, 27.0, 30.0, 31.0, 33.0, 35.0, 38.0, 39.0, 42.0, 44.0, 45.0, 46.0, 48.0, 49.0, 50.0, 55.0]
print(len(pdq_list))

crime_data = pd.read_csv("datasets/pdq_3.csv", index_col='Unnamed: 0').drop(columns=['jour','nuit','soir'])
print("crime data shape")
print(crime_data.shape)

preprocessed_weather_data_mtl = np.load("preprocessed_data/X_train_weather.npy")
preprocessed_weather_data_mtl[preprocessed_weather_data_mtl == np.inf] = 0
preprocessed_weather_data_mtl[preprocessed_weather_data_mtl == np.nan] = 0
preprocessed_weather_data_mtl = preprocessed_weather_data_mtl[np.logical_not(np.isnan(preprocessed_weather_data_mtl))].reshape(1648,20)

min_max_scaler = MinMaxScaler()
print(preprocessed_weather_data_mtl[:,0:10].shape)
preprocessed_weather_data_mtl = min_max_scaler.fit_transform(preprocessed_weather_data_mtl[:,0:10])

print("weather data shape")
print(preprocessed_weather_data_mtl.shape)

print()
print("ML Linear Regression...")

plt.figure(figsize=(20,10))

output_list = []

for index in range(len(pdq_list)):

	# model = Sequential()
	# model.add(Dense(10, input_shape=(10,), activation="sigmoid"))
	# model.add(Dense(10, activation="sigmoid"))
	# model.add(Dense(6, activation="relu"))
	# model.compile(optimizer="Adam", loss="mse", metrics=['acc', 'mse'])

	print("PDQ: "+str(pdq_list[index]))

	crime_data = pd.read_csv("datasets/pdq_"+str(int(pdq_list[index]))+".csv", index_col='Unnamed: 0').drop(columns=['jour','nuit','soir'])

	X_data = preprocessed_weather_data_mtl
	y_data = crime_data.values

	max_value = np.max(y_data)
	avg_value = np.mean(y_data)

	X_train, X_test, y_train, y_test = train_test_split(X_data[:,0:10], y_data, test_size=0.2)

	print(X_train.shape)
	print(y_train.shape)

	# model.fit(X_train, y_train, epochs=20, batch_size=16)
	# result = model.evaluate(X_test, y_test, batch_size=32)[0]

	acc = 0.0
	for x, y in zip(X_train, y_train):
		row_acc = 0.0
		for y_ in y:
			row_acc += (avg_value-y_)**2
		acc += row_acc/6

	result = acc/len(X_train)

	print(result)

	output_list.append(result)

print(dict(zip(pdq_list, output_list)))

