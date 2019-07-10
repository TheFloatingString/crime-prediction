import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

df_unique_crime_pdq = pd.read_csv("datasets/pdq_1.csv", index_col="Unnamed: 0").drop(columns=["jour", "nuit", "soir"])
print(df_unique_crime_pdq.head())
# df_nasdaq = pd.read_csv("datasets/NASDAQ_DATA.csv", index_col="Date")
# print(df_nasdaq.head())
nasdaq_data = np.load("preprocessed_data/X_train_nasdaq.npy")
print(nasdaq_data[0])
weather_array = np.load("preprocessed_data/X_train_weather.npy")

# x[~np.isnan(x).any(axis=1)]
# x[x == -inf] = 0


X_data = np.concatenate((nasdaq_data, weather_array), axis=1)
y_data = df_unique_crime_pdq.values

X_data[X_data == np.inf] = 0
X_data[X_data == -np.inf] = 0

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

print(weather_array[0])
clf = LinearRegression()
clf.fit(X_data, y_data)
print(clf.score(X_test, y_test))