
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import os
from sklearn.model_selection import train_test_split

from keras.models import load_model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
import time


# In[6]:


weather_data = np.load("../preprocessed_data/X_train_weather.npy")
nasdaq_data = np.load("../preprocessed_data/X_train_nasdaq.npy")
print(weather_data.shape, nasdaq_data.shape)


# In[3]:


base = datetime.datetime.strptime("2015-01-01", "%Y-%m-%d")
end = datetime.datetime.strptime("2019-07-06", "%Y-%m-%d")
delta = end - base
unique_PI_dates = [str(base + datetime.timedelta(days=x)).split(' ')[0] for x in range(delta.days + 1)]
print(len(unique_PI_dates))


# In[ ]:


def crime_prediction_model():
    pass


# In[7]:


pdq_list = [1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 
            10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 
            20.0, 21.0, 22.0, 23.0, 24.0, 26.0, 
            27.0, 30.0, 31.0, 33.0, 35.0, 38.0, 
            39.0, 42.0, 44.0, 45.0, 46.0, 48.0, 
            49.0, 50.0, 55.0]

for pdq in pdq_list:
    pdq_filename = "pdq_" + str(int(pdq)) + '.csv'
    
    pdq_df = pd.read_csv("../datasets/" + pdq_filename, index_col='Unnamed: 0')
    target_df = np.split(pdq_df, [6], axis=1)[0]
    daytime_df = np.split(pdq_df, [6], axis=1)[1]
    
    y = target_df.to_numpy()
    X = np.concatenate((weather_data, nasdaq_data, daytime_df.to_numpy()), axis=1)
    
    print(X.shape, y.shape)
    

