"""
Flask app file to be run to render web app
"""

# HWUR7ESMZY4OVM5S

# Import modules
from flask import Flask, flash, redirect, url_for, render_template, request, Response, send_file, make_response, abort, session
import glob
import os
import re
import json

import pandas as pd
import datetime

from src.draw_map import * 
from src.darksky_request import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from yahoo_fin import stock_info as si

import numpy as np


# configure Flask app
app = Flask(__name__)
app.secret_key = "zjd92kn"
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

X_min_max_scaler = joblib.load("static/saved_models/X_min_max_scaler.pkl")


# home page
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
	temp, wind, humidity = get_climate_data("https://api.darksky.net/forecast/2dfdcc237ebdb8ba4c7b2360152a9a87/45.508889,-73.561667")
	nasdaq = si.get_live_price('^ixic')
	return render_template('index.html', draw_pdq_code=return_coords(default_X_input_data), temp=str(temp)[:4], wind=wind, humidity=humidity, nasdaq=str(nasdaq)[:6])

@app.route('/get_pdq_data/<pdq_number>', methods=['GET'])
def get_pdq_data(pdq_number):
	temp, wind, humidity = get_climate_data("https://api.darksky.net/forecast/2dfdcc237ebdb8ba4c7b2360152a9a87/45.508889,-73.561667")
	nasdaq = si.get_live_price('^ixic')
	default_X_input_data = [ nasdaq, nasdaq ,nasdaq,nasdaq
	,1.93998000e+09,2.49303893e-02,8.78226158e-03,2.52015299e-02
	,7.71050944e-03,-4.06729205e-02, temp, temp
	,temp,0.00000000e+00,0.00000000e+00,1.00000000e+00
	,wind,3.90000000e+00,humidity,humidity
	,5.22388060e-02,3.52941176e-01,-4.52261307e-02,0.00000000e+00
	,0.00000000e+00,-9.62962963e-01,-4.12698413e-01,-1.52173913e-01
	,0.00000000e+00,0.00000000e+00]
	default_X_input_data = X_min_max_scaler.fit_transform(np.array(default_X_input_data).reshape(1,-1))[0]

	return compute_level(default_X_input_data, str(pdq_number))

# export heatmap as image file
@app.route('/export', methods=['GET', 'POST'])
def export_to_excel():
	return redirect(url_for('home'))


if __name__ == '__main__':
	app.run(debug=True)
