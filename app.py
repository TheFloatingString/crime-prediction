"""
Flask app file to be run to render web app
"""

# Import modules
from flask import Flask, flash, redirect, url_for, render_template, request, Response, send_file, make_response, abort, session
import glob
import os
import re
import json

import pandas as pd
import datetime

from src.draw_map import * 


# configure Flask app
app = Flask(__name__)
app.secret_key = "zjd92kn"
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


default_X_input_data = [ 7.24181982e+03,7.29174023e+03,7.23575977e+03,7.26520996e+03
,1.93998000e+09,2.49303893e-02,8.78226158e-03,2.52015299e-02
,7.71050944e-03,-4.06729205e-02,1.41000000e+01,9.20000000e+00
,1.90000000e+01,0.00000000e+00,0.00000000e+00,1.00000000e+00
,3.70000000e+01,3.90000000e+00,0.00000000e+00,0.00000000e+00
,5.22388060e-02,3.52941176e-01,-4.52261307e-02,0.00000000e+00
,0.00000000e+00,-9.62962963e-01,-4.12698413e-01,-1.52173913e-01
,0.00000000e+00,0.00000000e+00]

# home page
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
	return render_template('index.html', draw_pdq_code=return_coords(default_X_input_data))

@app.route('/get_pdq_data/<pdq_number>', methods=['GET'])
def get_pdq_data(pdq_number):
	return compute_level(default_X_input_data, str(pdq_number))

# export heatmap as image file
@app.route('/export', methods=['GET', 'POST'])
def export_to_excel():
	return redirect(url_for('home'))


if __name__ == '__main__':
	app.run(debug=True)