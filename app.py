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


# configure Flask app
app = Flask(__name__)
app.secret_key = "zjd92kn"
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# home page
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
	return render_template('index.html')


# export heatmap as image file
@app.route('/export', methods=['GET', 'POST'])
def export_to_excel():
	return redirect(url_for('home'))


if __name__ == '__main__':
	app.run(debug=True)