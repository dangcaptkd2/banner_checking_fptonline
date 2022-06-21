#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import logging.config
import yaml
import sys
import argparse
import json

from flask import Flask, request, render_template
from flask_restful import Api, Resource

import sys

import time
from nsfw import detect_nsfw, detect_flag, detect_weapon, detect_crypto

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch  
import random  
import numpy as np
torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)

app = Flask(__name__)
api = Api(app)

app.secret_key = "secret key"

class Stat(Resource):
    def get(self):
        return dict(error=0,message="server start")

@app.route('/nsfw/html', methods =["GET", "POST"])
def upload_check_ocr_url():
	if request.method == "POST":
		start = time.time()
		img_path = request.form.get("fname")
		result = {}
		result_nsfw = detect_nsfw(img_path, draw=True)
		
		if isinstance(result_nsfw, bool):
			result["status_sexy"] =  result_nsfw
			result["status_face_reg"] = None
		else:
			result["status_sexy"] =  False
			result["status_face_reg"] = result_nsfw
	
		result["time_detect_sexy_in"] = round(time.time()-start, 5)
		
		start = time.time()
		result_flag = detect_flag(img_path, draw=True)
		result['flag'] = result_flag
		result["time_detect_flag_in"] = round(time.time()-start, 5)

		start = time.time()
		#result_weapon = detect_weapon(img_path, draw=True)
		# result['weapon'] = result_weapon
		result['weapon'] = False
		result["time_detect_weapon_in"] = round(time.time()-start, 5)

		start = time.time()
		# result_crypto = detect_crypto(img_path, draw=True)
		# result['crypto'] = result_crypto
		result['crypto'] = False
		result["time_detect_crypto_in"] = round(time.time()-start, 5)
		return result
	return render_template("upload.html")
	
def main():
	api.add_resource(Stat, '/')
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', default=6050, help='port(default: 6050)')
	args = parser.parse_args()
	port = int(args.port)
	logging.info(f"Server start: {port}")
	app.debug = True
	app.run("0.0.0.0", port=port, threaded=True)

def test_full(filename):
	print(detect_nsfw(filename))
	print(detect_flag(filename))

if __name__ == "__main__":
	#filename = 'C:/Users/quyennt72/Desktop/Cosplay-meo-dom-sexy-TK2679-3.jpg'
	#test_full(filename)
	main()
