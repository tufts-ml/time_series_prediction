'''
This file, given the parameters, verify if each field in the json file
is accounted for, and that each csv file listen in JSON files are there

Currently checks if CSV Files from JSON file exists in the directory
'''
import numpy as np
import pandas as pd
import argparse
import json
import os
from os import path
parser = argparse.ArgumentParser()
parser.add_argument('--json_file', required=True)
parser.add_argument('--directory', required=True)
parser.add_argument('--step_size', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
json_name = args.json_file
abs_path = os.path.abspath(__file__) + "../../"
location = args.directory
with open(json_name) as json_file:
    data = json.load(json_file)
    fields = data['fields']
    csvFiles = data['names']
    for f in csvFiles:
    	if not os.path.isfile(abs_path+f):
    		print(abs_path+f+" Is not a file!")


        