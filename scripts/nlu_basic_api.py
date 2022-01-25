# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu.auto_nlu import AutoNLU
from flask import Flask, jsonify, request
import argparse
from flask_cors import CORS
import pandas as pd

# Create app
app = Flask(__name__)
CORS(app)

def initialize(load_folder_path):
    global nlu_model
    nlu_model = AutoNLU.load(load_folder_path)    

@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'hello from NLU service'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_json = request.json
    utterance = input_json["utterance"]
    df = pd.DataFrame(columns=["req","label","conf"])
    df["req"] = [utterance]

    print(utterance)
    response = nlu_model.predict(utterance)
    print(response)
    df["label"] = [response["intent"]["name"]]
    df["conf"] = [response["intent"]["confidence"]]

    df.to_csv("req_DB.csv", encoding="utf-8", quoting=1, header=False, mode="a", index=False)
    print(df)
    del df
    return jsonify(response)


if __name__ == '__main__':    
    # read command-line parameters
    # parser = argparse.ArgumentParser('Running JointNLU model basic service')
    # parser.add_argument('--model', '-m', help = 'Path to joint NLU model', type=str, required=True)
    # parser.add_argument('--port', '-p', help = 'port of the service', type=int, required=False, default=5000)
    #
    # args = parser.parse_args()
    # load_folder_path = args.model
    # port = args.port

    load_folder_path = r"C:\Users\Administrator\Desktop\chatbot2\dialog-nlu-master\saved_models\joint_trans_model"
    port = 5000
    print(('Starting the Server'))
    initialize(load_folder_path)
    # Run app
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)