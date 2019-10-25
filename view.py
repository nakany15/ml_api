# coding: utf-8
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# load trained model
clf = joblib.load('./model/brest_cancer.pkl')

@app.route('/', methods = ['POST'])
def predict():
    x = request.json['x']
    y = clf.predict([x])[0]
    ret = {'y': int(y)}
    return jsonify(ret)

# main
if __name__ == '__main__':
    app.run(debug = True)