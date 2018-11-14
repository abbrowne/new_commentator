import praw
import pandas as pd
import numpy as np
import sys
import re
import tensorflow as tf
import keras
import random
from keras.models import Sequential, load_model
from keras.layers import Dense
from praw.models import MoreComments
from flask import Flask, abort, request, render_template
from uuid import uuid4
import requests
import requests.auth
import urllib
import pandas as pd
import json
import predict

app = Flask(__name__)

@app.route('/test',methods = ['POST', 'GET'])
def test(name=None):
  if request.method == 'POST':
    result = request.form
    result = pd.Series(result)
    query_input = str(result[0])
    predictions = predict.main(query_input)
    predictions = json.dumps(predictions[0])
    return render_template('test.html',userinput=predictions)
  else:
    return render_template('test.html')

if __name__ == '__main__':
  app.run(debug = True)





