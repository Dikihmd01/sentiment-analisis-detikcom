import tensorflow as tf
import numpy as np
import os

from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json

app = Flask(__name__)
dir_name = 'static'
model = load_model('./model/model_v3.h5')
with open('./model/json_tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    sentences = []
    if request.method == 'POST':
        text = request.form['sentences']
        if text != '':         
            sentences.append(text)
            max_len = 89

            pred_sequence = tokenizer.texts_to_sequences(sentences)
            pred_padded = pad_sequences(pred_sequence, maxlen=max_len)
            predicted = model.predict(pred_padded).round()
            scores = model.predict(pred_padded)
            accuracy = max(scores[0])

            print(sentences)
            print(predicted)
            

            sentiment = ""
            if predicted[0][0] == 1.:
                sentiment += "negative"
            elif predicted[0][1] == 1.:
                sentiment += "neutral"
            elif predicted[0][2] == 1.:
                sentiment += "positive"
            
            data = {
                'text': text.lower(),
                'prediction': sentiment,
                'accuracy': round(accuracy, 2)
            }
            
            return render_template('index.html', data=data)

# if __name__ == '__main__':
#     app.run(port=3000, debug=True)