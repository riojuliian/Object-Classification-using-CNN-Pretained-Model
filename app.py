import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, flash
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')



# Model saved with Keras model.save()
# Model saved with Keras model.save()
MODEL_XCEPTION = 'models/Xception.h5'
MODEL_DENSENET = 'models/DenseNet201.h5'
MODEL_PATH = 'models/base_xception.h5'
MODEL_BASEDENS = 'models/DenseNet201Base.h5'
base_model = load_model(MODEL_PATH)


base_modelDens = load_model(MODEL_BASEDENS)


# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')
def model_predict(img, model):
    img = img.resize((224, 224))
    # Preprocessing the image
    img_tensor = image.img_to_array(img)
    img_tensor /= 255.
    features = base_model.predict(img_tensor.reshape(1, 224, 224, 3))
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #img_tensor = preprocess_input(x, mode='tf')
    try:
        preds = model.predict(features)
    except:
        preds = model.predict(features.reshape(1, 7*7*2048))

    return preds

def model_predict_densenet(img, model):
    img = img.resize((224, 224))
    # Preprocessing the image
    img_tensor = image.img_to_array(img)
    img_tensor /= 255.
    features = base_modelDens.predict(img_tensor.reshape(1, 224, 224, 3))

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='tf')
    try:
        preds = model.predict(features)
    except:
        preds = model.predict(features.reshape(1, 7*7*1920))

    return preds

@app.route('/', methods=['GET', 'POST'])
def get_model():
    # Main page

    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        model_xception = load_model(MODEL_XCEPTION)
        img = base64_to_pil(request.json)
        preds = model_predict(img, model_xception)
        # Save the image to ./uploads
        # img.save("./uploads/image.png")   
        # Make predictionpy
        
        #preds = model_predict(img, model
        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        classes = ['Abyssinian', 'Bengal', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Persian', 'Russian Blue', 'Siamese', 'Sphynx']

        #result = str(classes[0][0][1])               # Convert to string
        #result = str(classes[np.argmax(np.array(preds[0]))])
        #result = result.replace('_', ' ').capitalize()
    
        result = []
        for idx in preds.argsort()[0][::-1][:3]:
            arr = classes[idx].split("-")[-1]+ ' : ' + "{:.2f}%\n".format(preds[0][idx]*100)
            #arr = [ '\n{}'.format(x) for x in arr ]
            result.append(arr)
        a="Used Model: Xception"
        # Serialize the result, you can add additional fields
        return jsonify(result=result, prediction=pred_proba, a=a)
        return render_template("index.html", a=a)
    
    
@app.route('/predictDens', methods=['GET', 'POST'])
def predictDens():
    if request.method == 'POST':
        # Get the image from post request
        model_densenet = load_model(MODEL_DENSENET)
        img = base64_to_pil(request.json)
        preds = model_predict_densenet(img, model_densenet)
        # Save the image to ./uploads
        # img.save("./uploads/image.png")   
        # Make predictionpy
        
        #preds = model_predict(img, model
        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        classes = ['Abyssinian', 'Bengal', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Persian', 'Russian Blue', 'Siamese', 'Sphynx']

        #result = str(classes[0][0][1])               # Convert to string
        #result = str(classes[np.argmax(np.array(preds[0]))])
        #result = result.replace('_', ' ').capitalize()
        
        result = []
        for idx in preds.argsort()[0][::-1][:3]:
            arr = classes[idx].split("-")[-1]+ ' : ' + "{:.2f}%\n".format(preds[0][idx]*100)
            #arr = [ '\n{}'.format(x) for x in arr ]
            result.append(arr)
       
        # Serialize the result, you can add additional fields
        return jsonify(result=result, prediction=pred_proba)

    return None

if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()