import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
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

model_xception = load_model(MODEL_XCEPTION)
model_densenet = load_model(MODEL_DENSENET)
base_model = load_model(MODEL_PATH)


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
    #x = preprocess_input(x, mode='tf')
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
    features = base_model.predict(img_tensor.reshape(1, 224, 224, 3))


    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='tf')
    try:
        preds = model.predict(features)
    except:
        preds = model.predict(features.reshape(1, 7*7*1920))

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/get-model', methods=['GET', 'POST'])
def loadModel(img):
    tes = request.form['model-selector']
    MODEL_XCEPTION = 'models/Xception.h5'
    MODEL_DENSENET = 'models/DenseNet201.h5'

    if tes == 'Xception':
        model_xception = load_model(MODEL_XCEPTION)
        preds = model_predict(img, model_xception)
        
    elif tes == "DenseNet201":
        model_denseNet201 = load_model(MODEL_DENSENET)
        preds = model_predict_densenet(img, model_denseNet201)

    else:
        

    return preds


    #return namaModel

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")   
        # Make prediction
        
        #preds = model_predict(img, model)
        preds = loadModel(img)
        
        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        classes = ['Abyssinian', 'Bengal', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Persian', 'Russian Blue', 'Siamese', 'Sphynx']

        #result = str(classes[0][0][1])               # Convert to string
        result = str(classes[np.argmax(np.array(preds[0]))])
        result = result.replace('_', ' ').capitalize()
        
        ngetes = []
        for idx in preds.argsort()[0][::-1][:5]:
            arr = classes[idx].split("\n-")[-1],"\n{:.2f}%\n".format(preds[0][idx]*100)
            #arr = [ '\n{}'.format(x) for x in arr ]
            ngetes.append(arr)
       
        # Serialize the result, you can add additional fields
        return jsonify(result=ngetes, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()