import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit


model = keras.models.load_model("nn.h5")


def transform_image(pillow_image):
    data = np.asarray(pillow_image)
    data = data / 255.0
    data = data[np.newaxis, ..., np.newaxis]
    # --> [1, x, y, 1]
    data = tf.image.resize(data, [28, 28])
    return data


def predict(x):
    predictions = model(x)
    predictions = tf.nn.softmax(predictions)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    return label0

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
            tensor = transform_image(pillow_img)
            prediction = predict(tensor)
            data = {"prediction": int(prediction)}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"

@socketio.on('Hello')
def handle_message():
    for i in range(30):
        emit('Hello', {'number': i})
        print(f'Hello from {i}')
        time.sleep(1)


if __name__ == "__main__":
    socketio.run(app, debug=False)
