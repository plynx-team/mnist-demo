import keras
import numpy as np


def model_summary(model_path):
    model = keras.models.load_model(model_path)
    model.summary()


def batch_inference(model_path, x_val, y_output):
    model = keras.models.load_model(model_path)
    x_val = np.load(x_val)

    res = model.predict(x_val)
    np.save(y_output, res)
