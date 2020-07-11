'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
import logging
import keras
import numpy as np
from keras.callbacks import CSVLogger

import mnist_demo.models


def train(x_train, y_train, x_test, y_test, model_path=None, logs_path=None):
    batch_size = 128
    num_classes = 10
    epochs = 4

    # input image dimensions
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    x_train = np.load(x_train)
    x_test = np.load(x_test)
    y_train = np.load(y_train)
    y_test = np.load(y_test)

    logging.info('x_train shape: {}'.format(x_train.shape))
    logging.info('{} train samples'.format(x_train.shape[0]))
    logging.info('{} test samples'.format(x_test.shape[0]))

    model = mnist_demo.models.get_sequential_model(input_shape, num_classes)

    csv_logger = CSVLogger('log.csv', append=True, separator=',')
    callbacks = []
    if logs_path:
        csv_logger = CSVLogger(logs_path, separator=',')
        callbacks.append(csv_logger)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=callbacks,
              )
    score = model.evaluate(x_test, y_test, verbose=0)
    logging.info('Test loss: {}'.format(score[0]))
    logging.info('Test accuracy: {}'.format(score[1]))

    if model_path:
        model.save(model_path)
