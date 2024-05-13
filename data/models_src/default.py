from typing import List
import keras
import numpy as np
from keras import layers

def train_model_features(X, Y, model_name):
    X_val = X[-5:]
    Y_val = Y[-5:]

    xn = len(X[0])

    inputs = keras.Input(shape=(xn,))
    normalization = keras.layers.Normalization(axis=None)
    dense = layers.Dense(16, activation="relu")
    x = normalization(inputs)
    x = dense(inputs)
    #x = keras.layers.BatchNormalization()(x)
    x = layers.Dense(16, activation="relu")(x)
    #x = keras.layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation="sigmoid", name="parkinsons")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.BinaryCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.Accuracy()],
    )

    history = model.fit(
        X,
        Y,
        batch_size=1,
        epochs=5,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        #validation_data=(X_val, Y_val),
    )

    return (history, model)

def train_model_primitives(X, Y, WIDTH, HEIGHT, model_name, primitives_list, output_num):
    X_val = X[-5:]
    Y_val = Y[-5:]

    inputs = keras.Input(shape=(WIDTH,HEIGHT))
    flatten = layers.Flatten()(inputs)
    dense = layers.Dense(32, activation="relu")
    x = dense(flatten)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(output_num, activation="softmax", name="primitives")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.CategoricalAccuracy()],
    )

    history = model.fit(
        X,
        Y,
        batch_size=1,
        epochs=3,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
    )

    return (history, model)