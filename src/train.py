import pandas as pd 
from os import walk
import os

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
pd.set_option('display.max_columns', None)

from data_preprocessing import calculate_RUL, normalize_data, windowing, shuffle_data, split_data

# fix random seed for reproducibility
tf.random.set_seed(1)

def create_compiled_model(window_length):
    model = Sequential()
    model.add(LSTM(128, input_shape = (window_length, 26), activation = "tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = "relu"))
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def scheduler(epoch):
    if epoch < 5:
        return 0.001
    else:
        return 0.0001
    
def train(window_length):
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)

    model = create_compiled_model(window_length)
    history = model.fit(processed_train_data, processed_train_targets, epochs = 10,
                        #validation_split=0.05,
                        validation_data = (processed_val_data, processed_val_targets),
                        callbacks = callback,
                        batch_size = 128, verbose = 2)

    # save the model
    tf.keras.models.save_model(model, "../models/FD001_LSTM_2.h5")

def read_data(file_name):
    data = pd.read_csv(os.path.join("../Data/", file_name+".txt"), sep = "\s+", header = None)
    col_names = ["unit_number", "time"]
    col_names += [f"operation{i}" for i in range(1, 4)]
    col_names += [f"sensor{i}" for i in range(1, 22)]
    data.columns=col_names

    return data

if __name__ == "__main__":
    ## Read and Load Data
    file_names = []
    for (dirpath, dirnames, filenames) in walk("../Data/"):
        file_names.extend(filenames)

    # Training set
    train_FD001 = read_data("train_FD001")

    # Calculate RUL
    train_df = calculate_RUL(train_FD001)

    # Normalize data
    train_data = normalize_data(train_df)

    # Windowing
    processed_train_data, processed_train_targets, window_length = windowing(train_data)

    # Shuffle data
    processed_train_data, processed_train_targets = shuffle_data(processed_train_data, processed_train_targets)

    # Split the dataset into train and validation sets
    processed_train_data, processed_val_data, processed_train_targets, processed_val_targets = split_data(processed_train_data, processed_train_targets)

    # Create model, Train, and save the results
    train(window_length)



    