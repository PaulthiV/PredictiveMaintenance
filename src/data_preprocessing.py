import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)

# fix random seed for reproducibility
tf.random.set_seed(1)

def calculate_RUL(train_FD001):
    # Calculating RUL
    rul = train_FD001["unit_number"]
    max_rul = train_FD001.groupby("unit_number")["time"].count().values
    rul_c = []
    for i in rul:
        rul_c.append(max_rul[i-1])

    train_FD001["RUL"] = rul_c - train_FD001["time"]

    return train_FD001

def normalize_data(train_df):
    # Normalize dataset
    scaler = StandardScaler() # Try MinMaxScaler
    train_data = scaler.fit_transform(train_df.iloc[:,1:-1])

    train_data = pd.DataFrame(data = np.c_[train_df.iloc[:,0], train_data, train_df.iloc[:,-1]])
    
    return train_data

def windowing(train_data):
    # Unique engines
    num_train_machines = len(train_data[0].unique())

    # Windowing or reshaping into (samples, time steps, features)
    input_data = train_data.iloc[:,:-1]
    target_data = train_data.iloc[:,-1]
    window_length = 50
    shift = 10
    processed_train_data = []
    processed_train_targets = []

    # Windowing per engine
    for i in np.arange(1, num_train_machines+1):
        temp_train_data = train_data.loc[train_data[0] == i].drop(columns = [0]).values

        num_batches = int((len(input_data) - window_length)/shift)+ 1 
        num_features = input_data.shape[1]
        output_data = np.repeat(np.nan, repeats = num_batches * window_length * num_features).reshape(num_batches, window_length,
                                                                                                        num_features)
        output_targets = np.repeat(np.nan, repeats = num_batches)
        for batch in range(num_batches):
            output_data[batch,:,:] = input_data.iloc[(0+shift*batch):(0+shift*batch+window_length),:]
            output_targets[batch] = target_data.iloc[(shift*batch + (window_length-1))]
        
        processed_train_data.append(output_data)
        processed_train_targets.append(output_targets)
    processed_train_data = np.concatenate(processed_train_data)
    processed_train_targets = np.concatenate(processed_train_targets)

    return processed_train_data, processed_train_targets, window_length

def shuffle_data(processed_train_data, processed_train_targets):
    # Shuffle training data
    index = np.random.permutation(len(processed_train_targets))
    processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]

    print("Processed trianing data shape: ", processed_train_data.shape)
    print("Processed training ruls shape: ", processed_train_targets.shape)

    return processed_train_data, processed_train_targets

def split_data(processed_train_data, processed_train_targets):

    processed_train_data, processed_val_data, processed_train_targets, processed_val_targets = train_test_split(processed_train_data,
                                                                                                                processed_train_targets,
                                                                                                                test_size = 0.2,
                                                                                                                random_state = 83)
    print("Processed train data shape: ", processed_train_data.shape)
    print("Processed validation data shape: ", processed_val_data.shape)
    print("Processed train targets shape: ", processed_train_targets.shape)
    print("Processed validation targets shape: ", processed_val_targets.shape)

    return processed_train_data, processed_val_data, processed_train_targets, processed_val_targets

def remove_columns(train_df):
    # Remove columns
    columns_to_be_dropped = [0,1,2,3,4,5,9,10,14,20,22,23]
    train_df = train_df.drop(columns=columns_to_be_dropped)

    return train_df