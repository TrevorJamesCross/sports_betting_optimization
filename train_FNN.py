"""
Sports Betting Project: Train Functional Neural Network
Author: Trevor Cross
Last Updated: 01/30/22

Train neural network with functional topology. The data to use in training is
extracted using collect_feature_data.py.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import model building classes
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.distribute.experimental import CentralStorageStrategy

# import compilation classes
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

# import callback classes
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping

# import support libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

import time

# ------------------
# ---Prepare Data---
# ------------------

# define data path & read data
data_path = '/home/tjcross/sports_betting_optimization/differential_data.csv'
df = pd.read_csv(data_path)

# split into feature and label data (do not use last feature column)
feats, labs = df[df.columns[:-2]].to_numpy(), df[df.columns[-1]].to_numpy()

# split into train, validation, and test data
trn_feats, tst_feats, trn_labs, tst_labs = train_test_split(feats,
                                                            labs,
                                                            test_size=0.30, 
                                                            shuffle=False)

val_feats, tst_feats, val_labs, tst_labs = train_test_split(tst_feats,
                                                            tst_labs,
                                                            test_size=0.50,
                                                            shuffle=False)
# normalize features
scaler = MinMaxScaler(feature_range=(0,1))

trn_feats = scaler.fit_transform(trn_feats)
val_feats = scaler.transform(val_feats)
tst_feats = scaler.transform(tst_feats)

# --------------------------
# ---Build Model Topology---
# --------------------------

def build_model():
    
    # build model topology (SEQUENTIAL FOR NOW)
    input_layer = Input(shape=(1,trn_feats.shape[1]))
    hidden_layer = LSTM(units=1024, return_sequences=True)(input_layer)
    hidden_layer = LSTM(units=1024, return_sequences=True)(input_layer)
    hidden_layer = LSTM(units=1024, return_sequences=True)(input_layer)
    hidden_layer = LSTM(units=1024, return_sequences=True)(input_layer)
    output_layer = Dense(units=1, activation='sigmoid')(hidden_layer)
    
    return Model(inputs=input_layer, outputs=output_layer)

# ---------------------------
# ---Compile and Fit Model---
# ---------------------------

# compile model
init_lr = 0.001

with CentralStorageStrategy().scope():
    model = build_model()
    
    model.compile(optimizer=Adam(learning_rate=init_lr),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=BinaryAccuracy(threshold=0.5))

# define keras callbacks
tb_logs = '/home/tjcross/tb_logs'
decay_rate = 0.5
num_epochs = 30

callbacks = [TensorBoard(log_dir=tb_logs, write_graph=False),
             LearningRateScheduler(lambda epoch: init_lr * np.exp(-decay_rate * (epoch-1))),
             EarlyStopping(monitor='binary_accuracy', patience=num_epochs, restore_best_weights=True)]

# fit model
start_time = time.time()
model.fit(trn_feats.reshape(trn_feats.shape[0],1,trn_feats.shape[1]),
          trn_labs.reshape(len(trn_labs),1,1),
          epochs=num_epochs,
          callbacks=callbacks,
          validation_data=(val_feats.reshape(val_feats.shape[0],1,val_feats.shape[1]),
                           val_labs.reshape(len(val_labs),1,1)))
print(">>> Training Time: " + str(time.time() - start_time))

# --------------------
# ---Prompt to Save---
# --------------------

# prompt to evaluate test data
eval_tst = input("Evaluate Test Data? [y|n] \n")

if eval_tst == "y":
    print("Evaluating Test Data!")
    
    # evaluate test data
    model.evaluate(tst_feats.reshape(tst_feats.shape[0],1,tst_feats.shape[1]),
                   tst_labs.reshape(len(tst_labs),1,1))
    
else:
    print("Not Evaluating anything!")
    
# define saved model and scaler paths
model_path = "/home/tjcross/sports_betting_optimization/saved_model"
scaler_path = "/home/tjcross/sports_betting_optimization/saved_scaler"

# prompt to save model and scaler
save_model = input("Save model and scaler? [y|n] \n")

if save_model == "y":
    print("Saving trained model and transformed scaler!")
    
    # save model
    model.save(model_path)
    
    # save scaler
    joblib.dump(scaler, scaler_path)
    
else:
    print("Not saving anything!")
