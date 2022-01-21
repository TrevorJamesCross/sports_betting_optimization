"""
Sports Betting Project: Train Functional Neural Network
Author: Trevor Cross
Last Updated: 01/20/22

Train neural network with functional topology. The data to use in training is
extracted using collect_feature_data.py.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import ML libraries
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping

# import support libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ------------------
# ---Prepare Data---
# ------------------

# define data path & read data
data_path = '/home/tjcross/sports_betting_proj/NFL_diffs.csv'
df = pd.read_csv(data_path)

# split into feature and label data
feats, labs = df[df.columns[:-1]].to_numpy(), df[df.columns[-1]].to_numpy()

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
val_feats = scaler.fit_transform(val_feats)
tst_feats = scaler.fit_transform(tst_feats)

# --------------------------
# ---Build Model Topology---
# --------------------------

# build model topology
input_layer = Input(shape=(1,trn_feats.shape[1]))
hidden_layer = LSTM(units=256, return_sequences=True)(input_layer)
hidden_layer = LSTM(units=256, return_sequences=True)(input_layer)
hidden_layer = LSTM(units=256, return_sequences=True)(input_layer)
hidden_layer = LSTM(units=256, return_sequences=True)(input_layer)
output_layer = Dense(units=1, activation='sigmoid')(hidden_layer)

model = Model(inputs=input_layer, outputs=output_layer )

# ---------------------------
# ---Compile and Fit Model---
# ---------------------------

# compile model
init_lr = 0.01

model.compile(optimizer=Adam(learning_rate=init_lr),
              loss=BinaryCrossentropy(from_logits=True),
              metrics=BinaryAccuracy(threshold=0.5))

# define keras callbacks
tb_logs = '/home/tjcross/sports_betting_proj/tb_logs'
lr_dec = 0.75

callbacks = [TensorBoard(log_dir=tb_logs, write_graph=False),
             LearningRateScheduler(lambda epoch: init_lr * lr_dec ** epoch)]
# fit model
num_epochs = 10
model.fit(trn_feats.reshape(trn_feats.shape[0],1,trn_feats.shape[1]),
          trn_labs.reshape(len(trn_labs),1,1),
          epochs=num_epochs,
          callbacks=callbacks,
          validation_data=(val_feats.reshape(val_feats.shape[0],1,val_feats.shape[1]),
                           val_labs.reshape(len(val_labs,1,1))))