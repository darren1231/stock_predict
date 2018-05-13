# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:06:21 2018

@author: darren
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.models import load_model
import time
import math

def create_train_test_per_feature(data,seq_len,predict_days,train_test_ratio=0.8):
    """
    """
    
    sequence_length = seq_len + predict_days
    result = []
    for index in range(len(data) - sequence_length):
        result.append(list(data[index: index + sequence_length]))
    
    
#    result = scaler.fit_transform(result)

    result = np.array(result)

    row = round(train_test_ratio * result.shape[0])
    train = result[:int(row), :]    
   
    np.random.shuffle(train)
    x_train = train[:, :-predict_days]
    y_train = train[:, -predict_days:]
    x_test = result[int(row):, :-predict_days:]
    y_test = result[int(row):, -predict_days:]
    
    return [x_train, y_train, x_test, y_test]

def load_data(filename):
    
    data_frame = pd.DataFrame()
    dataframe = pd.read_csv(filename)  
    
    data_frame = pd.concat((data_frame,dataframe[r"收盤價"][::-1]),axis=1)
    data_frame = pd.concat((data_frame,np.log(dataframe[r"成交量"][::-1])),axis=1)    
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_frame = scaler.fit_transform(data_frame)
    
    
    return data_frame,scaler
    
def create_train_test(seq_len , predict_days):
    """
    Load data from XXX.csv file and return with training and test data
    
    Args:
        filename: XXX.csv
        seq_len: sequence length of the input
        normalise_window : boolean value, normalizeing data if true
        predict_days:
    Return:
        x_train: (N,seq_len,1)
        y_train: (M,seq_len,1)
        x_test:  (N,)
        y_test:  (M,)
    """    
    
    
    
    for i in range(data_frame.shape[1]):
        
        [temp_x_train, temp_y_train, temp_x_test, temp_y_test] = \
        create_train_test_per_feature(data_frame[:,i],seq_len,predict_days)
        
        if i ==0:
            y_train = temp_y_train
            y_test =  temp_y_test
            
            x_train = np.expand_dims(temp_x_train,axis=2)
            x_test = np.expand_dims(temp_x_test,axis=2)
            
            
        else:
            temp_x_train = np.expand_dims(temp_x_train,axis=2)
            temp_x_test = np.expand_dims(temp_x_test,axis=2)
            
            x_train = np.concatenate((x_train,temp_x_train),axis=2)
            x_test = np.concatenate((x_test,temp_x_test),axis=2)
    
    # sample_size,seqence_size, feature_size
    # x_train = np.reshape(x_train, (sample_size, seqence_size, feature_size))
  

    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))


    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model
    
data_frame,scaler=load_data("stock_csv/0050.csv")    

print (scaler.min_,scaler.scale_)
[x_train, y_train, x_test, y_test]= \
create_train_test(30,5)

#model = build_model([2, 30, 100, 5])
#
#
#epochs = 1
#
#result_dict = model.fit(x_train,
#	    y_train,
#	    batch_size=512,
#	    epochs=epochs,
#	    validation_split=0.1)
#
#train_loss = result_dict.history["loss"]
#val_loss = result_dict.history["val_loss"]
#
#x_plot_data = [i for i in range(epochs)]
#
#plt.plot(x_plot_data,train_loss,label="train_loss")
#plt.plot(x_plot_data,val_loss,label="val_loss")
#plt.legend(loc="best")
#plt.show()
#
#print (result_dict.history)
#model.save("stock.hdf5")
    
# Load the trained model
model = load_model("stock_50_0050.hdf5")


predict = model.predict(x_test)
print (y_test.shape)
y_test = (y_test-scaler.min_[0])/scaler.scale_[0]
predict = (predict-scaler.min_[0])/scaler.scale_[0]

predict_fix = np.empty(0)
for i in range(0,708,5):
    
    predict_fix = np.concatenate((predict_fix,(predict[i,:]))) 

plt.plot(y_test[:,0],label="real")
plt.plot(predict_fix,label="predict")
plt.legend(loc="best")
plt.show()

testScore = math.sqrt(mean_squared_error(y_test[:,0], predict_fix[:-2]))
# TEST RMSE
#testScore = math.sqrt(mean_squared_error(y_test[:,0], predict[:,0]))*0.1+ \
#            math.sqrt(mean_squared_error(y_test[:,1], predict[:,1]))*0.15+ \
#            math.sqrt(mean_squared_error(y_test[:,2], predict[:,2]))*0.20+ \
#            math.sqrt(mean_squared_error(y_test[:,3], predict[:,3]))*0.25+ \
#            math.sqrt(mean_squared_error(y_test[:,4], predict[:,4]))*0.30 \
            
print('Test RMSE: %.2f' % (testScore))