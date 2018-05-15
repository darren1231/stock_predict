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

def create_train_test_per_feature(data,seq_len,predict_days,shuffle=True,train_test_ratio=0.8):
    """
    Create training data and testing data per feature
    
    Args:
        data : numpy array [N,1]
        seq_len : sequence length of the input
        predict_days :the sequence length of the output
        train_test_ratio : the ratio to train and test
        
    Return:        
        x_train: (N,seq_len,1)
        y_train: (N,predict_days)
        
        x_test:  (M,seq_len,1)
        y_test:  (M,predict_days)
    """
    
    sequence_length = seq_len + predict_days
    result = []
    for index in range(len(data) - sequence_length):
        result.append(list(data[index: index + sequence_length]))
    
    
#    result = scaler.fit_transform(result)

    result = np.array(result)

    row = round(train_test_ratio * result.shape[0])
    train = result[:int(row), :]    
   
    if shuffle:        
        np.random.shuffle(train)
        
    x_train = train[:, :-predict_days]
    y_train = train[:, -predict_days:]
    x_test = result[int(row):, :-predict_days]
    y_test = result[int(row):, -predict_days:]
    
    return [x_train, y_train, x_test, y_test]

def load_data(filename):
    """
    Load data from csv file and normalize data with MinMaxScaler
    
    Args:
        filename: csv file name
    
    Return:
        data_frame: numpy data frame  [N,M]
        N: the number of data for each feature
        M: the number of feature
        
        scaler : sklearn scaler class
    """
    data_frame = pd.DataFrame()
    dataframe = pd.read_csv(filename)
    
    dataframe = dataframe.iloc[:360,:]
    
    data_frame = pd.concat((data_frame,dataframe[r"收盤價"][::-1]),axis=1)
    data_frame = pd.concat((data_frame,np.log(dataframe[r"成交量"][::-1])),axis=1)
    data_frame = pd.concat((data_frame,dataframe[r"外資"][::-1]),axis=1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_data_frame = scaler.fit_transform(data_frame)    
#    scaler_data_frame = np.array(data_frame)

    print ("Total data: row is ",data_frame.shape[0]," column is : ",data_frame.shape[1])
    return scaler_data_frame,scaler
    
def create_train_test(data_frame,seq_len , predict_days,shuffle=True):
    """
    Load data from XXX.csv file and return with training and test data
    
    Args:        
        seq_len: sequence length of the input        
        predict_days: the sequence length of the output
        
    Return:
        x_train: (N,seq_len,feature_number)
        y_train: (N,predict_days)
        
        x_test:  (M,seq_len,feature_number)
        y_test:  (M,predict_days)
    """    
    
    
    
    for i in range(data_frame.shape[1]):
        
        [temp_x_train, temp_y_train, temp_x_test, temp_y_test] = \
        create_train_test_per_feature(data_frame[:,i],seq_len,predict_days,shuffle)
        
        # Only keep the first feature data -- close data as output y
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
    """
    Build LSTM network using keras model
    
    Args:
        layers:
    
    Return:
        model: keras model
    """
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(
        layers[3],
        return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(
        output_dim=layers[4]))
    model.add(Activation("linear"))


    start = time.time()
    model.compile(loss="mse", optimizer="adam")
    print("> Compilation Time : ", time.time() - start)
    return model

def train_network(Input_sequence_length,Output_sequence_length):
    """
    Train network 
    """
#    Input_sequence_length = 30
#    Output_sequence_length = 5
    
    data_frame,scaler=load_data("stock_csv/0050.csv")
    
    feature_number = data_frame.shape[1]
    
    print (scaler.min_,scaler.scale_)
    [x_train, y_train, x_test, y_test]= \
    create_train_test(data_frame,Input_sequence_length,Output_sequence_length)
    
    model = build_model([feature_number, \
    Input_sequence_length, 100, 50,Output_sequence_length])
    
    
    epochs = 20
    
    result_dict = model.fit(x_train,
    	    y_train,
    	    batch_size=32,
    	    epochs=epochs,
    	    validation_split=0.2)
    
    train_loss = result_dict.history["loss"]
    val_loss = result_dict.history["val_loss"]
    
    x_plot_data = [i for i in range(epochs)]
    
    plt.plot(x_plot_data,train_loss,label="train_loss")
    plt.plot(x_plot_data,val_loss,label="val_loss")
    plt.legend(loc="best")
    plt.show()
    
    print (result_dict.history)
    model.save("stock_test.hdf5")

def plot_every_n_days(y_test,predict):
    
    predict_fix = np.empty(0)
    for i in range(0,y_test.shape[0],5):
        
        predict_fix = np.concatenate((predict_fix,(predict[i,:]))) 
    
    plt.plot(y_test[:,0],label="real")
    plt.plot(predict_fix,label="predict")
    plt.legend(loc="best")
    plt.show()

def plot_n_days_curve(y_test,predict):
    
    
    for i in range(0,y_test.shape[0],5):
        pad = [None for j in range(i)]
        plt.plot(pad+list(predict[i,:]),label = "predict")
#        plt.show()
    
    plt.plot(y_test[:,0],label="real")
    
    plt.legend(loc="best")
    plt.show()

def caculate_mse_error(y_test,predict):
    testScore = math.sqrt(mean_squared_error(y_test[:,0], predict[:,0]))*0.1+ \
                math.sqrt(mean_squared_error(y_test[:,1], predict[:,1]))*0.15+ \
                math.sqrt(mean_squared_error(y_test[:,2], predict[:,2]))*0.20+ \
                math.sqrt(mean_squared_error(y_test[:,3], predict[:,3]))*0.25+ \
                math.sqrt(mean_squared_error(y_test[:,4], predict[:,4]))*0.30
                
    return testScore
                
def test_network(Input_sequence_length,Output_sequence_length):
    """
    Test network
    """
    #Load the trained model
    model = load_model("stock_test.hdf5")
    
#    Input_sequence_length = 30
#    Output_sequence_length = 5
    
    data_frame,scaler=load_data("stock_csv/0050.csv")
    [x_train, y_train, x_test, y_test]= \
    create_train_test(data_frame,Input_sequence_length, \
    Output_sequence_length)
    
    predict = model.predict(x_test)
    print (y_test.shape)
    
    # De normalized
    y_test = (y_test-scaler.min_[0])/scaler.scale_[0]
    predict = (predict-scaler.min_[0])/scaler.scale_[0]
    
    predict_train = model.predict(x_train)
    
    y_train = (y_train-scaler.min_[0])/scaler.scale_[0]
    predict_train= (predict_train-scaler.min_[0])/scaler.scale_[0]

    print ("MSE error:",caculate_mse_error(y_test,predict))
#    plot_every_n_days(y_test,predict)
    plot_n_days_curve(y_test,predict)
#    plot_n_days_curve(y_train,predict_train)


#train_network(60,5)
test_network(60,5)