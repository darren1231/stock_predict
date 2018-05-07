# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:01:29 2018

@author: darren
"""
import numpy as np
import pandas as pd
import time
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from matplotlib import pyplot as plt

def load_data(filename, seq_len, normalise_window):
    """
    Load data from XXX.csv file and return with training and test data
    
    Args:
        filename: XXX.csv
        seq_len: sequence length of the input
        normalise_window : boolean value, normalizeing data if true
        
    Return:
        x_train: (N,seq_len,1)
        y_train: (M,seq_len,1)
        x_test:  (N,)
        y_test:  (M,)
    """    
        
    dataframe = pd.read_csv(filename)
    data = dataframe[r"收盤價"]
    
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(list(data[index: index + sequence_length]))
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]
    
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[-1])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

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
    
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    
def predict_point_by_point(model, data):
    """
    Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    
    Args:
        model:
        data: (N,seq_len,1)
    Return:
        predicted : (N,1)

    """
    
    predicted = model.predict(data) #(N,1)
    predicted = np.reshape(predicted, (predicted.size,)) # (N,)
    return predicted

def predict_sequence_len(model, data, predict_len):
    """
    Recurrent predict next N days
    
    Args:
        model: keras model
        data: [1,seq_len,1]
        predict_n_days: 
        
    Return:
        predicted: n_days predicted
    """
    curr_frame = data[0] #(N,seq_len,1) --> (seq_len,1)
    
    squ_len = data.shape[1]
    predicted = []
    for i in range(predict_len):
        input_data = curr_frame[newaxis,:,:]  #(1,seq_len,1)
        predicted.append(model.predict(input_data)[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, squ_len-1, predicted[-1], axis=0)

    
    return predicted


def caculate_up_down(last_prices,real_stock_predict):
    """
    Caculate if the price is up or down
    
    Args:
        last_prices: the price of last day 
        real_stock_predict: [N,1]
        
    return:
        up_down: describe the stock goes up or goes down
        1: go up
        0: the same
        -1: go down
        [1,0,1,-1..]     [N,1] list
    """
    
    real_stock_predict=np.insert(real_stock_predict,0,last_prices,axis=0)
    
    up_down = []
    for i in range(len(real_stock_predict)-1):
        
        diff = real_stock_predict[i+1]-real_stock_predict[i]
        if diff>0:
            up_down.append(1)
        elif diff<0:
            up_down.append(-1)
        else:
            up_down.append(0)
    
    return up_down


def create_per_line_pd(model,stock_code,seq_len):
    """
    Creat pandas data per line.
    
          
    Args:
        model: keras model
        stock_code : the code to the stock
        
    Return:
        csv_pd:pandas data frame
        
        ex:
        ["ETFid","Mon_ud","Mon_cprice","Tue_ud","Tue_cprice","Wed_ud",
          "Wed_cprice","Thu_ud","Thu_cprice","Fri_ud","Fri_cprice"]
        52  -1   53.74598  1     54.586  1  55.687443  1    56.49849  1 57.12508
    """    
    
    # prepare input
    stock = pd.read_csv("stock_csv/"+stock_code+".csv")
    input_data = stock[r"收盤價"][:seq_len][::-1]
    last_prices = stock[r"收盤價"][0]
    input_data = np.expand_dims(np.expand_dims(input_data,axis=0),axis=2)       
    
    
    predictions = predict_sequence_len(model, input_data,5)
    
    # de normalized
    real_stock_predict = (np.array(predictions)+1)*last_prices    
    up_down= caculate_up_down(last_prices,real_stock_predict)   
    
    
    csv_list=[]
    csv_list.append(str(int(stock_code)))
    for i in range(len(predictions)):
        csv_list.append(up_down[i])
        csv_list.append(real_stock_predict[i])
    
    csv_list = np.expand_dims(np.array(csv_list),axis=0)
    csv_pd = pd.DataFrame(csv_list)
    
    return csv_pd
    
def create_result_table(model,stock_code_list,seq_len):
    """
    Create result table for competition
    
    Args:
        model: keras model
        stock_code : the code to the stock ex:0050
        seq_len : the len to the input sequence
    
    Return:
        result_table: pandas data frame
    """
    
    # create empty pandas table
    result_table = pd.DataFrame()
    for stock_code in stock_code_list:        
        try:
            csv_line_pd = create_per_line_pd(model,stock_code,seq_len=seq_len)
            result_table=pd.concat([result_table,csv_line_pd],axis=0)
        except Exception:
            print ("wrong stock code",stock_code)
    
    
    column_str = ["ETFid","Mon_ud","Mon_cprice","Tue_ud","Tue_cprice","Wed_ud",
              "Wed_cprice","Thu_ud","Thu_cprice","Fri_ud","Fri_cprice"]
    
    result_table.columns =column_str
    result_table.to_csv("result_table.csv",index=None)
    
    return result_table

#epochs  = 50
#seq_len = 50
#
#print('> Loading data... ')
#X_train, y_train, X_test, y_test = load_data('2330.csv', seq_len, True)
#
#print('> Data Loaded. Compiling...')
#
model = build_model([1, 50, 100, 1])
##
#model.fit(X_train,
#	    y_train,
#	    batch_size=512,
#	    epochs=epochs,
#	    validation_split=0.05)
#model.save("test.hdf5")

model.load_weights("test.hdf5")

stock_code_list = ["0050","0051","0052","0053","0054","0055","0056","0057","0058","0059",
                   "006201","006203","006204","006208","00690","00692","00701","00713"]
create_result_table(model,stock_code_list,seq_len=50)