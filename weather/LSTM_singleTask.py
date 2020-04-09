"""
Keras LSTM, single task & multi-label prediction

"""
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from keras.layers import SimpleRNN,Convolution1D,MaxPooling1D,Flatten,Convolution2D
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Model
from keras.layers import Input, Dense
import helper_funcs
from numpy.random import seed
import os
import warnings
from tensorflow import set_random_seed
import glob


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings




def build_model(train):

    """
    Model 2: keras Function model
    """
    inputs = Input(shape=(train.shape[1], train.shape[2]))
    x = LSTM(64, activation='relu',dropout=0.2)(inputs) # 'return_sequences=True' if connectted to another LSTM or Conv layer.
    x = Dense(64, activation='relu')(x)
    outputs = Dense(6,activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
    print(model.summary())

    return model


def plot(y_true, y_predict,Smape):

    columns_predict = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    for i in range (len(columns_predict)):
        plt.figure(figsize=(24, 8))
        plt.plot(y_true[:,i], c='g', label='Actual')
        plt.plot(y_predict[:,i], c='r',  label='Predicted')
        plt.legend(fontsize='small')
        plt.title('Actual and Predicted ' + columns_predict[i]+ '_Smape:'+Smape[i])
        plt.savefig('results/shunyi_SingleTask_predicted_and_actural_'+columns_predict[i]+'.eps', format="eps", dpi=300)
        plt.show()


def main():

    look_back = 20  # number of previous timestamp used for training
    n_columns = 15  # total columns
    n_labels = 6  # number of labels
    split_ratio = 0.8  # train & test data split ratio

    file_list_train = glob.glob('preprocessed_data/train/*.csv')
    file_list_test = glob.glob('preprocessed_data/test/*.csv')
    
    
    file = open('time_cost/Single_LSTM.txt', 'w')
    sum_Smape = 0
    sum_Smape_PM25 = 0
    sum_Smape_PM10 = 0
    sum_Smape_NO2 = 0
    sum_Smape_CO = 0
    sum_Smape_O3 = 0
    sum_Smape_SO2 = 0
    sum_mae = 0
    sum_mae_PM25 = 0
    sum_mae_PM10 = 0
    sum_mae_NO2 = 0
    sum_mae_CO = 0
    sum_mae_O3 = 0
    sum_mae_SO2 = 0

    import time
    start_time = time.time()

    for i in range(len(file_list_train)):
        # train_data = 'data/preprocessed_data/train/bj_huairou.csv'
        # test_data = 'data/preprocessed_data/test/bj_huairou_201805.csv'

        locals()['dataset_train' + str(i)], locals()['scaled_train' + str(i)], locals()['scaler_train' + str(i)] = helper_funcs.load_dataset(file_list_train[i])
        locals()['dataset_test' + str(i)], locals()['scaled_test' + str(i)], locals()['scaler_test' + str(i)] = helper_funcs.load_dataset(file_list_test[i])

        # split into train and test sets
        locals()['train_X' + str(i)], locals()['train_y' + str(i)] = helper_funcs.split_dataset(locals()['scaled_train' + str(i)], look_back, n_columns, n_labels)
        locals()['test_X' + str(i)], locals()['test_y' + str(i)] = helper_funcs.split_dataset(locals()['scaled_test' + str(i)], look_back, n_columns, n_labels)

        model = build_model(locals()['train_X' + str(i)])

        

        # fit network
        history = model.fit(locals()['train_X' + str(i)], locals()['train_y' + str(i)], epochs=100, batch_size=120, validation_data=(locals()['test_X' + str(i)], locals()['test_y' + str(i)]), verbose=2,
                            shuffle=False,
                            callbacks=[
                                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2,
                                                              mode='min')]
                            )

        # plot history
        # plt.plot(history.history['loss'], label='train')
        # plt.plot(history.history['val_loss'], label='test')
        # plt.legend()
        # plt.show()

        # make a prediction
        y_predict = model.predict(locals()['test_X' + str(i)])
        # results = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], y_predict, look_back, n_columns, n_labels, locals()['scaler' + str(i)])

        locals()['Smape' + str(i)], locals()['mae' + str(i)] = helper_funcs.evaluation(locals()['test_X' + str(i)],
                                                                                       locals()['test_y' + str(i)],
                                                                                       y_predict,
                                                                                       look_back, n_columns, n_labels,
                                                                                       locals()['scaler_test' + str(i)])

        locals()['Smape_PM25' + str(i)], locals()['mae_PM25' + str(i)] = helper_funcs.evaluation_single(
            locals()['test_X' + str(i)], locals()['test_y' + str(i)],
            y_predict,
            look_back, n_columns, n_labels,
            locals()['scaler_test' + str(i)], 0)
        locals()['Smape_PM10' + str(i)], locals()['mae_PM10' + str(i)] = helper_funcs.evaluation_single(
            locals()['test_X' + str(i)], locals()['test_y' + str(i)],
            y_predict,
            look_back, n_columns, n_labels,
            locals()['scaler_test' + str(i)], 1)
        locals()['Smape_NO2' + str(i)], locals()['mae_NO2' + str(i)] = helper_funcs.evaluation_single(
            locals()['test_X' + str(i)], locals()['test_y' + str(i)],
            y_predict,
            look_back, n_columns, n_labels,
            locals()['scaler_test' + str(i)], 2)
        locals()['Smape_CO' + str(i)], locals()['mae_CO' + str(i)] = helper_funcs.evaluation_single(
            locals()['test_X' + str(i)], locals()['test_y' + str(i)],
            y_predict,
            look_back, n_columns, n_labels,
            locals()['scaler_test' + str(i)], 3)
        locals()['Smape_O3' + str(i)], locals()['mae_O3' + str(i)] = helper_funcs.evaluation_single(
            locals()['test_X' + str(i)], locals()['test_y' + str(i)],
            y_predict,
            look_back, n_columns, n_labels,
            locals()['scaler_test' + str(i)], 4)
        locals()['Smape_SO2' + str(i)], locals()['mae_SO2' + str(i)] = helper_funcs.evaluation_single(
            locals()['test_X' + str(i)], locals()['test_y' + str(i)],
            y_predict,
            look_back, n_columns, n_labels,
            locals()['scaler_test' + str(i)], 5)
        
        file.write('Current file index is: ' + str(i) + '\n')
        file.write('Smape:' + ' ' + str(locals()['Smape' + str(i)]) + '\n')
        file.write('Smape_PM25:' + ' ' + str(locals()['Smape_PM25' + str(i)]) + '\n')
        file.write('Smape_PM10:' + ' ' + str(locals()['Smape_PM10' + str(i)]) + '\n')
        file.write('Smape_NO2:' + ' ' + str(locals()['Smape_NO2' + str(i)]) + '\n')
        file.write('Smape_CO:' + ' ' + str(locals()['Smape_CO' + str(i)]) + '\n')
        file.write('Smape_O3:' + ' ' + str(locals()['Smape_O3' + str(i)]) + '\n')
        file.write('Smape_SO2:' + ' ' + str(locals()['Smape_SO2' + str(i)]) + '\n')
        file.write('mae:' + ' ' + str(locals()['mae' + str(i)]) + '\n')
        file.write('mae_PM25:' + ' ' + str(locals()['mae_PM25' + str(i)]) + '\n')
        file.write('mae_PM10:' + ' ' + str(locals()['mae_PM10' + str(i)]) + '\n')
        file.write('mae_NO2:' + ' ' + str(locals()['mae_NO2' + str(i)]) + '\n')
        file.write('mae_CO:' + ' ' + str(locals()['mae_CO' + str(i)]) + '\n')
        file.write('mae_O3:' + ' ' + str(locals()['mae_O3' + str(i)]) + '\n')
        file.write('mae_SO2:' + ' ' + str(locals()['mae_SO2' + str(i)]) + '\n')
        file.write('\n')

        sum_Smape = sum_Smape + locals()['Smape' + str(i)]
        sum_Smape_PM25 = sum_Smape_PM25 + locals()['Smape_PM25' + str(i)]
        sum_Smape_PM10 = sum_Smape_PM10 + locals()['Smape_PM10' + str(i)]
        sum_Smape_NO2 = sum_Smape_NO2 + locals()['Smape_NO2' + str(i)]
        sum_Smape_CO = sum_Smape_CO + locals()['Smape_CO' + str(i)]
        sum_Smape_O3 = sum_Smape_O3 + locals()['Smape_O3' + str(i)]
        sum_Smape_SO2 = sum_Smape_SO2 + locals()['Smape_SO2' + str(i)]
        sum_mae = sum_mae + locals()['mae' + str(i)]
        sum_mae_PM25 = sum_mae_PM25 + locals()['mae_PM25' + str(i)]
        sum_mae_PM10 = sum_mae_PM10 + locals()['mae_PM10' + str(i)]
        sum_mae_NO2 = sum_mae_NO2 + locals()['mae_NO2' + str(i)]
        sum_mae_CO = sum_mae_CO + locals()['mae_CO' + str(i)]
        sum_mae_O3 = sum_mae_O3 + locals()['mae_O3' + str(i)]
        sum_mae_SO2 = sum_mae_SO2 + locals()['mae_SO2' + str(i)]

    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    file.write('avg_Smape: ' + str(sum_Smape / len(file_list_test)) + '\n')
    file.write('avg_Smape_PM25: ' + str(sum_Smape_PM25 / len(file_list_test)) + '\n')
    file.write('avg_Smape_PM10: ' + str(sum_Smape_PM10 / len(file_list_test)) + '\n')
    file.write('avg_Smape_NO2: ' + str(sum_Smape_NO2 / len(file_list_test)) + '\n')
    file.write('avg_Smape_CO: ' + str(sum_Smape_CO / len(file_list_test)) + '\n')
    file.write('avg_Smape_O3: ' + str(sum_Smape_O3 / len(file_list_test)) + '\n')
    file.write('avg_Smape_SO2: ' + str(sum_Smape_SO2 / len(file_list_test)) + '\n')
    file.write('avg_mae: ' + str(sum_mae / len(file_list_test)) + '\n')
    file.write('avg_mae_PM25: ' + str(sum_mae_PM25 / len(file_list_test)) + '\n')
    file.write('avg_mae_PM10: ' + str(sum_mae_PM10 / len(file_list_test)) + '\n')
    file.write('avg_mae_NO2: ' + str(sum_mae_NO2 / len(file_list_test)) + '\n')
    file.write('avg_mae_CO: ' + str(sum_mae_CO / len(file_list_test)) + '\n')
    file.write('avg_mae_O3: ' + str(sum_mae_O3 / len(file_list_test)) + '\n')
    file.write('avg_mae_SO2: ' + str(sum_mae_SO2 / len(file_list_test)) + '\n')
    file.write('training time:' + str(end_time - start_time))



if __name__ == '__main__':
    main()
