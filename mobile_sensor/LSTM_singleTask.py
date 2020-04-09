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
    Model 1: keras Sequential model
    """
    # model = Sequential()
    # model.add(Convolution1D(nb_filter=128, filter_length=4, activation='relu',input_shape=(train.shape[1], train.shape[2])))
    # model.add(Convolution1D(nb_filter=64, filter_length=2, activation='relu'))
    # # model.add(MaxPooling1D(pool_size=4))
    # model.add(LSTM(64, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(51, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

    """
    Model 2: keras Function model
    """
    inputs = Input(shape=(train.shape[1], train.shape[2]))
    # x = Convolution1D(nb_filter=128, filter_length=3, activation='relu', input_shape=(train.shape[1], train.shape[2]))(inputs)
    # x = Convolution1D(nb_filter=64, filter_length=2, activation='relu')(x)
    # x = MaxPooling1D(3)(x)
    x = LSTM(64, activation='relu',dropout=0.2)(inputs) # 'return_sequences=True' if connectted to another LSTM or Conv layer.
    x = Dense(64, activation='relu')(x)
    outputs = Dense(51,activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
    print(model.summary())

    return model


def plot(y_true, y_predict):

    labels = ['1', '2', '3', '4', '5', '6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
              '21','22','23','24','25','26','27','28','29','30']
    for i in range (len(labels)):
        plt.figure(figsize=(24, 8))
        plt.plot(y_true[:,i], c='g', label='Actual')
        plt.plot(y_predict[:,i], c='r',  label='Predicted')
        plt.legend(fontsize='small')
        plt.title('Actual and Predicted ' + labels[i])
        plt.savefig('results/predicted_and_actural_'+labels[i] +'.eps', format="eps", dpi=200)


def main():

    look_back = 20 # number of previous timestamp used for training
    n_columns = 276 # total columns
    n_labels = 51 # number of labels
    split_ratio = 0.8 # train & test data split ratio

    file_list = glob.glob('data_csv/train/*.csv')

    file = open('time_cost/Single_LSTM_40users.txt', 'w')
    sum_bacc = 0
    sum_TPR = 0
    Num_tp = 0
    Num_fn = 0
    Num_fp = 0
    Num_tn = 0
    sum_precision = 0
    sum_F1 = 0
    train_time = 0

    import time
    start_time = time.time()
    for i in range(len(file_list)):
        locals()['dataset' + str(i)] = file_list[i]

        locals()['dataset' + str(i)], locals()['scaled' + str(i)], locals()['scaler' + str(i)] = helper_funcs.load_dataset(
            locals()['dataset' + str(i)])



        # split into train and test sets
        locals()['train_X' + str(i)], locals()['train_y' + str(i)], locals()['test_X' + str(i)], locals()[
            'test_y' + str(i)] = helper_funcs.split_dataset(locals()['dataset' + str(i)], locals()['scaled' + str(i)], look_back,
                                               n_columns, n_labels, split_ratio)


        model = build_model( locals()['train_X' + str(i)])

        

        # fit network
        history = model.fit( locals()['train_X' + str(i)], locals()['train_y' + str(i)], epochs=40, batch_size=60,
                            # validation_data=(test_X, test_y),
                            validation_split=0.25,
                            verbose=2,
                            shuffle=False,
                            callbacks=[
                                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                              mode='min')]
                            )

        

        # # plot history
        # plt.plot(history.history['loss'], label='train')
        # plt.plot(history.history['val_loss'], label='test')
        # plt.legend()
        # plt.show()

        y_predict = model.predict(locals()['test_X' + str(i)])

        results = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], y_predict, look_back, n_columns, n_labels, locals()['scaler' + str(i)])

        sum_bacc = sum_bacc + results[3]
        sum_TPR = sum_TPR + results[1]
        Num_tp = Num_tp + results[4]
        Num_fn = Num_fn + results[5]
        Num_fp = Num_fp + results[6]
        Num_tn = Num_tn + results[7]
        sum_precision = sum_precision + results[8]
        sum_F1 = sum_F1 + results[9]
        train_time = train_time + (end_time - start_time)

        file.write ('Accuracy:'+' ' + str(results[0])+' ')
        file.write ('TPR:'+' ' + str(results[1])+' ')
        file.write ('TNR:'+' '+ str(results[2])+' ')
        file.write ('Bacc:'+' ' + str(results[3])+ '\n')
        file.write('FP No.:' + ' ' + str(results[6]) + '\n')
        file.write('TN No.:' + ' ' + str(results[7]) + '\n')
        file.write('Precision:' + ' ' + str(results[8]) + '\n')
        file.write('F1:' + ' ' + str(results[9]) + '\n')

    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    file.write('avg_bacc: ' + str(sum_bacc / len(file_list)) + '\n')
    file.write('avg_TPR: ' + str(sum_TPR / len(file_list)) + '\n')
    file.write('avg_precision: ' + str(sum_precision / len(file_list)) + '\n')
    file.write('avg_F1: ' + str(sum_F1 / len(file_list)) + '\n')
    file.write('sum_Num_tp: ' + str(Num_tp) + '\n')
    file.write('sum_Num_fn: ' + str(Num_fn) + '\n')
    file.write('sum_Num_fp: ' + str(Num_fp) + '\n')
    file.write('sum_Num_tn: ' + str(Num_tn) + '\n')
    file.write('train_time: ' + str(train_time) + '\n')



if __name__ == '__main__':
    main()
