"""
Keras CNN & LSTM, multi-task & multi-label prediction

CNN is for the feature selection

"""
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import pandas as pd
import numpy as np
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Input, Embedding, LSTM, Dense,Convolution1D,MaxPooling1D
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Model
import helper_funcs
import os
import warnings
from keras import backend as K
import glob



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

# def load_dataset(datasource: str) -> (np.ndarray, MinMaxScaler):
#     """
#     The function loads dataset from given file name and uses MinMaxScaler to transform data
#     :param datasource: file name of data source
#     :return: tuple of dataset and the used MinMaxScaler
#     """
#
#     # load the dataset
#     dataframe = read_csv(datasource, index_col=0)
#     dataframe = dataframe.drop('label_source', axis=1)  # drop the last column
#
#     # dataframe = dataframe.fillna(method='ffill')
#     dataframe = dataframe.fillna(0)
#     dataframe = dataframe.iloc[0:3000]  # first 3000 rows of dataframe
#
#     dataset = dataframe.values
#     dataset = dataset.astype('float32')
#
#     # normalize the dataset
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled = scaler.fit_transform(dataset)
#     return dataset, scaled, scaler


# #take one input dataset and split it into train and test
# def split_dataset(dataset, scaled, look_back, n_columns,n_labels,ratio):
#
#     # frame as supervised learning
#     reframed = series_to_supervised(scaled, look_back, 1)
#     print(reframed.head())
#
#     # split into train and test sets
#     values = reframed.values
#     n_train_data = int(len(dataset) * ratio)
#     train = values[:n_train_data, :]
#     test = values[n_train_data:, :]
#     print ('test data-----:')
#     print (test[0:5,:])
#     # split into input and outputs
#     n_obs = look_back * n_columns
#     train_X, train_y = train[:, :n_obs], train[:, -n_labels:]  # labels are the last 6 columns
#     test_X, test_y = test[:, :n_obs], test[:, -n_labels:]
#     print(train_X.shape, len(train_X), train_y.shape)
#     # reshape input to be 3D [samples, timesteps, features]
#     train_X = train_X.reshape((train_X.shape[0], look_back, n_columns))
#     test_X = test_X.reshape((test_X.shape[0], look_back, n_columns))
#
#     return train_X, train_y, test_X, test_y

def build_model(trainX,
                task_num, con_layer1, con_layer1_filter, con_layer2, con_layer2_filter,
                lstm_layer, drop, r_drop, shared_layer,dense_num, n_labels):

    """
    Keras Function model
    """
    concate_list = []
    input_list = []
    for i in range(0, task_num):
        locals()['input' + str(i)] = Input(shape=(trainX[i].shape[1], trainX[i].shape[2]),
            name='input' + str(i))
        locals()['cnn_out' + str(i)] = Convolution1D(nb_filter=con_layer1, filter_length=con_layer1_filter,
                                                     activation='relu')(locals()['input' + str(i)])
        locals()['cnn_out' + str(i)] = MaxPooling1D(3)(locals()['cnn_out'+str(i)])

        # locals()['cnn_out' + str(i)] = Convolution1D(nb_filter=con_layer2, filter_length=con_layer2_filter,
        #                                              activation='relu')(locals()['cnn_out' + str(i)])
        locals()['lstm_out' + str(i)] = LSTM(lstm_layer, activation='relu',dropout=drop, recurrent_dropout=r_drop)(locals()['cnn_out' + str(i)])
        concate_list.append(locals()['lstm_out' + str(i)])
        input_list.append(locals()['input' + str(i)])

    concate_layer = keras.layers.concatenate(concate_list)

    dense_shared = Dense(shared_layer, activation='relu')(concate_layer)

    output_list = []
    for i in range(0, task_num):
        locals()['sub' + str(i)] = Dense(dense_num, activation='relu')(dense_shared)
        locals()['out' + str(i)] = Dense(n_labels, activation='sigmoid')(locals()['sub' + str(i)])
        output_list.append(locals()['out' + str(i)])

    model = Model(inputs=input_list, outputs=output_list)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    print(model.summary())

    return model


def main():
    # network parameters
    task_num = 40
    con_layer1 = 128
    con_layer1_filter = 3
    con_layer2 = 64
    con_layer2_filter = 4
    lstm_layer = 64
    drop = 0.3
    r_drop = 0.3
    shared_layer = 576
    dense_num = 128

    look_back = 20  # number of previous timestamp used for training
    n_columns = 276  # total columns
    n_labels = 51  # number of labels
    split_ratio = 0.8  # train & test data split ratio

    trainX_list = []
    trainy_list = []
    testX_list = []
    testy_list = []
    file_list = glob.glob('data_csv/train/*.csv')
    # print (file_list[0])

    for i in range(len(file_list)):
        locals()['dataset' + str(i)] = file_list[i]
        locals()['dataset' + str(i)], locals()['scaled' + str(i)], locals()['scaler' + str(i)] = helper_funcs.load_dataset(locals()['dataset' + str(i)])
        locals()['train_X'+str(i)], locals()['train_y'+str(i)], locals()['test_X'+str(i)], locals()['test_y'+str(i)] = helper_funcs.split_dataset(locals()['dataset' + str(i)], locals()['scaled' + str(i)], look_back, n_columns, n_labels, split_ratio)
        trainX_list.append(locals()['train_X'+str(i)])
        trainy_list.append(locals()['train_y'+str(i)])
        testX_list.append(locals()['test_X' + str(i)])
        testy_list.append(locals()['test_y' + str(i)])

    model = build_model(trainX_list,task_num, con_layer1, con_layer1_filter, con_layer2, con_layer2_filter,
                        lstm_layer, drop, r_drop, shared_layer, dense_num, n_labels)

    import time
    start_time = time.time()

    # fit network
    history = model.fit(trainX_list, trainy_list,
                        epochs=150,
                        batch_size=60,
                        validation_split=0.25,
                        # validation_data=(testX_list,testy_list),
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=2,
                                                          mode='min')]
                        )
    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    # make prediction
    # y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6 = model.predict(testX_list)
    # print (len(y_pred1))
    y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7, y_pred8, y_pred9, y_pred10, y_pred11, y_pred12, y_pred13, y_pred14, y_pred15, y_pred16, y_pred17, y_pred18, y_pred19, y_pred20, y_pred21, y_pred22, y_pred23, y_pred24, y_pred25, y_pred26, y_pred27 \
        , y_pred28, y_pred29, y_pred30, y_pred31, y_pred32, y_pred33, y_pred34, y_pred35, y_pred36, y_pred37, y_pred38, y_pred39, y_pred40 = model.predict(testX_list)
    # write parameters & results to file
    # file = open('results/cnn&lstm_results(6)--.txt', 'w')
    file = open('time_cost/CRNN.txt', 'w')

    file.write('task_num:' + str(task_num) + '\n')
    file.write('con_layer1:' + str(con_layer1) + '\n')
    file.write('con_layer1_filter:' + str(con_layer1_filter) + '\n')
    file.write('con_layer2:' + str(con_layer2) + '\n')
    file.write('con_layer2_filter:' + str(con_layer2_filter) + '\n')
    file.write('lstm_layer:' + str(lstm_layer) + '\n')
    file.write('drop:' + str(drop) + '\n')
    file.write('r_drop:' + str(r_drop) + '\n')
    file.write('shared_layer:' + str(shared_layer) + '\n')
    file.write('dense_num:' + str(dense_num) + '\n')

    sum_bacc = 0
    sum_TPR = 0
    Num_tp = 0
    Num_fn = 0
    Num_fp = 0
    Num_tn = 0
    sum_precision = 0
    sum_F1 = 0
    # balance accuracy
    for i in range(len(file_list)):
        locals()['Bacc' + str(i)] = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], locals()['y_pred' + str(i+1)], look_back, n_columns, n_labels, locals()['scaler' + str(i)])

        sum_bacc = sum_bacc + (locals()['Bacc' + str(i)])[3]
        sum_TPR = sum_TPR + (locals()['Bacc' + str(i)])[1]
        Num_tp = Num_tp + (locals()['Bacc' + str(i)])[4]
        Num_fn = Num_fn + (locals()['Bacc' + str(i)])[5]
        Num_fp = Num_fp + (locals()['Bacc' + str(i)])[6]
        Num_tn = Num_tn + (locals()['Bacc' + str(i)])[7]
        sum_precision = sum_precision + (locals()['Bacc' + str(i)])[8]
        sum_F1 = sum_F1 + (locals()['Bacc' + str(i)])[9]

        file.write ('Accuracy:'+' ' + str((locals()['Bacc' + str(i)])[0])+' ')
        file.write ('TPR:'+' ' + str((locals()['Bacc' + str(i)])[1])+' ')
        file.write ('TNR:'+' '+ str((locals()['Bacc' + str(i)])[2])+' ')
        file.write ('Bacc:'+' ' + str((locals()['Bacc' + str(i)])[3])+ '\n')
        file.write('FP No.:' + ' ' + str((locals()['Bacc' + str(i)])[6]) + '\n')
        file.write('TN No.:' + ' ' + str((locals()['Bacc' + str(i)])[7]) + '\n')
        file.write('Precision:' + ' ' + str((locals()['Bacc' + str(i)])[8]) + '\n')
        file.write('F1:' + ' ' + str((locals()['Bacc' + str(i)])[9]) + '\n')

    file.write ('avg_bacc: ' + str(sum_bacc/len(file_list)) +'\n')
    file.write ('avg_TPR: ' + str(sum_TPR/len(file_list))+'\n')
    file.write('avg_precision: ' + str(sum_precision/len(file_list)) + '\n')
    file.write('avg_F1: ' + str(sum_F1/len(file_list)) + '\n')
    file.write('sum_Num_tp: ' + str(Num_tp) + '\n')
    file.write('sum_Num_fn: ' + str(Num_fn) + '\n')
    file.write('sum_Num_fp: ' + str(Num_fp) + '\n')
    file.write('sum_Num_tn: ' + str(Num_tn) + '\n')
    file.write('training time:' + str(end_time - start_time))

if __name__ == '__main__':
    main()


# # code backup ----
# def build_model(data1,data2,data3,data4,data5,data6,data7,data8,data9):
#
#     """
#     Keras Function model
#     """
#     input1 = Input(shape=(data1.shape[1], data1.shape[2]))
#     cnn_out1 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input1)
#     cnn_out1 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out1)
#     lstm_out1 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out1)
#
#     input2 = Input(shape=(data2.shape[1], data2.shape[2]), name='input2')
#     cnn_out2 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input2)
#     cnn_out2 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out2)
#     lstm_out2 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out2)
#
#     input3 = Input(shape=(data3.shape[1], data3.shape[2]), name='input3')
#     cnn_out3 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input3)
#     cnn_out3 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out3)
#     lstm_out3 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out3)
#
#     input4 = Input(shape=(data4.shape[1], data4.shape[2]), name='input4')
#     cnn_out4 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input4)
#     cnn_out4 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out4)
#     lstm_out4 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out4)
#
#     input5 = Input(shape=(data5.shape[1], data5.shape[2]), name='input5')
#     cnn_out5 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input5)
#     cnn_out5 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out5)
#     lstm_out5 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out5)
#
#     input6 = Input(shape=(data6.shape[1], data6.shape[2]), name='input6')
#     cnn_out6 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input6)
#     cnn_out6 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out6)
#     lstm_out6 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out6)
#
#     input7 = Input(shape=(data7.shape[1], data7.shape[2]), name='input7')
#     cnn_out7 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input7)
#     cnn_out7 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out7)
#     lstm_out7 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out7)
#
#     input8 = Input(shape=(data8.shape[1], data8.shape[2]), name='input8')
#     cnn_out8 = Convolution1D(nb_filter=128, filter_length=8, activation='relu')(input8)
#     cnn_out8 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out8)
#     lstm_out8 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out8)
#
#     input9 = Input(shape=(data9.shape[1], data9.shape[2]), name='input9')
#     cnn_out9 = Convolution1D(nb_filter=128, filter_length=8,activation='relu')(input9)
#     cnn_out9 = Convolution1D(nb_filter=64, filter_length=4, activation='relu')(cnn_out9)
#     lstm_out9 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(cnn_out9)
#
#     # concate_layer = keras.layers.concatenate([cnn_out1, cnn_out2,cnn_out3,cnn_out4,cnn_out5,cnn_out6,cnn_out7,cnn_out8,cnn_out9])
#
#     concate_layer = keras.layers.concatenate([lstm_out1, lstm_out2,lstm_out3,lstm_out4,lstm_out5,lstm_out6,lstm_out7,lstm_out8,lstm_out9])
#     # x = LSTM(512, activation='relu')(concate_layer)
#     # x = LSTM(512, activation='relu')(x)
#
#     x = Dense(576, activation='relu')(concate_layer)
#     # x = Dense(576, activation='relu')(x)
#
#     sub1 = Dense(64,activation='relu')(x)
#     sub2 = Dense(64,activation='relu')(x)
#     sub3 = Dense(64,activation='relu')(x)
#     sub4 = Dense(64,activation='relu')(x)
#     sub5 = Dense(64,activation='relu')(x)
#     sub6 = Dense(64,activation='relu')(x)
#     sub7 = Dense(64,activation='relu')(x)
#     sub8 = Dense(64,activation='relu')(x)
#     sub9 = Dense(64,activation='relu')(x)
#     out1 = Dense(51, activation='sigmoid')(sub1)
#     out2 = Dense(51, activation='sigmoid')(sub2)
#     out3 = Dense(51, activation='sigmoid')(sub3)
#     out4 = Dense(51, activation='sigmoid')(sub4)
#     out5 = Dense(51, activation='sigmoid')(sub5)
#     out6 = Dense(51, activation='sigmoid')(sub6)
#     out7 = Dense(51, activation='sigmoid')(sub7)
#     out8 = Dense(51, activation='sigmoid')(sub8)
#     out9 = Dense(51, activation='sigmoid')(sub9)
#
#     model = Model(inputs=[input1,input2,input3,input4,input5,input6,input7,input8,input9],
#                   outputs=[out1,out2,out3,out4,out5,out6,out7,out8,out9])
#     model.compile(loss=mycrossentropy, optimizer='adam', metrics=['binary_accuracy'])
#
#     return model