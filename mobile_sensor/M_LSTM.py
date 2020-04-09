"""
Keras LSTM, multi-task & multi-outputs prediction (also can be used in multi-label situation)

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
from keras.layers import Input, Embedding, LSTM, Dense
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Model
import helper_funcs
import os
import warnings
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings



def build_model(trainX,
                task_num, lstm_layer, drop, r_drop, shared_layer,dense_num, n_labels):

    """
    Keras Function model
    """
    concate_list = []
    input_list = []
    for i in range(0, task_num):
        locals()['input' + str(i)] = Input(shape=(trainX[i].shape[1], trainX[i].shape[2]),
            name='input' + str(i))
        locals()['lstm_out' + str(i)] = LSTM(lstm_layer, activation='relu', dropout=drop, recurrent_dropout=r_drop)(
            locals()['input' + str(i)])
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
        locals()['dataset' + str(i)], locals()['scaled' + str(i)], locals()['scaler' + str(i)] = helper_funcs.load_dataset(
            locals()['dataset' + str(i)])
        locals()['train_X' + str(i)], locals()['train_y' + str(i)], locals()['test_X' + str(i)], locals()[
            'test_y' + str(i)] = helper_funcs.split_dataset(locals()['dataset' + str(i)], locals()['scaled' + str(i)], look_back,
                                               n_columns, n_labels, split_ratio)
        trainX_list.append(locals()['train_X' + str(i)])
        trainy_list.append(locals()['train_y' + str(i)])
        testX_list.append(locals()['test_X' + str(i)])
        testy_list.append(locals()['test_y' + str(i)])

    model = build_model(trainX_list,task_num, lstm_layer, drop, r_drop, shared_layer, dense_num, n_labels)

    import time
    start_time = time.time()

    # fit network
    history = model.fit(trainX_list, trainy_list,
                        epochs=150,
                        batch_size=60,
                        validation_split=0.25,
                        # validation_data=(testX_list, testy_list),
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=2,
                                                          mode='min')]
                        )
    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    # make prediction
    pred_time = time.time()

    # y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6 = model.predict(testX_list)
    y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7, y_pred8, y_pred9, y_pred10, y_pred11, y_pred12, y_pred13, y_pred14, y_pred15, y_pred16, y_pred17, y_pred18, y_pred19, y_pred20, y_pred21, y_pred22, y_pred23, y_pred24, y_pred25, y_pred26, y_pred27 \
        , y_pred28, y_pred29, y_pred30, y_pred31, y_pred32, y_pred33, y_pred34, y_pred35, y_pred36, y_pred37, y_pred38, y_pred39, y_pred40 = model.predict(testX_list)

    pred_end_time = time.time()

    #=====================================================================================#
    # write parameters & results to file
    # file = open('results/Baseline_results(6)_F1.txt', 'w')
    file = open('time_cost/M_LSTM.txt', 'w')

    file.write('task_num:' + str(task_num) + '\n')
    file.write('lstm_layer:' + str(lstm_layer) + '\n')
    file.write('drop:' + str(drop) + '\n')
    file.write('r_drop:' + str(r_drop) + '\n')
    file.write('shared_layer:' + str(shared_layer) + '\n')
    file.write('dense_num:' + str(dense_num) + '\n')
    file.write('running time:' + str(end_time - start_time) + '\n')


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
        locals()['Bacc' + str(i)] = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], locals()['y_pred' + str(i+1)],
                                               look_back, n_columns, n_labels, locals()['scaler' + str(i)])
        sum_bacc = sum_bacc + (locals()['Bacc' + str(i)])[3]
        sum_TPR = sum_TPR + (locals()['Bacc' + str(i)])[1]
        Num_tp = Num_tp + (locals()['Bacc' + str(i)])[4]
        Num_fn = Num_fn + (locals()['Bacc' + str(i)])[5]
        Num_fp = Num_fp + (locals()['Bacc' + str(i)])[6]
        Num_tn = Num_tn + (locals()['Bacc' + str(i)])[7]
        sum_precision = sum_precision + (locals()['Bacc' + str(i)])[8]
        sum_F1 = sum_precision + (locals()['Bacc' + str(i)])[9]

        file.write ('Accuracy:'+' ' + str((locals()['Bacc' + str(i)])[0])+' ')
        file.write ('TPR:'+' ' + str((locals()['Bacc' + str(i)])[1])+' ')
        file.write ('TNR:'+' '+ str((locals()['Bacc' + str(i)])[2])+' ')
        file.write ('Bacc:'+' ' + str((locals()['Bacc' + str(i)])[3])+ '\n')
        file.write('TP No.:' + ' ' + str((locals()['Bacc' + str(i)])[4]) + '\n')
        file.write('FN No.:' + ' ' + str((locals()['Bacc' + str(i)])[5]) + '\n')
        file.write('FP No.:' + ' ' + str((locals()['Bacc' + str(i)])[6]) + '\n')
        file.write('TN No.:' + ' ' + str((locals()['Bacc' + str(i)])[7]) + '\n')
        file.write('Precision:' + ' ' + str((locals()['Bacc' + str(i)])[8]) + '\n')
        file.write('F1:' + ' ' + str((locals()['Bacc' + str(i)])[9]) + '\n')

    file.write ('avg_bacc: ' + str(sum_bacc/len(file_list)) +'\n')
    file.write ('avg_TPR: ' + str(sum_TPR/len(file_list))+'\n')
    file.write ('avg_precision: ' + str(sum_precision/len(file_list))+'\n')
    file.write ('avg_F1: ' + str(sum_F1/len(file_list))+'\n')
    file.write('sum_Num_tp: ' + str(Num_tp) + '\n')
    file.write('sum_Num_fn: ' + str(Num_fn) + '\n')
    file.write('sum_Num_fp: ' + str(Num_fp) + '\n')
    file.write('sum_Num_tn: ' + str(Num_tn) + '\n')
    file.write('training time:' + str(end_time - start_time))
    file.write('prediction time:' + str(pred_end_time - pred_time))
    
if __name__ == '__main__':
    main()


# back-up code

# def build_model(data1,data2,data3,data4,data5,data6,data7,data8,data9):
#
#     """
#     Keras Function model
#     """
#     input1 = Input(shape=(data1.shape[1], data1.shape[2]), name='input1')
#     lstm_out1 = LSTM(64, activation='relu',dropout=0.2, recurrent_dropout=0.2)(input1)
#     input2 = Input(shape=(data2.shape[1], data2.shape[2]), name='input2')
#     lstm_out2 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input2)
#     input3 = Input(shape=(data3.shape[1], data3.shape[2]), name='input3')
#     lstm_out3 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input3)
#     input4 = Input(shape=(data4.shape[1], data4.shape[2]), name='input4')
#     lstm_out4 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input4)
#     input5 = Input(shape=(data5.shape[1], data5.shape[2]), name='input5')
#     lstm_out5 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input5)
#     input6 = Input(shape=(data6.shape[1], data6.shape[2]), name='input6')
#     lstm_out6 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input6)
#     input7 = Input(shape=(data7.shape[1], data7.shape[2]), name='input7')
#     lstm_out7 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input7)
#     input8 = Input(shape=(data8.shape[1], data8.shape[2]), name='input8')
#     lstm_out8 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input8)
#     input9 = Input(shape=(data9.shape[1], data9.shape[2]), name='input9')
#     lstm_out9 = LSTM(64, activation='relu',dropout=0.2,recurrent_dropout=0.2)(input9)
#
#     concate_layer = keras.layers.concatenate([lstm_out1, lstm_out2,lstm_out3,lstm_out4,lstm_out5,lstm_out6,lstm_out7,lstm_out8,lstm_out9])
#     x = Dense(576, activation='relu')(concate_layer)
#     x = Dense(576, activation='relu')(x)
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
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
#
#     return model




    # dataset1 = 'data_csv/train/10users/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv'
    # dataset2 = 'data_csv/train/10users/0E6184E1-90C0-48EE-B25A-F1ECB7B9714E.features_labels.csv'
    # dataset3 = 'data_csv/train/10users/1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842.features_labels.csv'
    # dataset4 = 'data_csv/train/10users/2C32C23E-E30C-498A-8DD2-0EFB9150A02E.features_labels.csv'
    # dataset5 = 'data_csv/train/10users/4FC32141-E888-4BFF-8804-12559A491D8C.features_labels.csv'
    # dataset6 = 'data_csv/train/10users/5EF64122-B513-46AE-BCF1-E62AAC285D2C.features_labels.csv'
    # dataset7 = 'data_csv/train/10users/7CE37510-56D0-4120-A1CF-0E23351428D2.features_labels.csv'
    # dataset8 = 'data_csv/train/10users/11B5EC4D-4133-4289-B475-4E737182A406.features_labels.csv'
    # dataset9 = 'data_csv/train/10users/74B86067-5D4B-43CF-82CF-341B76BEA0F4.features_labels.csv'
    #
    #
    # dataset1, scaled1, scaler1 = load_dataset(dataset1)
    # dataset2, scaled2, scaler2 = load_dataset(dataset2)
    # dataset3, scaled3, scaler3 = load_dataset(dataset3)
    # dataset4, scaled4, scaler4 = load_dataset(dataset4)
    # dataset5, scaled5, scaler5 = load_dataset(dataset5)
    # dataset6, scaled6, scaler6 = load_dataset(dataset6)
    # dataset7, scaled7, scaler7 = load_dataset(dataset7)
    # dataset8, scaled8, scaler8 = load_dataset(dataset8)
    # dataset9, scaled9, scaler9 = load_dataset(dataset9)
    #
    #
    # look_back = 20  # number of previous timestamp used for training
    # n_columns = 276  # total columns
    # n_labels = 51  # number of labels
    # split_ratio = 0.8 # train & test data split ratio
    #
    #
    # # get train and test sets
    # train_X1, train_y1, test_X1, test_y1 = split_dataset(dataset1, scaled1, look_back, n_columns, n_labels, split_ratio)
    # train_X2, train_y2, test_X2, test_y2 = split_dataset(dataset2, scaled2, look_back, n_columns, n_labels, split_ratio)
    # train_X3, train_y3, test_X3, test_y3 = split_dataset(dataset3, scaled3, look_back, n_columns, n_labels, split_ratio)
    # train_X4, train_y4, test_X4, test_y4 = split_dataset(dataset4, scaled4, look_back, n_columns, n_labels, split_ratio)
    # train_X5, train_y5, test_X5, test_y5 = split_dataset(dataset5, scaled5, look_back, n_columns, n_labels, split_ratio)
    # train_X6, train_y6, test_X6, test_y6 = split_dataset(dataset6, scaled6, look_back, n_columns, n_labels, split_ratio)
    # train_X7, train_y7, test_X7, test_y7 = split_dataset(dataset7, scaled7, look_back, n_columns, n_labels, split_ratio)
    # train_X8, train_y8, test_X8, test_y8 = split_dataset(dataset8, scaled8, look_back, n_columns, n_labels, split_ratio)
    # train_X9, train_y9, test_X9, test_y9 = split_dataset(dataset9, scaled9, look_back, n_columns, n_labels, split_ratio)
    #
    #
    #
    # model = build_model(train_X1,train_X2,train_X3,train_X4,train_X5,train_X6,train_X7,train_X8,train_X9)
    #
    # import time
    # start_time = time.time()
    #
    # # fit network
    # history = model.fit([train_X1,train_X2,train_X3,train_X4,train_X5,train_X6,train_X7,train_X8,train_X9],
    #                     [train_y1,train_y2,train_y3,train_y4,train_y5,train_y6,train_y7,train_y8,train_y9],
    #                     epochs=50,
    #                     batch_size=72,
    #                     validation_data=([test_X1,test_X2,test_X3,test_X4,test_X5,test_X6,test_X7,test_X8,test_X9],
    #                                      [test_y1,test_y2,test_y3,test_y4,test_y5,test_y6,test_y7,test_y8,test_y9]),
    #                     verbose=2,
    #                     shuffle=False,
    #                     callbacks=[
    #                         keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2,
    #                                                       mode='min')]
    #                     )
    # end_time = time.time()
    # print('--- %s seconds ---' % (end_time - start_time))
    #
    # # plot history
    # plt.plot(history.history['loss'], label='train')
    # plt.legend()
    # plt.show()
    #
    # # make prediction
    # y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7,y_pred8,y_pred9 = model.predict([test_X1,test_X2,test_X3,test_X4,test_X5,test_X6,test_X7,test_X8,test_X9])
    # # print (len(y_pred1))
    #
    #
    # # balance accuracy
    # Bacc1 = evaluation(test_X1, test_y1, y_pred1, look_back, n_columns, n_labels, scaler1)
    # Bacc2 = evaluation(test_X2, test_y2, y_pred2, look_back, n_columns, n_labels, scaler2)
    # Bacc3 = evaluation(test_X3, test_y3, y_pred3, look_back, n_columns, n_labels, scaler3)
    # Bacc4 = evaluation(test_X4, test_y4, y_pred4, look_back, n_columns, n_labels, scaler4)
    # Bacc5 = evaluation(test_X5, test_y5, y_pred5, look_back, n_columns, n_labels, scaler5)
    # Bacc6 = evaluation(test_X6, test_y6, y_pred6, look_back, n_columns, n_labels, scaler6)
    # Bacc7 = evaluation(test_X7, test_y7, y_pred7, look_back, n_columns, n_labels, scaler7)
    # Bacc8 = evaluation(test_X8, test_y8, y_pred8, look_back, n_columns, n_labels, scaler8)
    # Bacc9 = evaluation(test_X9, test_y9, y_pred9, look_back, n_columns, n_labels, scaler9)

