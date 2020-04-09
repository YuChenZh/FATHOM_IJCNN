import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import glob
from keras.layers.normalization import BatchNormalization
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import helper_funcs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_dataset(datasource):

    # load the dataset
    dataframe = read_csv(datasource)
    dataframe.set_index('utc_time',inplace = True)

    # dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(0)

    dataset = dataframe.values
    # integer encode direction
    encoder = LabelEncoder()
    dataset[:, 8] = encoder.fit_transform(dataset[:, 8])
    dataset[:, 7] = encoder.fit_transform(dataset[:, 7])
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)

    dataset = pd.DataFrame(scaled)

    return dataset


if __name__ == '__main__':
    file_list_train = glob.glob('preprocessed_data/train/*.csv')
    file_list_test = glob.glob('preprocessed_data/test/*.csv')
    
    file = open('results/Single_LR_2.txt', 'w')
    sum_Smape = 0
    sum_Smape_PM25 = 0
    sum_Smape_PM10 = 0
    sum_Smape_NO2 = 0
    sum_Smape_CO = 0
    sum_Smape_O3 = 0
    sum_Smape_SO2 = 0

    for i in range(len(file_list_train)):

        ## Part 2 ************** fixed train & test prediction *****************
        # data = pd.read_csv('data_csv/train/5users/0E6184E1-90C0-48EE-B25A-F1ECB7B9714E.features_labels.csv')
        locals()['dataset_train' + str(i)] = load_dataset(file_list_train[i])

        locals()['X_train' + str(i)] = locals()['dataset_train' + str(i)].iloc[:,0:9]
        locals()['Y_train' + str(i)] = locals()['dataset_train' + str(i)].iloc[:,9:15]

        locals()['dataset_test' + str(i)] = load_dataset(file_list_train[i])

        locals()['X_test' + str(i)] = locals()['dataset_test' + str(i)].iloc[:, 0:9]
        locals()['Y_test' + str(i)] = locals()['dataset_test' + str(i)].iloc[:, 9:15]

        ## ------- Fit an independent logistic regression model or MLP for each class using the OneVsRestClassifier wrapper.
        import time

        start_time = time.time()
        ovr = OneVsRestClassifier(LogisticRegression())
        # ------ evaluation of test data predicion  -----------------------#

        ovr.fit(locals()['X_train' + str(i)], locals()['Y_train' + str(i)])
        Y_pred = ovr.predict(locals()['X_test' + str(i)])

        end_time = time.time()
        print('--- %s seconds ---' % (end_time - start_time))

        locals()['Smape' + str(i)] = helper_funcs.smape(locals()['Y_test' + str(i)],Y_pred)
        locals()['Smape_PM25' + str(i)] = helper_funcs.smape(locals()['Y_test' + str(i)][0],Y_pred[0])
        locals()['Smape_PM10' + str(i)] = helper_funcs.smape(locals()['Y_test' + str(i)][1],Y_pred[1])
        locals()['Smape_NO2' + str(i)] = helper_funcs.smape(locals()['Y_test' + str(i)][2],Y_pred[2])
        locals()['Smape_CO' + str(i)] = helper_funcs.smape(locals()['Y_test' + str(i)][3],Y_pred[3])
        locals()['Smape_O3' + str(i)] = helper_funcs.smape(locals()['Y_test' + str(i)][4],Y_pred[4])
        locals()['Smape_SO2' + str(i)] = helper_funcs.smape(locals()['Y_test' + str(i)][5],Y_pred[5])

        file.write('Current file index is: ' + str(i) + '\n')
        file.write('Smape:' + ' ' + str(locals()['Smape' + str(i)]) + '\n')
        file.write('Smape_PM25:' + ' ' + str(locals()['Smape_PM25' + str(i)]) + '\n')
        file.write('Smape_PM10:' + ' ' + str(locals()['Smape_PM10' + str(i)]) + '\n')
        file.write('Smape_NO2:' + ' ' + str(locals()['Smape_NO2' + str(i)]) + '\n')
        file.write('Smape_CO:' + ' ' + str(locals()['Smape_CO' + str(i)]) + '\n')
        file.write('Smape_O3:' + ' ' + str(locals()['Smape_O3' + str(i)]) + '\n')
        file.write('Smape_SO2:' + ' ' + str(locals()['Smape_SO2' + str(i)]) + '\n')
        file.write('\n')

        sum_Smape = sum_Smape + locals()['Smape' + str(i)]
        sum_Smape_PM25 = sum_Smape_PM25 + locals()['Smape_PM25' + str(i)]
        sum_Smape_PM10 = sum_Smape_PM10 + locals()['Smape_PM10' + str(i)]
        sum_Smape_NO2 = sum_Smape_NO2 + locals()['Smape_NO2' + str(i)]
        sum_Smape_CO = sum_Smape_CO + locals()['Smape_CO' + str(i)]
        sum_Smape_O3 = sum_Smape_O3 + locals()['Smape_O3' + str(i)]
        sum_Smape_SO2 = sum_Smape_SO2 + locals()['Smape_SO2' + str(i)]

    file.write('avg_Smape: ' + str(sum_Smape / len(file_list_test)) + '\n')
    file.write('avg_Smape_PM25: ' + str(sum_Smape_PM25 / len(file_list_test)) + '\n')
    file.write('avg_Smape_PM10: ' + str(sum_Smape_PM10 / len(file_list_test)) + '\n')
    file.write('avg_Smape_NO2: ' + str(sum_Smape_NO2 / len(file_list_test)) + '\n')
    file.write('avg_Smape_CO: ' + str(sum_Smape_CO / len(file_list_test)) + '\n')
    file.write('avg_Smape_O3: ' + str(sum_Smape_O3 / len(file_list_test)) + '\n')
    file.write('avg_Smape_SO2: ' + str(sum_Smape_SO2 / len(file_list_test)) + '\n')
    file.write('training time:' + str(end_time - start_time))





