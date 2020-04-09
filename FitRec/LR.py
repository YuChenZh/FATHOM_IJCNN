import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
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
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_dataset(datasource):
    # load the dataset
    dataframe = read_csv(datasource)
    # dataframe.set_index('timestamp',inplace = True)
    dataframe = dataframe.drop(['id', 'since_begin', 'since_last', 'time_elapsed'], axis=1)
    dataframe = dataframe.sort_values(by='timestamp')
    # dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(0)
    dataframe = dataframe.iloc[0:30000]  # first 30000 rows of dataframe
    dataframe.tar_derived_speed = dataframe.tar_derived_speed.round()
    dataframe.tar_heart_rate = dataframe.tar_heart_rate.round()

    features = dataframe.iloc[:,0:10]
    labels = dataframe.iloc[:,10:]

    features = features.values
    # labels = labels.values


    # integer encode direction
    encoder = LabelEncoder()
    features[:, 3] = encoder.fit_transform(features[:, 3])
    features[:, 7] = encoder.fit_transform(features[:, 7])
    # labels[:, 0] = encoder.fit_transform(labels[:, 0])
    # labels[:, 1] = encoder.fit_transform(labels[:, 1])

    features = features.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(features)
    normalized_features = pd.DataFrame(scaled)
    # labels = pd.DataFrame(labels)
    return (normalized_features,labels)


if __name__ == '__main__':
    file_list_train = glob.glob('data/data_user/*.csv')

    file = open('results/Single_LR_HRate.txt', 'w')
    sum_Smape = 0
    sum_Smape_speed = 0
    sum_Smape_heartRate = 0
    sum_mae = 0
    sum_mae_speed = 0
    sum_mae_heartRate = 0

    for i in range(len(file_list_train)):

        ## Part 2 ************** fixed train & test prediction *****************
        # data = pd.read_csv('data_csv/train/5users/0E6184E1-90C0-48EE-B25A-F1ECB7B9714E.features_labels.csv')
        # locals()['dataset_train' + str(i)] = load_dataset(file_list_train[i])

        locals()['features' + str(i)],locals()['labels' + str(i)] = load_dataset(file_list_train[i])
        locals()['X_train' + str(i)] = locals()['features' + str(i)].iloc[0:18000,:]
        locals()['Y_train' + str(i)] = locals()['labels' + str(i)].iloc[0:18000,:]

        locals()['X_test' + str(i)] = locals()['features' + str(i)].iloc[24000:30000, :]
        locals()['Y_test' + str(i)] = locals()['labels' + str(i)].iloc[24000:30000,:]

        ## ------- Fit an independent logistic regression model or MLP for each class using the OneVsRestClassifier wrapper.
        import time

        start_time = time.time()
        ovr = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto')

        # ovr = MultiOutputRegressor(LogisticRegression())
        # ------ evaluation of test data predicion  -----------------------#

        ovr.fit(locals()['X_train' + str(i)], locals()['Y_train' + str(i)].iloc[:,1])
        Y_pred = ovr.predict(locals()['X_test' + str(i)])

        end_time = time.time()
        print('--- %s seconds ---' % (end_time - start_time))

        locals()['Smape_Speed' + str(i)] = helper_funcs.smape(locals()['Y_test' + str(i)].iloc[:,1],Y_pred)
        # locals()['Smape_HRate' + str(i)] = helper_funcs.smape(locals()['Y_test' + str(i)].iloc[:,1],Y_pred[1])
        locals()['mae_Speed' + str(i)] = mean_absolute_error(locals()['Y_test' + str(i)].iloc[:,1],Y_pred)
        # locals()['mae_HRate' + str(i)] = mean_absolute_error(locals()['Y_test' + str(i)].iloc[:,1],Y_pred[1])

        file.write('Current file index is: ' + str(i) + '\n')
        # file.write('Smape_HRate:' + ' ' + str(locals()['Smape_HRate' + str(i)]) + '\n')
        file.write('Smape_Speed:' + ' ' + str(locals()['Smape_Speed' + str(i)]) + '\n')
        # file.write('mae_HRate:' + ' ' + str(locals()['mae_HRate' + str(i)]) + '\n')
        file.write('mae_Speed:' + ' ' + str(locals()['mae_Speed' + str(i)]) + '\n')
        file.write('\n')

        sum_Smape_speed = sum_Smape_speed + locals()['Smape_Speed' + str(i)]
        # sum_Smape_heartRate = sum_Smape_heartRate + locals()['Smape_HRate' + str(i)]
        sum_mae_speed = sum_mae_speed + locals()['mae_Speed' + str(i)]
        # sum_mae_heartRate = sum_mae_heartRate + locals()['mae_HRate' + str(i)]

    file.write('avg_sum_Smape_speed: ' + str(sum_Smape_speed / len(file_list_train)) + '\n')
    # file.write('avg_sum_Smape_heartRate: ' + str(sum_Smape_heartRate / len(file_list_train)) + '\n')
    file.write('avg_sum_mae_speed: ' + str(sum_mae_speed / len(file_list_train)) + '\n')
    # file.write('avg_sum_mae_heartRate: ' + str(sum_mae_heartRate / len(file_list_train)) + '\n')

    file.write('training time:' + str(end_time - start_time))





