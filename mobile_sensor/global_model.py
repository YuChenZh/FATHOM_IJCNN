"""
Proposed model: Hierarchical Attention model. First attention layer applied on each task's input_dim, second attention layer applied on joined tasks' TIME_step dimension.

"""
import keras
from keras.layers.core import *
from keras.models import *
from keras.layers import Input, Embedding, Dense,Convolution1D,MaxPooling1D,merge
from keras.layers.recurrent import LSTM
import os
import warnings
import glob
import helper_funcs
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import shutil
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings


def load_dataset(datasource,i,j):

    # load the dataset
    dataframe = read_csv(datasource, index_col=0)
    dataframe = dataframe.drop('label_source', axis=1)  # drop the last column

    dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(method='bfill')
    dataframe = dataframe.fillna(0)
    dataframe = dataframe.iloc[i:j]  # first 5000 rows of dataframe

    return dataframe

def min_max(dataframe):
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    return dataset, scaled, scaler

def split_dataset(scaled, look_back, n_columns,n_labels):

    # frame as supervised learning
    reframed = helper_funcs.series_to_supervised(scaled, look_back, 1)

    # split into train and test sets
    values = reframed.values

    # split into input and outputs
    n_obs = look_back * n_columns
    data_X, data_y = values[:, :n_obs], values[:, -n_labels:]  # labels are the last 51 columns
    print(data_X.shape, len(data_X), data_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    data_X = data_X.reshape((data_X.shape[0], look_back, n_columns))

    return data_X, data_y

TIME_STEPS = 20
SINGLE_ATTENTION_VECTOR = False

def attention_3d_block(shared,inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    activation_weights= Flatten()(shared)
    activation_weights=Dense(TIME_STEPS,activation='tanh')(activation_weights)
    activation_weights=Activation('softmax')(activation_weights)
    activation_weights= RepeatVector(input_dim)(activation_weights)
    activation_weights=Permute([2,1])(activation_weights)
    activation_weighted=keras.layers.multiply([inputs, activation_weights])

    # sum_weighted = Lambda(lambda x: K.sum(x, axis=-2), output_shape=(input_dim,))(activation_weighted)
    return activation_weighted


def build_model(trainX,lstm_layer, drop, r_drop, l2_value,dense_num, n_labels):
    """
    Keras Function model
    """

    input = Input(shape=(trainX.shape[1], trainX.shape[2]), name='input')
    lstm_layer1 = LSTM(lstm_layer, activation='relu', dropout=drop,
                                       recurrent_dropout=r_drop,kernel_regularizer=regularizers.l2(l2_value),
                                       return_sequences=True)(input)


    concate_layer = attention_3d_block(lstm_layer1,lstm_layer1)
    LSTM_layer2 = LSTM(dense_num,activation='relu',dropout=0.2,recurrent_dropout=0.2)(concate_layer)
    sub = Dense(dense_num,activation='relu')(LSTM_layer2)
    out= Dense(n_labels, activation='sigmoid')(sub)

    model = Model(inputs=input,outputs=out)
    # adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=helper_funcs.mycrossentropy, optimizer='adam', metrics=[helper_funcs.BA_metric])
    print(model.summary())

    return model



def main():

    # network parameters
    task_num = 40
    con_layer1 = 128 # for 6 users
    # con_layer1 = 256
    con_layer1_filter = 5
    con_layer2 = 64
    con_layer2_filter = 4
    lstm_layer = 64
    drop = 0.3
    r_drop = 0.3
    l2_value = 0.001
    shared_layer = 576
    dense_num = 64

    look_back = 20  # number of previous timestamp used for training
    n_columns = 276  # total columns
    n_labels = 51  # number of labels
    split_ratio = 0.8  # train & test data split ratio

    train_list = []
    # trainy_list = []
    # test_list = []
    # testy_list = []
    file_list = glob.glob('data_csv/train/*.csv')

    for i in range(len(file_list)):
        data = load_dataset(file_list[i], 0, 2400)
        train_list.append(data)

    result = pd.concat(train_list)

    # train_dataset = load_dataset(result)
    train_dataset, scaled_train, scaler_train = min_max(result)
    # split into train and test sets
    train_X, train_y = split_dataset(scaled_train, look_back, n_columns, n_labels)

    model = build_model(train_X, lstm_layer, drop, r_drop, l2_value, dense_num, n_labels)

    import time
    start_time = time.time()

    # fit network
    history = model.fit(train_X, train_y,
                        epochs=100,
                        batch_size=60,
                        # validation_data=(test_X, test_y),
                        validation_split=0.25,
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                          mode='min')]
                        )

    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))


    file = open('results/globalAtt_5.txt', 'w')
    sum_bacc = 0
    sum_TPR = 0
    Num_tp = 0
    Num_fn = 0
    Num_fp = 0
    Num_tn = 0
    sum_precision = 0
    sum_F1 = 0
    train_time = 0

    for i in range(len(file_list)):

        locals()['dataset_test' + str(i)] = load_dataset(file_list[i],2400,3000)

        locals()['dataset_test' + str(i)], locals()['scaled_test' + str(i)], locals()[
            'scaler_test' + str(i)] = min_max(locals()['dataset_test' + str(i)])

        locals()['test_X' + str(i)], locals()['test_y' + str(i)] = split_dataset(
            locals()['scaled_test' + str(i)], look_back, n_columns, n_labels)




        y_predict = model.predict(locals()['test_X' + str(i)])

        results = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], y_predict,
                                          look_back, n_columns, n_labels, locals()['scaler_test' + str(i)])

        sum_bacc = sum_bacc + results[3]
        sum_TPR = sum_TPR + results[1]
        Num_tp = Num_tp + results[4]
        Num_fn = Num_fn + results[5]
        Num_fp = Num_fp + results[6]
        Num_tn = Num_tn + results[7]
        sum_precision = sum_precision + results[8]
        sum_F1 = sum_F1 + results[9]
        train_time = train_time + (end_time - start_time)

        file.write('Accuracy:' + ' ' + str(results[0]) + ' ')
        file.write('TPR:' + ' ' + str(results[1]) + ' ')
        file.write('TNR:' + ' ' + str(results[2]) + ' ')
        file.write('Bacc:' + ' ' + str(results[3]) + '\n')
        file.write('FP No.:' + ' ' + str(results[6]) + '\n')
        file.write('TN No.:' + ' ' + str(results[7]) + '\n')
        file.write('Precision:' + ' ' + str(results[8]) + '\n')
        file.write('F1:' + ' ' + str(results[9]) + '\n')

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
