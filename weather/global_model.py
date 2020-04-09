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
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings



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


def build_model(trainX,
                lstm_layer, drop, r_drop, l2_value, dense_num, n_labels):
    """
    Keras Function model
    """
    input = Input(shape=(trainX.shape[1], trainX.shape[2]), name='input')
    lstm_layer1= LSTM(lstm_layer, activation='relu', dropout=drop,
                                       recurrent_dropout=r_drop,kernel_regularizer=regularizers.l2(l2_value),
                                       return_sequences=True)(input)
    # concate_layer = keras.layers.concatenate(lstm_layer1)


    # for i in range(0,task_num):
    concate_layer = attention_3d_block(lstm_layer1,input)
    LSTM_layer2 = LSTM(dense_num,activation='relu')(concate_layer)
    sub = Dense(dense_num,activation='relu')(LSTM_layer2)
    out = Dense(n_labels, activation='sigmoid')(sub)

    model = Model(inputs=input,outputs=out)
    # adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    return model



def main():

    # network parameters
    task_num = 9
    lstm_layer = 64
    drop = 0.2
    r_drop = 0.2
    l2_value = 0.001
    shared_layer = 576
    dense_num = 64

    look_back = 20  # number of previous timestamp used for training
    n_columns = 15  # total columns
    n_labels = 6  # number of labels
    split_ratio = 0.8  # train & test data split ratio

    # trainX_list = []
    # trainy_list = []
    # testX_list = []
    # testy_list = []
    file_list_train = glob.glob('preprocessed_data/train/*.csv')
    file_list_test = glob.glob('preprocessed_data/test/*.csv')

    # path = r'data/US/market/merged_data'
    # allFiles = glob.glob(path + "/*.csv")
    with open('train_combined.csv', 'wb') as outfile:
        for i, fname in enumerate(file_list_train):
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()  # Throw away header on all but first file
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)
                print(fname + " has been imported.")

    train_data,scaled,scaler =helper_funcs.load_dataset('train_combined.csv')

    trainX,trainy = helper_funcs.split_dataset(scaled,look_back,n_columns, n_labels)

    file = open('results/globalAtt_1.txt', 'w')
    sum_Smape = 0
    sum_Smape_PM25 = 0
    sum_Smape_PM10 = 0
    sum_Smape_NO2 = 0
    sum_Smape_CO = 0
    sum_Smape_O3 = 0
    sum_Smape_SO2 = 0

    for i in range(len(file_list_train)):
        # train_data = 'data/preprocessed_data/train/bj_huairou.csv'
        # test_data = 'data/preprocessed_data/test/bj_huairou_201805.csv'

        # locals()['dataset_train' + str(i)], locals()['scaled_train' + str(i)], locals()[
        #     'scaler_train' + str(i)] = helper_funcs.load_dataset(file_list_train[i])
        locals()['dataset_test' + str(i)], locals()['scaled_test' + str(i)], locals()[
            'scaler_test' + str(i)] = helper_funcs.load_dataset(file_list_test[i])

        # split into train and test sets
        # locals()['train_X' + str(i)], locals()['train_y' + str(i)] = helper_funcs.split_dataset(
        #     locals()['scaled_train' + str(i)], look_back, n_columns, n_labels)
        locals()['test_X' + str(i)], locals()['test_y' + str(i)] = helper_funcs.split_dataset(
            locals()['scaled_test' + str(i)], look_back, n_columns, n_labels)

        model = build_model(trainX,lstm_layer, drop, r_drop, l2_value, dense_num, n_labels)

        import time
        start_time = time.time()

        # fit network
        history = model.fit(trainX, trainy,
                            epochs=100,
                            batch_size=120,
                            validation_data=(locals()['test_X' + str(i)], locals()['test_y' + str(i)]), verbose=2,
                            shuffle=False,
                            callbacks=[
                                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2,
                                                              mode='min')]
                            )

        end_time = time.time()
        print('--- %s seconds ---' % (end_time - start_time))

        # plot history
        # plt.plot(history.history['loss'], label='train')
        # plt.plot(history.history['val_loss'], label='test')
        # plt.legend()
        # plt.show()

        # make a prediction
        y_predict = model.predict(locals()['test_X' + str(i)])
        # results = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], y_predict, look_back, n_columns, n_labels, locals()['scaler' + str(i)])

        locals()['Smape' + str(i)] = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)],
                                                             y_predict,
                                                             look_back, n_columns, n_labels,
                                                             locals()['scaler_test' + str(i)])

        locals()['Smape_PM25' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)],
                                                                         locals()['test_y' + str(i)],
                                                                         y_predict,
                                                                         look_back, n_columns, n_labels,
                                                                         locals()['scaler_test' + str(i)], 0)
        locals()['Smape_PM10' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)],
                                                                         locals()['test_y' + str(i)],
                                                                         y_predict,
                                                                         look_back, n_columns, n_labels,
                                                                         locals()['scaler_test' + str(i)], 1)
        locals()['Smape_NO2' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)],
                                                                        locals()['test_y' + str(i)],
                                                                        y_predict,
                                                                        look_back, n_columns, n_labels,
                                                                        locals()['scaler_test' + str(i)], 2)
        locals()['Smape_CO' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)],
                                                                       locals()['test_y' + str(i)],
                                                                       y_predict,
                                                                       look_back, n_columns, n_labels,
                                                                       locals()['scaler_test' + str(i)], 3)
        locals()['Smape_O3' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)],
                                                                       locals()['test_y' + str(i)],
                                                                       y_predict,
                                                                       look_back, n_columns, n_labels,
                                                                       locals()['scaler_test' + str(i)], 4)
        locals()['Smape_SO2' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)],
                                                                        locals()['test_y' + str(i)],
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

if __name__ == '__main__':
    main()
