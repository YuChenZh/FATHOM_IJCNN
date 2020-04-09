"""
Baseline model 3: No attention layers, the other settings are the sanme as Hierarchical Attention model.

"""
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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings



TIME_STEPS = 20
SINGLE_ATTENTION_VECTOR = False

# attention applied on TIME_STEP dimension
def attention_time(inputs,i):
    ## inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction'+str(i))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vect'+str(i))(a)
    output_attention_mul = keras.layers.multiply([inputs, a_probs], name='attention_mul'+str(i))
    return output_attention_mul

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

# attention applied on input_dim dimension
def attention_dim(inputs,i):
    # hidden_states.shape = (batch_size, time_steps, hidden_size)
    hidden_size = int(inputs.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec'+str(i))(inputs)
    # h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(inputs)
    # score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_vecd'+str(i))(score_first_part)
    context_vector = keras.layers.multiply([inputs, attention_weights], name='context_vector'+str(i))
    # pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh',name='attention_vector'+str(i))(context_vector)

    return attention_vector

def build_model(trainX,
                task_num,
                lstm_layer, drop, r_drop, l2_value, shared_layer,dense_num, n_labels):
    """
    Keras Function model
    """

    concate_list = []
    input_list = []
    for i in range(0,task_num):
        locals()['input'+str(i)] = Input(shape=(trainX[i].shape[1], trainX[i].shape[2]), name='input'+str(i))
        # locals()['attentionD' + str(i)] = attention_dim(locals()['input'+str(i)],i)
        locals()['lstm_layer1'+str(i)] = LSTM(lstm_layer, activation='relu', dropout=drop,
                                           recurrent_dropout=r_drop,kernel_regularizer=regularizers.l2(l2_value),
                                           return_sequences=True)(locals()['input'+str(i)])
        concate_list.append(locals()['lstm_layer1'+str(i)])
        input_list.append(locals()['input'+str(i)])


    concate_layer = keras.layers.concatenate(concate_list)


    output_list = []
    for i in range(0,task_num):
        # locals()['concate_layer'+str(i)] = attention_3d_block(concate_layer,locals()['input'+str(i)])
        locals()['LSTM_layer2'+str(i)] = LSTM(dense_num,activation='relu')(concate_layer)
        locals()['sub'+str(i)] = Dense(dense_num,activation='relu')(locals()['LSTM_layer2'+str(i)])
        locals()['out'+str(i)] = Dense(n_labels, activation='sigmoid')(locals()['sub'+str(i)])
        output_list.append(locals()['out'+str(i)])

    model = Model(inputs=input_list,outputs=output_list)
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

    trainX_list = []
    trainy_list = []
    testX_list = []
    testy_list = []
    file_list_train = glob.glob('preprocessed_data/train/*.csv')
    file_list_test = glob.glob('preprocessed_data/test/*.csv')

    for i in range(len(file_list_train)):
        locals()['dataset' + str(i)] = file_list_train[i]
        locals()['dataset' + str(i)], locals()['scaled' + str(i)], locals()['scaler' + str(i)] = helper_funcs.load_dataset(
            locals()['dataset' + str(i)])
        locals()['train_X' + str(i)], locals()['train_y' + str(i)] = helper_funcs.split_dataset(locals()['scaled' + str(i)],
                                                                                   look_back, n_columns, n_labels)

        trainX_list.append(locals()['train_X' + str(i)])
        trainy_list.append(locals()['train_y' + str(i)])

    for i in range(len(file_list_test)):
        locals()['dataset_test' + str(i)] = file_list_test[i]
        locals()['dataset_test' + str(i)], locals()['scaled_test' + str(i)], locals()['scaler_test' + str(i)] = helper_funcs.load_dataset(locals()['dataset_test' + str(i)])
        locals()['test_X' + str(i)], locals()['test_y' + str(i)] = helper_funcs.split_dataset(locals()['scaled_test' + str(i)],
                                                                                 look_back,
                                                                                 n_columns, n_labels)
        testX_list.append(locals()['test_X' + str(i)])
        testy_list.append(locals()['test_y' + str(i)])

    model = build_model(trainX_list,task_num,lstm_layer, drop, r_drop, l2_value, shared_layer, dense_num, n_labels)


    import time
    start_time = time.time()

    # fit network
    history = model.fit(trainX_list, trainy_list,
                        epochs=200,
                        batch_size=120,
                        validation_split = 0.25,
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

    y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7, y_pred8, y_pred9 = model.predict(testX_list)


   #===========================================================================================#
    # write parameters & results to file
    # file = open('results/Attention_results(12)_F1.txt', 'w')
    file = open('results/Base3(NoAttention)-1.txt', 'w')

    file.write('task_num:' + str(task_num) + '\n')
    file.write('lstm_layer:' + str(lstm_layer) + '\n')
    file.write('drop:' + str(drop) + '\n')
    file.write('r_drop:' + str(r_drop) + '\n')
    file.write('l2_value:' + str(l2_value) + '\n')
    file.write('shared_layer:' + str(shared_layer) + '\n')
    file.write('dense_num:' + str(dense_num) + '\n')

    sum_Smape = 0
    sum_Smape_PM25 = 0
    sum_Smape_PM10 = 0
    sum_Smape_NO2 = 0
    sum_Smape_CO = 0
    sum_Smape_O3 = 0
    sum_Smape_SO2 = 0


    for i in range(len(file_list_test)):

        locals()['Smape' + str(i)] = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)],
                                                locals()['y_pred' + str(i + 1)],
                                                look_back, n_columns, n_labels, locals()['scaler_test' + str(i)])

        locals()['Smape_PM25' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)], locals()['test_y' + str(i)],
                                                            locals()['y_pred' + str(i + 1)],
                                                            look_back, n_columns, n_labels,
                                                            locals()['scaler_test' + str(i)], 0)
        locals()['Smape_PM10' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)], locals()['test_y' + str(i)],
                                                            locals()['y_pred' + str(i + 1)],
                                                            look_back, n_columns, n_labels,
                                                            locals()['scaler_test' + str(i)], 1)
        locals()['Smape_NO2' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)], locals()['test_y' + str(i)],
                                                           locals()['y_pred' + str(i + 1)],
                                                           look_back, n_columns, n_labels,
                                                           locals()['scaler_test' + str(i)], 2)
        locals()['Smape_CO' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)], locals()['test_y' + str(i)],
                                                          locals()['y_pred' + str(i + 1)],
                                                          look_back, n_columns, n_labels,
                                                          locals()['scaler_test' + str(i)], 3)
        locals()['Smape_O3' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)], locals()['test_y' + str(i)],
                                                          locals()['y_pred' + str(i + 1)],
                                                          look_back, n_columns, n_labels,
                                                          locals()['scaler_test' + str(i)], 4)
        locals()['Smape_SO2' + str(i)] = helper_funcs.evaluation_single(locals()['test_X' + str(i)], locals()['test_y' + str(i)],
                                                           locals()['y_pred' + str(i + 1)],
                                                           look_back, n_columns, n_labels,
                                                           locals()['scaler_test' + str(i)], 5)

        file.write('Current file index is: ' + str(i) + '\n')
        file.write('Smape:'+' ' + str(locals()['Smape' + str(i)]) + '\n')
        file.write('Smape_PM25:'+' ' + str(locals()['Smape_PM25' + str(i)]) + '\n')
        file.write('Smape_PM10:'+' ' + str(locals()['Smape_PM10' + str(i)]) + '\n')
        file.write('Smape_NO2:'+' ' + str(locals()['Smape_NO2' + str(i)]) + '\n')
        file.write('Smape_CO:'+' ' + str(locals()['Smape_CO' + str(i)]) + '\n')
        file.write('Smape_O3:'+' ' + str(locals()['Smape_O3' + str(i)]) + '\n')
        file.write('Smape_SO2:'+' ' + str(locals()['Smape_SO2' + str(i)]) + '\n')
        file.write('\n')

        sum_Smape = sum_Smape + locals()['Smape' + str(i)]
        sum_Smape_PM25 = sum_Smape_PM25 + locals()['Smape_PM25' + str(i)]
        sum_Smape_PM10 = sum_Smape_PM10 + locals()['Smape_PM10' + str(i)]
        sum_Smape_NO2 = sum_Smape_NO2 +  locals()['Smape_NO2' + str(i)]
        sum_Smape_CO = sum_Smape_CO + locals()['Smape_CO' + str(i)]
        sum_Smape_O3 = sum_Smape_O3 + locals()['Smape_O3' + str(i)]
        sum_Smape_SO2 = sum_Smape_SO2 + locals()['Smape_SO2' + str(i)]

    file.write ('avg_Smape: ' + str(sum_Smape/len(file_list_test)) +'\n')
    file.write ('avg_Smape_PM25: ' + str(sum_Smape_PM25/len(file_list_test)) +'\n')
    file.write ('avg_Smape_PM10: ' + str(sum_Smape_PM10/len(file_list_test)) +'\n')
    file.write ('avg_Smape_NO2: ' + str(sum_Smape_NO2/len(file_list_test)) +'\n')
    file.write ('avg_Smape_CO: ' + str(sum_Smape_CO/len(file_list_test)) +'\n')
    file.write ('avg_Smape_O3: ' + str(sum_Smape_O3/len(file_list_test)) +'\n')
    file.write ('avg_Smape_SO2: ' + str(sum_Smape_SO2/len(file_list_test)) +'\n')
    file.write('training time:' + str(end_time - start_time))

if __name__ == '__main__':
    main()

