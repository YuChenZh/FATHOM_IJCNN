"""
Keras CNN & LSTM, with shared attention layer, multi-task & multi-label prediction

CNN is for the data sample selection

"""
import keras
from keras.layers.core import *
from keras.models import *
from keras.layers import Input, Embedding, Dense,Convolution1D,MaxPooling1D,merge
from keras.layers.recurrent import LSTM
import os
import warnings
import glob
import sys
sys.path.insert(0, '/Users/yujingchen/PycharmProjects/WATCH_proj/mobile_sensor/')
import helper_funcs


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings



TIME_STEPS = 30
SINGLE_ATTENTION_VECTOR = False

# # attention applied on TIME_STEP dimension
# def attention_time(inputs,i):
#     ## inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[2])
#     a = Permute((2, 1))(inputs)
#     a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(TIME_STEPS, activation='softmax')(a)
#     if SINGLE_ATTENTION_VECTOR:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction'+str(i))(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((2, 1), name='attention_vect'+str(i))(a)
#     output_attention_mul = keras.layers.multiply([inputs, a_probs], name='attention_mul'+str(i))
#     return output_attention_mul

# # attention applied on TIME_STEP dimension
def attention_3d_block(shared,inputs,i):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    activation_weights= Flatten()(shared)
    activation_weights=Dense(TIME_STEPS,activation='tanh')(activation_weights)
    activation_weights=Activation('softmax')(activation_weights)
    print(activation_weights)
    print('shape after softmax is:',activation_weights)
    activation_weights= RepeatVector(input_dim)(activation_weights)
    print('shape after RepeatVector is:',activation_weights)
    activation_weights=Permute([2,1],name='attention_vect'+str(i))(activation_weights)
    activation_weighted=keras.layers.multiply([inputs, activation_weights])
    print('shape after multiply is:',activation_weights)

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
                task_num, con_layer1, con_layer1_filter, con_layer2, con_layer2_filter,
                lstm_layer, drop, r_drop, l2_value, shared_layer,dense_num, n_labels):
    """
    Keras Function model
    """

    concate_list = []
    input_list = []
    for i in range(0,task_num):
        locals()['input'+str(i)] = Input(shape=(trainX[i].shape[1], trainX[i].shape[2]), name='input'+str(i))
        locals()['attentionD' + str(i)] = attention_dim(locals()['input'+str(i)],i)
        locals()['lstm_layer1'+str(i)] = LSTM(lstm_layer, activation='relu', dropout=drop,
                                           recurrent_dropout=r_drop,kernel_regularizer=regularizers.l2(l2_value),
                                           return_sequences=True)(locals()['attentionD'+str(i)])
        concate_list.append(locals()['lstm_layer1'+str(i)])
        input_list.append(locals()['input'+str(i)])


    concate_layer = keras.layers.concatenate(concate_list)


    output_list = []
    for i in range(0,task_num):
        locals()['concate_layer'+str(i)] = attention_3d_block(concate_layer,locals()['input'+str(i)],i)
        locals()['LSTM_layer2'+str(i)] = LSTM(dense_num,activation='relu',dropout=drop,
                                              recurrent_dropout=r_drop, kernel_regularizer=regularizers.l2(l2_value)
                                              )(locals()['concate_layer'+str(i)])
        locals()['sub'+str(i)] = Dense(dense_num,activation='relu')(locals()['LSTM_layer2'+str(i)])
        locals()['out'+str(i)] = Dense(n_labels, activation='sigmoid')(locals()['sub'+str(i)])
        output_list.append(locals()['out'+str(i)])

    model = Model(inputs=input_list,outputs=output_list)
    # adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=helper_funcs.mycrossentropy, optimizer='adam', metrics=[helper_funcs.BA_metric])
    print(model.summary())

    return model



def main():

    # network parameters
    task_num = 6
    con_layer1 = 128 # for 6 users
    # con_layer1 = 256
    con_layer1_filter = 5
    con_layer2 = 64
    con_layer2_filter = 4
    lstm_layer = 64
    drop = 0.2
    r_drop = 0.2
    l2_value = 0.001
    shared_layer = 576
    dense_num = 64

    look_back = 30  # number of previous timestamp used for training
    n_columns = 276  # total columns
    n_labels = 51  # number of labels
    split_ratio = 0.8  # train & test data split ratio

    trainX_list = []
    trainy_list = []
    testX_list = []
    testy_list = []
    file_list = glob.glob('../data_csv/train/6users/*.csv')

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

    model = build_model(trainX_list,task_num, con_layer1, con_layer1_filter, con_layer2, con_layer2_filter,
                    lstm_layer, drop, r_drop, l2_value, shared_layer, dense_num, n_labels)


    import time
    start_time = time.time()

    # fit network
    history = model.fit(trainX_list, trainy_list,
                        epochs=100,
                        batch_size=60,
                        validation_split = 0.25,
                        # validation_data=(testX_list, testy_list),
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2,
                                                          mode='min')]
                        )
    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    # ####################  attention plot - input dimension ################################
    #
    # attention_vectorsd = []
    # layer_name_list = ['attention_vecd0','attention_vecd1','attention_vecd2','attention_vecd3','attention_vecd4','attention_vecd5']
    # for i in range(len(file_list)):
    #     for j in range(10):
    #         activations = helper_funcs.get_activations(model, locals()['test_X' + str(i)], \
    #                                                    print_shape_only=True, layer_name=layer_name_list[i])
    #         attention_vec = np.mean(activations[0], axis=0).squeeze()
    #         print('attention =', attention_vec)
    #         # assert (np.sum(attention_vec) - 1.0) < 1e-5
    #         attention_vectorsd.append(attention_vec)
    #
    #     attention_vector_final = np.mean(np.array(attention_vectorsd), axis=0)
    #     print(len(attention_vector_final[0]))
    #
    #     # plot part.
    #     import matplotlib.pyplot as plt
    #     import pandas as pd
    #
    #     df = pd.DataFrame(attention_vector_final[26], columns=['attention (%)']) ## attention_vector_final[i], i is the index of test datapoint
    #     df.to_csv('results/attention_plot/input_dim_attention'+str(i)+'.csv')
    #     df.plot(kind='bar',title='Attention Mechanism as ''a function of input'' dimensions.')
    #     # plt.figure(figsize=(100, 100))
    #     plt.xticks(rotation=90)
    #     # plt.savefig('results/attention_plot/input_dim_attention_plot'+ str(i)+'.png', dpi=150)
    #     plt.show()
    #
    # ###################################### attention plot-input dimension ends #########################################

    # ####################  attention plot - TIME STEP ################################
    #
    # attention_vectorst = []
    # for i in range(len(file_list)):
    #     for j in range(10):
    #         activations = helper_funcs.get_activations(model, locals()['test_X' + str(i)], \
    #                                                    print_shape_only=True, layer_name='attention_vect'+str(i))
    #         attention_vec = np.mean(activations[0], axis=2).squeeze()
    #         print('attention_vec shape:', attention_vec.shape)
    #         print('attention =', attention_vec)
    #         # assert (np.sum(attention_vec) - 1.0) < 1e-5
    #         attention_vectorst.append(attention_vec)
    #
    #     attention_vector_final = np.mean(np.array(attention_vectorst), axis=0)
    #     print('attention_vector_final shape:', attention_vector_final.shape)
    #
    #     # plot part.
    #     import matplotlib.pyplot as plt
    #     import pandas as pd
    #
    #     df = pd.DataFrame(attention_vector_final[0], columns=['attention (%)'])
    #     df.to_csv('results/attention_plot/TIME_STEP_attention' + str(i) + '.csv')
    #     df.plot(kind='bar', title='Attention Mechanism as ''a function of input'' dimensions.')
    #     # plt.figure(figsize=(100, 100))
    #     plt.xticks(rotation=90)
    #     plt.savefig('results/attention_plot/TIME_STEP_attention_plot' + str(i) + '.png', dpi=150)
    #     plt.show()
    #
    # ###################################### attention plot-TIME STEP ends #########################################

    # make prediction

    # y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7,y_pred8,y_pred9,y_pred10,y_pred11,y_pred12,y_pred13,y_pred14,y_pred15,y_pred16,y_pred17,y_pred18,y_pred19,y_pred20,y_pred21,y_pred22,y_pred23,y_pred24,y_pred25,y_pred26,y_pred27 \
    #     , y_pred28,y_pred29,y_pred30,y_pred31,y_pred32,y_pred33,y_pred34,y_pred35,y_pred36,y_pred37,y_pred38,y_pred39,y_pred40 = model.predict(testX_list)
    # print (len(y_pred1))
    y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6 = model.predict(testX_list)
    # y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6,y_pred7,y_pred8,y_pred9,y_pred10,y_pred11,y_pred12 = model.predict(testX_list)


   #===========================================================================================#
    # write parameters & results to file
    # file = open('results/Attention_results(12)_F1.txt', 'w')
    file = open('TimeStep_30.txt', 'w')

    file.write('task_num:' + str(task_num) + '\n')
    file.write('con_layer1:' + str(con_layer1) + '\n')
    file.write('con_layer1_filter:' + str(con_layer1_filter) + '\n')
    file.write('con_layer2:' + str(con_layer2) + '\n')
    file.write('con_layer2_filter:' + str(con_layer2_filter) + '\n')
    file.write('lstm_layer:' + str(lstm_layer) + '\n')
    file.write('drop:' + str(drop) + '\n')
    file.write('r_drop:' + str(r_drop) + '\n')
    file.write('l2_value:' + str(l2_value) + '\n')
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
        locals()['Bacc' + str(i)] = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], locals()['y_pred' + str(i+1)],
                                               look_back, n_columns, n_labels, locals()['scaler' + str(i)])
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

if __name__ == '__main__':
    main()
