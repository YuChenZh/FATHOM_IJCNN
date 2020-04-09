"""
Baseline model 2: Attention model. only keep the second attention layer which is applied on joined tasks' TIME_step dimension.

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
        locals()['lstm_layer1'+str(i)] = LSTM(lstm_layer, activation='relu', dropout=drop,
                                           recurrent_dropout=r_drop,kernel_regularizer=regularizers.l2(l2_value),
                                           return_sequences=True)(locals()['input'+str(i)])
        concate_list.append(locals()['lstm_layer1'+str(i)])
        input_list.append(locals()['input'+str(i)])


    concate_layer = keras.layers.concatenate(concate_list)


    output_list = []
    for i in range(0,task_num):
        locals()['concate_layer'+str(i)] = attention_3d_block(concate_layer,locals()['input'+str(i)])
        locals()['LSTM_layer2'+str(i)] = LSTM(dense_num,activation='relu')(locals()['concate_layer'+str(i)])
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
    task_num = 40
    con_layer1 = 128 # for 6 users
    # con_layer1 = 256
    con_layer1_filter = 5
    con_layer2 = 64
    con_layer2_filter = 4
    lstm_layer = 64
    drop = 0.25
    r_drop = 0.25
    l2_value = 0.001
    shared_layer = 576
    dense_num = 64

    look_back = 20  # number of previous timestamp used for training
    n_columns = 276  # total columns
    n_labels = 51  # number of labels
    split_ratio = 0.8  # train & test data split ratio

    trainX_list = []
    trainy_list = []
    testX_list = []
    testy_list = []
    file_list = glob.glob('data_csv/train/*.csv')

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
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=2,
                                                          mode='min')]
                        )
    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))





    # make prediction
    pred_time = time.time()

    y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7,y_pred8,y_pred9,y_pred10,y_pred11,y_pred12,y_pred13,y_pred14,y_pred15,y_pred16,y_pred17,y_pred18,y_pred19,y_pred20,y_pred21,y_pred22,y_pred23,y_pred24,y_pred25,y_pred26,y_pred27 \
        , y_pred28,y_pred29,y_pred30,y_pred31,y_pred32,y_pred33,y_pred34,y_pred35,y_pred36,y_pred37,y_pred38,y_pred39,y_pred40 = model.predict(testX_list)
    # print (len(y_pred1))
    # y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6 = model.predict(testX_list)
    # y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6,y_pred7,y_pred8,y_pred9,y_pred10,y_pred11,y_pred12 = model.predict(testX_list)
    pred_end_time = time.time()


   #===========================================================================================#
    # write parameters & results to file
    # file = open('results/Attention_results(12)_F1.txt', 'w')
    file = open('time_cost/FATHOMb2_40users.txt', 'w')

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
    file.write('prediction time:' + str(pred_end_time - pred_time))

if __name__ == '__main__':
    main()
