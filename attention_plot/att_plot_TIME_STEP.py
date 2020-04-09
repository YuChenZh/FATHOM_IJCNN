"""
Keras CNN & LSTM, with shared attention layer, multi-task & multi-label prediction

CNN is for the data sample selection

"""
import keras
from keras.layers.core import *
from keras.models import *
from keras.layers import Input, Embedding, Dense,Convolution1D,MaxPooling1D,merge,concatenate,dot,multiply
from keras.layers.recurrent import LSTM
import os
import warnings
import glob
import sys
sys.path.insert(0, '/Users/yujingchen/PycharmProjects/WATCH_proj/mobile_sensor/')
import helper_funcs


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings


# attention applied on input_dim dimension
def attention_dim(inputs,i):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    # a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(inputs)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #     a = RepeatVector(input_dim)(a)
    # a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = keras.layers.multiply([inputs, a],name='attention_vecd'+str(i))
    attention_vector = Dense(128, use_bias=False, activation='tanh')(output_attention_mul)

    return attention_vector

# def attention_3d_block(inputs):
#     # hidden_states.shape = (batch_size, time_steps, hidden_size)
#     hidden_size = int(inputs.shape[2])
#     score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(inputs)
#     # h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(inputs)
#     # score = dot([score_first_part, h_t], [2, 1], name='attention_score')
#     attention_weights = Activation('softmax', name='attention_vec')(score_first_part)
#     context_vector = multiply([inputs, attention_weights], name='context_vector')
#     # pre_activation = concatenate([context_vector, h_t], name='attention_output')
#     attention_vector = Dense(128, use_bias=False, activation='tanh',name='attention_vector')(context_vector)
#
#     return attention_vector

# # attention applied on TIME_STEP dimension
def attention_3d_block(shared,inputs,i):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    flattened= Flatten()(shared)
    activation_vect=Dense(TIME_STEPS,activation='tanh')(flattened)
    activation_weights=Activation('softmax')(activation_vect)
    activation_weights= RepeatVector(input_dim)(activation_weights)
    activation_weights=Permute([2,1])(activation_weights)
    activation_weighted=keras.layers.multiply([inputs, activation_weights],name='attention_vect'+str(i))

    # sum_weighted = Lambda(lambda x: K.sum(x, axis=-2), output_shape=(input_dim,))(activation_weighted)
    return activation_weighted

TIME_STEPS = 30
SINGLE_ATTENTION_VECTOR = False


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
        locals()['att_dim'+str(i)] = attention_dim(locals()['input'+str(i)],i)

        locals()['lstm_out'+str(i)] = LSTM(lstm_layer, activation='relu', dropout=drop,
                                           recurrent_dropout=r_drop,kernel_regularizer=regularizers.l2(l2_value),return_sequences=True,name='lstm1'+str(i))(locals()['att_dim'+str(i)])
        concate_list.append(locals()['lstm_out'+str(i)])
        input_list.append(locals()['input'+str(i)])

    concate_layer = keras.layers.concatenate(concate_list,axis=-1)


    output_list = []
    for i in range(0,task_num):
        locals()['concate_layer'+str(i)] = attention_3d_block(locals()['lstm_out'+str(i)],locals()['input'+str(i)],i)
        # print(locals()['concate_layer'+str(i)])
        locals()['flatten_layer'+str(i)] = LSTM(dense_num,activation='relu')(locals()['concate_layer'+str(i)])
        locals()['sub'+str(i)] = Dense(dense_num,activation='relu')(locals()['flatten_layer'+str(i)])
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
    con_layer1_filter = 1
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
                        epochs=50,
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

    y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6 = model.predict(testX_list)
    # y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7,y_pred8,y_pred9,y_pred10,y_pred11,y_pred12,y_pred13,y_pred14,y_pred15,y_pred16,y_pred17,y_pred18,y_pred19,y_pred20,y_pred21,y_pred22,y_pred23,y_pred24,y_pred25,y_pred26,y_pred27 \
    #     , y_pred28,y_pred29,y_pred30,y_pred31,y_pred32,y_pred33,y_pred34,y_pred35,y_pred36,y_pred37,y_pred38,y_pred39,y_pred40 = model.predict(testX_list)

    ####################  attention plot - TIME STEP level ################################

    attention_vectors = []

    ##### Case Study, input_dimension ################
    #### axis = 1 is ploting the attention on input_dim, axis = 2 is ploting the attention on TIME_STEP dimension
    for k in range(len(file_list)):
        for j in range(10):
            attention_vector = np.mean(helper_funcs.get_activationsT(k,model,testX_list[k][0:20,:,:],
                                                       print_shape_only=True,
                                                       layer_name='attention_vect'+str(k))[0], axis=1).squeeze()
            # print('attention =', attention_vector)
            # assert (np.sum(attention_vector) - 1.0) < 1e-5
            attention_vectors.append(attention_vector)
            print('.....')
            print(len(attention_vector))

        attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
        print('attention final=', attention_vector_final)
        # print('attention final length=', len(attention_vector_final))

        import seaborn as sns
        import matplotlib.pylab as plt

        # attention_vector_final = np.delete(attention_vector_final, np.s_[225:], axis=1)
        ax = sns.heatmap(attention_vector_final, cmap="BuPu")
        plt.savefig('time_dim_heatmap2'+str(k)+'.png', dpi=150)
        plt.show()

        # # plot part.
        # import matplotlib.pyplot as plt
        # import pandas as pd
        #
        #
        # df = pd.DataFrame(attention_vector_final[1], columns=['attention (%)'])
        # df.to_csv('../results/attention_plot/TimeAtt1/30_TIME_STEP_attention'+str(k)+'.csv')
        # df.plot(kind='bar',title='Attention Mechanism as ''a function of input'' dimensions.')
        # # plt.figure(figsize=(100, 100))
        # plt.xticks(rotation=90)
        # plt.savefig('../results/attention_plot/TimeAtt1/30_TIME_STEP_attention'+str(k)+'.png',dpi=150)
        # plt.show()

    ################################# attention plot ends #################################################



        helper_funcs.evaluation(locals()['test_X' + str(k)], locals()['test_y' + str(k)], locals()['y_pred' + str(k + 1)],
                                look_back, n_columns, n_labels, locals()['scaler' + str(k)])

if __name__ == '__main__':
    main()
