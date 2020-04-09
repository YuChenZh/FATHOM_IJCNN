from keras.layers import concatenate, dot,multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *


INPUT_DIM = 100
TIME_STEPS = 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = True


def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.randint(input_dim, size=(n, time_steps))
    x = np.eye(input_dim)[x]
    y = x[:, attention_column, :]
    # print(y)
    print(x.shape)
    print(y.shape)
    return x, y


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

# def attention_3d_block(inputs):
#     # hidden_states.shape = (batch_size, time_steps, hidden_size)
#     hidden_size = int(inputs.shape[2])
#     # Inside dense layer
#     #              hidden_states            dot               W            =>           score_first_part
#     # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
#     # W is the trainable weight matrix of attention
#     # Luong's multiplicative style score
#     score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(inputs)
#     #            score_first_part           dot        last_hidden_state     => attention_weights
#     # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
#     h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(inputs)
#     score = dot([score_first_part, h_t], [2, 1], name='attention_score')
#     attention_weights = Activation('softmax', name='attention_weight')(score)
#     # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
#     context_vector = dot([inputs, attention_weights], [1, 1], name='context_vector')
#     pre_activation = concatenate([context_vector, h_t], name='attention_output')
#     attention_vector = Dense(128, use_bias=False, activation='tanh',name='attention_vector')(pre_activation)
#
#     return attention_vector

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    print(inputs.shape)
    print(a_probs.shape)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

# def model_attention_applied_before_lstm():
#     inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
#     attention_mul = attention_3d_block(inputs)
#     lstm_units = 32
#     attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
#     output = Dense(1, activation='sigmoid')(attention_mul)
#     model = Model(input=[inputs], output=output)
#     return model

def model_attention_applied_after_lstm(data):
    inputs = Input(shape=(data.shape[1], data.shape[2],))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(INPUT_DIM, activation='sigmoid', name='output')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

if __name__ == '__main__':

    N = 30000
    # N = 300 -> too few = no training
    inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)


    m = model_attention_applied_after_lstm(inputs_1)

    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0)

    attention_vectors = []
    for i in range(10):
        testing_inputs_1, testing_outputs = get_data_recurrent(10, TIME_STEPS, INPUT_DIM)
        attention_vector = np.mean(get_activations(m,
                                                   testing_inputs_1,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec')[0], axis=2).squeeze()
        print('attention =', attention_vector)
        # assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector[2])

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)

    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()