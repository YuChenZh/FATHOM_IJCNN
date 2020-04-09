
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.layers.core import *
from pandas import read_csv
from keras import backend as K
from sklearn.metrics import mean_absolute_error


def load_dataset(datasource):

    # load the dataset
    dataframe = read_csv(datasource)
    # dataframe.set_index('timestamp',inplace = True)
    dataframe  = dataframe.drop(['id','since_begin','since_last','time_elapsed'],axis=1)
    dataframe = dataframe.sort_values(by='timestamp')
    # dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(0)
    dataframe = dataframe.iloc[0:30000]  # first 30000 rows of dataframe


    dataset = dataframe.values
    # integer encode direction
    encoder = LabelEncoder()
    dataset[:, 3] = encoder.fit_transform(dataset[:, 3])
    dataset[:, 7] = encoder.fit_transform(dataset[:, 7])
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    return dataset, scaled, scaler


#take one input dataset and split it into train and test
def split_dataset(dataset, scaled, look_back, n_columns,n_labels,ratio):

    # frame as supervised learning
    reframed = series_to_supervised(scaled, look_back, 1)

    # split into train and test sets
    values = reframed.values
    n_train_data = int(len(dataset) * ratio)
    train = values[:n_train_data, :]
    test = values[n_train_data:, :]
    # split into input and outputs
    n_obs = look_back * n_columns
    train_X, train_y = train[:, :n_obs], train[:, -n_labels:]  # labels are the last 6 columns
    test_X, test_y = test[:, :n_obs], test[:, -n_labels:]

    print(train_X.shape, len(train_X), train_y.shape)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], look_back, n_columns))
    test_X = test_X.reshape((test_X.shape[0], look_back, n_columns))

    return train_X, train_y, test_X, test_y

def tensor_similarity(t1,t2,data1,data2):
    p1 = tf.placeholder(dtype=t1.dtype, shape=t1.shape)
    p2 = tf.placeholder(dtype=t2.dtype, shape=t2.shape)

    # s = tf.losses.cosine_distance(tf.nn.l2_normalize(p1, 0), tf.nn.l2_normalize(p2, 0), axis=1,reduction=Reduction.Mean)
    # ss = keras.layers.dot([p1, p2], axes=2, normalize=True)

    # Method1: using keras dot
    ss = keras.layers.dot([p1, p2], axes=2, normalize=True)

    # Method2: using TF/Keras backend
    square_sum1 = K.sum(K.square(p1), axis=2, keepdims=True)
    norm1 = K.sqrt(K.maximum(square_sum1, K.epsilon()))
    square_sum2 = K.sum(K.square(p2), axis=2, keepdims=True)
    norm2 = K.sqrt(K.maximum(square_sum2, K.epsilon()))

    num = K.batch_dot(p1, K.permute_dimensions(p2, (0, 2, 1)))
    den = (norm1 * K.permute_dimensions(norm2, (0, 2, 1)))
    cos_similarity = num / den

    with tf.Session().as_default() as sess:
        similarity1 = sess.run(ss, feed_dict={p1: data1, p2: data2})
        similarity2 = sess.run(cos_similarity, feed_dict={p1: data1, p2: data2})

    similarity1 = np.average(similarity1)
    similarity2 = np.average(similarity2)

    return similarity1,similarity2


def evaluation(test_X, test_y, y_pred, timestamps, n_columns, n_labels, scaler):

    test_X = test_X.reshape((test_X.shape[0], timestamps * n_columns))
    # invert scaling for forecast
    y_predict = concatenate((test_X[:, -n_columns:-n_labels], y_pred), axis=1)
    y_predict = scaler.inverse_transform(y_predict)
    y_predict = y_predict[:, -n_labels:]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), n_labels))
    y_true = concatenate((test_X[:, -n_columns:-n_labels], test_y), axis=1)
    y_true = scaler.inverse_transform(y_true)
    y_true = y_true[:, -n_labels:]

    Smape = smape(y_predict, y_true)
    mae = mean_absolute_error(y_true, y_pred)

    return Smape,mae

def evaluation_single(test_X, test_y, y_pred, timestamps, n_columns, n_labels, scaler,i):

    test_X = test_X.reshape((test_X.shape[0], timestamps * n_columns))
    # invert scaling for forecast
    y_predict = concatenate((test_X[:, -n_columns:-n_labels], y_pred), axis=1)
    # print('before')
    # print(y_predict.shape)
    y_predict = scaler.inverse_transform(y_predict)
    y_predict = y_predict[:, -n_labels:]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), n_labels))
    y_true = concatenate((test_X[:, -n_columns:-n_labels], test_y), axis=1)
    y_true = scaler.inverse_transform(y_true)
    y_true = y_true[:, -n_labels:]
    Smape = smape(y_predict[i],y_true[i])
    mae = mean_absolute_error(y_predict[i],y_true[i])


    return Smape,mae

def smape(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator =  np.abs(np.array(actual)) +  np.abs(np.array(predicted))

    return 2 * np.mean(
        np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def mycrossentropy(y_true, y_pred, e=0.3):
    nb_classes = 51
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return (1-e)*loss1 + e*loss2


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
