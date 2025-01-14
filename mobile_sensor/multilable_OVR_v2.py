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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def features_labels_process(data):
    # data = data[0:3000]
    all_lables_list = [col for col in data if col.startswith('label')]
    all_features_list = [col for col in data if col not in all_lables_list]
    # all_features_list.remove('timestamp')
    all_lables_list.remove('label_source')

    features = data[all_features_list]
    labels = data[all_lables_list]

    # For missing sensor-features, replace "nan" with the upper cell(ffill) or below cell(bfill) value
    features = features.fillna(method='ffill')
    # drop the feature column with all "nan"
    # features = features.dropna(axis=1, how='all')
    features = features.fillna(method='bfill')
    features = features.fillna(0) # after forward-filling and back-filling, fill the rest nan cells with 0

    # normalize feature vaules, by subtracting mean, then divide std
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(features)
    normalized_features = pd.DataFrame(np_scaled)

    # fill empty labels with 0
    labels = labels.fillna(0)
    return (normalized_features,labels)

def evaluation(y_pred,y_true):

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
    y_true = y_true.as_matrix()
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            if y_pred[i][j] == 1 and y_true[i][j] == 1:
                tp += 1
            elif y_pred[i][j] == 1 and y_true[i][j] == 0:
                fp += 1
            elif y_pred[i][j] == 0 and y_true[i][j] == 0:
                tn += 1
            elif y_pred[i][j] == 0 and y_true[i][j] == 1:
                fn += 1
            total += 1

    # true positive rate
    sensitivity = float(tp) / (tp + fn)
    # true negative rate
    specificity = float(tn) / (tn + fp)
    precision = float(tp) / (tp + fp)

    if (precision + sensitivity) ==0:
        F1 = 0
    else:
        F1 = (2 * precision * sensitivity) / (precision + sensitivity)
    # naive accuracy
    accuracy = float(tn + tp) / total

    # # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    #
    print("-" * 10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision: %.2f' % precision);
    print('F1: %.2f' % F1);
    print("-" * 10);
    return accuracy, sensitivity, specificity, balanced_accuracy,tp,fn,fp,tn,precision,F1


if __name__ == '__main__':
    file_list = glob.glob('data_csv/train/*.csv')

    file = open('results/Single_LR_40users1.txt', 'w')
    sum_bacc = 0
    sum_TPR = 0
    Num_tp = 0
    Num_fn = 0
    Num_fp = 0
    Num_tn = 0
    sum_precision = 0
    sum_F1 = 0
    train_time = 0

    import time

    start_time = time.time()

    for i in range(len(file_list)):

        ## Part 2 ************** fixed train & test prediction *****************
        locals()['dataset' + str(i)] = pd.read_csv(file_list[i])
        # data = pd.read_csv('data_csv/train/5users/0E6184E1-90C0-48EE-B25A-F1ECB7B9714E.features_labels.csv')
        locals()['features' + str(i)], locals()['labels' + str(i)] = features_labels_process(locals()['dataset' + str(i)])
        # X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=.6,random_state=0)
        locals()['X_train' + str(i)] = locals()['features' + str(i)][0:1800]
        locals()['Y_train' + str(i)] = locals()['labels' + str(i)][0:1800]
        locals()['X_test' + str(i)] = locals()['features' + str(i)][2401:3000]
        locals()['Y_test' + str(i)] = locals()['labels' + str(i)][2401:3000]


        ## ------- Fit an independent logistic regression model or MLP for each class using the OneVsRestClassifier wrapper.
        
        # ovr = OneVsRestClassifier(svm.SVC())
        ovr = OneVsRestClassifier(LogisticRegression())
        # ovr = OneVsRestClassifier(
        #     MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1,
        #                   max_iter=10, learning_rate_init=.1))
        # ovr = OneVsRestClassifier(MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50,50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1))

        # ------ evaluation of test data predicion  -----------------------#

        ovr.fit(locals()['X_train' + str(i)], locals()['Y_train' + str(i)])
        Y_pred = ovr.predict(locals()['X_test' + str(i)])

        

        results = evaluation(Y_pred, locals()['Y_test' + str(i)])

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

    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))
    
    file.write('avg_bacc: ' + str(sum_bacc / len(file_list)) + '\n')
    file.write('avg_TPR: ' + str(sum_TPR / len(file_list)) + '\n')
    file.write('avg_precision: ' + str(sum_precision / len(file_list)) + '\n')
    file.write('avg_F1: ' + str(sum_F1 / len(file_list)) + '\n')
    file.write('sum_Num_tp: ' + str(Num_tp) + '\n')
    file.write('sum_Num_fn: ' + str(Num_fn) + '\n')
    file.write('sum_Num_fp: ' + str(Num_fp) + '\n')
    file.write('sum_Num_tn: ' + str(Num_tn) + '\n')
    file.write('train_time: ' + str(train_time) + '\n')




