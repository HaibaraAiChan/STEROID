from __future__ import print_function

import sys
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
# from voxelization import Vox3DBuilder
import pickle
import cPickle
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.models import load_model
from data_generator import DataGenerator
from sklearn import metrics


def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein',
                        required=True,
                        help='location of the protein pdb file path')
    parser.add_argument('--aux',
                        required=True,
                        help='location of the auxilary input file')
    parser.add_argument('--r',
                        required=False,
                        help='radius of the grid to be generated', default=15,
                        type=int,
                        dest='r')
    parser.add_argument('--N',
                        required=False,
                        help='number of points long the dimension the generated grid', default=31,
                        type=int,
                        dest='N')
    args = parser.parse_args()
    return args

def print_data_list(data_list, labels):
    # for i in range(len(data_list)):
    #     print (i, data_list[i])
    print("**********************************************************")
    for i, v in enumerate(labels):
        print (i, v)



def gen_folder_list(data_folder, times):
    f_list = []
    folder_list = os.listdir(data_folder)
    folder_list = [f for f in folder_list if 'rotate_voxel_data' in f]
    folder_list.sort()
    f_list = folder_list[0:times]
    return f_list


def load_list(data_folder, labels, a_times, o_times):
    data_list = []
    label_list = []

    folder_list = os.listdir(data_folder)
    folder_pos = [f for f in folder_list if 'pos' in f][0]
    a_f_list = gen_folder_list(data_folder + folder_pos, a_times)

    folder_neg = [f for f in folder_list if 'neg' in f][0]
    o_f_list = gen_folder_list(data_folder + folder_neg, o_times)

    numm = 0

    for folder in a_f_list:
        for filename in os.listdir(data_folder + 'positive/' + folder):

            data_list.append(filename)
            labels[filename] = 1
            label_list.append(int(1))
            numm = numm + 1
            print(numm, end=' ')
            if numm % 20 == 0:
                print()

    print('\nthe steroid list done')

    num = 0
    for folder in o_f_list:
        for filename in os.listdir(data_folder + 'negative/' + folder):

            data_list.append(filename)
            labels[filename] = 0
            label_list.append(int(0))
            num = num + 1
            print(num, end=' ')
            if num % 20 == 0:
                print()

    print('the other list done')

    return data_list, (numm+num), label_list


# def load_predict_data(voxel_folder, batch_size, a_times, o_times):
#     labels = {}
#     predict_list = load_list(voxel_folder, labels, a_times, o_times)
#
#     partition = {"predict": predict_list}
#
#     predict_params = {'dim': (32, 32, 32),
#                       'n_channels': 14,
#                       'batch_size': batch_size,
#                       'n_classes': 2,
#                       'shuffle': True,
#                       'path': voxel_folder}
#
#     # Generators
#     predict_generator = DataGenerator(partition['train'], labels, **predict_params)
#     print("the predicting data is ready")


def predict(voxel_folder,  batch_size, a_times, o_times , model_path):
    # load the data
    labels = {}
    predict_list, total_len, label_list = load_list(voxel_folder, labels, a_times, o_times)

    print_data_list(predict_list, label_list)

    partition = {"predict": predict_list}

    predict_params = {'dim': (32, 32, 32),
                      'n_channels': 14,
                      'batch_size': batch_size,
                      'n_classes': 2,
                      'shuffle': True,
                      'path': voxel_folder}

    # Generators
    pre_generator = DataGenerator(partition['predict'], labels, **predict_params)
    print("the predicting data is ready")
    # print(labels)

    # valid_voxel, valid_label = load_valid_data(adenines, others, path, 1, 1)
    p_y = np_utils.to_categorical(label_list, num_classes=2)
    #
    # # true_classes = pre_generator.classes
    #
    model = load_model(model_path)
    score = model.predict_generator(pre_generator,
                                    steps=np.ceil(total_len/batch_size))
    print(score)
    oname = "./score.pkl"
    cPickle.dump(score, open(oname, "wb"))

    auc = roc_auc_score(y_true=p_y, y_score=score)
    print('auc score from generator', auc)



    # score = mdl.predict(valid_voxel)
    #
    # # Compute ROC curve and ROC area for each class
    # n_classes = 2
    #
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     y_score = np.array(score[:, i])
    #     y_test = np.array(p_y[:, i])
    #     fpr[i], tpr[i], _ = roc_curve(y_test, y_score, pos_label=1)
    #     # fpr_t, tpr_t, _t = metrics.roc_curve(y_test, y_score, pos_label=1)
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(p_y.ravel(), score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # plt.figure()
    # lw = 2
    # # plt.plot(fpr[0], tpr[0], color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    # plt.plot(fpr[1], tpr[1], color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig('positive_84.png', frameon=False, pad_inches=False)
    # plt.close()

if __name__ == "__main__":

    predict_folder = '../data_prepare/test/'

    batch_size = 128
    a_times = 72
    o_times = 1

    # batch_size = 2
    # a_times = 1
    # o_times = 0

    model_path = '../model_backup/model_128/model_v_total_bs_128_84/deepdrug3d.h5'
    # model_path = './model_v_total_bs_128_84/deepdrug3d.h5'
    predict(predict_folder, batch_size, a_times, o_times, model_path)
