
from __future__ import print_function

import sys
import numpy as np
import argparse
import os
# import matplotlib.pyplot as plt
# from voxelization import Vox3DBuilder
import cPickle
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import roc_curve, auc
from keras.models import load_model
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


def gen_folder_list(data_folder, times):
    f_list = []
    folder_list = os.listdir(data_folder)
    folder_list = [f for f in folder_list if 'rotate_voxel_data' in f]
    folder_list.sort()
    f_list = folder_list[0:times]
    return f_list


def load_predict_data(voxel_folder, a_times, o_times):

    L = a_times*7+o_times*1640

    voxel = np.zeros(shape=(L, 14, 32, 32, 32),
                     dtype=np.float64)
    label = np.zeros(shape=(L,), dtype=int)

    folder_list = os.listdir(voxel_folder)
    folder_pos = [f for f in folder_list if 'pos' in f][0]
    a_f_list = gen_folder_list(voxel_folder + folder_pos, a_times)

    folder_neg = [f for f in folder_list if 'neg' in f][0]
    o_f_list = gen_folder_list(voxel_folder + folder_neg, o_times)

    numm = 0
    cnt =0
    for folder in a_f_list:
        if a_times == 0:
            break
        for filename in os.listdir(voxel_folder + 'positive/' + folder):

            full_path = voxel_folder + 'positive/' + folder +'/'+ filename
            temp = np.load(full_path)
            voxel[cnt, :] = temp
            label[cnt] = 1
            cnt = cnt + 1
            numm = numm + 1
            print(numm, end=' ')
            if numm % 20 == 0:
                print()



    print('\nthe steroid list done')

    num = 0
    for folder in o_f_list:
        if o_times == 0:
            break
        for filename in os.listdir(voxel_folder + 'negative/' + folder):

            full_path = voxel_folder + 'negative/' + folder +'/'+ filename
            temp = np.load(full_path)
            voxel[cnt, :] = temp
            label[cnt] = 0
            cnt = cnt + 1
            num = num + 1
            print(num, end=' ')
            if num % 20 == 0:
                print()

    print('the other list done')

    print("valid data total " + str(numm + num) + ' ligands')
    return voxel, label


def predict(path, a_times, o_times, model_path, oname):

    valid_voxel, valid_label = load_predict_data(path, a_times, o_times)
    p_y = np_utils.to_categorical(valid_label, num_classes=2)

    mdl = load_model(model_path)

    score = mdl.predict(valid_voxel)
#    oname = "./score.pkl"
    cPickle.dump(score, open(oname, "wb"))

#    auc = roc_auc_score(y_true=p_y, y_score=score)
#    print('auc score from generator', auc)
    # # Compute ROC curve and ROC area for each class
    # n_classes = 2
    #
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     y_score = np.array(score[:, i])
    #     y_test = np.array(v_y[:, i])
    #     fpr[i], tpr[i], _ = roc_curve(y_test, y_score, pos_label=1)
    #     # fpr_t, tpr_t, _t = metrics.roc_curve(y_test, y_score, pos_label=1)
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(v_y.ravel(), score.ravel())
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

    path = '../data_prepare/test/'
    # path = './data_prepare/valid/rotate_voxel_data_z_180/'
 #   model_path = './model_v_total_bs_128_84/deepdrug3d.h5'
    model_path = './model_v_total_bs_64_82/deepdrug3d.h5'
    a_times = 72
    o_times = 1
    oname = "./64_direct_total_score.pkl"

    
    predict(path, a_times, o_times, model_path,oname)
