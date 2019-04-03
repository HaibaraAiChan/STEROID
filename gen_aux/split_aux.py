import os
import numpy as np
from shutil import copyfile


def split_aux_list(aux_files_path, num=200):

    folders = 0
    aux_list = []

    for filename in os.listdir(aux_files_path):
        if filename:
            aux_list.append(filename)
    total_len = len(aux_list)
    f_num = total_len / num
    tail = total_len % num
    if tail == 0:
        folders = f_num
    else:
        folders = f_num + 1

    f_list = []

    for i in range(folders):
        f = 'aux_files_'+str(i+1)
        f_list.append(f)
        o_folder = '../data/' + f
        if not os.path.exists(o_folder):
            os.makedirs(o_folder)
        files_list = aux_list[i*num:((i+1)*num-1)]
        for fi in files_list:

            copyfile(aux_files_path+fi,o_folder+'/'+fi)








    return folders


if __name__ == "__main__":

    aux_files_path = '../data/aux_files/'
    folders = split_aux_list(aux_files_path=aux_files_path, num=250)
    print folders

