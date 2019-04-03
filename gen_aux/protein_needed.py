
import os
import shutil
import numpy as np
from biopandas.pdb import PandasPdb


def gen_protein_list(aux_files, protein_path):

    protein_list = []
    for filename in os.listdir(protein_path):
        if filename:
            protein_list.append(filename[0:-4])


    aux_list = []
    for filename in os.listdir(aux_files):
        if filename:
            file = filename.split('.')[0]
            pro = file.split('_')
            aux_list.append(pro[0])
    no_list = list(set(protein_list)-set(aux_list))
    return no_list

if __name__ == "__main__":
    # pocket_path = '../data/original_data/pockets_st/'
    protein_path = '../data/original_data/proteins/'
    aux_files_path = '../data/original_data/aux_files/'
    pro_list = gen_protein_list(aux_files=aux_files_path, protein_path=protein_path)
    for i in range(len(pro_list)):
        print pro_list[i]
    print len(pro_list)
