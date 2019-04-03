import sys
import os
import argparse
import numpy as np
import shutil
import cPickle
from voxelization import Vox3DBuilder
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")


def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein',
                        required=False,
                        default='./data/Proteins/',
                        help='location of the protein pdb file path')
    parser.add_argument('--aux',
                        required=False,
                        default='./data/aux_files/',
                        help='location of the auxilary input file')
    parser.add_argument('--r',
                        required=False,
                        help='radius of the grid to be generated',
                        default=15,
                        type=int,
                        dest='r')
    parser.add_argument('--N',
                        required=False,
                        help='number of points long the dimension the generated grid',
                        default=31,
                        type=int,
                        dest='N')
    parser.add_argument('--output',
                        required=False,
                        default='./voxel_output/',
                        help='location of the protein pdb file path')
    args = parser.parse_args()
    return args


def gen_one_voxel(protein_path, aux_path, output, r, N):
    voxel = Vox3DBuilder.voxelization(protein_path, aux_path, r, N)
    print "the type of voxel data--------------------------------"
    print type(voxel)
    print voxel.shape
    print "the type of voxel data--------------------------------"
    if not os.path.exists(output):
        os.makedirs(output)
    outname = os.path.splitext(os.path.basename(aux_path))[0][0:-4]

    oname = output + outname + ".pkl"
    # if os.path.exists(oname):
    #    os.remove(oname)
    cPickle.dump(voxel, open(oname, "wb"))
    # Y = cPickle.load(open(oname, "rb"))
    # print Y[0][0][0][0]

    print('The end of one file ')


def worker(aux_filename_list, args):
    protein_path = args.protein
    aux_path = args.aux

    for aux_file in aux_filename_list:
        base = aux_file[0:-4].split('_')
        pro_file = base[0] + '.pdb'
        print "PID:		 " + str(os.getpid())
        print "protein file: " + pro_file
        print "aux_file:     " + aux_file

        gen_one_voxel(protein_path + pro_file, aux_path + aux_file, args.output, args.r, args.N)


if __name__ == "__main__":
    args = myargs()

    protein_path = args.protein
    aux_path = args.aux
    aux_filename_list = []
    protein_list = []
    for aux_file in os.listdir(aux_path):
        if aux_file:
           aux_filename_list.append(aux_file)
    print "the number of aux files is " + str(len(aux_filename_list))

    # aux_filename_list = [
    #     '1kqfA_6MO_aux.txt',
    #     '1kqfA_MGD_aux.txt', '1kqfA_SF4_aux.txt']
    # print "the number of aux files is " + str(len(aux_filename_list))

    size = len(aux_filename_list)

    P_NUM = 1
    p = Pool(P_NUM)
    for i in range(P_NUM):
        p.apply_async(worker, args=(aux_filename_list[size / P_NUM * i: size / P_NUM * (i + 1)], args,))

    p.close()
    p.join()


