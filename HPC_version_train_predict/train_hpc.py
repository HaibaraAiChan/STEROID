from __future__ import print_function
import numpy as np
from deepdrug3d import DeepDrug3DBuilder
import os
from keras import callbacks
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.models import Sequential
from data_generator import DataGenerator
# from valid_generator import V_DataGenerator
from keras.models import load_model
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf


config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 20})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def gen_folder_list(data_folder, times):
    f_list=[]
    folder_list = os.listdir(data_folder)
    folder_list = [f for f in folder_list if 'rotate_voxel_data' in f]
    folder_list.sort()
    f_list = folder_list[0:times]
    return f_list

def load_list(data_folder, labels, a_times, o_times):
    data_list = []

    folder_list = os.listdir(data_folder)
    folder_pos = [f for f in folder_list if 'pos' in f][0]
    a_f_list = gen_folder_list(data_folder+folder_pos, a_times)

    folder_neg = [f for f in folder_list if 'neg' in f][0]
    o_f_list = gen_folder_list(data_folder+folder_neg,  o_times)

    numm = 0

    for folder in a_f_list:
        for filename in os.listdir(data_folder + 'positive/'+folder):

            data_list.append(filename)
            labels[filename] = 1

            numm = numm + 1
            print(numm, end=' ')
            if numm % 20 == 0:
                print()

    print('\nthe steroid list done')

    num = 0
    for folder in o_f_list:
        for filename in os.listdir(data_folder + 'negative/'+ folder):

            data_list.append(filename)
            labels[filename] = 0

            num = num + 1
            print(num, end=' ')
            if num % 20 == 0:
                print()

    print('the other list done')
    return data_list


def train(train_folder, valid_folder, output, batch_size, epoch, lr, a_times, o_times):
    labels = {}

    train_list = load_list(train_folder, labels, a_times, o_times)
    valid_list = load_list(valid_folder, labels, 18, 1)

    partition = {"train": train_list, "validation": valid_list}

    # Parameters
    train_params = {'dim': (32, 32, 32),
                    'n_channels': 14,
                    'batch_size': batch_size,
                    'n_classes': 2,
                    'shuffle': True,
                    'path': train_folder}
    valid_params = {'dim': (32, 32, 32),
                    'n_channels': 14,
                    'batch_size': batch_size,
                    'n_classes': 2,
                    'shuffle': True,
                    'path': valid_folder}

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **train_params)
    print("the training data is ready")
    validation_generator = DataGenerator(partition['validation'], labels, **valid_params)
    print("the validating data is ready")

    model = DeepDrug3DBuilder.build()
    model = multi_gpu_model(model, gpus=2)
    print(model.summary())
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # We add metrics to get more results you want to see
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss',
                                  patience=10,
                                  verbose=0,
                                  mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5',
                               save_best_only=True,
                               monitor='val_loss',
                               mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2,
                                       patience=20,
                                       verbose=1,
                                       min_delta=1e-4,
                                       mode='min')

    print("ready to fit generator")
    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=epoch,
                        verbose=2,
                        use_multiprocessing=True,
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                        workers=8)

    if output == None:
        model.save('deepdrug3d.h5')
    else:
        if not os.path.exists(output):
            os.mkdir(output)
        if os.path.exists('deepdrug3d.h5'):
            os.remove('deepdrug3d.h5')
        model.save(output + 'deepdrug3d.h5')
        model.save_weights(output + 'weights.h5')
        mm = load_model(output + 'deepdrug3d.h5')
        print(mm.summary())


if __name__ == "__main__":
    train_folder = '../data_prepare/train/'

    valid_folder = '../data_prepare/valid/'

    output = './save_model/'

    batch_size = 64
    epoch = 30
    lr = 0.00001

    a_times = 72
    o_times = 1

    train(train_folder, valid_folder, output, batch_size, epoch, lr, a_times, o_times)

