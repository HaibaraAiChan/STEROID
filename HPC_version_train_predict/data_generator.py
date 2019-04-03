import numpy as np
import keras
import os


class DataGenerator(keras.utils.Sequence):
    # '''Generates data for Keras'''

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True, path=''):
        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path = path
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

#        if self.shuffle == True:
#            np.random.shuffle(self.indexes)
#
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        ll = len(self.dim)
        tmp = []
        for i in range(ll):
            tmp.append(int(self.dim[i]))
        X = np.empty((self.batch_size, self.n_channels, tmp[0], tmp[1], tmp[2]))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            y[i] = self.labels[ID]
            # print(ID)
            # print(y[i])
            tttt = ''
            if y[i] == 1:
                tttt = 'positive/'
            elif y[i] == 0:
                tttt = 'negative/'

            tmp_ID = ID[0:-4].split('_')
            path = '_' + tmp_ID[2] + "_"

            tp = "{:03d}".format(int(tmp_ID[-1]))
            path = path + tp

            folder = [f for f in os.listdir(self.path + tttt) if path in f]
            path = self.path + tttt + folder[0] + '/' + ID

            X[i,] = np.load(path)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
