from numpy import zeros
from matplotlib.pyplot import imread
from os import listdir
from os.path import join


class Data:



    def grayscale_jpg_load(self, path, dimensions):

        types = []
        for type in listdir(path):
            if type[0] != '.':
                types.append(type)

        data = []
        labels = []
        num_pixels = dimensions[0]*dimensions[1]

        for type in types:
            print('Data type \''+type+'\' loading...')
            type_index = types.index(type)
            for item in listdir(join(path, type))[:6000]:
                if item[0] != '.':
                    data.append(imread(join(path, type, item)).reshape(num_pixels)/255.0)
                    label = zeros(len(types))
                    label[type_index] = 1
                    labels.append(label)
            print('Data type \''+type+'\' loaded...')

        return types, data, labels

    def rgb_jpg_load(self, path, dimensions):

        types = []
        for type in listdir(path):
            if type[0] != '.':
                types.append(type)

        data = []
        labels = []
        num_pixels = dimensions[0]*dimensions[1]*3

        for type in types:
            print('Data type \''+type+'\' loading...')
            type_index = types.index(type)
            for item in listdir(join(path, type)):
                if 'Vincent_van_Goh' in item:
                    thing = imread(join(path, type, item)).reshape(num_pixels)/255.0
                    data.append(thing)
                    labels.append(thing)
            print('Data type \''+type+'\' loaded...')

        return types, data, labels



    def rgb_jpg_load(self, path):


        return data, labels
