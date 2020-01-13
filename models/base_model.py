import os
from abc import ABC, abstractmethod
from keras import backend as K
import numpy as np

class BaseModel(ABC):

    def __init__(self, opt):

        self.opt = opt
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0 

    def setup(self, opt):

        self.print_networks(opt.verbose)


    def print_networks(self, verbose):
        print('---------- Modelo inicializado -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)


                trainable_count = int(
                    np.sum([K.count_params(p) for p in set(net.trainable_weights)]))
                non_trainable_count = int(
                    np.sum([K.count_params(p) for p in set(net.non_trainable_weights)]))

                print('Total params: {:,}'.format(trainable_count + non_trainable_count))
                print('Trainable params: {:,}'.format(trainable_count))
                print('Non-trainable params: {:,}'.format(non_trainable_count))