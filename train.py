from __future__ import absolute_import, division, print_function, unicode_literals
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    opt = TrainOptions().parse()   # opciones de entrenamiento
    dataset = create_dataset(opt)  # crear dataset con el formato de la data
    #dataset_size = len(dataset)    # obtenemos el número de imágenes del dataset
    #print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # creando el modelo
    model.setup(opt)          
    visualizer = Visualizer(opt)
    total_iters = 0 

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()


