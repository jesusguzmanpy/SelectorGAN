import argparse
import os

class BaseOptions():


    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # parametros básicos
        parser.add_argument('--dataroot', required=True, help='folder de imagenes(debe tener la forma)')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='los modelos se guardaran aqui')
        parser.add_argument('--name', type=str, required=True, default='experiment_name', help='nombre de los experimentos')
        # parametros del modelo
        parser.add_argument('--model', type=str, default='selector_gan', help='elige el modelo a usar')
        parser.add_argument('--input_nc', type=int, default=3, help='# imaganes de entrada de: 3 para RGB y 1 para escala de grises')
        parser.add_argument('--reference_nc', type=int, default=3, help='# imaganes de referencia de: 3 para RGB y 1 para escala de grises')
        parser.add_argument('--output_nc', type=int, default=3, help='# imaganes de salida de: 3 para RGB y 1 para escala de grises')
        parser.add_argument('--ngf', type=int, default=64, help='# de filtros del generador en la ùltima capa conv')
        parser.add_argument('--ndf', type=int, default=64, help='# of de filtros del discrim en la primera capa conv')
        parser.add_argument('--netD', type=str, default='patchgan', help='especifique la arquitectura del discriminación')
        parser.add_argument('--netG', type=str, default='unet', help='especifique la arquitectura del generador')
        parser.add_argument('--norm', type=str, default='instance', help='especifique el tipo de normalización')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout para el generador')
        parser.add_argument('--init_gain', type=float, default=0.02, help='factor de escala para normal, xavier y ortogonal')
        parser.add_argument('--init_type', type=str, default='normal', help='inicialización de la arquitectura')
        # parametros del dataset
        parser.add_argument('--dataset_mode', type=str, default='aligned_reference', help='cuando se trabaja con el selector gan se debe tener aligned_reference')
        parser.add_argument('--verbose', action='store_true', help='mostrar informción')
        parser.add_argument('--display_winsize', type=int, default=256, help='mostrar el tamaño de la ventana para visdom y HTML')

        self.initialized = True
        return parser

    def gather_options(self):

        if not self.initialized:  # verificar si ha inicilizado
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        opt, _ = parser.parse_known_args()


        self.parser = parser
        return parser.parse_args()


    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train o test

        self.opt = opt
        return self.opt