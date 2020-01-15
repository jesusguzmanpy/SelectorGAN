import argparse

class BaseOptions():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
    	
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--model', type=str, default='selector_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--selector_nc', type=int, default=3, help='# of selector image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--netG', type=str, default='unet', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', type=int, default=1, help='if specified, print more debugging information')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)    	
        return parser.parse_args()

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        
        self.opt = opt
        return self.opt