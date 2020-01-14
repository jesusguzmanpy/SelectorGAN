import argparse

class BaseOptions():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
    	
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--model', type=str, default='selector_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')

        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
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