from .base_model import BaseModel
from . import networks

class SelectorGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if self.isTrain:
            self.model_names = ['G']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain)