from .base_model import BaseModel
from . import networks


class SelectorGANModel(BaseModel):

	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		if self.isTrain:
			self.model_names = ['G', 'D']
		else:
			self.model_names = ['G']

		self.netG = networks.define_G(opt.input_nc, opt.selector_nc, opt.output_nc, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain)

	def train_step(self, input_img, selector_img, target_img):
		self.real_input_train = input_img
		self.real_selector_train = selector_img
		self.real_target_train = target_img

	def test_step(self, input_img, selector_img, target_img):
		self.real_input_test = input_img
		self.real_selector_test = selector_img
		self.real_target_test = target_img


