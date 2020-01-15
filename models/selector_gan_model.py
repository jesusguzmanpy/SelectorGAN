from .base_model import BaseModel
from . import networks


class SelectorGANModel(BaseModel):

	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		
		self.visual_names = ['real_source_test', 'real_selector_test','real_generate_test', 'fake_generate_test']

		if self.isTrain:
			self.model_names = ['G', 'D']
		else:
			self.model_names = ['G']

		self.netG = networks.define_G( opt.netG, opt.input_nc, opt.selector_nc, opt.output_nc, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain)


	def train_step(self, input_img, selector_img, target_img):
		self.real_input_train = input_img
		self.real_selector_train = selector_img
		self.real_generate_train = target_img

		self.fake_generate_train = self.netG([self.real_input_train, self.real_selector_train], training=True)


	def test_step(self, input_img, selector_img, target_img):
		self.real_source_test = input_img
		self.real_selector_test = selector_img
		self.real_generate_test = target_img


		self.fake_generate_test = self.netG([self.real_input_train, self.real_selector_train], training=True)
