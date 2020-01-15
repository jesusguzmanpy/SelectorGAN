import os
from abc import ABC, abstractmethod
from collections import OrderedDict

class BaseModel(ABC):

	def __init__(self, opt):
		self.opt = opt
		self.isTrain = opt.isTrain
		self.visual_names = []

	@abstractmethod
	def train_step(self, input_img, selector_img, target_img):
		pass

	@abstractmethod
	def test_step(self, input_img, selector_img, target_img):
		pass

	def setup(self, opt):

		if not self.isTrain or opt.continue_train:
			load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
			self.load_networks(load_suffix)
		self.print_networks(opt.verbose)

	def get_current_visuals(self):
		visual_ret = OrderedDict()
		for name in self.visual_names:
			if isinstance(name, str):
				visual_ret[name] = getattr(self, name)
		return visual_ret