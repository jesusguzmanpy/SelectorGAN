from abc import ABC, abstractmethod

class BaseModel(ABC):

	def __init__(self, opt):
		self.opt = opt
		self.isTrain = opt.isTrain

	@abstractmethod
	def set_input(self, input):
		pass

	def setup(self, opt):

		if not self.isTrain or opt.continue_train:
			load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
			self.load_networks(load_suffix)
		self.print_networks(opt.verbose)