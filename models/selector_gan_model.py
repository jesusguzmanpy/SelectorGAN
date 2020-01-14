from .base_model import BaseModel


class SelectorGANModel(BaseModel):

	def __init__(self, opt):
		BaseModel.__init__(self, opt)
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

	def set_input(self, input):

		self.real_S = input['S']
		self.real_R = input['R']
		self.real_T = input['T']