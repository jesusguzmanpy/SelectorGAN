from .base_options import BaseOptions


class TrainOptions(BaseOptions):

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

		self.isTrain = True
		return parser