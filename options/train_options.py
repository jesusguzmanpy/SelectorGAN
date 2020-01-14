from .base_options import BaseOptions


class TrainOptions(BaseOptions):

	def initialize(self, parser):

		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
		parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs with the initial learning rate')

		self.isTrain = True
		return parser