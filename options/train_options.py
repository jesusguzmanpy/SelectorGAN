from .base_options import BaseOptions


class TrainOptions(BaseOptions):

	def initialize(self, parser):

		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
		parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs with the initial learning rate')
		parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')

		parser.add_argument('--display_server', type=str, default="localhost", help='visdom server of the web display')
		parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
		parser.add_argument('--display_port', type=int, default=8080, help='visdom port of the web display')
 

		self.isTrain = True
		return parser