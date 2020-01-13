from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import tensorflow as tf


if __name__ == '__main__':

	opt = TrainOptions().parse()
	train_dataset, test_dataset = create_dataset(opt)
	train_dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
	test_dataset_size = tf.data.experimental.cardinality(test_dataset).numpy()

	print('The number of training images = %d' % train_dataset_size)
	print('The number of testing images = %d' % test_dataset_size)


	model = create_model(opt)
	model.setup(opt)