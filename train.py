import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import tensorflow as tf
from util.visualizer import Visualizer


if __name__ == '__main__':

	opt = TrainOptions().parse()
	train_dataset, test_dataset = create_dataset(opt)
	train_dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
	test_dataset_size = tf.data.experimental.cardinality(test_dataset).numpy()

	print('The number of training images = %d' % train_dataset_size)
	print('The number of testing images = %d' % test_dataset_size)

	model = create_model(opt)
	visualizer = Visualizer(opt)

	print("ffffffffffffff")

	for epoch in range(opt.epoch_count, opt.n_epochs + 1):
		epoch_start_time = time.time()
		iter_data_time = time.time()
		epoch_iter = 0

		for input_img, selector_img, target_img in train_dataset:
			model.train_step(input_img, selector_img, target_img)

		for input_img, selector_img, target_img in test_dataset:
			model.test_step(input_img, selector_img, target_img)

		save_result = 0			
		visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)


