import tensorflow as tf
import os.path
from os import path



def verbose_dataset(opt, txt):
    if(opt.verbose == 1):
        print(txt)


def verify_dataset(opt):

    aux_status = False
    if(path.exists('dataset/'+opt.dataroot)):
        aux_status = True
    else:
        verbose_dataset(opt, "- Folder " + opt.dataroot + " not found")

    if(path.exists('dataset/'+opt.dataroot+'/train')):
        aux_status = True
        if [f for f in os.listdir('dataset/'+opt.dataroot+'/train') if not f.startswith('.')] == []:
            verbose_dataset(opt, "- Folder " + opt.dataroot + "/train is empty")
            aux_status = False
    else:
        verbose_dataset(opt, "- Folder " + opt.dataroot + "/train not found")

    if(path.exists('dataset/'+opt.dataroot+'/test')):
        aux_status = True
        if [f for f in os.listdir('dataset/'+opt.dataroot+'/test') if not f.startswith('.')] == []:
            verbose_dataset(opt, "- Folder " + opt.dataroot + "/test is empty")
            aux_status = False
    else:
        verbose_dataset(opt, "- Folder " + opt.dataroot + "/test not found")

    return aux_status

@tf.function()
def normalize(input_image, ref_image, output_image):
  input_image = (input_image / 127.5) - 1
  ref_image = (ref_image / 127.5) - 1
  output_image = (output_image / 127.5) - 1
  return input_image, ref_image, output_image

@tf.function()
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
  w = tf.shape(image)[1]//3

  input_image = image[:,:w,:]
  reference_image = image[:,w:w*2,:]
  output_image = image[:,w*2:w*3,:]

  input_image = tf.cast(input_image, tf.float32)
  reference_image = tf.cast(reference_image, tf.float32)
  output_image = tf.cast(output_image, tf.float32)

  return input_image, reference_image, output_image

@tf.function()
def load_image_train(image_file):
  input_image, reference_image, output_image = load(image_file)
  input_image, reference_image, output_image = normalize(input_image, reference_image, output_image)

  return input_image, reference_image, output_image

@tf.function()
def load_image_test(image_file):
  input_image, reference_image, output_image = load(image_file)
  input_image, reference_image, output_image = normalize(input_image, reference_image, output_image)

  return input_image, reference_image, output_image


def create_dataset(opt):
    if(verify_dataset(opt)):
        train_dataset = tf.data.Dataset.list_files('dataset/'+opt.dataroot+'/train/*')
        test_dataset = tf.data.Dataset.list_files('dataset/'+opt.dataroot+'/test/*')

        train_dataset = train_dataset.map(load_image_train,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.map(load_image_test,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_dataset = train_dataset.shuffle(tf.data.experimental.cardinality(train_dataset).numpy())
        test_dataset = test_dataset.shuffle(tf.data.experimental.cardinality(test_dataset).numpy())

        train_dataset = train_dataset.batch(opt.batch_size)
        test_dataset = test_dataset.batch(opt.batch_size)
            
        return train_dataset, test_dataset
    else:
        exit()
        
