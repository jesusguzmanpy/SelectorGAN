import tensorflow as tf

def ResnetGenerator():

  return tf.keras.Model(inputs=[inp, tar], outputs=last)



def UnetGenerator(input_nc, selector_nc, output_nc):

    inp = tf.keras.layers.Input(shape=[256,256,input_nc])
    select = tf.keras.layers.Input(shape=[256,256,selector_nc])

    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv2DTranspose(output_nc, 4,
                                         strides=2,
                                         padding='same',
                                       kernel_initializer=initializer,
                                         activation='tanh')
    x = last(inp)

    return tf.keras.Model(inputs=[inp, select], outputs=x)


def define_G(netG, input_nc, selector_nc ,output_nc, ngf,  norm='batch', use_dropout=False, init_type='normal', init_gain=0.02):
    net = None

    if netG == 'unet':
        net = UnetGenerator(input_nc, selector_nc, output_nc)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, selector_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return net