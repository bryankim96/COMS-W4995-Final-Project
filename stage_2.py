import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

tfgan = tf.contrib.gan

# output shape of generator images
IMAGE_SHAPE = 64

# dimension of the compressed embedding/ conditioning vector
# input to both generator and discriminator
EMBEDDING_DIM = 128

# size/num_parameters factor for generator (number of filters)
GENERATOR_DIM = 128
# size/num_parameters factor for discriminator (number of filters)
DISCRIMINATOR_DIM = 64

# NOTE: ordering of Batch norm and ReLU differs from original paper implementation
# see https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
# here we apply batch normalization after the activation, rather than before as the
# authors did originally in https://arxiv.org/pdf/1612.03242.pdf

# NOTE: Our usage of conv2d layers omits the use of a bias term because in
# the authors' github repo their custom conv2d layer does not use one.
# The motivation behind this is unclear

# residual block used in the stage II generator function
def residual_block(x, is_training=True):
    x_0 = x
    x_1 = tf.layers.conv2d(x, 
            filters=GENERATOR_DIM * 4, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.relu)
    x_1 = tf.layers.batch_normalization(x_1, training=is_training)
    x_1 = tf.layers.conv2d(x_1, 
            filters=GENERATOR_DIM * 4, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    x_1 = tf.layers.batch_normalization(x_1, training=is_training)
 
    # residual connection (see https://arxiv.org/pdf/1512.03385.pdf for motivation)
    x_out = tf.add(x_0, x_1)
    x_out = tf.nn.relu(x_out)

    return x_out

def generator_stage2(x_var, conditioning_vector, is_training=True):

    # encode/reduce the input image
    # 3 stacked convolutional layers with ReLU activations and batch normalization
    # reduces from shape (BATCH_SIZE, 64, 64, 3) to (BATCH_SIZE, 16, 16, GENERATOR_DIM * 4)
    x = tf.layers.conv2d(x_var, 
            filters=GENERATOR_DIM, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 
            filters=GENERATOR_DIM * 2, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.relu)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.layers.conv2d(x, 
            filters=GENERATOR_DIM * 4, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    x = tf.layers.batch_normalization(x, training=is_training)
 
    # get conditioning vector of shape (BATCH_SIZE, 128)
    # and tile to shape (BATCH_SIZE, 16, 16, 128)
    c_var = tf.expand_dims(tf.expand_dims(conditioning_vector, 1), 1)
    c_var = tf.tile(c_var, [1, IMAGE_SHAPE/4, IMAGE_SHAPE/4, 1])

    # concatenate reduced image of shape (BATCH_SIZE, 16, 16, GENERATOR_DIM * 4)
    # and expanded conditioning vector (BATCH_SIZE, 16, 16, 128) along 4th dimension to get
    # (BATCH_SIZE, 16, 16, GENERATOR_DIM * 4 + 128)
    x = tf.concat(3, [x, c_var])

    # apply 1 conv layer on the combined tensor (reduced image + expanded conditioning vector)
    # outputs (BATCH_SIZE, 16, 16, GENERATOR_DIM * 4)
    x = tf.layers.conv2d(x,
            filters=GENERATOR_DIM * 4,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    x = tf.layers.batch_normalization(x, training=is_training)

    # apply 4 residual blocks
    # takes in (BATCH_SIZE, 16, 16, GENERATOR_DIM * 4)
    # outputs (BATCH_SIZE, 16, 16, GENERATOR_DIM * 4)
    x_1 = residual_block(x)
    x_2 = residual_block(x_1)
    x_3 = residual_block(x_2)
    x_4 = residual_block(x_3)

    # upsample to high resolution image by alternating convolutions and
    # nearest neighbor resizing
    # takes in (BATCH_SIZE, 16, 16, GENERATOR_DIM * 4)
    # outputs (BATCH_SIZE, 256, 256, 3) 

    x_5 = tf.image.resize_nearest_neighbor(x_4, [(IMAGE_SHAPE // 2),(IMAGE_SHAPE // 2)])
    x_5 = tf.layers.conv2d(x_5, 
            filters=GENERATOR_DIM * 2, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.relu)
    x_5 = tf.layers.batch_normalization(x_5, training=is_training)

    x_6 = tf.image.resize_nearest_neighbor(x_5, [IMAGE_SHAPE,IMAGE_SHAPE])
    x_6 = tf.layers.conv2d(x_6, 
            filters=GENERATOR_DIM, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.relu)
    x_6 = tf.layers.batch_normalization(x_6, training=is_training)

    x_7 = tf.image.resize_nearest_neighbor(x_6, [(IMAGE_SHAPE * 2), (IMAGE_SHAPE * 2)])
    x_7 = tf.layers.conv2d(x_7, 
            filters=(GENERATOR_DIM // 2), 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.relu)
    x_7 = tf.layers.batch_normalization(x_7, training=is_training)

    x_8 = tf.image.resize_nearest_neighbor(x_7, [(IMAGE_SHAPE * 4), (IMAGE_SHAPE * 4)])
    x_8 = tf.layers.conv2d(x_8, 
            filters=(GENERATOR_DIM // 4), 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.relu)
    x_8 = tf.layers.batch_normalization(x_8, training=is_training)

    x_out = tf.layers.conv2d(x_8, 
            filters=3, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.tanh)

    return x_out

def discriminator_stage2(image, embedding_vector, is_training=True):

    # process embedding vector by passing it through a fully-connected layer
    compressed_embedding = tf.layers.dense(embedding_vector, units=EMBEDDING_DIM, activation=tf.nn.leaky_relu)
    # expand from shape [BATCH_SIZE, EMBEDDING_DIM] to [BATCH_SIZE, 4, 4, EMBEDDING_DIM]
    compressed_embedding = tf.expand_dims(tf.expand_dims(compressed_embedding, 1), 1)
    compressed_embedding = tf.tile(compressed_embedding, [1, (IMAGE_SHAPE/16), (IMAGE_SHAPE/16), 1])

    # downsample and convolve image input with strided convolutions + batch norm
    # from [BATCH_SIZE, 256, 256, 3] to [BATCH_SIZE, 4, 4, DISCRIMINATOR_DIM * 8]
    # NOTE: no bias is used and the activation is leaky ReLU
    x_0 = tf.layers.conv2d(image, 
            filters=DISCRIMINATOR_DIM, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_0 = tf.layers.conv2d(x_0, 
            filters=DISCRIMINATOR_DIM * 2, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_0 = tf.layers.batch_normalization(x_0, training=is_training)

    x_0 = tf.layers.conv2d(x_0, 
            filters=DISCRIMINATOR_DIM * 4, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_0 = tf.layers.batch_normalization(x_0, training=is_training)
   
    x_0 = tf.layers.conv2d(x_0, 
            filters=DISCRIMINATOR_DIM * 8, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_0 = tf.layers.batch_normalization(x_0, training=is_training) 

    x_0 = tf.layers.conv2d(x_0, 
            filters=DISCRIMINATOR_DIM * 16, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_0 = tf.layers.batch_normalization(x_0, training=is_training)

    x_0 = tf.layers.conv2d(x_0, 
            filters=DISCRIMINATOR_DIM * 32, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_0 = tf.layers.batch_normalization(x_0, training=is_training)

    # Use 1x1 convolutions to reduce number of channels from DISCRIMINATOR_DIM * 32 to DISCRIMINATOR_DIM * 8
    # Shape goes from (BATCH_SIZE, 4, 4, DISCRIMINATOR_DIM * 32) to (BATCH_SIZE, 4, 4, DISCRIMINATOR_DIM * 8)

    x_0 = tf.layers.conv2d(x_0, 
            filters=DISCRIMINATOR_DIM * 16, 
            kernel_size=1, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_0 = tf.layers.batch_normalization(x_0, training=is_training)

    x_0 = tf.layers.conv2d(x_0, 
            filters=DISCRIMINATOR_DIM * 8, 
            kernel_size=1, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_0 = tf.layers.batch_normalization(x_0, training=is_training)

    # Apply a residual block


    x_1 = tf.layers.conv2d(x_0, 
            filters=DISCRIMINATOR_DIM * 2, 
            kernel_size=1, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_1 = tf.layers.batch_normalization(x_1, training=is_training) 

    x_1 = tf.layers.conv2d(x_1, 
            filters=DISCRIMINATOR_DIM * 2, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    x_1 = tf.layers.batch_normalization(x_1, training=is_training)

    x_1 = tf.layers.conv2d(x_1, 
            filters=DISCRIMINATOR_DIM * 8, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    x_1 = tf.layers.batch_normalization(x_1, training=is_training)

    # note residual connection see https://arxiv.org/pdf/1512.03385.pdf for motivation)
    x = tf.add(x_0, x_1)
    x = tf.nn.leakyrelu(x)

    # concatenate together the downsampled image features and the compressed embedding vector along
    # the channels dimension to shape [BATCH_SIZE, 4, 4, EMBEDDING_DIM + DISCRIMINATOR_DIM * 8] 
    x_and_embedding = tf.concat([x, compressed_embedding], axis=3)

    # apply conv layers on combined image features and embedding to get discriminator output
    # 1x1 convolution outputs [BATCH_SIZE, 4, 4, DISCRIMINATOR_DIM * 8]
    output = tf.layers.conv2d(x_and_embedding, 
            filters=DISCRIMINATOR_DIM * 8, 
            kernel_size=1, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation=tf.nn.leaky_relu)
    output = tf.layers.batch_normalization(output, training=is_training)

    # use kernel of size (4,4) to convolve over entire region and output a single channel dimension
    # gives output of shape [BATCH_SIZE, 1, 1, 1], which is essentially a scalar logit.
    # It represents the discriminator output probability
    output = tf.layers.conv2d(x_0_and_embedding, 
            filters=1, 
            kernel_size=(IMAGE_SHAPE // 16), 
            strides=(IMAGE_SHAPE // 16), 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))

    # reduce shape to [BATCH_SIZE,] by removing extra dimensions
    output = tf.squeeze(output)

    return output

