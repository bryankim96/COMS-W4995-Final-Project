import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

from conditioning import get_conditioning_vector

tfgan = tf.contrib.gan


# output shape of generator images
IMAGE_SHAPE = 64

# dimension of z (noise) vector for generator input
Z_DIM = 100
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
def generator_stage1(features, mode):

    z = features['z']
    conditioning_vector = features['c']
    kl_div = features['kl_div']

    is_training = True if mode == 'train' else False

    # concatenate noise/latent vector and conditioning vector
    z_var = tf.concat([z, conditioning_vector], axis=1)

    # residual block 1
    ############################################################################

    # upsamples from [BATCH_SIZE, 128 + 100] (z_var)
    # to [BATCH_SIZE, 4, 4, 128 * 8]

    # this portion does the upsampling from the z_var input vector to a
    # [BATCH_SIZE, 4, 4, 128 * 8] image
    x_1_0 = tf.layers.flatten(z_var)
    x_1_0 = tf.layers.dense(x_1_0, (IMAGE_SHAPE // 16) * (IMAGE_SHAPE // 16) * GENERATOR_DIM * 8)
    x_1_0 = tf.layers.batch_normalization(x_1_0, training=is_training)
    x_1_0 = tf.reshape(x_1_0, [-1, (IMAGE_SHAPE // 16), (IMAGE_SHAPE // 16), GENERATOR_DIM * 8])

    # 3 stacked convolutional layers with ReLU activations and batch normalization
    x_1_1 = tf.layers.conv2d(x_1_0, 
            filters=GENERATOR_DIM * 2, 
            kernel_size=1, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.relu)
    x_1_1 = tf.layers.batch_normalization(x_1_1, training=is_training)
    x_1_1 = tf.layers.conv2d(x_1_1, 
            filters=GENERATOR_DIM * 2, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.relu)
    x_1_1 = tf.layers.batch_normalization(x_1_1, training=is_training)
    x_1_1 = tf.layers.conv2d(x_1_1, 
            filters=GENERATOR_DIM * 8, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    x_1_1 = tf.layers.batch_normalization(x_1_1, training=is_training)
 
    # note residual connection back to x_1_0 (see https://arxiv.org/pdf/1512.03385.pdf for motivation)
    x_1 = tf.add(x_1_0, x_1_1)
    x_1 = tf.nn.relu(x_1)

    # residual block 2
    ################################################################################

    # upsamples using nearest neighbor interpolation from [BATCH_SIZE, 4, 4, 128 * 8]
    # to [BATCH_SIZE, 8, 8, 128 * 8] and applies convolutions
    x_2_0 = tf.image.resize_nearest_neighbor(x_1, ((IMAGE_SHAPE // 8),(IMAGE_SHAPE // 8)))
    x_2_0 = tf.layers.conv2d(x_2_0, 
            filters=GENERATOR_DIM * 8, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    x_2_0 = tf.layers.batch_normalization(x_2_0, training=is_training)

    # 3 convolutional layers with batch norm and ReLU activations
    x_2_1 = tf.layers.conv2d(x_2_0, 
            filters=GENERATOR_DIM * 2, 
            kernel_size=1, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.relu)
    x_2_1 = tf.layers.batch_normalization(x_2_1, training=is_training)
    x_2_1 = tf.layers.conv2d(x_2_1, 
            filters=GENERATOR_DIM * 2, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.relu)
    x_2_1 = tf.layers.batch_normalization(x_2_1, training=is_training)
    x_2_1 = tf.layers.conv2d(x_2_1, 
            filters=GENERATOR_DIM * 8, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    x_2_1 = tf.layers.batch_normalization(x_2_1, training=is_training)
 
    x_2 = tf.add(x_2_0, x_2_1)
    x_2 = tf.nn.relu(x_2)

    # upsample, apply conv + ReLU + BN
    x_3 = tf.image.resize_nearest_neighbor(x_2, ((IMAGE_SHAPE // 4),(IMAGE_SHAPE // 4)))
    x_3 = tf.layers.conv2d(x_3, 
            filters=GENERATOR_DIM * 2, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.relu)
    x_3 = tf.layers.batch_normalization(x_3, training=is_training)
 
    # upsample, apply conv + ReLU + BN
    x_4 = tf.image.resize_nearest_neighbor(x_3, ((IMAGE_SHAPE // 2),(IMAGE_SHAPE // 2)))
    x_4 = tf.layers.conv2d(x_4, 
            filters=GENERATOR_DIM, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.relu)
    x_4 = tf.layers.batch_normalization(x_4, training=is_training)

    # upsample, apply conv + ReLU + BN
    x_5 = tf.image.resize_nearest_neighbor(x_4, ((IMAGE_SHAPE),(IMAGE_SHAPE)))
    x_5 = tf.layers.conv2d(x_5, 
            filters=3, 
            kernel_size=3, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.tanh)

    return x_5

def discriminator_stage1(image, features, mode):

    embedding_vector = features['c']

    is_training = True if mode == 'train' else False

    # process embedding vector by passing it through a fully-connected layer
    compressed_embedding = tf.layers.dense(embedding_vector, units=EMBEDDING_DIM, activation=tf.nn.leaky_relu)
    # expand from shape [BATCH_SIZE, EMBEDDING_DIM] to [BATCH_SIZE, 4, 4, EMBEDDING_DIM]
    compressed_embedding = tf.expand_dims(tf.expand_dims(compressed_embedding, 1), 1)
    compressed_embedding = tf.tile(compressed_embedding, [1, (IMAGE_SHAPE // 16), (IMAGE_SHAPE // 16), 1])

    # downsample and convolve image input with strided convolutions + batch norm
    # from [BATCH_SIZE, 64, 64, 3] to [BATCH_SIZE, 4, 4, DISCRIMINATOR_DIM * 8]
    # NOTE: no bias is used and the activation is leaky ReLU
    image_features = tf.layers.conv2d(image, 
            filters=DISCRIMINATOR_DIM, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.leaky_relu)
    image_features = tf.layers.conv2d(image_features, 
            filters=DISCRIMINATOR_DIM * 2, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.leaky_relu)
    image_features = tf.layers.batch_normalization(image_features, training=is_training)

    image_features = tf.layers.conv2d(image_features, 
            filters=DISCRIMINATOR_DIM * 4, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.leaky_relu)
    image_features = tf.layers.batch_normalization(image_features, training=is_training)
   
    image_features = tf.layers.conv2d(image_features, 
            filters=DISCRIMINATOR_DIM * 8, 
            kernel_size=4, 
            strides=2, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.leaky_relu)
    image_features = tf.layers.batch_normalization(image_features, training=is_training)

    # concatenate together the downsampled image features and the compressed embedding vector along
    # the channels dimension to shape [BATCH_SIZE, 4, 4, EMBEDDING_DIM + DISCRIMINATOR_DIM * 8] 
    image_features_and_embedding = tf.concat([image_features, compressed_embedding], axis=3)

    # apply conv layers on combined image features and embedding to get discriminator output
    # 1x1 convolution outputs [BATCH_SIZE, 4, 4, DISCRIMINATOR_DIM * 8]
    output = tf.layers.conv2d(image_features_and_embedding, 
            filters=DISCRIMINATOR_DIM * 8, 
            kernel_size=1, 
            strides=1, 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            activation=tf.nn.leaky_relu)
    output = tf.layers.batch_normalization(output, training=is_training)

    # use kernel of size (4,4) to convolve over entire region and output a single channel dimension
    # gives output of shape [BATCH_SIZE, 1, 1, 1], which is essentially a scalar logit.
    # It represents the discriminator output probability
    output = tf.layers.conv2d(image_features_and_embedding, 
            filters=1, 
            kernel_size=(IMAGE_SHAPE // 16), 
            strides=(IMAGE_SHAPE // 16), 
            padding="same", 
            use_bias=False, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    # reduce shape to [BATCH_SIZE,] by removing extra dimensions
    output = tf.squeeze(output)

    return output

