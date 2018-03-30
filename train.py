import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

import argparse

tfgan = tf.contrib.gan

from stage_1 import generator_stage1, discriminator_stage1
from stage_2 import generator_stage2, discriminator_stage2

BATCH_SIZE = 64

# output shape of generator images
IMAGE_SHAPE = 64

# size factor for the KL-divergence regularization term
# in the stage 1 generator loss
KL_REG_LAMBDA = 0.01

NUM_STEPS = 600

DATA_DIR = "./Data/birds"

TRAIN_DIR = DATA_DIR + "/train"
TEST_DIR = DATA_DIR + "/test"

# We define the generator loss used in the paper by adding the KL regularization term to
# the standard minimax GAN loss from https://arxiv.org/abs/1406.2661
def custom_generator_loss(gan_model, add_summaries=False):

    standard_generator_loss = tfgan.losses.modified_generator_loss(gan_model.discriminator_gen_outputs)
    
    # gan_model.generator_inputs[2] is the KL divergence
    reg_loss = KL_REG_LAMBDA * gan_model.generator_inputs[2] 

    custom_loss = tf.add(standard_generator_loss, reg_loss) 

    return custom_loss

def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "embedding": tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.decode_raw(parsed_features['image'], tf.float32)
    image = tf.reshape(image, [76,76,3])

    embedding = tf.decode_raw(parsed_features['embedding'], tf.float32)
    embedding = tf.reshape(embedding, [1024])

    return image, embedding

# train input function
def train_input_fn():

    train_filenames = [TRAIN_DIR + '/data.tfrecord']
    dataset = tf.data.TFRecordDataset(train_filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()

    batch_images, batch_embeddings = iterator.get_next()

    # get randomly sampled noise/latent vector
    batch_z = tf.random_normal([BATCH_SIZE, Z_DIM])
    # get conditioning vector (from embedding) and KL divergence for use as a
    # regularization term in the generator loss
    batch_conditioning_vectors, kl_div = get_conditioning_vector(batch_embeddings, conditioning_vector_size=EMBEDDING_DIM)

    return batch_z, batch_images, batch_conditioning_vectors, kl_div 

def predict_input_fn():

    test_filenames = [TEST_DIR + '/data.tfrecord']
    dataset = tf.data.TFRecordDataset(test_filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()

    batch_images, batch_embeddings = iterator.get_next()

    # get conditioning vector (from embedding) and KL divergence for use as a
    # regularization term in the generator loss
    batch_conditioning_vectors, kl_div = get_conditioning_vector(batch_embeddings, conditioning_vector_size=EMBEDDING_DIM)

    return batch_images, batch_embeddings
        
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a StackGAN model.')
    parser.add_argument('model', help='Whether to train Stage I or Stage I + II. Choose from stage1 or stage2.')
    parser.add_argument('logdir', help='Directory for storing/reading checkpoint files.')
    parser.add_argument('--mode', default='train', help='Whether to train or predict. Defaults to train')
    parser.add_argument('--num_steps', type=int, help='Number of steps to train for.', default=NUM_STEPS)

    args = parser.parse_args()

    if args.model == 'stage1':
        generator_function = generator_stage1
        discriminator_function = discriminator_stage1
        generator_loss_function = custom_generator_loss
        discriminator_loss_function = tfgan.losses.modified_discriminator_loss
    elif args.model == 'stage2':
        raise NotImplementedError('Not yet implemented.')
    else:
        raise ValueError('Invalid model.')

    # setup gan Estimator
    gan_estimator = tfgan.estimator.GANEstimator(
        generator_fn=generator_function,
        discriminator_fn=discriminator_function,
        generator_loss_fn=generator_loss_function,
        discriminator_loss_fn=discriminator_loss_function,
        generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
        add_summaries=tfgan.estimator.SummaryType.IMAGES,
        model_dir=args.logdir)

    if args.mode == 'train':
        gan_estimator.train(train_input_fn, max_steps=args.num_steps)
    elif args.mode == 'predict':
        # predict (generate) and visualize
        prediction_gen = gan_estimator.predict(predict_input_fn, hooks=[tf.train.StopAtStepHook(last_step=1)])
        predictions = [prediction_gen.next() for _ in xrange(36)]

        # Visualize 36 images together
        image_rows = [np.concatenate(predictions[i:i+6], axis=0) for i in range(0, 36, 6)]
        tiled_images = np.concatenate(image_rows, axis=1)

        # Visualize.
        plt.axis('off')
        plt.imshow(np.squeeze(tiled_images), cmap='gray')

    else:
        raise ValueError('Invalid mode.')

