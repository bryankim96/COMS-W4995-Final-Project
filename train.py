import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

import argparse

tfgan = tf.contrib.gan

from conditioning import get_conditioning_vector
from stage_1 import generator_stage1, discriminator_stage1
from stage_2 import generator_stage2, discriminator_stage2

BATCH_SIZE = 64

Z_DIM = 100
EMBEDDING_DIM = 128

# output shape of generator images
IMAGE_SHAPE = 64

# size factor for the KL-divergence regularization term
# in the stage 1 generator loss
KL_REG_LAMBDA = 2.0

#NUM_STEPS = 600
NUM_STEPS = -1

DATA_DIR = "./Data/birds"

TRAIN_DIR = DATA_DIR + "/train"
TEST_DIR = DATA_DIR + "/test"

# We define the generator loss used in the paper by adding the KL regularization term to
# the standard minimax GAN loss from https://arxiv.org/abs/1406.2661
def custom_generator_loss(gan_model, add_summaries=False):

    standard_generator_loss = tfgan.losses.modified_generator_loss(gan_model) 

    # gan_model.generator_inputs[2] is the KL divergence
    reg_loss = KL_REG_LAMBDA * gan_model.generator_inputs[2] 

    custom_loss = tf.add(standard_generator_loss, reg_loss) 

    return custom_loss

def _parse_function(example_proto, image_type='lr'):

    if image_type=='lr':
        raw_shape = 76
        cropped_shape = 64
    elif image_type=='hr':
        raw_shape = 304
        cropped_shape = 256

    features = {"embeddings": tf.FixedLenFeature([], tf.string),
                "image": tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.decode_raw(parsed_features['image'], tf.float32)
    image = tf.reshape(image, [raw_shape, raw_shape, 3])

    # normalize to (-1,1)
    image = (image * (2.0/255.0)) - 1.0

    # randomly crop from (76, 76, 3) to (64, 64, 3)
    image = tf.random_crop(image, [cropped_shape, cropped_shape, 3])

    # randomly flip left right w/ 50% probability
    image = tf.image.random_flip_left_right(image)

    embeddings = tf.decode_raw(parsed_features['embeddings'], tf.float32)
    embeddings = tf.reshape(embeddings, [-1,1024])

    # randomly sample 4 embeddings and take the average
    embeddings = tf.random_shuffle(embeddings)
    embeddings = embeddings[:4,:]
    avg_embedding = tf.reduce_mean(embeddings, axis=0)

    return image, avg_embedding
      
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
        
        def parse_fn(example_proto):
            return _parse_function(example_proto, image_type='lr')

        data_filename = '/data_76.tfrecord'

    elif args.model == 'stage2':
        raise NotImplementedError('Not yet implemented.')
    else:
        raise ValueError('Invalid model.')

    if args.mode == 'train':

        train_filenames = [TRAIN_DIR + data_filename]
        dataset = tf.data.TFRecordDataset(train_filenames)
        dataset = dataset.map(parse_fn, num_parallel_calls=4)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_initializable_iterator()

        batch_images, batch_embeddings = iterator.get_next()

        # get randomly sampled noise/latent vector
        batch_z = tf.random_normal([BATCH_SIZE, Z_DIM])
        # get conditioning vector (from embedding) and KL divergence for use as a
        # regularization term in the generator loss
        batch_conditioning_vectors, kl_div = get_conditioning_vector(batch_embeddings, conditioning_vector_size=EMBEDDING_DIM)

        model = tfgan.gan_model(
            generator_fn=generator_function,
            discriminator_fn=discriminator_function,
            real_data=batch_images,
            generator_inputs=(batch_z, batch_conditioning_vectors, kl_div))

        loss = tfgan.gan_loss(model,
                generator_loss_fn=generator_loss_function,
                discriminator_loss_fn=discriminator_loss_function)

        generator_optimizer = tf.train.AdamOptimizer(0.002, beta1=0.5)
        discriminator_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            gan_train_ops = tfgan.gan_train_ops(model, loss, generator_optimizer, discriminator_optimizer)

            global_step = tf.train.get_or_create_global_step()
            train_step_fn = tfgan.get_sequential_train_steps()

	# set up image summaries
        tf.summary.image('real_images', batch_images)
        tf.summary.image('generated_images', model.generated_data)
        summary_op = tf.summary.merge_all()
        summary_hook = tf.train.SummarySaverHook(save_secs=300,output_dir=args.logdir,summary_op=summary_op)

        with tf.train.MonitoredTrainingSession(hooks=[summary_hook], checkpoint_dir=args.logdir) as sess:
            sess.run(iterator.initializer)
            if NUM_STEPS < 0:
                while True:
                    cur_loss, _ = train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})
            else:
                for i in range(NUM_STEPS):
                    cur_loss, _ = train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})

    elif args.mode == 'predict':

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
        
    else:
        raise ValueError('Invalid mode.')

