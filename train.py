import tensorflow as tf
import numpy as np
import pickle
from PIL import Image, ImageDraw
import scipy
import torch

from torch.utils.serialization import load_lua
 
import argparse
import random

from tensorflow.python import debug as tf_debug
tfgan = tf.contrib.gan

from conditioning import get_conditioning_vector
from stage_1 import generator_stage1, discriminator_stage1
from stage_2 import generator_stage2, discriminator_stage2

from misc.preprocess_birds import LOAD_SIZE, LR_HR_RATIO, load_bbox
from misc.utils import get_image

BATCH_SIZE = 64

NUM_EVAL = 10

Z_DIM = 100
EMBEDDING_DIM = 128

# output shape of generator images
IMAGE_SHAPE = 64

# size factor for the KL-divergence regularization term
# in the stage 1 generator loss
KL_REG_LAMBDA = 1.0

#NUM_STEPS = 600
NUM_STEPS = -1

DATA_DIR = "./Data/birds"

TRAIN_DIR = DATA_DIR + "/train"
TEST_DIR = DATA_DIR + "/test"

with open(TRAIN_DIR + '/76images.pickle', 'rb') as f:
    IMAGES = pickle.load(f, encoding='latin1')
    IMAGES = np.array(IMAGES)
    print('images: ', IMAGES.shape)

with open(TRAIN_DIR + '/char-CNN-RNN-embeddings.pickle', 'rb') as f:
    EMBEDDINGS = pickle.load(f, encoding='latin1')
    EMBEDDINGS = np.array(EMBEDDINGS)
    print('embeddings: ', EMBEDDINGS.shape)

with open(TRAIN_DIR + '/class_info.pickle', 'rb') as f:
    CLASSES = pickle.load(f, encoding='latin1')

NUM_TRAINING_EXAMPLES = IMAGES.shape[0]

# We define the generator loss used in the paper by adding the KL regularization term to
# the standard minimax GAN loss from https://arxiv.org/abs/1406.2661

def custom_generator_loss(gan_model, add_summaries=False):

    standard_generator_loss = tfgan.losses.modified_generator_loss(gan_model) 

    # gan_model.generator_inputs[2] is the KL divergence
    kl_div_loss = tf.multiply(KL_REG_LAMBDA, gan_model.generator_inputs[2], name="kl_div_loss") 
    generator_loss = tf.add(standard_generator_loss, kl_div_loss, name="generator_loss")

    if add_summaries:
        tf.summary.scalar("kl_div_loss", kl_div_loss)
        tf.summary.scalar("generator_loss", generator_loss) 

    return generator_loss

# discriminator loss with mismatched pairs
def custom_discriminator_loss(gan_model, real_data, batch_mismatched_conditioning_vectors, add_summaries=False):

    mismatched_inputs = (gan_model.generator_inputs[0], batch_mismatched_conditioning_vectors, gan_model.generator_inputs[2])

    discriminator_real_outputs = gan_model.discriminator_real_outputs,
    discriminator_gen_outputs = gan_model.discriminator_gen_outputs,  
    with tf.variable_scope('Discriminator', reuse=True):    
        discriminator_mismatched_outputs = gan_model.discriminator_fn(real_data, mismatched_inputs) 

    #label_smoothing=0.25
    label_smoothing=0.2
    real_weights=1.0
    generated_weights=1.0
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
  
    with tf.name_scope(None, 'discriminator_minimax_loss', (
      discriminator_real_outputs, discriminator_gen_outputs, real_weights,
      generated_weights, label_smoothing)) as scope:

        # -log((1 - label_smoothing) - sigmoid(D(x)))
        loss_on_real = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_real_outputs),
            discriminator_real_outputs, real_weights, label_smoothing, scope,
            loss_collection=None, reduction=reduction)
        # -log(- sigmoid(D(G(z,h))))
        loss_on_generated = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_gen_outputs),
            discriminator_gen_outputs, generated_weights, scope=scope,
            loss_collection=None, reduction=reduction)    
        # -log(- sigmoid(D(G(z,h_hat))))
        loss_mismatched = tf.losses.sigmoid_cross_entropy(tf.zeros_like(discriminator_mismatched_outputs),
            discriminator_mismatched_outputs, generated_weights, label_smoothing, scope=scope,
            loss_collection=None, reduction=reduction) 
     
        loss = loss_on_real + (loss_on_generated + loss_mismatched)/2
        tf.losses.add_loss(loss, tf.GraphKeys.LOSSES)

    if add_summaries:
        tf.summary.scalar('discriminator_gen_minimax_loss', loss_on_generated)
        tf.summary.scalar('discriminator_real_minimax_loss', loss_on_real)
        tf.summary.scalar('discriminator_mismatched_minimax_loss', loss_mismatched)
        tf.summary.scalar('discriminator_minimax_loss', loss)

    return loss


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

def map_fn(index):
    return tuple(tf.py_func(load_data, [index], [tf.float32, tf.float32, tf.float32]))

def load_data(index, image_type='lr'):

    if image_type=='lr':
        raw_shape = 76
        cropped_shape = 64
    elif image_type=='hr':
        raw_shape = 304
        cropped_shape = 256

    image = IMAGES[index]
    image = np.reshape(image, [raw_shape, raw_shape, 3])

    # normalize to (-1,1)
    image = (image * (2.0/255.0)) - 1.0

    # randomly crop from (76, 76, 3) to (64, 64, 3)
    x = np.random.randint(12)
    y = np.random.randint(12)
    image = image[x:x+64,y:y+64,:]
    image = image.astype(np.float32)

    # randomly flip left right w/ 50% probability
    flip = np.random.randint(2)
    if flip == 1:
        image = np.fliplr(image)

    embeddings = EMBEDDINGS[index]

    # randomly sample 4 embeddings and take the average
    np.random.shuffle(embeddings)
    embeddings = embeddings[:4,:]
    embedding = np.mean(embeddings, axis=0)

    # get class of selected image
    img_class = CLASSES[index]

    # get wrong embeddings (of different class)
    wrong_index = np.random.randint(NUM_TRAINING_EXAMPLES)
    if img_class == CLASSES[wrong_index]:
        wrong_index = (wrong_index + np.random.randint(100, 200)) % NUM_TRAINING_EXAMPLES
    j = np.random.randint(10)
    mismatched_embedding = EMBEDDINGS[wrong_index][j,:]

    return image, embedding, mismatched_embedding
  
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a StackGAN model.')
    parser.add_argument('model', help='Whether to train Stage I or Stage I + II. Choose from stage1 or stage2.')
    parser.add_argument('logdir', help='Directory for storing/reading checkpoint files.')
    parser.add_argument('--mode', default='train', help='Whether to train or predict. Defaults to train')
    parser.add_argument('--num_steps', type=int, help='Number of steps to train for.', default=NUM_STEPS)
    parser.add_argument('--test_description', help='text description sentence to predict for in eval mode', default=None)
    parser.add_argument('--test_embedding', help='path to a test embedding to predict on when in eval mode', default=None)
    parser.add_argument('--base_learning_rate', type=int, default=0.0002)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if args.model == 'stage1':
        
        generator_function = generator_stage1
        discriminator_function = discriminator_stage1
        generator_loss_function = custom_generator_loss
        #discriminator_loss_function = custom_discriminator_loss_fn
        
        def parse_fn(example_proto):
            return _parse_function(example_proto, image_type='lr')

        data_filename = '/data_76.tfrecord'

    elif args.model == 'stage2':
        raise NotImplementedError('Not yet implemented.')
    else:
        raise ValueError('Invalid model.')

    if args.mode == 'train':

        """
        train_filenames = [TRAIN_DIR + data_filename]
        dataset = tf.data.TFRecordDataset(train_filenames)
        dataset = dataset.map(parse_fn, num_parallel_calls=4)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_initializable_iterator()
        """

        def gen():
            for i in range(NUM_TRAINING_EXAMPLES):
                yield i

        dataset = tf.data.Dataset.from_generator(gen, (tf.int32), (tf.TensorShape([])))
        dataset = dataset.map(map_fn, num_parallel_calls=4)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()    

        batch_images, batch_embeddings, batch_mismatched_embeddings = iterator.get_next()

        batch_images.set_shape([None,64,64,3])
        batch_embeddings.set_shape([None,1024])
        batch_mismatched_embeddings.set_shape([None,1024])

        batch_images = tf.identity(batch_images, name="batch_images")
        batch_embeddings = tf.identity(batch_embeddings, name="batch_embeddings")
        batch_mismatched_embeddings = tf.identity(batch_mismatched_embeddings, name="batch_mismatched_embeddings")

        # get randomly sampled noise/latent vector        
        batch_z = tf.random_normal([BATCH_SIZE, Z_DIM], name="batch_z")
        # get conditioning vector (from embedding) and KL divergence for use as a
        # regularization term in the generator loss
        with tf.variable_scope("conditioning_augmentation", reuse=tf.AUTO_REUSE):
            batch_conditioning_vectors, kl_div = get_conditioning_vector(batch_embeddings, conditioning_vector_size=EMBEDDING_DIM)
            batch_mismatched_conditioning_vectors, _ = get_conditioning_vector(batch_mismatched_embeddings, conditioning_vector_size=EMBEDDING_DIM)

        batch_conditioning_vectors = tf.identity(batch_conditioning_vectors, name="batch_conditioning_vectors")
        kl_div = tf.identity(kl_div, name="kl_div")
        batch_mismatched_conditioning_vectors = tf.identity(batch_mismatched_conditioning_vectors, name="batch_mismatched_conditioning_vectors")

        def custom_discriminator_loss_fn(gan_model, add_summaries=False):
            return custom_discriminator_loss(gan_model, batch_images, batch_mismatched_conditioning_vectors, add_summaries)

        model = tfgan.gan_model(
            generator_fn=generator_function,
            discriminator_fn=discriminator_function,
            real_data=batch_images,
            generator_inputs=(batch_z, batch_conditioning_vectors, kl_div))

        loss = tfgan.gan_loss(model,
                generator_loss_fn=generator_loss_function,
                discriminator_loss_fn=custom_discriminator_loss_fn)

        global_step = tf.train.get_or_create_global_step()
        boundaries = list(range((NUM_TRAINING_EXAMPLES//BATCH_SIZE) * 100, (NUM_TRAINING_EXAMPLES//BATCH_SIZE) * 601, (NUM_TRAINING_EXAMPLES//BATCH_SIZE) * 100))
        generator_lr_values = [args.base_learning_rate * (0.5)**i for i in range(6)]
        discriminator_lr_values = [i for i in generator_lr_values] 

        generator_learning_rate = tf.train.piecewise_constant(global_step, boundaries, generator_lr_values, name="generator_learning_rate")
        discriminator_learning_rate = tf.train.piecewise_constant(global_step, boundaries, discriminator_lr_values, name="discriminator_learning_rate")

        tf.summary.scalar("generator_learning_rate", generator_learning_rate)
        tf.summary.scalar("discriminator_learning_rate", discriminator_learning_rate)

        generator_optimizer = tf.train.AdamOptimizer(generator_learning_rate, beta1=0.5)
        #discriminator_optimizer = tf.train.AdamOptimizer(discriminator_learning_rate, beta1=0.5)
        discriminator_optimizer = tf.train.AdamOptimizer(discriminator_learning_rate, beta1=0.5)

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

        hooks = [summary_hook]
        if args.debug:
            hooks.append(tf_debug.LocalCLIDebugHook())

        with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=args.logdir) as sess:
            if NUM_STEPS < 0:
                while True:
                    cur_loss, _ = train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})
            else:
                for i in range(NUM_STEPS):
                    cur_loss, _ = train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})

    elif args.mode == 'eval':

        if args.test_description and args.test_embedding:
            caption = args.test_description
            embedding = load_lua('test_embedding.t7').numpy()[0]

        elif args.test_description and (not args.test_embedding):
            raise ValueError("Both a description and embedding need to be provided.")
        elif (not args.test_description) and args.test_embedding:
            raise ValueError("Both a description and embedding need to be provided.")
        else:

            # get class ids as list of len (2933)
            with open(TEST_DIR + '/class_info.pickle', 'rb') as f:
                class_ids = pickle.load(f)

            # load embeddings as numpy array of shape (2933, 10, 1024)
            with open(TEST_DIR + '/char-CNN-RNN-embeddings.pickle', 'rb') as f:
                embeddings = pickle.load(f,encoding='latin1')
                embeddings = np.array(embeddings)

            # get list of filenames of len (2933)
            with open(TEST_DIR + '/filenames.pickle', 'rb') as f:
                list_filenames = pickle.load(f)
                
            # choose a random test image filename
            index, filename = random.choice(list(enumerate(list_filenames)))	

            # get the corresponding captions
            with open(DATA_DIR + '/text_c10/' + filename + '.txt', "r") as f:
                captions = f.read().split('\n')
            captions = [cap for cap in captions if len(cap) > 0]
        
            # randomly select 1 caption and the corresponding embedding
            j, caption = random.choice(list(enumerate(captions)))
            embedding = embeddings[index][j]

            # load the example true image (process as usual down to 76 x 76
            lr_size = int(LOAD_SIZE / LR_HR_RATIO)
            filename_bbox = load_bbox('Data/birds/')
            bbox = filename_bbox[filename]
            f_name = 'Data/birds/CUB_200_2011/images/%s.jpg' % filename
            img = get_image(f_name, LOAD_SIZE, is_crop=True, bbox=bbox) 
            img = img.astype(np.float32)
            true_img = scipy.misc.imresize(img, [lr_size, lr_size], 'bicubic').astype(np.float32)
            
        print("Caption: ", caption)

        # convert the embedding to a tensor and repeat BATCH_SIZE times to shape (BATCH_SIZE, 1024)
        embedding = tf.constant(embedding)
        batch_embeddings =  tf.tile(tf.expand_dims(embedding, axis=0),[BATCH_SIZE,1])
       
        batch_z = tf.random_normal([BATCH_SIZE, Z_DIM])
        
        # get conditioning vector (from embedding) and KL divergence for use as a
        # regularization term in the generator loss
        # get batch of conditioning vectors of shape (NUM_EVAL, EMBEDDING_DIM)
        batch_conditioning_vectors, kl_div = get_conditioning_vector(batch_embeddings, conditioning_vector_size=EMBEDDING_DIM)

        with tf.variable_scope('Generator'):
            eval_images = generator_function((batch_z, batch_conditioning_vectors, kl_div), is_training=False)
        reshaped_eval_imgs = tfgan.eval.image_reshaper(eval_images[:NUM_EVAL,:,:,:], num_cols=NUM_EVAL)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess,tf.train.latest_checkpoint(args.logdir))
            # tf.contrib.framework.init_from_checkpoint(args.logdir,{'Generator/': 'Generator/'})
           
            # get out BATCH_SIZE predicted images 
            eval_images_array = sess.run(reshaped_eval_imgs)

        # draw composite image with PIL
        # display true image, NUM_EVAL example generated images, and caption
        # in a single figure
        composite_img = Image.new('RGB', (NUM_EVAL*64 + 200 + 76, 200))

        d = ImageDraw.Draw(composite_img)
        d.text((20,30), caption, fill=(255,255,0))

        if not args.test_description and not args.test_embedding:
            true_img = Image.fromarray(true_img.astype(np.uint8))    
            composite_img.paste(true_img,(50,112))

        eval_images_array = Image.fromarray(((eval_images_array[0] + 1.0) * (255.0 / 2.0)).astype(np.uint8))
        composite_img.paste(eval_images_array,(176,118))

        composite_img.save('eval.png')

    else:
        raise ValueError('Invalid mode.')

