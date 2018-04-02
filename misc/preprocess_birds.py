from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import os
import pickle
import scipy

from .utils import get_image

import pandas as pd

LR_HR_RATIO = 4
IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)
BIRD_DIR = '../Data/birds/'

def load_embeddings(data_dir):
    filepath = data_dir + 'char-CNN-RNN-embeddings.pickle'
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f,encoding='latin1')
    print('Load embeddings from: %s (%d)' % (filepath, len(embeddings)))
    return embeddings


def load_filenames(data_dir):
    filepath = data_dir + 'filenames.pickle'
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f,encoding='latin1')
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def load_bbox(data_dir):
    bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    #
    filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
    df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])
    #
    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in range(0, numImgs):
        # bbox = [x-left, y-top, width, height]
        bbox = df_bounding_boxes.iloc[i][1:].tolist()

        key = filenames[i][:-4]
        filename_bbox[key] = bbox
    #
    return filename_bbox


def save_data_tfrecords(inpath, outpath, filenames, filename_bbox):
     
    # get list of embeddings
    # list of len NUM_IMAGES of numpy arrays of shape [NUM_CAPTIONS, 1024]
    embeddings = load_embeddings(outpath)

    lr_size = int(LOAD_SIZE / LR_HR_RATIO)

    # open the TFRecords files
    writer_lr = tf.python_io.TFRecordWriter(outpath + 'data_' + str(lr_size) + '.tfrecord')
    writer_hr = tf.python_io.TFRecordWriter(outpath + 'data_' + str(LOAD_SIZE) + '.tfrecord')
 
    cnt = 0
    for i, key in enumerate(filenames):
        bbox = filename_bbox[key]
        f_name = '%s/CUB_200_2011/images/%s.jpg' % (inpath, key)
        img = get_image(f_name, LOAD_SIZE, is_crop=True, bbox=bbox) 
        img = img.astype(np.float32)
        lr_img = scipy.misc.imresize(img, [lr_size, lr_size], 'bicubic').astype(np.float32)

        # Create features
        features_lr = {'embeddings': tf.train.Feature(bytes_list=tf.train.BytesList(value=[embeddings[i].tostring()])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lr_img.tostring()]))
                    }

        features_hr = {'embeddings': tf.train.Feature(bytes_list=tf.train.BytesList(value=[embeddings[i].tostring()])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
                    }

        # Create an example protocol buffer
        example_lr = tf.train.Example(features=tf.train.Features(feature=features_lr))
        example_hr = tf.train.Example(features=tf.train.Features(feature=features_hr))
 
        # Serialize to string and write on the file
        writer_lr.write(example_lr.SerializeToString())
        writer_hr.write(example_hr.SerializeToString())

        cnt += 1
        if cnt % 100 == 0:
            print('Loaded %d......' % cnt)

    print('Done loading. %d images loaded' % cnt)

def convert_birds_dataset_tfrecords(inpath):
    # Load dictionary between image filename to its bbox
    filename_bbox = load_bbox(inpath)
    # ## For Train data
    train_dir = os.path.join(inpath, 'train/')
    train_filenames = load_filenames(train_dir)
    save_data_tfrecords(inpath, train_dir, train_filenames, filename_bbox)

    # ## For Test data
    test_dir = os.path.join(inpath, 'test/')
    test_filenames = load_filenames(test_dir)
    save_data_tfrecords(inpath, test_dir, test_filenames, filename_bbox)


if __name__ == '__main__':
    convert_birds_dataset_tfrecords(BIRD_DIR)
