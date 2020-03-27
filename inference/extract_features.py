"""
 utilities to extract features from SavedModel and save it in a pickle file
 """

import csv
import pickle
import argparse

from collections import defaultdict

import tensorflow as tf
import numpy as np
import pandas as pd


class ReadImage:
    """ class to read the images for inference """

    def __init__(self, args):
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.image_depth = args.image_depth
        self.image_dir = args.image_dir
        self.batch_size = args.batch_size
        self.cache_dir = args.log_dir
        self.tune = tf.data.experimental.AUTOTUNE

    def decode_image(self,
                     filename,
                     filename_id):
        """This function will decode the image using image path and normalize
            pixel values between 0 and 1
        args:
            filename        : path to an image
            filename_id     : unique filename
        returns:
            img          : image tensor of the type float
        """
        image_raw = tf.io.read_file(filename)
        image = tf.io.decode_image(image_raw,
                                   channels=self.image_depth)
        image = tf.image.convert_image_dtype(image,
                                             tf.float32)
        image = tf.image.resize_with_pad(image,
                                         self.image_height,
                                         self.image_width,
                                         antialias=True)
        filename_id = tf.cast(filename_id, dtype=tf.int32)
        return image, filename_id

    def map_fn_serving(self,
                       image_path,
                       image_id):
        """
        function to map the images as per serving model needs
        args:
            image_path   : string image path array to read the images, size : (None, )
            image_id     : string image id to identify images uniquely, size : (None, )
        return:
            dataset iterator to iterate over the images
        """

        dataset = tf.data.Dataset.from_tensor_slices((image_path, image_id))
        dataset = dataset.map(self.decode_image, num_parallel_calls=self.tune)
        if self.cache_dir:
            if isinstance(self.cache_dir, str):
                dataset = dataset.cache(self.cache_dir)
            else:
                dataset = dataset.cache()
        dataset = dataset.repeat(1).batch(self.batch_size).prefetch(buffer_size=self.tune)
        return dataset


def extract_features(args):
    """ extract image features from the given csv
    args:
        saved_model_path    : tensorflow saved model absolute path
        csv_path            : csv of images to get the features for
        image_path_col      : column name corresponding to relevant image path
        image_dir           : parent directory where all the images are stored
        batch_size          : batch size to feed forward images to the network
        pickle_file_path    : pickle file path to save the feature vectors

    """
    # read the csv using pandas
    data_frame = pd.read_csv(args.csv_path)

    # make a tf.data.Dataset iterator
    read_image = ReadImage(args)
    dataset = read_image.map_fn_serving(data_frame[args.image_path_col].values,
                                        data_frame['class_id'].values)

    # load the saved model
    model = tf.saved_model.load(args.saved_model_path)

    for step, (image, image_id) in enumerate(dataset):
        _, base_features, base_features_l2, head_features, head_features_l2 = model([image,
                                                                                     image_id])
        if step == 0:
            features_b = np.zeros((data_frame.shape[0], base_features.numpy().shape[1]))
            features_b_l2 = np.zeros((data_frame.shape[0], base_features_l2.numpy().shape[1]))
            features_h = np.zeros((data_frame.shape[0], head_features.numpy().shape[1]))
            features_h_l2 = np.zeros((data_frame.shape[0], head_features_l2.numpy().shape[1]))

        int_size = base_features.shape[0]
        features_b[step * int_size:(step + 1) * int_size] = base_features.numpy()
        features_b_l2[step * int_size:(step + 1) * int_size] = base_features_l2.numpy()
        features_h[step * int_size:(step + 1) * int_size] = head_features.numpy()
        features_h_l2[step * int_size:(step + 1) * int_size] = head_features_l2.numpy()

    # extract features
    feature_dict = defaultdict(defaultdict)
    # store embeddings in dict
    with open(args.csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for i, row in enumerate(csv_reader):
            feature_dict[data_frame.path.values[i]] = row

            dummy = {'base_embeddings': features_b[i, :],
                     'base_embeddings_l2': features_b_l2[i, :],
                     'head_embeddings': features_h[i, :],
                     'head_embeddings_l2': features_h_l2[i]}

            # append embedding to dictionary
            feature_dict[data_frame.path.values[i]].update(dummy)

    with open(args.pickle_file_path, "wb+") as file:
        pickle.dump(feature_dict, file, pickle.HIGHEST_PROTOCOL)
    print("Embeddings are stored at {0}".format(args.pickle_file_path))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--log_dir',
        type=str,
        help='log directory to store logs',
        default='/media/jay/rnd/Recognition_Datasets/MNIST/metric')
    PARSER.add_argument(
        '--image_dir',
        type=str,
        help='base path for the img directory',
        default='/media/jay/rnd/Recognition_Datasets/MNIST/images')
    PARSER.add_argument(
        '--image_height',
        type=int,
        help='Number of channels images have',
        default=28)
    PARSER.add_argument(
        '--image_width',
        type=int,
        help='Number of channels images have',
        default=28)
    PARSER.add_argument(
        '--image_depth',
        type=int,
        help='Final image size (height or width) in pixels.',
        default=1)
    PARSER.add_argument(
        '--batch_size',
        type=int,
        help='batch size for the mini batch of images',
        default=256)
    PARSER.add_argument(
        '--csv_path',
        type=str,
        help='path to csv containing image info to get the features for',
        default="/media/jay/rnd/Recognition_Datasets/MNIST/metric/test_dataset.csv")
    PARSER.add_argument(
        '--image_path_col',
        type=str,
        help='column corresponding to rel path of an image in csv',
        default="path")
    PARSER.add_argument(
        '--saved_model_path',
        type=str,
        help='parent path to image directory',
        default="/media/jay/rnd/Recognition_Datasets/MNIST/metric/export-1076")
    PARSER.add_argument(
        '--pickle_file_path',
        type=str,
        help='path to csv containing image info to get the features for',
        default="/media/jay/rnd/Recognition_Datasets/MNIST/metric/test_embeddings.pkl")

    ARGS = PARSER.parse_args()

    extract_features(ARGS)
