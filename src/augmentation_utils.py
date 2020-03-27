""" data augmentation helper functions for training / test and inference routines"""

import sys
import argparse
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def plot_images(images_arr):
    """Plot images to analyse the dataset
    args:
     images_arr: images as a list of concatenated numpy arrays
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 12))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


class Augmentation:

    def __init__(self, args):
        """ augmentation class init
        :param
            - args : augument namespace
        """
        self.log_dir = args.log_dir
        self.image_dir = args.img_dir
        self.image_size = args.image_size
        self.image_depth = args.image_depth
        self.random_hflip = False
        self.batch_size = args.batch_size
        self.seed = 4

    def map_fn_train(self, train_data):
        """training time augmentations
        args:
            train_data  : training dataframe
        returns:
            train_generator : generator for training images
            train_data      : pandas dataframe for training images
        """
        train_data['class_name'] = train_data.class_name.values.astype(str)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            brightness_range=[0.8, 1.2],
            shear_range=0.1,
            zoom_range=[0.7, 1.3],
            fill_mode='constant',
            cval=0.0,
            horizontal_flip=self.random_hflip)
        train_generator = train_datagen.flow_from_dataframe(
            train_data,
            directory=self.image_dir,
            x_col='path',
            y_col='class_id',
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode='raw',
            shuffle=True,
            seed=self.seed,
            interpolation="nearest",
            validate_filenames=True)
        return train_generator, train_data

    def map_fn_test(self, test_data):
        """training time augmentations
        args:
            test_data  : test dataframe
        returns:
            test_generator : generator for test images
            test_data      : pandas dataframe for test images
        """
        test_data['class_name'] = test_data.class_name.values.astype(str)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            fill_mode='constant',
            cval=0.0)
        test_generator = train_datagen.flow_from_dataframe(
            test_data,
            directory=self.image_dir,
            x_col='path',
            y_col='class_id',
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode='raw',
            shuffle=False,
            seed=None,
            interpolation="nearest",
            validate_filenames=True)
        return test_generator, test_data


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        type=str,
                        help='log directory to store logs',
                        default='/media/jay/data/Dataset/Fashion-MNIST/logs')
    parser.add_argument('--img_dir',
                        type=str,
                        help='base path for the img directory',
                        default='/media/jay/data/Dataset/Fashion-MNIST')
    parser.add_argument('--image_size',
                        type=int,
                        help='Number of channels images have',
                        default=28)
    parser.add_argument('--image_depth',
                        type=int,
                        help='Final image size (height or width) in pixels.',
                        default=1)
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size for the mini batch of images',
                        default=32)
    return parser.parse_args(argv)


if __name__ == '__main__':
    data = Augmentation(parse_arguments(sys.argv[1:]))
    image_data = data.map_fn_train()
