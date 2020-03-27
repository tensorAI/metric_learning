"""Image class for data reading and augmentation utils
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import sys
import glob
import os
import matplotlib.pyplot as plt
import generate_pbtxt as pbtxt

from PIL import Image


class Dataset:

    def __init__(self, args):
        """
        read the argument and load the dataset using pandas dataframe
        :param args: argument namespace
        """
        self.img_dir = args.img_dir
        self.log_dir = args.log_dir
        self.dataset = self.make_dataset()
        self.fraction = args.frac
        self.args = args
        self.class_names = self.dataset.class_name.unique().tolist()
        self.class_id = self.dataset.class_id.unique().tolist()

    def make_dataset(self):
        """ make a dataset csv from the image directory
        :returns
            dataset : dataframe consists of following columns
                - path  : image paths (absolute)
                - class_id  : image labels / classes as int
                - class_name: image labels / classes as str
                - image_id  : unique image names as str
        Note that this format of dataset csv is consistant across the project
        """
        list_ds = glob.glob(self.img_dir + '/*/*')
        dataset = pd.DataFrame({'path': list_ds})
        dataset['class_name'] = dataset.path.map(lambda x: x.split('/')[-2])
        dataset['image_id'] = dataset.path.map(lambda x: os.path.relpath(x, self.img_dir))
        labels, _ = pd.factorize(dataset.class_name.values)
        dataset['class_id'] = labels
        with open(os.path.join(self.log_dir, 'dataset.csv'), 'w') as f:
            dataset.to_csv(f, sep=',', index=False)
        return dataset

    def write_args(self):
        """Write arguments before every run"""
        with tf.io.gfile.GFile(os.path.join(self.log_dir, 'arguments.txt'), 'w') as f:
            for key, value in vars(self.args).items():
                f.write('%s: %s\n' % (key, str(value)))

    def get_class_info(self):
        """returns dataset classes with image count
         - stores class_counts.csv with image counts in each class
         - stores a pbtxt file with label_mappings in log_dir
        """
        counts = self.dataset.class_name.value_counts()[self.class_names]
        with open(os.path.join(self.log_dir, 'class_counts.csv'), 'w+') as f:
            df = pd.DataFrame({'class_id': self.class_id, 'class_name': counts.index, 'count': counts.values})
            df.to_csv(f, sep=',', index=False)
        pbtxt.write_class_names(os.path.join(self.log_dir, 'class_counts.csv'))
        return dict(zip(counts.index, counts.values))

    def split_dataset(self):
        """ Divide a dataframe in test-train dataframe according to fraction set in args.frac """
        train_dataframe, test_dataframe = [], []
        np.random.seed(4)

        for name in self.class_names:
            df = self.dataset[self.dataset['class_name'].values == name]
            train_msk = np.random.choice(np.asarray(range(len(df))),
                                         size=(int(len(df)*self.fraction),), replace=False)
            test_msk = np.delete(range(len(df)),
                                 train_msk)
            train_dataframe.append(df.iloc[train_msk.tolist(), :]), \
                test_dataframe.append(df.iloc[test_msk.tolist(), :])

        # write both the split to the disk
        train_data, test_data = pd.concat(train_dataframe).reset_index(drop=True), \
            pd.concat(test_dataframe).reset_index(drop=True)

        with open(os.path.join(self.log_dir, 'train_dataset.csv'), 'w') as f:
            train_data.to_csv(f, sep=',', index=False)
        with open(os.path.join(self.log_dir, 'test_dataset.csv'), 'w') as f:
            test_data.to_csv(f, sep=',', index=False)
        return train_data, test_data

    def plot_img(self, class_name):
        """plot dataset images before training
        :arg
        class_name  : class names to plot the images from
        """
        try:
            df = self.dataset[self.dataset['class_name'].values == class_name]
            assert (df.shape[0] > 0), "Dataset must contain a classname given as 'class_name'"
        except ValueError:
            return None

        np.random.seed(None)
        images = np.random.choice(df.path.values, size=(10,), replace=False)
        img_data = [Image.open(os.path.join(self.img_dir, i)) for i in images]

        fig, axes = plt.subplots(2, 5, figsize=(12, 12))
        fig.suptitle(class_name, fontsize=25)
        axes = axes.flatten()
        for img, ax in zip(img_data, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        type=str,
                        help='log directory to store logs',
                        default='/media/jay/data/Dataset/object_detection/CUB_200_2011/logs')
    parser.add_argument('--img_dir',
                        type=str,
                        help='base path for the img directory',
                        default='/media/jay/data/Dataset/object_detection/CUB_200_2011')
    parser.add_argument('--dataset_csv',
                        type=str,
                        help='dataset csv path containing columns {class_id, class_name, path}',
                        default='/media/jay/data/Dataset/object_detection/CUB_200_2011/dataset.csv')
    parser.add_argument('--frac',
                        type=float,
                        help='fraction to split the dataset',
                        default=0.8)
    return parser.parse_args(argv)


if __name__ == '__main__':
    data = Dataset(parse_arguments(sys.argv[1:]))
    data.get_class_info()
    data.write_args()
    train, test = data.split_dataset()
    data.plot_img('Horned_Lark')