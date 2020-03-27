""" project features on 2D plane using PCA """

import pickle
import itertools
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn import preprocessing


class ProjectFeatures:
    """ a class to project features"""

    def __init__(self):
        pass

    @staticmethod
    def read_pickle_file(pickle_file_path,
                         feature_node,
                         class_name_node):
        """ read the pickle file to be used for PCA
        Args:
            pickle_file_path: pickle file path to read
            feature_node:   node to be used for PCA e.g embedding / logit etc
            class_name_node: node corresponding to class name
        Returns:
              features: features extracted from pickle file
              class_names: class names corresponding to features extracted
        """
        # read feature values and class  from pickle file
        with open(pickle_file_path, 'rb') as file:
            feature_bank = pickle.load(file)
        class_name, feature = [], []
        for _, values in itertools.islice(feature_bank.items(), len(feature_bank)):
            feature.append(values[feature_node]),
            class_name.append(values[class_name_node])
        return np.asarray(feature), np.asarray(class_name)

    @staticmethod
    def perform_pca(features,
                    components=2):
        """ analyse pca and compress the dimensions to given components
        Args:
            features: features extracted from model
                      to be compressed in terms of dimensions
            components: dimensions to be reduced to
        Return:
              features_pca : features after reducing the dimensions
        """
        pca = PCA(n_components=components)
        features_pca = pca.fit(features).transform(features)
        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio_))
        return features_pca

    @staticmethod
    def plot_pca(features_pca,
                 class_names):
        """ Given features and class names it wil project the features on 2D space
        Args:
            features_pca:    features extracted from a model
            class_names:    class names corresponding to each class
        Return:
              plot 2d graph for the features
        """
        plt.figure()
        encoder = preprocessing.LabelEncoder()
        encoder.fit(class_names)
        # get the unique class names for each data point
        target_names = list(map(int, encoder.classes_))
        label = encoder.transform(class_names)
        # get colors for each class name
        colors = cm.rainbow(np.linspace(0,
                                        class_names.shape[0] // len(target_names),
                                        class_names.shape[0]))
        for color, i, target_name in zip(colors, range(len(target_names)), target_names):
            plt.scatter(features_pca[label == i, 0],
                        features_pca[label == i, 1],
                        color=color,
                        alpha=.8,
                        lw=1,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, linestyle='--')
        plt.xlim(-np.amax(np.absolute(features_pca)), np.amax(np.absolute(features_pca)))
        plt.ylim(-np.amax(np.absolute(features_pca)), np.amax(np.absolute(features_pca)))
        plt.title('PCA for Bottleneck Features')
        plt.show()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        '--pickle_file_path',
        type=str,
        help='absolute pickle file path containing all the feature information',
        default="/media/jay/rnd/Recognition_Datasets/MNIST/metric/test_embeddings.pkl")
    PARSER.add_argument(
        '--feature_node',
        type=str,
        help='output node corresponding to image embeddings',
        default="head_embeddings")
    PARSER.add_argument(
        '--class_node',
        type=str,
        help='output node corresponding to feature classes',
        default="class_name")
    PARSER.add_argument(
        '--n_components',
        type=int,
        help='number of principle components to reduce the dimensions to',
        default=2)

    ARGS = PARSER.parse_args()

    # get the projection class
    PROJECT_FEATURES = ProjectFeatures()

    FEATURES, LABELS = PROJECT_FEATURES.read_pickle_file(ARGS.pickle_file_path,
                                                         ARGS.feature_node,
                                                         ARGS.class_node)

    FEATURES_2D = PROJECT_FEATURES.perform_pca(FEATURES,
                                               ARGS.n_components)

    PROJECT_FEATURES.plot_pca(FEATURES_2D, LABELS)
