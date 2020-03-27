"""
    utilities to sample positive / negative or centroids for the given anchor features

"""

import tensorflow as tf


class SampleCentroids:

    def __init__(self, feature_dims, class_count):
        self.feature_dims = feature_dims
        self.class_count = class_count
        self.array_shape = (self.class_count, self.feature_dims)
        self.centers = tf.Variable(initial_value=tf.random.normal(shape=self.array_shape, mean=0.0, stddev=0.01),
                                   dtype=tf.float32,
                                   trainable=True)

    def get_centroids(self, label):
        label = tf.cast(label, dtype=tf.int32)
        centers_batch = tf.gather(self.centers, label)
        return centers_batch
