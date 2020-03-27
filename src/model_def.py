"""
    define convolution neural network(CNN) graph for the model training
"""
import tensorflow as tf

from tensorflow.keras import layers


def metric_learning(args, feature_dims=2):
    """ keras model definition for metric learning training
           :arg
               args        : argument namespace
               feature_dims : feature dimensions to be used
           :returns
               model   : model built using hub modules for metric learning application
    """

    # Functional API
    # define inputs
    image = tf.keras.Input(shape=(args.image_size, args.image_size, args.image_depth), name='image')
    image_id = tf.keras.Input(shape=(), dtype=tf.int32, name='image_id')

    # --------------------define base model ----------------------- #
    x = layers.Flatten()(image)
    x = layers.Dense(128, activation='relu')(x)
    feature_extractor_layer = layers.Dense(128, activation=None, use_bias=False, name="base_features")(x)
    feature_extractor_layer_l2 = tf.math.l2_normalize(feature_extractor_layer,
                                                      axis=1,
                                                      epsilon=1e-12,
                                                      name="base_features_l2")
    head_layer = layers.Dense(feature_dims,
                              activation=None,
                              use_bias=False,
                              name="features")(feature_extractor_layer)
    head_layer_l2 = tf.math.l2_normalize(head_layer,
                                         axis=1,
                                         epsilon=1e-12,
                                         name="head_features_l2")

    model = tf.keras.Model(inputs=[image, image_id],
                           outputs=[image_id,
                                    feature_extractor_layer,
                                    feature_extractor_layer_l2,
                                    head_layer,
                                    head_layer_l2])
    model.summary()

    return model
