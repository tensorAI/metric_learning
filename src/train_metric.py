# -*- coding: utf-8 -*-
"""
training script for the metric learning training using centroids / agent vectors

TODO: Add read me for
    - Augmentations utils
    - Dataset utils
    - Running a classifier
    - Model export/restore and fine tuning
    - Model Export for serving
    - Model inference
TODO: make train_utils.py and log all the optimizers / lr_schedules
TODO: Label Smoothing
TODO: warm up steps
TODO: GCP support

"""
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import sys
import os
import time
import argparse
import augmentation_utils as augment
import dataset_utils as data
import model_def
import train_utils as utils

from tqdm import tqdm


def prepare_data_for_training(args):
    """ make dataset ready for training and validation"""
    # Form the train/test splits and write them to disk
    dataset = data.Dataset(args)
    # get image classes and image counts in each class
    label_map = dataset.get_class_info()
    class_count = len(list(label_map.values()))
    # split the data and store it in log dir
    df_train, df_test = dataset.split_dataset()

    # perform dataset augmentations
    image_data = augment.Augmentation(args)
    # get the data gens for training and test images
    train_data_gen, _ = image_data.map_fn_train(df_train)
    test_data_gen, _ = image_data.map_fn_test(df_test)

    return train_data_gen, test_data_gen, df_train, df_test, class_count


def main(args):
    """this function will train the base model and classifier head"""
    # create log dir
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    def _write_arguments_to_file(filename):
        with tf.io.gfile.GFile(filename, 'w') as f:
            for key, value in vars(args).items():
                f.write('%s: %s\n' % (key, str(value)))

    _write_arguments_to_file(os.path.join(args.log_dir, 'arguments.txt'))

    # get the dataset for training
    train_data_gen, test_data_gen, df_train, df_test, class_count = prepare_data_for_training(args)
    steps_per_epoch = int(np.ceil(df_train.shape[0] / args.batch_size))

    # get the model data and centers
    model = model_def.metric_learning(args)
    # initiate the centers array
    centers = tf.Variable(initial_value=tf.zeros([class_count, 2], dtype=tf.float32), dtype=tf.float32)

    # define learning rate schedule
    learning_rate = utils.configure_schedule(args, steps_per_epoch)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # for checkpoints
    checkpoint_directory = os.path.join(args.log_dir, 'tf_ckpts')
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_directory, max_to_keep=3)

    # define summary writers
    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, 'train'))

    for i in range(args.epochs):
        with train_summary_writer.as_default():
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")
            train(train_data_gen,
                  model,
                  centers,
                  optimizer,
                  ckpt,
                  manager,
                  i + 1,
                  steps_per_epoch)
            train_summary_writer.flush()

    # final checkpoint save
    ckpt.step.assign(tf.cast(optimizer.iterations, tf.int32))
    print("Saved Checkpoint for Step {}: {}".format(
        int(ckpt.step), manager.save(checkpoint_number=int(ckpt.step))))

    # model export for serving
    export_prefix = os.path.join(args.log_dir, 'export-{}'.format(optimizer.iterations.numpy()))
    if not os.path.exists(export_prefix):
        os.mkdir(export_prefix)
    tf.saved_model.save(model, export_prefix)


def train(train_data_gen,
          model,
          centers,
          optimizer,
          checkpoint,
          manager,
          epoch,
          nofsteps,
          save_ckpt_freq=100,
          log_freq=25):
    """train network for an epoch
    :arg
        - train_data_gen    : training data generator
        - model             : keras model
        - optimizer         : optimizer object from keras
        - loss_fn           : loss function object
        - checkpoint        : keras checkpoint object
        - manager           : checkpoint manager object from keras
        - epoch             : epoch number
        - nofsteps          : total number of steps in an epoch
        - save_ckpt_freq    : checkpoints are saved after _ steps
        - log_freq          : steps after summary is written in an event file
    :returns
        None
        Train a network for an epoch.
    """
    avg_loss = tf.keras.metrics.Mean(name='total_loss', dtype=tf.float32)
    for steps in tqdm(range(nofsteps), desc='Training Epoch {}'.format(epoch)):
        start_time = time.time()
        images, labels = next(train_data_gen)
        metric_loss, centers = train_step(images, labels, model, centers, optimizer)
        duration = time.time() - start_time
        # update loss states
        avg_loss.update_state(metric_loss)

        # save running speed for every iterations
        tf.summary.scalar('train/learning_rate',
                          optimizer.learning_rate(optimizer.iterations),
                          step=optimizer.iterations)
        tf.summary.scalar('train/time_per_step',
                          duration,
                          step=optimizer.iterations)

        if tf.equal(optimizer.iterations % log_freq, 0) or tf.equal(optimizer.iterations % save_ckpt_freq, 0) or \
                tf.equal(optimizer.iterations % (nofsteps - 1), 0):
            tf.summary.scalar('train/loss/total_loss', avg_loss.result(), step=optimizer.iterations)
            checkpoint.step.assign(tf.cast(optimizer.iterations, tf.int32))
            print('\n Global Step : %s \t [ Epoch / Step ]  %s / %s  \t Training Loss: %f'
                  '\t Learning Rate: %f' % (int(checkpoint.step), epoch, steps + 1, float(metric_loss),
                                            optimizer.learning_rate(optimizer.iterations).numpy()))
            # save checkpoints
            print("Saved Checkpoint for Step {}: {}".format(
                int(checkpoint.step), manager.save(checkpoint_number=int(checkpoint.step))))
            avg_loss.reset_states()


@tf.function
def train_step(images,
               labels,
               model,
               centers,
               optimizer):
    """train model for a step
    :arg
        - images    : image data
        - labels    : label data
        - model     : keras model object
        - loss_fn   : loss function
        - optimizer : optimizer object in keras
    """
    with tf.GradientTape(persistent=True) as tape:
        _, _, _, embeddings, _ = model([images,
                                        labels],
                                       training=True)
        lab_with_centers = tf.concat([labels, labels], axis=0)
        # combine features and centers to computer a loss number
        center_batch = tf.gather(centers, labels)
        feat_with_centers = tf.concat([embeddings, tf.squeeze(center_batch)], axis=0)
        # metric learning loss
        metric_loss = tf.reduce_sum(tf.square(embeddings - tf.squeeze(center_batch)))
        # metric_loss = tfa.losses.lifted_struct_loss(lab_with_centers,
        #                                            feat_with_centers,
        #                                            margin=0.5)

    gradients = tape.gradient(metric_loss, model.trainable_variables)
    processed_grads = [tf.clip_by_norm(g, 4.0) for g in gradients]
    optimizer.apply_gradients(zip(processed_grads, model.trainable_variables))

    # track the moving averages of trainable variables
    variable_averages = tf.Graph().get_collection('trainable_variables')
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    ema.apply(variable_averages)

    return metric_loss, centers


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        type=str,
                        help='log directory to store logs',
                        default='/media/jay/rnd/Recognition_Datasets/MNIST/metric')
    parser.add_argument('--img_dir',
                        type=str,
                        help='base path for the img directory',
                        default='/media/jay/rnd/Recognition_Datasets/MNIST/images')
    parser.add_argument('--frac',
                        type=float,
                        help='fraction to split the dataset',
                        default=0.8)
    parser.add_argument('--image_depth',
                        type=int,
                        help='Final image size (height or width) in pixels.',
                        default=1)
    parser.add_argument('--image_size',
                        type=int,
                        help='Number of channels images have',
                        default=28)
    parser.add_argument('--feature_dims',
                        type=int,
                        help='feature dimensions for metric learning training',
                        default=2)

    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size for the mini batch of images',
                        default=256)

    parser.add_argument('--lr_rate',
                        type=float,
                        help='initial learning rate to train network',
                        default=1e-6)
    parser.add_argument('--lr_schedule',
                        type=str,
                        help='Learning rate schedule',
                        default='polynomial')
    parser.add_argument('--drop_rate',
                        type=float,
                        help='drop out rate for the CNN',
                        default=0.2)
    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to run training for',
                        default=5)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
