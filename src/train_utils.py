"""
    training utility function to improve the training performance

"""

import tensorflow as tf


def configure_schedule(args,
                       steps_per_epoch):
    """Configure optimizer learning rate schedule
    :arg
        - args  - argument namespace
    :returns
        - learning_rate_fn - learning rate schedule function for the optimizer
    """
    if args.lr_schedule == "exponential":
        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr_rate,
            decay_steps=args.epochs * steps_per_epoch,
            decay_rate=0.96,
            staircase=False)
    elif args.lr_schedule == "piecewise_constant":
        boundaries = [500]
        values = [0.01, 0.001]
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)
    elif args.lr_schedule == "inverse_time":
        decay_steps = 1.0  # How often to apply decay
        decay_rate = 0.5  # The decay rate
        learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
            args.lr_rate, decay_steps, decay_rate)
    elif args.lr_schedule == "polynomial":
        decay_steps = args.epochs * steps_per_epoch
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            args.lr_rate,
            decay_steps,
            end_learning_rate=0.00007,
            power=3.0,
            cycle=False)
    elif args.lr_schedule == "cosine":
        decay_steps = args.epochs * steps_per_epoch
        learning_rate_fn = tf.keras.experimental.CosineDecay(
            args.lr_rate, decay_steps, alpha=0.05)
    elif args.lr_schedule == "linear_cosine":
        decay_steps = args.epochs * steps_per_epoch
        learning_rate_fn = tf.keras.experimental.LinearCosineDecay(
            args.lr_rate, decay_steps, num_periods=2, alpha=0.0, beta=0.001)
    elif args.lr_schedule == "cosine_with_restarts":
        decay_steps = args.epochs * steps_per_epoch
        learning_rate_fn = tf.keras.experimental.CosineDecayRestarts(
            args.lr_rate, decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0)
    elif args.lr_schedule == "fixed":
        learning_rate_fn = args.lr_rate
    else:
        raise ValueError("Not a valid learning rate schedule. Input valid argument --lr_schedule.")
    return learning_rate_fn

# call optimizers through args
