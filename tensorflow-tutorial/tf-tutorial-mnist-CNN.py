from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np


def cnn_model_fn(features, labels, mode):
    tf.random.set_random_seed(1234)
    img_input = tf.reshape(features["x"], [100, 28, 28, 1])

    # Input layer - here as a conv2d layer #1
    conv1 = tf.layers.conv2d(inputs=img_input, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Dropout layer #1
    dropout1 = tf.layers.dropout(inputs=conv1, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN, seed=(1234))

    # Conv2d layer #2
    conv2 = tf.layers.conv2d(inputs=img_input, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # Dropout layer #2
    dropout2 = tf.layers.dropout(inputs=conv1, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN, seed=(1234))

    # Dense layer
    dense = tf.layers.dense(inputs=dropout2, units = 1024, activation = tf.nn.relu)

    # Logits layer
    logits = tf.layers.dense(inputs=dense, units=10)

    # Generate predictions
    tf.argmax(input=logits, axis=1)

    # Apply the softmax activation function
    tf.nn.softmax(logits, name="softmax_tensor")

    # Compile the predictions in a dict and return EstimatorSpec as object
    predictions = {
        'class': tf.arg_max(input=logits, axis=1), 'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Training sgd
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation - accuracy metrics
    eval_metrice = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrice)
