#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a custom Estimator for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    # Fetch the data
    data = [[-3, -2,-1], [-5,-4,-3], [5,6,7],
          [-4.5, -2, -3], [9, 11, 10], [16, 25, 20],
          [-8, -10, -2], [9, 2, 19], [10, 2, 25],
          [10, 2, 6], [100,10, 50], [25, -10, 10],
          [-5, 5, -10], [8, 20, -6], [20, 40, 10],
          [10, 2, -5], [8, 2, -10], [100, 50, -50]]
    target = [0,0,0,
           1,1,1,
           2,2,2,
           3,3,3,
           4,4,4,
           5,5,5]

    # Feature columns describe how to use the input.
    #my_feature_columns = []
    my_feature_columns = [tf.feature_column.numeric_column("x", shape=[3])]
    #for key in train_x.keys():
    #    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 6,
        })

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(data)},
      y=np.array(target),
      num_epochs=None,
      shuffle=True)
    # Train the Model.
    classifier.train(
        #input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
        input_fn=train_input_fn,
        steps=1000)

    # Evaluate the model.
    #eval_result = classifier.evaluate(
    #    input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(data)},
      y=np.array(target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)

    #print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
    new_samples = np.array(
      [[6.4, 3.2, 4.5],
       [5.8, 3.1, 55.0]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["class_ids"] for p in predictions]

    print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))
    print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
