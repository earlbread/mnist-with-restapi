"""A simple MNIST classifier.
"""
import argparse
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

def train(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

    # Save trained model
    saver = tf.train.Saver()
    saver.save(sess, FLAGS.save_path,
               write_meta_graph=False, write_state=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')

    default_save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data')

    if not os.path.exists(default_save_dir):
        os.makedirs(default_save_dir)

    default_save_path = os.path.join(default_save_dir, 'regression')

    parser.add_argument('--save_path',
                        type=str,
                        default=default_save_path,
                        help='Path for storing trained model')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
