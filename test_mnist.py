import tensorflow as tf
import embedder
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
BATCH_SIZE = 64

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_path', "/tmp/embed_test", "path")

if not os.path.exists(FLAGS.test_path):
    os.makedirs(FLAGS.test_path)

data_sets = input_data.read_data_sets(FLAGS.test_path, validation_size=BATCH_SIZE)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, os.path.join(FLAGS.test_path, 'model.ckpt'))

batch_dataset, batch_labels = data_sets.validation.next_batch(BATCH_SIZE)
batch_dataset = batch_dataset.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)

embedder.summary_embedding_with_labels(sess, batch_dataset, batch_labels, FLAGS.test_path, IMAGE_SIZE, NUM_CHANNELS)
