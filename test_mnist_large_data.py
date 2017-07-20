import tensorflow as tf
import embedder
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
BATCH_SIZE = 64

test_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(os.path.join(test_path, 'embedding')):
    os.makedirs(os.path.join(test_path, 'embedding'))


# 1. load model graph
def model():
    input_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
    fc1_weights = tf.Variable(
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1, dtype=tf.float32))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, dtype=tf.float32))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32))

    conv = tf.nn.conv2d(input_placeholder, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    return input_placeholder, tf.matmul(hidden, fc2_weights) + fc2_biases


input_placeholder, logits = model()

# 2. load dataset to visualize embedding
data_sets = input_data.read_data_sets(test_path, validation_size=BATCH_SIZE)

# 3. init session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 4. load pre-trained model file
saver = tf.train.Saver()
saver.restore(sess, os.path.join(test_path, 'model.ckpt'))

# 6. if you want to use large data
total_dataset = None
total_labels = None
total_activations = None
for i in range(10):
    batch_dataset, batch_labels = data_sets.validation.next_batch(BATCH_SIZE)
    feed_dict = {input_placeholder: batch_dataset.reshape([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])}
    activations = sess.run(logits, feed_dict)

    if not total_dataset:
        total_dataset = batch_dataset
        total_labels = batch_labels
        total_activations = activations
    else:
        total_dataset = np.append(batch_dataset, total_dataset, axis=0)
        total_labels = np.append(batch_labels, total_labels, axis=0)
        total_activations = np.append(activations, total_activations, axis=0)

embedder.summary_embedding(sess=sess, dataset=total_dataset, embedding_list=[total_activations],
                           embedding_path=os.path.join(test_path, 'embedding'), image_size=IMAGE_SIZE,
                           channel=NUM_CHANNELS, labels=total_labels)
