from tensorflow.contrib.tensorboard.plugins import projector
import scipy.misc
import os
import numpy as np
import tensorflow as tf


def summary_embedding_with_labels(sess, dataset, labels, input_placeholder, summary_dir, image_size,
                                  channel=3, batch_size=100):
    summary_embedding(sess, dataset, image_size, channel, summary_dir, labels)


def summary_embedding_no_labels(sess, dataset, input_placeholder, summary_dir, image_size,
                                argmax_op=None, channel=3, batch_size=100):
    summary_embedding(sess, dataset, image_size, channel, summary_dir)


def summary_embedding(sess, dataset, image_size, channel, output_path, labels=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    summary_writer = tf.summary.FileWriter(output_path, sess.graph)

    config = projector.ProjectorConfig()

    # enbed_dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # embed_tensor = tf.Variable(enbed_dataset, name='dataset')
    # embed_tensors.append(embed_tensor)
    # sess.run(embed_tensor.initializer)

    tensor_name = make_embed_tensor(sess, dataset, image_size)
    write_projector_config(config, tensor_name, output_path, image_size, channel, summary_writer, labels)
    #
    # for embed_idx, embed_vectors in enumerate(layer_outputs):
    #     tensor_name = make_embed_tensor(sess, embed_vectors, embed_idx, embed_tensors)
    #     write_projector_config(config, tensor_name, output_path, image_size, channel, summary_writer, labels)
    summary_writer.close()

    save_model(sess, None, output_path)

    # Make sprite and labels.
    make_sprite(dataset, image_size, channel, output_path)
    if labels is not None and len(labels) > 0:
        make_metadata(labels, output_path)


def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def make_sprite(dataset, image_size, channel, output_path):
    if channel == 1:
        images = np.array(dataset).reshape((-1, image_size, image_size)).astype(np.float32)
    else:
        images = np.array(dataset).reshape((-1, image_size, image_size, channel)).astype(np.float32)
    sprite = images_to_sprite(images)
    scipy.misc.imsave(os.path.join(output_path, 'sprite.png'), sprite)


def make_metadata(labels, output_path):
    labels = np.array(labels).flatten()
    metadata_file = open(os.path.join(output_path, 'labels.tsv'), 'w')
    metadata_file.write('Name\tClass\n')
    for i in range(len(labels)):
        metadata_file.write('%06d\t%d\n' % (i, labels[i]))
    metadata_file.close()


def make_embed_tensor(sess, dataset, image_size):
    enbed_dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    embed_tensor = tf.Variable(enbed_dataset, name='embed')
    # sess.run(embed_tensor.initializer)
    return embed_tensor.name


def write_projector_config(config, tensor_name, output_path, image_size, channel, summary_writer, labels):
    embedding = config.embeddings.add()
    embedding.tensor_name = tensor_name
    if labels is not None and len(labels) > 0:
        embedding.metadata_path = os.path.join(output_path, 'labels.tsv')
    embedding.sprite.image_path = os.path.join(output_path, 'sprite.png')
    if channel == 1:
        embedding.sprite.single_image_dim.extend([image_size, image_size])
    else:
        embedding.sprite.single_image_dim.extend([image_size, image_size, channel])
    projector.visualize_embeddings(summary_writer, config)


def save_model(sess, embed_tensors, output_path):
    # sess.run(embed_tensors)
    # saver = tf.train.Saver(embed_tensors)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(output_path, 'model_embed.ckpt'))
