from tensorflow.contrib.tensorboard.plugins import projector
import scipy.misc
import os
import numpy as np
import tensorflow as tf


def summary_embedding_with_labels(sess, dataset, labels, summary_dir, image_size, channel=3, batch_size=64,
                                  input_placeholder=None, layer_op_list=None):
    if len(dataset.shape) == 2:
        dataset = dataset.reshape((-1, image_size * image_size * channel))
        
    if layer_op_list is not None:
        layer_outputs = run_layers(sess=sess, layer_op_list=layer_op_list, images=dataset,
                                   input_placeholder=input_placeholder, batch_size=batch_size)

        embeddings = [dataset] + layer_outputs

    else:
        embeddings = [dataset]

    summary_embedding(sess, embeddings, image_size, channel, summary_dir, labels)


def summary_embedding_no_labels(sess, dataset, summary_dir, image_size, channel=3, batch_size=64,
                                input_placeholder=None, layer_op_list=None):
    if len(dataset.shape) == 2:
        dataset = dataset.reshape((-1, image_size * image_size * channel))

    if layer_op_list is not None:
        layer_outputs = run_layers(sess=sess, layer_op_list=layer_op_list, images=dataset,
                                   input_placeholder=input_placeholder, batch_size=batch_size)

        embeddings = [dataset] + layer_outputs

    else:
        embeddings = [dataset]

    summary_embedding(sess, embeddings, image_size, channel, summary_dir)


def summary_embedding_test(sess, dataset, input_placeholder, argmax_op, summary_dir, image_size, channel=3,
                           batch_size=64, layer_op_list=None):
    if len(dataset.shape) == 2:
        dataset = dataset.reshape((-1, image_size * image_size * channel))

    layer_outputs, labels = run_layers_test(sess=sess, layer_op_list=layer_op_list, images=dataset,
                                            input_placeholder=input_placeholder, argmax_op=argmax_op,
                                            batch_size=batch_size)

    if layer_outputs is not None and len(layer_outputs) > 0:
        embeddings = [dataset] + layer_outputs
    else:
        embeddings = [dataset]

    summary_embedding(sess, embeddings, image_size, channel, summary_dir, labels)


def summary_embedding(sess, embedding_list, image_size, channel, output_path, labels=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    summary_writer = tf.summary.FileWriter(output_path, sess.graph)

    config = projector.ProjectorConfig()

    for embed_idx, embed_vectors in enumerate(embedding_list):
        embed_tensor = make_embed_tensor(sess, embed_vectors, image_size, channel, embed_idx)
        write_projector_config(config, embed_tensor.name, output_path, image_size, channel, summary_writer, labels)

    summary_writer.close()

    save_model(sess, output_path)

    # Make sprite and labels.
    make_sprite(embedding_list[0], image_size, channel, output_path)
    if labels is not None and len(labels) > 0:
        make_metadata(labels, output_path)


def run_layers(sess, layer_op_list, images, input_placeholder, batch_size=32):
    layer_outputs = [[] for i in range(len(layer_op_list))]

    for i in range(0, len(images), batch_size):
        batch_data = images[i:i + batch_size]
        if len(batch_data) < batch_size:
            break
        feed_dict = {input_placeholder: batch_data}
        results = sess.run(layer_op_list, feed_dict=feed_dict)

        for j, layer_output in enumerate(layer_outputs):
            layer_output.append(results[j])

    return layer_outputs


def run_layers_test(sess, images, input_placeholder, argmax_op, layer_op_list=None, batch_size=32):
    if argmax_op is None:
        return None, None

    if layer_op_list is not None:
        layer_outputs = [[] for i in range(len(layer_op_list))]
    else:
        layer_outputs = None

    argmax_outputs = []

    for i in range(0, len(images), batch_size):
        batch_data = images[i:i + batch_size]
        if len(batch_data) < batch_size:
            break
        feed_dict = {input_placeholder: batch_data}
        if layer_op_list is not None:
            results = sess.run(layer_op_list + [argmax_op], feed_dict=feed_dict)
            for j, layer_output in enumerate(layer_outputs):
                layer_output.append(results[j])
        else:
            results = sess.run([argmax_op], feed_dict=feed_dict)

        argmax_outputs.append(results[-1])

    return layer_outputs, argmax_outputs


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
    min_data = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min_data).transpose(3, 0, 1, 2)
    max_data = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max_data).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
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


def make_embed_tensor(sess, embed_vectors, image_size, channel, embed_idx):
    if embed_idx == 0:
        embed_dataset = embed_vectors.reshape((-1, image_size * image_size * channel)).astype(np.float32)
        embed_tensor = tf.Variable(embed_dataset, name='embed')
    else:
        embed_tensor = tf.Variable(np.array(embed_vectors).reshape(len(embed_vectors) * embed_vectors[0].shape[0], -1),
                                   name=('embed_%s' % embed_idx))

    sess.run(embed_tensor.initializer)
    return embed_tensor


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


def save_model(sess, output_path):
    # saver = tf.train.Saver([embed_tensor])
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(output_path, 'model_embed.ckpt'))
