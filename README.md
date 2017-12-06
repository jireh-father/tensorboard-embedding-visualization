# tensorboard-embedding-visualization
Easily visualize embedding on tensorboard with thumbnail images and labels.

Currently this repo is compatible with Tensorflow r1.0.1

![alt text](https://raw.githubusercontent.com/jireh-father/tensorboard-embedding-visualization/master/mnist_embedding_visualization.jpg)


## Getting Started

```python
import embedder

# create the model graph and get the last layer's output.
logits = model()

# init session and restore pre-trained model file
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, os.path.join(test_path, 'model.ckpt'))

# read your dataset and labels
data_sets, labels = read_data_sets()

# run your model
feed_dict = {input_placeholder: dataset, label_placeholder: labels}
activations = sess.run(logits, feed_dict)

embedder.summary_embedding(sess=sess, dataset=data_sets, embedding_list=[activations],
                                       embedding_path="your embedding path", image_size=your_image_size, channel=3,
                                       labels=labels)
```


### If you want to use large data.
```python
import embedder

# create the model graph
logits = model()

# init session and restore pre-trained model file
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, os.path.join(test_path, 'model.ckpt'))

total_dataset = None
total_labels = None
total_activations = None
for i in range(10)
    data_sets, labels = get_batch(i)
    feed_dict = {input_placeholder: dataset, label_placeholder: labels}
    activations = sess.run(logits, feed_dict)
    if total_dataset is None:
      total_dataset = data_sets
      total_labels = labels
      total_activations = activations
    else:
      total_dataset = np.append(data_sets, total_dataset, axis=0)
      total_labels = np.append(labels, total_labels, axis=0)
      total_activations = np.append(activations, total_activations, axis=0)

embedder.summary_embedding(sess=sess, dataset=total_dataset, embedding_list=[total_activations],
                                       embedding_path="your embedding path", image_size=your_image_size, channel=3,
                                       labels=total_labels)
```

---


## Running mnist test

```shell
python test_mnist.py
(python test_mnist_large_data.py)
tensorboard --log_dir=./
```

This should print that TensorBoard has started. Next, connect http://localhost:6006 and click the EMBEDDINGS menu.


---


## API Reference

```python
def summary_embedding(sess, dataset, embedding_list, embedding_path, image_size, channel=3, labels=None):
    pass

```


---


## Acknowledgments
http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
https://github.com/tensorflow/tensorflow/issues/6322


