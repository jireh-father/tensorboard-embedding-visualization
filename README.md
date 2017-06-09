# tensorboard-embedding-visualization
Visualize embedding on tensorboard with thumbnail images.

![alt text](https://raw.githubusercontent.com/jireh-father/tensorboard-embedding-visualization/master/mnist_embedding_visualization.jpg)


## Getting Started

### 1. Load your model code, model file and dataset.
```python
# load model code
model()

# init session and restore pre-trained model file
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, os.path.join(test_path, 'model.ckpt'))

# load you dataset
data_sets = read_data_sets()
```


### 2. Just import embedder.py and call summary_embedding_with_labels or summary_embedding_no_labels.
```python
import embedder

# if you have labels.
embedder.summary_embedding_with_labels(sess, batch_dataset, batch_labels, test_path, IMAGE_SIZE, NUM_CHANNELS)

# if you want embedding other layers.
input_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
logits = model()
embedder.summary_embedding_with_labels(sess, batch_dataset, batch_labels, test_path, IMAGE_SIZE, NUM_CHANNELS, input_placeholder=input_placeholder, layer_op_list=[logits])

# if you have no labels.
embedder.summary_embedding_no_labels(sess, batch_dataset, test_path, IMAGE_SIZE, NUM_CHANNELS)

# if you have no labels, but you want to get the result to labels.
input_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
embedder.summary_embedding_test(sess, batch_dataset, input_placeholder, argmax_op, test_path, IMAGE_SIZE, NUM_CHANNELS)
```


## Running the tests

```shell
python test_mnist.py
tensorboard --log_dir=./
```

This should print that TensorBoard has started. Next, connect http://localhost:6006 and click the EMBEDDINGS menu.


## API Reference

```python
def summary_embedding_with_labels(sess, dataset, labels, summary_dir, image_size, channel=3, batch_size=64, input_placeholder=None, layer_op_list=None):
    pass

def summary_embedding_no_labels(sess, dataset, summary_dir, image_size, channel=3, batch_size=64, input_placeholder=None, layer_op_list=None):
    pass

def summary_embedding_test(sess, dataset, input_placeholder, argmax_op, summary_dir, image_size, channel=3, batch_size=64, layer_op_list=None):
```


## Acknowledgments
http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
https://github.com/tensorflow/tensorflow/issues/6322


