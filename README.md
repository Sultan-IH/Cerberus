# Cerberus
The AI core for any Machine Learning Project. This Project is intended for machine learning research and implementation.
Essentially offers a high level api for TensorFlow.
# Library features:
* Convolutional, Deep, Recurrent and Deep Neural Networks (with hybrids)

* A variety of Layers

* Ability to save, load and train models at any time

* Docker containerization support

* Off-site training on a cluster of servers and support for multiple GPU training

* Interactive Python shell/ IPython notebook server support
# Sample use

```python
from Vengine.Layers import DenseLayer, ConvLayer, PoolLayer
from Vengine.main import train, Network, save
from Vengine.Engines import Adam_engine
from Vengine.Costs import CrossEntropy

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("../Cases/MNIST/MNIST_data/", one_hot=True)

batch_xs, batch_ys = mnist.train.next_batch(50000)

layers = [
    ConvLayer((5, 5, 1, 15)),
    PoolLayer(),
    ConvLayer((5, 5, 15, 30)),
    PoolLayer(),
    DenseLayer([7 * 7 * 30, 250])  # the shape of this tensor was the problem

]

data_sets = {
    "Train_data": [batch_xs, batch_ys],
    "Test_data": [mnist.test.images, mnist.test.labels],
    "Validation_data": False
}

data_sets["Train_data"][0] = np.reshape(data_sets["Train_data"][0], [-1, 28, 28, 1])
data_sets["Test_data"][0] = np.reshape(data_sets["Test_data"][0], [-1, 28, 28, 1])

model = Network(layers=layers)

model.add_layer(DenseLayer([250, 10]), 5)
model.fit_engine(engine=Adam_engine(CrossEntropy, lr=1e-3))

train(net=model, data_sets=data_sets, epochs=30, batch_size=32)
save(sess=model.sess, name="my_awesome_model")
```