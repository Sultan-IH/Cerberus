from Vengine import Network, DenseLayer, Adam_engine, CrossEntropy, ConvLayer, PoolLayer
import Vengine.main.train as v8
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../Cases/MNIST/MNIST_data/", one_hot=True)

batch_xs, batch_ys = mnist.train.next_batch(50000)

layers = [
    ConvLayer(filter_size=(5, 5, 1, 15)),
    DenseLayer([784, 1000]),
    DenseLayer([1000, 10])

]

data_sets = {
    "Train_data": [batch_xs, batch_ys],
    "Test_data": [mnist.test.images, mnist.test.labels],
    "Validation_data": False

}

model = Network(Adam_engine(CrossEntropy, lr=1e-3), layers)

v8.train(net=model, data_sets=data_sets, epochs=30, batch_size=32, log=True)
