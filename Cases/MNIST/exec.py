from Vengine import Network, DenseLayer, SGD_engine, CrossEntropy
import Vengine.main.train as v8
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../Cases/MNIST/MNIST_data/", one_hot=True)

batch_xs, batch_ys = mnist.train.next_batch(50000)

Train_data = zip(batch_xs, batch_ys)

layers = [
    DenseLayer([784, 1000]),
    DenseLayer([1000, 10])

]

model = Network(SGD_engine(CrossEntropy), layers)

v8.train(model, 30, Train_data=Train_data, Test_data=[mnist.test.images, mnist.test.lables], batch_size=32)
