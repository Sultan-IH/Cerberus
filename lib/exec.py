from lib.Engines.SGD_engine import SGD_engine
from  lib.Costs.CrossEntropy import CrossEntropy
from lib.main.FF_Network_Constructor import NetworkConstructor as Builder
from lib.Layers.ConvLayer import ConvLayer

layers = [ConvLayer(filter_size=(5, 5, 1, 15))]
net = Builder()
net.fit_layers(layers)
net.fit_engine(SGD_engine(CrossEntropy))
net.construct()
