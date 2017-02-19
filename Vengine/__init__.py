__version__ = 0.1

# TODO: some layers have different dimensions so the data would require flattening first
# TODO: implement dropout
# TODO: finish load save and join
# TODO:figure out how to properly set out the init
# TODO: stop when accuracy reaches a peak
# TODO: training depending on the number of gpus

__all__ = ['Network', 'load', 'CrossEntropy', 'SGD_engine', 'ConvLayer', 'DenseLayer', 'train', 'Adam_engine',
           'PoolLayer']

from Vengine.main.Network import Network
from Vengine.main.load import load
from Vengine.main.train import train
from Vengine.Costs.CrossEntropy import CrossEntropy
from Vengine.Layers.ConvLayer import ConvLayer
from Vengine.Layers.PoolLayer import PoolLayer
from Vengine.Layers.DenseLayer import DenseLayer
from Vengine.Engines.SGD_engine import SGD_engine
from Vengine.Engines.Adam_engine import Adam_engine
