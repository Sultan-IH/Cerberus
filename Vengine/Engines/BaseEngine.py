import tensorflow as tf
import numpy as np


class Engine():
    """Base class for all engines"""

    def __init__(self, cost_class,lr):
        """Must check if the machine we are running on has multiple GPUs
        if so then create a tensorflow server and distribute the workflow"""
        self.cost_class = cost_class
        self.lr = lr
        print("Engine inited")

    def propagate(self, _dict, compute_op):
        """Use distrubuted workload when possible, need to find out more about that"""
        #NOT SURE IF NEEDED
        return compute_op.eval(feed_dict=_dict)
