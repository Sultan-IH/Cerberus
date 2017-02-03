import tensorflow as tf
import numpy as np


class Cost:
    def __init__(self, params):

        self.params = params

    def l2_reg(self, LAMBDA):
        reg_params = [tf.nn.l2_loss(param[0]) for param in self.params]
        l2_reg = tf.divide(sum(reg_params) * LAMBDA, len(self.params))
        return l2_reg
