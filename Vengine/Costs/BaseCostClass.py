import tensorflow as tf
import numpy as np


class Cost():
    def l2_reg(self, params, LAMBDA):
        reg_params = [tf.nn.l2_loss(param) for param in params]
        l2_reg = tf.divide(sum(reg_params) * LAMBDA, len(params))
        return l2_reg
