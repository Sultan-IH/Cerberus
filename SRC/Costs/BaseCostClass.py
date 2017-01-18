import tensorflow as tf
import numpy as np

class Cost():
    @staticmethod
    def l2_reg(param):
        return tf.nn.l2_loss(param)