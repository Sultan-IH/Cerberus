import tensorflow as tf
from SRC.Costs.BaseCostClass import Cost


class CrossEntropy(Cost):
    def __init__(self, Y, y_, params):
        self.Y = Y
        self.y_ = y_
        self.params = params

    def cost_op(self):
        l2_weight_reg = [self.l2_reg(param) for param in self.params]
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.Y, self.y_) + tf.divide(l2_weight_reg, len(self.params)))
        return cost


