import tensorflow as tf
from Vengine.Costs.BaseCostClass import Cost


class CrossEntropy(Cost):
    def __init__(self, params):

        self.params = params

    def cost_op(self, Y, y_, LAMBDA):
        l2_reg = self.l2_reg(params=self.params, LAMBDA=LAMBDA)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(Y, y_) + l2_reg)
        return cost
